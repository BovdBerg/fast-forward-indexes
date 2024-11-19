import argparse
import time
import warnings
from copy import copy
from pathlib import Path
from typing import List, Tuple

import ir_datasets
import numpy as np
import pyterrier as pt
import torch
from ir_measures import calc_aggregate, measures

from fast_forward.encoder.avg import W_METHOD, WeightedAvgEncoder
from fast_forward.encoder.transformer import TCTColBERTQueryEncoder
from fast_forward.index import Index
from fast_forward.index.disk import OnDiskIndex
from fast_forward.ranking import Ranking
from fast_forward.util import to_ir_measures
from fast_forward.util.pyterrier import FFInterpolate, FFScore


def parse_args():
    """
    Parse command-line arguments for the re-ranking script.

    Returns:
        argparse.Namespace: Parsed command-line arguments.

    Arguments:
        Run the script with --help or -h to see the full list of arguments.
    """
    parser = argparse.ArgumentParser(
        description="Re-rank documents based on query embeddings."
    )
    # TODO [final]: Remove default paths (index_path) form the arguments
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print more information during re-ranking.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="msmarco_passage",
        help="Dataset (using package ir-datasets).",
    )
    parser.add_argument(
        "--index_path",
        type=Path,
        default="/home/bvdb9/indices/msm-psg/ff_index_msmpsg_TCTColBERT_opq.h5",
        help="Path to the index file.",
    )
    parser.add_argument(
        "--in_memory", action="store_true", help="Whether to load the index in memory."
    )
    parser.add_argument(
        "--rerank_cutoff",
        type=int,
        default=1000,
        help="Number of documents to re-rank per query.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for encoding queries.",
    )
    # WeightedAvgEncoder
    parser.add_argument(
        "--w_method",
        type=W_METHOD,
        choices=list(W_METHOD),
        default="SOFTMAX_SCORES",
        help="Method to estimate query embeddings. Only used for EncodingMethod.WEIGHTED_AVERAGE.",
    )
    parser.add_argument(
        "--k_avg",
        type=int,
        default=30,
        help="Number of top-ranked documents to use. Only used for EncodingMethod.WEIGHTED_AVERAGE.",
    )
    parser.add_argument(
        "--avg_chains",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--avg_int_alphas",
        type=float,
        nargs="+",
        default=[0.1, 0.8, 0.9],
        help='List of interpolation "alpha" parameters we initialize the WeightedAvgEncoder chains with. If --avg_chains is larger than its length, the remaining values are initialized as 0.5.',
    )
    # VALIDATION
    parser.add_argument(
        "--val_pipelines",
        type=str,
        nargs="+",
        default=[],
        help="List of pipelines to validate, based on exact pipeline names.",
    )
    parser.add_argument(
        "--dev_sample_size",
        type=int,
        default=1024,
        help="Number of queries to sample for validation.",
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[round(x, 2) for x in np.arange(0, 1.0001, 0.1)],
        help="List of interpolation parameters for evaluation.",
    )
    parser.add_argument(
        "--dev_eval_metric",
        type=str,
        default="recip_rank",  # Find official metrics for dataset version on https://ir-datasets.com/msmarco-passage.html
        help="Evaluation metric for pt.GridSearch on dev set.",
    )
    # EVALUATION
    parser.add_argument(
        "--test_datasets",
        type=str,
        nargs="+",
        default=["irds:msmarco-passage/trec-dl-2019/judged"],
        help="Datasets to evaluate the rankings. May never be equal to dev_dataset (=msmarco_passage/dev or msmarco_passage/dev.small).",
    )
    parser.add_argument(
        "--eval_metrics",
        type=str,
        nargs="+",
        default=[
            "nDCG@10",
            "RR(rel=2)@10",
            "AP(rel=2)@10",
        ],  # Official metrics for TREC '19 according to https://ir-datasets.com/msmarco-passage.html#msmarco-passage/trec-dl-2019/judged
        help="Metrics used for evaluation.",
    )
    return parser.parse_args()


def print_settings() -> None:
    """
    Print general settings used for re-ranking.

    Args:
        pipeline (pt.Transformer): The pipeline used for re-ranking.
    """
    # General settings
    settings_description: List[str] = [
        f"verbose={args.verbose}",
        f"in_memory={args.in_memory}",
        f"rerank_cutoff={args.rerank_cutoff}",
        "TCTColBERTQueryEncoder:",
        f"\tdevice={args.device}",
        "WeightedAvgEncoder:",
        f"\tw_method={args.w_method.name}",
        f"\tk_avg={args.k_avg}",
        f"\tavg_chains={args.avg_chains}",
    ]
    # Validation settings
    settings_description.append(f"val_pipelines={args.val_pipelines}")
    if args.val_pipelines:
        settings_description[-1] += ":"
        settings_description.extend(
            [
                f"\tdev_sample_size={args.dev_sample_size}",
                f"\talphas={args.alphas}",
                f"\tdev_eval_metric={args.dev_eval_metric}",
            ]
        )
    settings_description.append(f"test_datasets={args.test_datasets}")

    print(f"Settings:\n\t{'\n\t'.join(settings_description)}")


def estimate_best_alpha(
    sparse_ranking: Ranking,
    dense_ranking: Ranking,
    dataset: ir_datasets.Dataset,
    eval_metrics: List[measures.Measure],
) -> None:
    """
    Calculate and print the evaluation results for different interpolation parameters.

    Args:
        sparse_ranking (Ranking): The initial sparse ranking of documents.
        dense_ranking (Ranking): The re-ranked dense ranking of documents.
        dataset (ir_datasets.Dataset): Dataset to evaluate the rankings.
        eval_metrics (List[measures.Measure]): Evaluation metrics.
    """
    warnings.warn("This method is still experimental and may not work as expected.")

    # Estimate best interpolation alpha as weighted average of sparse- and dense-nDCG scores
    dense_score = calc_aggregate(
        eval_metrics, dataset.qrels_iter(), to_ir_measures(dense_ranking)
    )
    sparse_score = calc_aggregate(
        eval_metrics, dataset.qrels_iter(), to_ir_measures(sparse_ranking)
    )
    dense_nDCG10 = dense_score[measures.nDCG @ 10]
    sparse_nDCG10 = sparse_score[measures.nDCG @ 10]
    weights = [dense_nDCG10, sparse_nDCG10]
    if sparse_nDCG10 == dense_nDCG10:
        best_alpha = 0.5
    elif dense_nDCG10 > sparse_nDCG10:
        best_alpha = np.average([0, 0.5], weights=weights)
    else:
        best_alpha = np.average([0.5, 1], weights=weights)
    assert 0 <= best_alpha <= 1, f"Invalid best_alpha: {best_alpha}"
    best_score = calc_aggregate(
        eval_metrics,
        dataset.qrels_iter(),
        to_ir_measures(sparse_ranking.interpolate(dense_ranking, best_alpha)),
    )
    print(
        f"\tEstimated best-nDCG@10 interpolated ranking (α~={best_alpha}): {best_score}"
    )


# TODO [later]: Further improve efficiency of re-ranking step. Discuss with ChatGPT and Jurek.
def main(args: argparse.Namespace) -> None:
    """
    Re-ranking Stage: Create query embeddings and re-rank documents based on similarity to query embeddings.

    This script takes the initial ranking of documents and re-ranks them based on the similarity to the query embeddings.
    It uses various encoding methods and evaluation metrics to achieve this.

    See parse_args() for command-line arguments.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Input:
        ranking (List[Tuple]): A ranking of documents for each given query.
            - Format: (q_id, q0, d_id, rank, score, name)
        ff_index (Index): Used to retrieve document embeddings.

    Output:
        ranking (List[Tuple]): A re-ranked ranking of documents for each given query.
    """
    start_time = time.time()
    print_settings()
    pt.init()

    # Parse eval_metrics (e.g. "nDCG@10", "RR(rel=2)", "AP") to ir-measures' measure objects.
    eval_metrics = []
    for metric_str in args.eval_metrics:
        if "(" in metric_str:
            metric_name, rest = metric_str.split("(")
            params, at_value = rest.split(")@")
            param_dict = {
                k: int(v) for k, v in (param.split("=") for param in params.split(","))
            }
            eval_metrics.append(
                getattr(measures, metric_name)(**param_dict) @ int(at_value)
            )
        else:
            metric_name, at_value = metric_str.split("@")
            eval_metrics.append(getattr(measures, metric_name) @ int(at_value))

    # Load dataset and create sparse retriever (e.g. BM25)
    dataset = pt.get_dataset(args.dataset)
    print("Creating BM25 retriever via PyTerrier index...")
    try:
        bm25 = pt.BatchRetrieve.from_dataset(
            dataset, "terrier_stemmed", wmodel="BM25", verbose=True
        )
    except:
        indexer = pt.IterDictIndexer(
            str(Path.cwd()),  # ignored but must be a valid path
            type=pt.index.IndexingType.MEMORY,
        )
        index_ref = indexer.index(dataset.get_corpus_iter(), fields=["text"])
        bm25 = pt.BatchRetrieve(index_ref, wmodel="BM25", verbose=True)
    bm25_cut = ~bm25 % args.rerank_cutoff

    # Create re-ranking pipeline based on TCTColBERTQueryEncoder (normal FF approach)
    index_tct = OnDiskIndex.load(
        args.index_path,
        TCTColBERTQueryEncoder("castorini/tct_colbert-msmarco", device=args.device),
        verbose=args.verbose,
    )
    if args.in_memory:
        index_tct = index_tct.to_memory(2**14)
    ff_tct = FFScore(index_tct)
    int_tct = FFInterpolate(alpha=0.1)
    tct = bm25_cut >> ff_tct >> int_tct

    # TODO: Add profiling to re-ranking step
    # Create re-ranking pipeline based on WeightedAvgEncoder
    index_avg = copy(index_tct)
    index_avg.query_encoder = WeightedAvgEncoder(index_tct, args.k_avg, args.w_method)

    # TODO: Check if PyTerrier supports caching now. Or try https://github.com/seanmacavaney/pyterrier-caching

    # Create int_avg array of length 4 with each alpha value
    avg_chains = max([1, args.avg_chains])
    avg_int_alphas = args.avg_int_alphas + [0.5] * (
        avg_chains - len(args.avg_int_alphas)
    )
    int_avg = [FFInterpolate(alpha=a) for a in avg_int_alphas[:avg_chains]]
    ff_avg = FFScore(index_avg)
    avg_pipelines = [bm25_cut]
    for i in range(len(int_avg)):
        avg_pipelines.append(avg_pipelines[-1] >> ff_avg >> int_avg[i])
    avg_pipelines = avg_pipelines[1:]  # Remove 1st pipeline (bm25) from avg_pipelines

    int_combo_tct = FFInterpolate(alpha=0.3)
    combo = avg_pipelines[0] >> ff_tct >> int_combo_tct

    # Validation and parameter tuning on dev set
    if args.val_pipelines:
        # TODO [later]: Tune k_avg for WeightedAvgEncoder
        dataset_str = "irds:msmarco-passage/dev/judged"
        print(f"Loading dev queries and qrels from {dataset_str}...")
        dev_dataset = pt.get_dataset(dataset_str)
        dev_queries = dev_dataset.get_topics()
        dev_qrels = dev_dataset.get_qrels()

        # Sample dev queries if dev_sample_size is set
        if args.dev_sample_size is not None:
            dev_queries = dev_queries.sample(
                n=args.dev_sample_size, random_state=42
            )  # Fixed seed for reproducibility.
            dev_qrels = dev_qrels[dev_qrels["qid"].isin(dev_queries["qid"])]

        print(f"Adding {len(dev_queries)} sampled queries to BM25 ranking...")
        bm25_df = bm25_cut(dev_queries).rename(columns={"qid": "q_id", "docid": "id"})
        print("Creating BM25 ranking for dev queries...")
        index_avg.query_encoder.sparse_ranking = Ranking(bm25_df)

        # Validate pipelines in args.val_pipelines.
        pipelines_to_validate = [
            # bm25 has no tunable parameters, so it is not included here
            (tct, [int_tct], "tct"),
            (avg_pipelines[0], [int_avg[0]], "avg_1"),
            (combo, [int_combo_tct], "combo"),
        ] + [
            (pipeline, [int_avg[i]], f"avg_{i+1}")
            for i, pipeline in enumerate(avg_pipelines[1:], start=1)
        ]

        for pipeline, tunable_alphas, name in pipelines_to_validate:
            if name in args.val_pipelines:
                print(f"\nValidating pipeline: {name}...")
                # TODO [IMPORTANT!]: Find why this reaches ~0.5 performance (ndCG@10 ~= 0.35 instead of 0.7)
                # TODO: metric should be RR@10 or MRR@10 for dev/small.
                pt.GridSearch(
                    pipeline,
                    {tunable: {"alpha": args.alphas} for tunable in tunable_alphas},
                    dev_queries,
                    dev_qrels,
                    metric=args.dev_eval_metric,  # Find official metrics for dataset version on https://ir-datasets.com/msmarco-passage.html
                    verbose=True,
                    batch_size=128,
                )

    if args.test_datasets:
        # Define which pipelines to evaluate on test sets
        test_pipelines: List[Tuple[str, pt.Transformer, str]] = [
            (~bm25, "bm25"),
            (tct, f"tct, α={int_tct.alpha}"),
            (int_avg[0], f"avg_1, α={int_avg[0].alpha}"),
            (combo, f"combo, α_AVG={int_avg[0].alpha}, α_TCT={int_combo_tct.alpha}"),
        ] + [
            (
                avg_pipelines[i],
                f"avg_{i+1}, α=[{','.join(str(int_avg[j].alpha) for j in range(i+1))}]",
            )
            for i in range(1, len(int_avg))
        ]
        test_pipelines = [(pipeline, desc) for pipeline, desc in test_pipelines]

        # Final evaluation on test sets
        for test_dataset_name in args.test_datasets:
            test_dataset = pt.get_dataset(test_dataset_name)
            test_queries = test_dataset.get_topics()
            index_avg.query_encoder.sparse_ranking = Ranking(
                df=bm25_cut(test_queries).rename(columns={"qid": "q_id", "docid": "id"})
            )

            print(f"\nRunning final tests on {test_dataset_name}...")
            results = pt.Experiment(
                [pipeline for pipeline, _ in test_pipelines],
                test_queries,
                test_dataset.get_qrels(),
                eval_metrics=eval_metrics,
                names=[desc for _, desc in test_pipelines],
                verbose=True,
            )
            print_settings()
            print(f"\nFinal results on {test_dataset_name}:\n{results}\n")
            # TODO: Save experiment results to new row in Excel file.

    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
