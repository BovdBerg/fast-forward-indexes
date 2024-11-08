import argparse
import warnings
from copy import copy
from pathlib import Path
from typing import List, Tuple

import ir_datasets
import numpy as np
import pandas as pd
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
    # TODO [final]: update pipelines choices here
    pipelines = ["bm25", "tct", "combo"] + [f"avg_{i}" for i in range(1, 6)]

    parser = argparse.ArgumentParser(
        description="Re-rank documents based on query embeddings."
    )
    # TODO [at hand-in]: Remove default paths (index_path) form the arguments
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
        default=[0.1, 0.8, 0.9, 0.5, 0.5],
        help="List of interpolation \"alpha\" parameters we initialize the WeightedAvgEncoder chains with. Must be larger than --avg_chains: len(avg_alphas) >= avg_chains.",
    )
    # VALIDATION
    parser.add_argument(
        "--val_pipelines",
        type=str,
        nargs="+",
        default=[],
        choices=pipelines,
        help="List of pipelines to validate, based on exact pipeline names.",
    )
    parser.add_argument(
        "--dev_dataset",
        type=str,
        default="irds:msmarco-passage/dev/small",
        help="Dataset to validate and tune parameters. May never be equal to test_dataset.",
    )
    parser.add_argument(
        "--dev_sample_size",
        type=int,
        default=32,
        help="Number of queries to sample for validation.",
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[round(x, 2) for x in np.arange(0, 1.0001, 0.1)],
        help="List of interpolation parameters for evaluation.",
    )
    # EVALUATION
    parser.add_argument(
        "--test_datasets",
        type=str,
        nargs="+",
        default=["irds:msmarco-passage/trec-dl-2019/judged"],
        help="Datasets to evaluate the rankings. May never be equal to dev_dataset.",
    )
    parser.add_argument(
        "--eval_metrics",
        type=str,
        nargs="+",
        default=["nDCG@10", "RR@10", "AP@1000"],
        help="Metrics used for evaluation.",
    )
    parser.add_argument(
        "--test_pipelines",
        type=str,
        nargs="+",
        default=pipelines,
        help="List of pipelines to evaluate, based on exact pipeline names.",
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
    ]
    # Validation settings
    settings_description.append(f"val_pipelines={args.val_pipelines}")
    if args.val_pipelines:
        settings_description[-1] += ":"
        settings_description.extend(
            [
                f"\tdev_dataset={args.dev_dataset}",
                f"\tdev_sample_size={args.dev_sample_size}",
                f"\talphas={args.alphas}",
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


def validate(
    pipeline: pt.Transformer,
    tunable_alphas: List[pt.Transformer],
    name: str,
    dev_queries: pd.DataFrame,
    dev_dataset: ir_datasets.Dataset,
) -> None:
    """
    Validate the pipeline and tune the alpha parameters based on the dev set.
    Only validate if the pipeline name is in args.val_pipelines.

    Args:
        pipeline (pt.Transformer): The pipeline to validate.
        tunable_alphas (List[pt.Transformer]): List of FFInterpolate blocks with tunable alpha parameters.
        name (str): Name of the pipeline for logging purposes. Must EXACTLY match args.val_pipelines.
        dev_queries (pd.DataFrame): DataFrame with dev queries.
        dev_dataset (ir_datasets.Dataset): Dataset to validate the pipeline.
    """
    print(f"\nValidating pipeline: {name}...")
    param_grid = {tunable: {"alpha": args.alphas} for tunable in tunable_alphas}
    pt.GridSearch(
        pipeline,
        param_grid,
        dev_queries,
        dev_dataset.get_qrels(),
        metric="ndcg_cut_10",  # TODO: Find why this scores so horribly.
        verbose=True,
        batch_size=128,
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
    print_settings()
    pt.init()

    # Load index
    index: Index = OnDiskIndex.load(args.index_path, verbose=args.verbose)
    if args.in_memory:
        index = index.to_memory(buffer_size=2**14)

    # Parse eval_metrics to ir-measures' measure objects
    eval_metrics = []
    for metric_str in args.eval_metrics:
        metric_name, at_value = metric_str.split("@")
        eval_metrics.append(getattr(measures, metric_name) @ int(at_value))

    # Load dataset and create sparse retriever (e.g. BM25)
    dataset = pt.get_dataset(args.dataset)
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
    index_tct = copy(index)
    index_tct.query_encoder = TCTColBERTQueryEncoder(
        "castorini/tct_colbert-msmarco", device=args.device
    )
    ff_tct = FFScore(index_tct)
    int_tct = FFInterpolate(alpha=0.1)
    tct = bm25_cut >> ff_tct >> int_tct

    # TODO: Add profiling to re-ranking step
    # Create re-ranking pipeline based on WeightedAvgEncoder
    index_avg = copy(index)
    index_avg.query_encoder = WeightedAvgEncoder(index, args.k_avg, args.w_method)
    ff_avg = FFScore(index_avg)

    # TODO: Check if PyTerrier supports caching now. Or try https://github.com/seanmacavaney/pyterrier-caching
    # TODO: Try bm25 >> rm3 >> bm25 from lecture notebook 5.

    # Create int_avg array of length 4 with each alpha value
    # TODO: tune params again and update their defaults here
    int_avg = [FFInterpolate(alpha=a) for a in args.avg_int_alphas[:args.avg_chains]]
    avg_pipelines = [bm25_cut >> ff_avg >> int_avg[0]]
    for i in range(1, len(int_avg)):
        avg_pipelines.append(avg_pipelines[-1] >> ff_avg >> int_avg[i])

    int_combo_tct = FFInterpolate(alpha=0.3)
    combo = avg_pipelines[0] >> ff_tct >> int_combo_tct

    # Validation and parameter tuning on dev set
    # TODO: Tune k_avg for WeightedAvgEncoder
    dev_dataset = pt.get_dataset(args.dev_dataset)
    dev_queries = dev_dataset.get_topics()
    index_avg.query_encoder.sparse_ranking = Ranking(
        df=bm25_cut(dev_queries).rename(columns={"qid": "q_id", "docid": "id"})
    )

    # Sample dev queries if dev_sample_size is set
    if args.dev_sample_size is not None:
        dev_queries = dev_queries.sample(n=args.dev_sample_size)

    # Validate pipelines in args.val_pipelines.
    pipelines_to_validate = [
        # bm25 has no tunable parameters, so it is not included here
        (tct, [int_tct], "tct"),
        (combo, [int_combo_tct], "combo"),
    ] + [
        (pipeline, [int_avg[i]], f"avg_{i+1}") for i, pipeline in enumerate(avg_pipelines)
    ]
    for pipeline, tunable_alphas, name in pipelines_to_validate:
        if name in args.val_pipelines:
            validate(pipeline, tunable_alphas, name, dev_queries, dev_dataset)

    # Define which pipelines to evaluate on test sets
    test_pipelines: List[Tuple[str, pt.Transformer, str]] = [
        ("bm25", ~bm25, "bm25"),
        ("tct", tct, f"tct, α={int_tct.alpha}"),
        ("combo", combo, f"combo, α_AVG={int_avg[0].alpha}, α_TCT={int_combo_tct.alpha}"),
    ] + [
        (f"avg_{i+1}", avg_pipelines[i], f"avg_{i+1}, α=[{','.join(str(int_avg[j].alpha) for j in range(i+1))}]")
        for i in range(len(int_avg))
    ]
    test_pipelines = [
        (pipeline, desc)
        for name, pipeline, desc in test_pipelines
        if name in args.test_pipelines
    ]

    # Final evaluation on test sets
    for test_dataset_name in args.test_datasets:
        test_dataset = pt.get_dataset(test_dataset_name)
        test_queries = test_dataset.get_topics()
        index_avg.query_encoder.sparse_ranking = Ranking(
            df=bm25_cut(test_queries).rename(columns={"qid": "q_id", "docid": "id"})
        )

        print(f"\nRunning final evaluations on {test_dataset_name}...")
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


if __name__ == "__main__":
    args = parse_args()
    main(args)
