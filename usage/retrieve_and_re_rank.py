import argparse
import warnings
from copy import copy
from pathlib import Path
from typing import List

import ir_datasets
import numpy as np
import pandas as pd
import pyterrier as pt
import torch
from ir_measures import calc_aggregate, measures

from fast_forward.encoder.avg import ProbDist, WeightedAvgEncoder
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
    # TODO [at hand-in]: Remove default paths (sparse_ranking_path, index_path) form the arguments
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print more information during re-ranking.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="msmarco_passage",
        help="Dataset (using package ir-datasets). Must match the sparse_ranking.",
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
        "--prob_dist",
        type=ProbDist,
        choices=list(ProbDist),
        default="UNIFORM",
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
        default=1,
        help="Number of chained (AVG >> INT) blocks with shared weights. Only used for EncodingMethod.WEIGHTED_AVERAGE.",
    )
    parser.add_argument(
        "--avg_shared_int_alpha",
        type=float,
        default=0.1,
        help=(
            "Shared weight for chained (AVG >> INT) blocks. Only used for EncodingMethod.WEIGHTED_AVERAGE. Overwritten on parameter tuning. "
            "Best combinations (--avg_chains, --avg_shared_int_alpha, nDCG@10):"
            "(1, 0.1, 0.552501), (2, 0.5, 0.543245), (3, 0.8, 0.506276), (4, 0.6, 0.550304), (5, 0.8, 0.531738)"
        ),
    )
    # VALIDATION
    parser.add_argument(
        "--validate_pipelines",
        type=str,
        nargs="+",
        default=[],
        # TODO [final]: update pipelines choices here
        choices=[
            "pipeline_tct",
            "pipeline_avg_1",
            "pipeline_chained_avg_un2",
            "pipeline_chained_avg_un3",
            "pipeline_chained_avg_shN",
        ],
        help="List of pipelines to validate, based on exact pipeline names.",
    )
    parser.add_argument(
        "--dev_dataset",
        type=str,
        default="irds:msmarco-passage/dev/small",
        help="Dataset to validate and tune parameters. May never be equal to test_dataset.",
    )
    parser.add_argument(
        "--dev_sparse_ranking_path",
        type=Path,
        default="/home/bvdb9/sparse_rankings/msmarco_passage-dev.small-BM25-top100.tsv",
        help="Path to the sparse ranking file.",
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
    # TODO: Add option to evaluate on multiple datasets, accepting multiple test_datasets and test_sparse_ranking_paths.
    parser.add_argument(
        "--test_dataset",
        type=str,
        default="irds:msmarco-passage/trec-dl-2019/judged",
        help="Dataset to evaluate the rankings. May never be equal to dev_dataset.",
    )
    parser.add_argument(
        "--test_sparse_ranking_path",
        type=Path,
        default="/home/bvdb9/sparse_rankings/msmarco_passage-trec-dl-2019.judged-BM25-top10000.tsv",
        help="Path to the sparse ranking file.",
    )
    parser.add_argument(
        "--eval_metrics",
        type=str,
        nargs="+",
        default=["nDCG@10", "RR@10", "AP@1000"],
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
        f"\tprob_dist={args.prob_dist.name}",
        f"\tk_avg={args.k_avg}",
        f"\tavg_chains={args.avg_chains}",
    ]
    # Validation settings
    settings_description.append(f"validate_pipelines={args.validate_pipelines}")
    if args.validate_pipelines:
        settings_description[-1] += ":"
        settings_description.extend(
            [
                f"\tdev_dataset={args.dev_dataset}",
                f"\tdev_sample_size={args.dev_sample_size}",
                f"\talphas={args.alphas}",
            ]
        )

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
    Only validate if the pipeline name is in args.validate_pipelines.

    Args:
        pipeline (pt.Transformer): The pipeline to validate.
        tunable_alphas (List[pt.Transformer]): List of FFInterpolate blocks with tunable alpha parameters.
        name (str): Name of the pipeline for logging purposes. Must EXACTLY match args.validate_pipelines.
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
        metric="ndcg_cut_10",
        verbose=True,
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
    pipeline_bm25 = ~bm25 % args.rerank_cutoff

    # Create re-ranking pipeline based on TCTColBERTQueryEncoder (normal FF approach)
    index_tct = copy(index)
    index_tct.query_encoder = TCTColBERTQueryEncoder(
        "castorini/tct_colbert-msmarco", device=args.device
    )
    ff_score_tct = FFScore(index_tct)
    ff_int_tct = FFInterpolate(alpha=0.1)
    pipeline_tct = pipeline_bm25 >> ff_score_tct >> ff_int_tct

    # TODO: Add profiling to re-ranking step
    # Create re-ranking pipeline based on WeightedAvgEncoder
    index_avg = copy(index)
    index_avg.query_encoder = WeightedAvgEncoder(index, args.k_avg, args.prob_dist)
    ff_score_avg = FFScore(index_avg)

    # TODO: Check if PyTerrier supports caching now.
    # TODO: Try bm25 >> rm3 >> bm25 from lecture notebook 5.
    # TODO: find bug when validating on WEIGHTED_AVERAGE
    # TODO: Add query_encoder as arg to FFScore.__init__.
    # TODO: Try chained ff_score_avg + ff_score_tct
    # TODO: Encode as weighted average of WeightedAvgEncoder and (lightweight) QueryEncoder
    ff_int_avg_1 = FFInterpolate(alpha=0.1)
    pipeline_avg_1 = pipeline_bm25 >> ff_score_avg >> ff_int_avg_1

    # Create re-ranking pipeline with individual tuning of FFInterpolate.
    # WARNING: validation time on this pipeline scales exponentially: args.alphas ** avg_chains.
    # TODO: Run big evaluation with different amount of chains and overwrite with best outcomes
    ff_int_avg_un2_1 = FFInterpolate(alpha=0.1)
    ff_int_avg_un2_2 = FFInterpolate(alpha=0.9)
    pipeline_chained_avg_un2 = (
        pipeline_bm25
        >> ff_score_avg
        >> ff_int_avg_un2_1
        >> ff_score_avg
        >> ff_int_avg_un2_2
    )

    # WARNING: validation time on this pipeline scales exponentially: args.alphas ** avg_chains.
    # TODO: Find defaults for below parameters by running evaluation once.
    ff_int_avg_un3_1 = FFInterpolate(alpha=0.5)
    ff_int_avg_un3_2 = FFInterpolate(alpha=0.5)
    ff_int_avg_un3_3 = FFInterpolate(alpha=0.5)
    pipeline_chained_avg_un3 = (
        pipeline_bm25
        >> ff_score_avg
        >> ff_int_avg_un3_1
        >> ff_score_avg
        >> ff_int_avg_un3_2
        >> ff_score_avg
        >> ff_int_avg_un3_3
    )

    ff_int_avg_shN = FFInterpolate(alpha=args.avg_shared_int_alpha)
    pipeline_chained_avg_shN = pipeline_bm25
    for chain in range(args.avg_chains):
        pipeline_chained_avg_shN = (
            pipeline_chained_avg_shN >> ff_score_avg >> ff_int_avg_shN
        )

    # Validation and parameter tuning on dev set
    # TODO: Tune k_avg for WeightedAvgEncoder
    dev_dataset = pt.get_dataset(args.dev_dataset)
    index_avg.query_encoder.sparse_ranking = Ranking.from_file(
        args.dev_sparse_ranking_path,
        queries={q.qid: q.query for q in dev_dataset.get_topics().itertuples()},
    )

    # Sample dev queries if dev_sample_size is set
    dev_queries = dev_dataset.get_topics()
    if args.dev_sample_size is not None:
        dev_queries = dev_queries.sample(n=args.dev_sample_size)

    # Validate pipelines in args.validate_pipelines.
    pipelines_to_validate = [
        (pipeline_tct, [ff_int_tct], "pipeline_tct"),
        (pipeline_avg_1, [ff_int_avg_1], "pipeline_avg_1"),
        (
            pipeline_chained_avg_un2,
            [ff_int_avg_un2_1, ff_int_avg_un2_2],
            "pipeline_chained_avg_un2",
        ),
        (
            pipeline_chained_avg_un3,
            [ff_int_avg_un3_1, ff_int_avg_un3_2, ff_int_avg_un3_3],
            "pipeline_chained_avg_un3",
        ),
        (pipeline_chained_avg_shN, [ff_int_avg_shN], "pipeline_chained_avg_shN"),
    ]
    for pipeline, tunable_alphas, name in pipelines_to_validate:
        if name in args.validate_pipelines:
            validate(pipeline, tunable_alphas, name, dev_queries, dev_dataset)

    # Final evaluation on test set
    test_dataset = pt.get_dataset(args.test_dataset)
    index_avg.query_encoder.sparse_ranking = Ranking.from_file(
        args.test_sparse_ranking_path,
        queries={q.qid: q.query for q in test_dataset.get_topics().itertuples()},
    )
    print(f"\nRunning final evaluations on {args.test_dataset}...")
    results = pt.Experiment(
        [
            ~bm25,
            pipeline_tct,
            pipeline_avg_1,
            pipeline_chained_avg_un2,
            pipeline_chained_avg_un3,
            pipeline_chained_avg_shN,
        ],
        test_dataset.get_topics(),
        test_dataset.get_qrels(),
        eval_metrics=eval_metrics,
        names=[
            "BM25",
            f"pipeline_tct, α={ff_int_tct.alpha}",
            f"pipeline_avg_1, α={ff_int_avg_1.alpha}",
            f"pipeline_chained_avg_un2, α=[{ff_int_avg_un2_1.alpha},{ff_int_avg_un2_2.alpha}]",
            f"pipeline_chained_avg_un3, α=[{ff_int_avg_un3_1.alpha},{ff_int_avg_un3_2.alpha},{ff_int_avg_un3_3.alpha}]",
            f"pipeline_chained_avg_sh{args.avg_chains}, α={ff_int_avg_shN.alpha}",
        ],
    )
    print_settings()
    print(f"\nFinal results on {args.test_dataset}:\n{results}\n")


if __name__ == "__main__":
    args = parse_args()
    main(args)
