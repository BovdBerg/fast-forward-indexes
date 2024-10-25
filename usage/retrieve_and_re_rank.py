import argparse
from copy import copy
import warnings
from enum import Enum
from pathlib import Path
from typing import List

import ir_datasets
import numpy as np
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
        default=4,
        help="Number of chained FF-Score and FF-Interpolate blocks. Only used for EncodingMethod.WEIGHTED_AVERAGE.",
    )
    # VALIDATION
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
        default=[round(x, 1) for x in np.arange(0, 1.0001, 0.1)],
        help="List of interpolation parameters for evaluation.",
    )
    # EVALUATION
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
        f"in_memory={args.in_memory}",
        f"rerank_cutoff={args.rerank_cutoff}",
        f"dev_dataset={args.dev_dataset}",
        f"dev_sample_size={args.dev_sample_size}",
        f"alphas={args.alphas}",
        "TCTColBERTQueryEncoder:",
        f"\tdevice={args.device}",
        "WeightedAvgEncoder:",
        f"\tprob_dist={args.prob_dist.name}",
        f"\tk_avg={args.k_avg}",
        f"\tavg_chains={args.avg_chains}",
    ]

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
    print_settings()
    pt.init()

    # Load index
    index: Index = OnDiskIndex.load(args.index_path)
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

    # Create re-ranking pipeline based on TCTColBERTQueryEncoder (normal FF approach)
    index_tct = copy(index)
    index_tct.query_encoder = TCTColBERTQueryEncoder(
        "castorini/tct_colbert-msmarco", device=args.device
    )
    ff_score_tct = FFScore(index_tct)
    ff_int_tct = FFInterpolate(alpha=0.1) # Alpha will be tuned and overwritten later, but this was the best result so far
    pipeline_tct = ~bm25 % args.rerank_cutoff >> ff_score_tct >> ff_int_tct

    # TODO: Add profiling to re-ranking step
    # Create re-ranking pipeline based on WeightedAvgEncoder
    index_avg = copy(index)
    index_avg.query_encoder = WeightedAvgEncoder(index, args.k_avg, args.prob_dist)
    ff_score_avg = FFScore(index_avg)
    ff_int_avg = FFInterpolate(alpha=0.4) # Alpha will be tuned and overwritten later, but this was the best result so far
    # TODO: Check if PyTerrier supports caching now.
    # TODO: Try bm25 >> rm3 >> bm25 from lecture notebook 5.
    # TODO: find bug when validating on WEIGHTED_AVERAGE
    # TODO: Add query_encoder as arg to FFScore.__init__.
    # TODO: Should each iteration of ff_int_avg have its own alpha weight?
    # TODO: Try chained ff_score_avg + ff_score_tct
    # TODO: Encode as weighted average of WeightedAvgEncoder and (lightweight) QueryEncoder
    pipeline_chained_avg = ~bm25 % args.rerank_cutoff
    for chain in range(args.avg_chains):
        pipeline_chained_avg = pipeline_chained_avg >> ff_score_avg >> ff_int_avg

    # TODO: Tune k_avg for WeightedAvgEncoder
    # Validation and parameter tuning on dev set
    dev_dataset = pt.get_dataset(args.dev_dataset)
    index_avg.query_encoder.sparse_ranking = Ranking.from_file(
        args.dev_sparse_ranking_path,
        queries={q.qid: q.query for q in dev_dataset.get_topics().itertuples()},
    )

    # Sample dev queries if dev_sample_size is set
    dev_queries = dev_dataset.get_topics()
    if args.dev_sample_size is not None:
        dev_queries = dev_queries.sample(n=args.dev_sample_size)

    print(f"\nValidating pipeline: BM25 >> FFScoreTCT >> FFIntTCT...")
    pt.GridSearch(
        pipeline_tct,
        {ff_int_tct: {"alpha": args.alphas}},
        dev_queries,
        dev_dataset.get_qrels(),
        verbose=True,
    )

    print(f"\nValidating pipeline: BM25 >> {args.avg_chains}X (FFScoreAVG >> FFIntAVG)...")
    pt.GridSearch(
        pipeline_chained_avg,
        {ff_int_avg: {"alpha": args.alphas}},
        dev_queries,
        dev_dataset.get_qrels(),
        verbose=True,
    )

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
            pipeline_chained_avg
        ],
        test_dataset.get_topics(),
        test_dataset.get_qrels(),
        eval_metrics=eval_metrics,
        names=[
            "BM25",
            f"BM25 >> FFScoreTCT >> FFIntTCT(α={ff_int_tct.alpha})",
            f"BM25 >> {args.avg_chains}X (FFScoreAVG >> FFIntAVG(α={ff_int_avg.alpha}))",
        ],
    )
    print_settings()
    print(f"\nFinal results, on {args.test_dataset}:\n{results}\n")


if __name__ == "__main__":
    args = parse_args()
    main(args)
