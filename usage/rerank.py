from enum import Enum
from pathlib import Path
from typing import List
import numpy as np
import warnings
import torch
from fast_forward.encoder.avg import ProbDist, WeightedAvgEncoder
from fast_forward.encoder.transformer import TCTColBERTQueryEncoder
from fast_forward.index import Index
from fast_forward.index.disk import OnDiskIndex
from fast_forward.ranking import Ranking
from fast_forward.util.pyterrier import FFInterpolate, FFScore
import ir_datasets
from ir_measures import calc_aggregate, measures
from fast_forward.util import to_ir_measures
import argparse
import pyterrier as pt


class EncodingMethod(Enum):
    """
    Enumeration for different methods to estimate query embeddings.

    Attributes:
        TCTColBERT: Use TCT-ColBERT method for encoding queries.
        WEIGHTED_AVERAGE: Use weighted average method for encoding queries. Averages over top-ranked document embeddings.
    """

    TCTCOLBERT = "TCTCOLBERT"
    WEIGHTED_AVERAGE = "WEIGHTED_AVERAGE"
    # TODO: Encode as weighted average of WeightedAvgEncoder and (lightweight) QueryEncoder


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
    parser.add_argument(
        "--encoding_method",
        type=EncodingMethod,
        choices=list(EncodingMethod),
        default="WEIGHTED_AVERAGE",
        help="Method to estimate query embeddings.",
    )
    # EncodingMethod.WEIGHTED_AVERAGE
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
        default=None,
        help="Number of top-ranked documents to use. Only used for EncodingMethod.WEIGHTED_AVERAGE.",
    )
    # VALIDATION
    parser.add_argument(
        "--dev_dataset",
        type=str,
        default="irds:msmarco-passage/dev/judged",
        help="Dataset to validate and tune parameters. May never be equal to test_dataset.",
    )
    parser.add_argument(
        "--dev_sparse_ranking_path",
        type=Path,
        default="/home/bvdb9/sparse_rankings/msmarco_passage-dev.judged-BM25-top100.tsv",
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
        default=np.arange(0, 1.0001, 0.25).tolist(),
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
    """
    # General settings
    settings_description: List[str] = [
        f"in_memory={args.in_memory}",
        f"rerank_cutoff={args.rerank_cutoff}",
        f"dev_dataset={args.dev_dataset}",
        f"dev_sample_size={args.dev_sample_size}",
        f"alphas={args.alphas}",
    ]

    # Encoding method settings
    settings_description.append(f"encoding_method={args.encoding_method.name}")
    match args.encoding_method:  # Append method-specific settings
        case EncodingMethod.TCTCOLBERT:
            settings_description.extend(
                [
                    f"device={args.device}",
                ]
            )
        case EncodingMethod.WEIGHTED_AVERAGE:
            settings_description.extend(
                [
                    f"prob_dist={args.prob_dist.name}",
                    f"k_avg={args.k_avg}",
                ]
            )

    print(f"Settings:\n\t{',\n\t'.join(settings_description)}")


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


def add_ranking_to_enc(
    index: Index,
    dataset: ir_datasets.Dataset,
    sparse_ranking_path: Path,
) -> None:
    """
    Add the sparse ranking to the query encoder for re-ranking.

    Args:
        index (Index): The index containing document embeddings.
        dataset (ir_datasets.Dataset): Dataset to evaluate the rankings.
        sparse_ranking_path (Path): Path to the sparse ranking file.
    """
    if args.encoding_method == EncodingMethod.WEIGHTED_AVERAGE:
        index.query_encoder.sparse_ranking = Ranking.from_file(
            sparse_ranking_path,
            queries={q.qid: q.query for q in dataset.get_topics().itertuples()},
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
        index = index.to_memory(buffer_size=2**12)

    # Parse eval_metrics to ir-measures' measure objects
    eval_metrics = []
    for metric_str in args.eval_metrics:
        metric_name, at_value = metric_str.split("@")
        eval_metrics.append(getattr(measures, metric_name) @ int(at_value))

    # Choose query encoder based on encoding_method
    match args.encoding_method:
        case EncodingMethod.TCTCOLBERT:
            index.query_encoder = TCTColBERTQueryEncoder(
                "castorini/tct_colbert-msmarco", device=args.device
            )
        case EncodingMethod.WEIGHTED_AVERAGE:
            index.query_encoder = WeightedAvgEncoder(index, args.k_avg, args.prob_dist)
        case _:
            raise ValueError(f"Unsupported encoding method: {args.encoding_method}")
    assert index.query_encoder is not None, "Query encoder not set in index."

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

    # TODO: Add profiling to re-ranking step
    # Create pipeline for re-ranking
    ff_int = FFInterpolate(alpha=0.5)
    ff_score = FFScore(index)
    # TODO: Check if PyTerrier supports caching now.
    # TODO: find bug when validating on WEIGHTED_AVERAGE
    ff_pipeline = ~bm25 % args.rerank_cutoff >> ff_score >> ff_int

    # TODO: Tune k_avg for WeightedAvgEncoder
    # Validation and parameter tuning on dev set
    dev_dataset = pt.get_dataset(args.dev_dataset)
    add_ranking_to_enc(index, dev_dataset, args.dev_sparse_ranking_path)

    # Sample dev queries if dev_sample_size is set
    dev_queries = dev_dataset.get_topics()
    if args.dev_sample_size is not None:
        dev_queries = dev_queries.sample(n=args.dev_sample_size)

    pt.GridSearch(
        ff_pipeline,
        {ff_int: {"alpha": args.alphas}},
        dev_queries,
        dev_dataset.get_qrels(),
        verbose=True,
    )

    # Final evaluation on test set
    test_dataset = pt.get_dataset(args.test_dataset)
    add_ranking_to_enc(index, test_dataset, args.test_sparse_ranking_path)
    results = pt.Experiment(
        [~bm25, ff_pipeline],
        test_dataset.get_topics(),
        test_dataset.get_qrels(),
        eval_metrics=eval_metrics,
        names=["BM25", f"BM25 >> FFScore >> FFInt(α={ff_int.alpha})"],
    )
    print_settings()
    print(f"\nFinal results, on {args.test_dataset}:\n{results}\n")


if __name__ == "__main__":
    args = parse_args()
    main(args)
