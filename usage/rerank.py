from enum import Enum, auto
from pathlib import Path
from typing import List
import torch
from fast_forward.encoder.avg import AvgEncoder
from fast_forward.encoder.transformer import TCTColBERTQueryEncoder
from fast_forward.index import Index
from fast_forward.index.disk import OnDiskIndex
from fast_forward.ranking import Ranking
import ir_datasets
from ir_measures import calc_aggregate, measures
from fast_forward.util import to_ir_measures
import argparse


class EncodingMethod(Enum):
    """
    Enumeration for different methods to estimate query embeddings.

    Attributes:
        TCTColBERT: Use TCT-ColBERT method for encoding queries.
        AVERAGE: Use average of top-ranked documents for encoding queries.
    """
    TCTColBERT = auto()
    AVERAGE = auto()


def parse_args():
    """
    Parse command-line arguments for the re-ranking script.

    Returns:
        argparse.Namespace: Parsed command-line arguments.

    Arguments:
        --ranking_path (Path): Path to the first-stage ranking file.
        --index_path (Path): Path to the index file.
        --ranking_output_path (Path): Path to save the re-ranked ranking.
        --dataset (str): Dataset to evaluate the re-ranked ranking.
        --rerank_cutoff (int): Number of documents to re-rank per query.
        --encoding_method (EncodingMethod): Method to estimate query embeddings.
        --k_avg (int): Number of top-ranked documents to use for EncodingMethod.AVERAGE.
        --in_memory (bool): Whether to load the index in memory.
        --device (str, choices=["cuda", "cpu"], default="cuda" if torch.cuda.is_available() else "cpu"): Device to use for encoding queries.
        --eval_metrics (list of str): Metrics used for evaluation.
        --alphas (list of float): List of interpolation parameters for evaluation.
    """
    parser = argparse.ArgumentParser(description="Re-rank documents based on query embeddings.")
    # TODO [at hand-in]: Remove default paths (ranking_path, index_path) form the arguments
    parser.add_argument("--ranking_path", type=Path, default="/home/bvdb9/sparse_rankings/msmarco-passage-test2019-sparse10000.txt", help="Path to the first-stage ranking file (.tsv or .txt).")
    parser.add_argument("--index_path", type=Path, default="/home/bvdb9/indices/msm-psg/ff/ff_index_msmpsg_TCTColBERT_opq.h5", help="Path to the index file.")
    parser.add_argument("--ranking_output_path", type=Path, default="dense_ranking.tsv", help="Path to save the re-ranked ranking.")
    parser.add_argument("--dataset", type=str, default="msmarco-passage/trec-dl-2019", help="Dataset (using package ir-datasets).")
    parser.add_argument("--rerank_cutoff", type=int, default=1000, help="Number of documents to re-rank per query.")
    parser.add_argument("--encoding_method", type=EncodingMethod, choices=list(EncodingMethod), default=EncodingMethod.AVERAGE, help="Method to estimate query embeddings.")
    parser.add_argument("--k_avg", type=int, default=8, help="Number of top-ranked documents to use for EncodingMethod.AVERAGE.")
    parser.add_argument("--in_memory", action="store_true", help="Whether to load the index in memory.")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for encoding queries.")
    parser.add_argument("--eval_metrics", type=str, nargs='+', default=["nDCG@10"], help="Metrics used for evaluation.")
    parser.add_argument("--alphas", type=float, nargs='+', default=[0, 0.25, 0.5, 0.75, 1], help="List of interpolation parameters for evaluation.")
    return parser.parse_args()


def print_settings(
    dataset: str,
    ranking_path: Path, 
    index_path: Path, 
    rerank_cutoff: int, 
    encoding_method: EncodingMethod, 
    device: str, 
    k_avg: int
    ) -> None:
    """
    Print the settings used for re-ranking.

    Args:
        dataset (str): Dataset to evaluate the re-ranked ranking.
        ranking_path (Path): Path to the first-stage ranking file.
        index_path (Path): Path to the index file.
        rerank_cutoff (int): Number of documents to re-rank per query.
        encoding_method (EncodingMethod): Method to estimate query embeddings.
        device (str): Device to use for encoding queries.
        k_avg (int): Number of top-ranked documents to use for EncodingMethod.AVERAGE.
    """
    settings_description: List[str] = [
        f"dataset={dataset}",
        f"ranking={ranking_path.name}",
        f"index={index_path.name}",
        f"rerank_cutoff={rerank_cutoff}",
        f"encoding_method={encoding_method}",
    ]
    match encoding_method:  # Append method-specific settings
        case EncodingMethod.TCTColBERT:
            settings_description.append(f"device={device}")
        case EncodingMethod.AVERAGE:
            settings_description.append(f"k_avg={k_avg}")
    print("\nSettings:\n\t" + ",\n\t".join(settings_description))


def print_results(
    alphas: List[float], 
    sparse_ranking: Ranking, 
    dense_ranking: Ranking, 
    eval_metrics: List[str], 
    dataset
    ) -> None:
    """
    Print the evaluation results for different interpolation parameters.

    Args:
        alphas (List[float]): List of interpolation parameters for evaluation.
        sparse_ranking (Ranking): The initial sparse ranking of documents.
        dense_ranking (Ranking): The re-ranked dense ranking of documents.
        eval_metrics (List[str]): Metrics used for evaluation.
        dataset: Dataset to evaluate the re-ranked ranking (provided by ir_datasets package).
    """
    eval_metrics_objects = []
    for metric_str in eval_metrics:
        metric_name, at_value = metric_str.split('@')
        eval_metrics_objects.append(getattr(measures, metric_name) @ int(at_value))

    print('Results:')
    for alpha in alphas:
        interpolated_ranking = sparse_ranking.interpolate(dense_ranking, alpha)
        score = calc_aggregate(eval_metrics_objects, dataset.qrels_iter(), to_ir_measures(interpolated_ranking))
        ranking_type = (
            "Sparse" if alpha == 1 else 
            "Dense" if alpha == 0 else 
            "Interpolated"
        )
        print(f"\t{ranking_type} ranking (alpha={alpha}): {score}")


def main(
        args: argparse.Namespace
    ) -> None:
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
            - Saved to ranking_output_path
    """
    # Load dataset
    dataset = ir_datasets.load(args.dataset)

    # Load ranking and attach queries
    sparse_ranking: Ranking = Ranking.from_file(
        args.ranking_path,
        queries={q.query_id: q.text for q in dataset.queries_iter()},
    )
    sparse_ranking_cut = sparse_ranking.cut(args.rerank_cutoff) # Cut ranking to rerank_cutoff

    # Load index
    index: Index = OnDiskIndex.load(args.index_path)
    if args.in_memory:
        index = index.to_memory()

    # Choose query encoder based on encoding_method
    match args.encoding_method:
        case EncodingMethod.TCTColBERT:
            index.query_encoder = TCTColBERTQueryEncoder("castorini/tct_colbert-msmarco", device=args.device)
        case EncodingMethod.AVERAGE:
            index.query_encoder = AvgEncoder(sparse_ranking_cut, index, args.k_avg)
        case _:
            raise ValueError(f"Unsupported encoding method: {args.encoding_method}")
    assert index.query_encoder is not None, "Query encoder not set in index."

    # Create and save dense_ranking by ranking on similarity between (q_rep, d_rep)
    dense_ranking = index(sparse_ranking_cut)
    dense_ranking.save(args.ranking_output_path)

    print_settings(args.dataset, args.ranking_path, args.index_path, args.rerank_cutoff, args.encoding_method, args.device, args.k_avg)
    print_results(args.alphas, sparse_ranking, dense_ranking, args.eval_metrics, dataset)


if __name__ == '__main__':
    args = parse_args()
    main(args)
