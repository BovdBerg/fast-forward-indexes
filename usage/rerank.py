from enum import Enum
from pathlib import Path
from typing import List
import numpy as np
import torch
from fast_forward.encoder import TCTColBERTQueryEncoder
from fast_forward.index import Index
from fast_forward.index.disk import OnDiskIndex
from fast_forward.ranking import Ranking
from tqdm import tqdm
import ir_datasets
from ir_measures import calc_aggregate, measures
from fast_forward.util import to_ir_measures
import pandas as pd
import argparse


class EncodingMethod(Enum):
    """Enum for encoding method.

    TCTColBERT: Use the TCTColBERT query encoder.
    AVERAGE: Estimate the query embeddings as the average of the top-ranked document embeddings.
    """
    TCTColBERT = 1
    AVERAGE = 2


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


def load_index(index_path: Path, in_memory: bool) -> Index:
    """
    Load the index from disk and optionally move it to memory.

    Args:
        index_path (Path): Path to the index file.
        in_memory (bool): Whether to load the index in memory.

    Returns:
        Index: The loaded index.
    """
    index: Index = OnDiskIndex.load(index_path)
    if in_memory:
        index = index.to_memory()
    return index


def load_and_prepare_ranking(
        ranking_path: Path, 
        dataset, 
        rerank_cutoff: int
    ) -> Ranking:
    """
    Load the ranking, attach the queries, and prepare the data for re-ranking.

    Args:
        ranking_path (Path): Path to the first-stage ranking file.
        dataset: Dataset to evaluate the re-ranked ranking (provided by ir_datasets package).
        rerank_cutoff (int): Number of documents to re-rank per query.

    Returns:
        Ranking: Prepared ranking with unique queries and their indices.
    """
    # Load the ranking and attach the queries
    sparse_ranking: Ranking = Ranking.from_file(
        ranking_path,
        queries={q.query_id: q.text for q in dataset.queries_iter()},
    ).cut(rerank_cutoff)  # Cutoff to top_k docs per query

    # Find unique queries and save their index in q_no column
    uniq_q = sparse_ranking._df[["q_id", "query"]].drop_duplicates().reset_index(drop=True)
    uniq_q["q_no"] = uniq_q.index
    print(f"uniq_q shape: {uniq_q.shape}, head:\n{uniq_q.head()}")

    # Merge q_no into the sparse ranking
    sparse_ranking._df = sparse_ranking._df.merge(uniq_q, on=["q_id", "query"])
    print(f"sparse_ranking._df shape: {sparse_ranking._df.shape}, head:\n{sparse_ranking._df.head()}")

    return sparse_ranking, uniq_q


def create_query_representations(
        sparse_ranking: Ranking, 
        uniq_q: pd.DataFrame, 
        index: Index, 
        encoding_method: EncodingMethod, 
        k_avg: int, 
        device: str
    ) -> np.ndarray:
    """
    Create query representations based on the specified encoding method.

    Args:
        sparse_ranking (Ranking): The initial ranking of documents.
        uniq_q (pd.DataFrame): DataFrame containing unique queries and their indices.
        index (Index): The index used to retrieve document embeddings.
        encoding_method (EncodingMethod): Method to estimate query embeddings.
        k_avg (int): Number of top-ranked documents to use for EncodingMethod.AVERAGE.
        device (str): Device to use for encoding queries.

    Returns:
        np.ndarray: Query representations.
    """
    # Create q_reps as np.ndarray with shape (len(ranking), index.dim) where index.dim is the dimension of the embeddings, often 768.
    q_reps: np.ndarray = np.zeros((len(sparse_ranking), index.dim), dtype=np.float32)
    match encoding_method:
        case EncodingMethod.TCTColBERT:
            # Default approach: encode queries using a query_encoder
            index.query_encoder = TCTColBERTQueryEncoder("castorini/tct_colbert-msmarco", device=device)
            q_reps = index.encode_queries(uniq_q["query"])
        case EncodingMethod.AVERAGE:
            # Estimate the query embeddings as the average of the top-ranked document embeddings
            top_docs = sparse_ranking.cut(k_avg)
            for q_id, query, q_no in tqdm(
                uniq_q.itertuples(index=False), 
                desc="Estimating query embeddings", 
                total=len(uniq_q)
            ):
                # get the embeddings of the top_docs from the index
                top_docs_ids = top_docs[q_id].keys()
                d_reps: np.ndarray = index._get_vectors(top_docs_ids)[0]
                if index.quantizer is not None:
                    d_reps = index.quantizer.decode(d_reps)

                # calculate the average of the embeddings and save it
                q_reps[q_no] = np.mean(d_reps, axis=0)
    print(f"qreps shape: {q_reps.shape}, head:\n{pd.DataFrame(q_reps).head()}")
    return q_reps


def rerank(
        index: Index, 
        sparse_ranking: Ranking, 
        q_reps: np.ndarray,
        ranking_output_path: Path
    ) -> Ranking:
    """
    Re-ranks a given sparse ranking using dense representations and saves the result.

    Args:
        index (Index): The index object used to compute scores.
        sparse_ranking (Ranking): The initial sparse ranking to be re-ranked.
        q_reps (np.ndarray): Query representations used for re-ranking.
        ranking_output_path (Path): The file path where the re-ranked results will be saved.

    Returns:
        Ranking: The re-ranked results as a Ranking object.
    """
    # Compute scores
    result = index._compute_scores(sparse_ranking._df, q_reps)
    result["score"] = result["ff_score"]

    # Create dense ranking object
    dense_ranking = Ranking(
        result,
        name="fast-forward",
        dtype=sparse_ranking._df.dtypes["score"],
        copy=False,
        is_sorted=False,
    )

    # Save dense ranking to output file
    dense_ranking.save(ranking_output_path)

    return dense_ranking


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


def get_measure(
        metric_str: str
    ) -> measures.Measure:
    """
    Parses a metric string and returns the corresponding measure function.

    Args:
        metric_str (str): A string representing the metric and its parameter, 
                            formatted as 'metric_name@value'.

    Returns:
        function: The measure function corresponding to the metric name, 
                    parameterized by the given value.

    Raises:
        AttributeError: If the metric name does not correspond to an attribute in `measures`.
        ValueError: If the metric string is not properly formatted or the value is not an integer.
    """
    metric_name, at_value = metric_str.split('@')
    return getattr(measures, metric_name) @ int(at_value)


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
    eval_metrics_objects = [get_measure(metric_str) for metric_str in eval_metrics]

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

    Input:
        ranking (List[Tuple]): A ranking of documents for each given query.
            - Format: (q_id, q0, d_id, rank, score, name)
        ff_index (Index): Used to retrieve document embeddings.

    Output:
        ranking (List[Tuple]): A re-ranked ranking of documents for each given query.
            - Saved to ranking_output_path
    """
    # Convert dataset argument to actual dataset object
    dataset = ir_datasets.load(args.dataset)

    index = load_index(args.index_path, args.in_memory)
    sparse_ranking, uniq_q = load_and_prepare_ranking(args.ranking_path, dataset, args.rerank_cutoff)
    q_reps = create_query_representations(sparse_ranking, uniq_q, index, args.encoding_method, args.k_avg, args.device)
    dense_ranking = rerank(index, sparse_ranking, q_reps, args.ranking_output_path)

    print_settings(args.dataset, args.ranking_path, args.index_path, args.rerank_cutoff, args.encoding_method, args.device, args.k_avg)
    print_results(args.alphas, sparse_ranking, dense_ranking, args.eval_metrics, dataset)


if __name__ == '__main__':
    args = parse_args()
    main(args)
