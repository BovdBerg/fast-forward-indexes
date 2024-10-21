from enum import Enum
from pathlib import Path
from typing import List, Tuple
import numpy as np
import warnings
import torch
from fast_forward.encoder.avg import ProbDist, WeightedAvgEncoder
from fast_forward.encoder.transformer import TCTColBERTQueryEncoder
from fast_forward.index import Index
from fast_forward.index.disk import OnDiskIndex
from fast_forward.ranking import Ranking
from fast_forward.util.pyterrier import FFInterpolate
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
    parser = argparse.ArgumentParser(description="Re-rank documents based on query embeddings.")
    # TODO [at hand-in]: Remove default paths (sparse_ranking_path, index_path) form the arguments
    parser.add_argument("--dataset", type=str, default="msmarco-passage", help="Dataset (using package ir-datasets). Must match the sparse_ranking.")
    parser.add_argument("--index_path", type=Path, default="/home/bvdb9/indices/msm-psg/ff_index_msmpsg_TCTColBERT_opq.h5", help="Path to the index file.")
    parser.add_argument("--rerank_cutoff", type=int, default=1000, help="Number of documents to re-rank per query.")
    parser.add_argument("--encoding_method", type=EncodingMethod, choices=list(EncodingMethod), default="WEIGHTED_AVERAGE", help="Method to estimate query embeddings.")
    parser.add_argument("--in_memory", action="store_true", help="Whether to load the index in memory.")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for encoding queries.")
    # Arguments for EncodingMethod.WEIGHTED_AVERAGE
    parser.add_argument("--k_avg", type=int, default=8, help="Number of top-ranked documents to use. Only used for EncodingMethod.WEIGHTED_AVERAGE.")
    parser.add_argument("--prob_dist", type=ProbDist, choices=list(ProbDist), default="UNIFORM", help="Method to estimate query embeddings. Only used for EncodingMethod.WEIGHTED_AVERAGE.")
    # VALIDATION
    parser.add_argument("--enable_validation", action="store_true", default=False, help="Whether to validate and tune parameters.")
    parser.add_argument("--dev_dataset", type=str, default="irds:msmarco-passage/dev/judged", help="Dataset to validate and tune parameters. May never be equal to test_dataset.")
    parser.add_argument("--dev_sparse_ranking_path", type=Path, default="/home/bvdb9/sparse_rankings/msmarco_passage-dev.judged-BM25-top100.tsv", help="Path to the sparse ranking file.")
    # EVALUATION
    parser.add_argument("--test_dataset", type=str, default="irds:msmarco-passage/trec-dl-2019/judged", help="Dataset to evaluate the rankings. May never be equal to dev_dataset.")
    parser.add_argument("--test_sparse_ranking_path", type=Path, default="/home/bvdb9/sparse_rankings/msmarco_passage-trec-dl-2019.judged-BM25-top10000.tsv", help="Path to the sparse ranking file.")
    parser.add_argument("--eval_metrics", type=str, nargs='+', default=["nDCG@10"], help="Metrics used for evaluation.")
    parser.add_argument("--alphas", type=float, nargs='+', default=np.arange(0.0, 1.00001, 0.2).tolist(), help="List of interpolation parameters for evaluation.")
    return parser.parse_args()


# TODO: remove this method
def print_settings(
    ) -> None:
    """
    Print the settings used for re-ranking.

    Deprecated:
        This method is deprecated and will be removed in future versions.
    """
    warnings.warn(
        "The print_settings() method is deprecated and will be removed in future versions.",
        FutureWarning,
    )

    settings_description: List[str] = [
        f"dataset={args.dataset}",
        f"index={args.index_path.name}",
        f"rerank_cutoff={args.rerank_cutoff}",
        f"encoding_method={args.encoding_method.name}",
    ]
    match args.encoding_method:  # Append method-specific settings
        case EncodingMethod.TCTCOLBERT:
            settings_description.extend([
                f"device={args.device}",
            ])
        case EncodingMethod.WEIGHTED_AVERAGE:
            settings_description.extend([
                f"k_avg={args.k_avg}",
                f"prob_dist={args.prob_dist.name}",
            ])
    print("\nSettings:\n\t" + ",\n\t".join(settings_description))


# TODO: remove this method
def results(
        sparse_ranking: Ranking, 
        dense_ranking: Ranking, 
        dataset: ir_datasets.Dataset,
    ) -> None:
    """
    Calculate and print the evaluation results for different interpolation parameters.

    Args:
        sparse_ranking (Ranking): The initial sparse ranking of documents.
        dense_ranking (Ranking): The re-ranked dense ranking of documents.
        dataset (ir_datasets.Dataset): Dataset to evaluate the rankings.

    Deprecated:
        This method is deprecated and will be removed in future versions.
    """
    warnings.warn(
        "The results() method is deprecated and will be removed in future versions.",
        FutureWarning,
    )

    print('Results:')
    eval_metrics_objects = []
    for metric_str in args.eval_metrics:
        metric_name, at_value = metric_str.split('@')
        eval_metrics_objects.append(getattr(measures, metric_name) @ int(at_value))

    # Print interpolated results for all different alpha values
    for alpha in args.alphas:
        interpolated_ranking = sparse_ranking.interpolate(dense_ranking, alpha)
        score = calc_aggregate(eval_metrics_objects, dataset.qrels_iter(), to_ir_measures(interpolated_ranking))
        ranking_type = (
            "Sparse" if alpha == 1 else 
            "Dense" if alpha == 0 else 
            "Interpolated"
        )
        print(f"\t{ranking_type} ranking (alpha={alpha}): {score}")

    # Estimate best interpolation alpha as weighted average of sparse- and dense-nDCG scores
    dense_score = calc_aggregate(eval_metrics_objects, dataset.qrels_iter(), to_ir_measures(dense_ranking))
    sparse_score = calc_aggregate(eval_metrics_objects, dataset.qrels_iter(), to_ir_measures(sparse_ranking))
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
    best_score = calc_aggregate(eval_metrics_objects, dataset.qrels_iter(), to_ir_measures(sparse_ranking.interpolate(dense_ranking, best_alpha)))
    print(f"\tEstimated best-nDCG@10 interpolated ranking (alpha~={best_alpha}): {best_score}")


def load_data(
        dataset_path: Path,
        ranking_path: Path,
    ) -> Tuple[ir_datasets.Dataset, Ranking]:
    """
    Load the dataset and the initial sparse ranking of documents.

    Args:
        dataset_path (Path): Path to the dataset.
        ranking_path (Path): Path to the initial sparse ranking of documents.

    Returns:
        Tuple[ir_datasets.Dataset, Ranking]: The dataset and the initial sparse ranking of documents.
    """
    dataset = pt.get_dataset(dataset_path)
    sparse_ranking = Ranking.from_file(
        ranking_path,
        queries={q.qid: q.query for q in dataset.get_topics().itertuples()}
    )
    return dataset, sparse_ranking


# TODO: Add profiling to re-ranking step
def rerank(
        index: Index,
        sparse_ranking: Ranking,
    ) -> Ranking:
    """
    Re-rank the documents based on the similarity to query embeddings.

    Args:
        index (Index): The index containing document embeddings.
        sparse_ranking (Ranking): The initial sparse ranking of documents.
    
    Returns:
        Ranking: The re-ranked dense ranking of documents.
    """
    sparse_ranking_cut = sparse_ranking.cut(args.rerank_cutoff)

    if isinstance(index.query_encoder, WeightedAvgEncoder):
        index.query_encoder.sparse_ranking = sparse_ranking_cut.cut(args.k_avg)

    dense_ranking = index(sparse_ranking_cut)
    return dense_ranking


# TODO [later]: Further improve efficiency of re-ranking step. Discuss with ChatGPT and Jurek.
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
    """
    pt.init()

    # Load index
    index: Index = OnDiskIndex.load(args.index_path)
    if args.in_memory:
        index = index.to_memory()

    # Parse eval_metrics to ir-measures' measure objects
    eval_metrics = []
    for metric_str in args.eval_metrics:
        metric_name, at_value = metric_str.split('@')
        eval_metrics.append(getattr(measures, metric_name) @ int(at_value))

    # Choose query encoder based on encoding_method
    match args.encoding_method:
        case EncodingMethod.TCTCOLBERT:
            index.query_encoder = TCTColBERTQueryEncoder("castorini/tct_colbert-msmarco", device=args.device)
        case EncodingMethod.WEIGHTED_AVERAGE:
            index.query_encoder = WeightedAvgEncoder(index, args.k_avg, args.prob_dist)
        case _:
            raise ValueError(f"Unsupported encoding method: {args.encoding_method}")
    assert index.query_encoder is not None, "Query encoder not set in index."

    if args.enable_validation:
        ### Validation and parameter tuning on dev set
        # TODO: Tune k_avg for WeightedAvgEncoder
        dev_dataset, dev_sparse_ranking = load_data(args.dev_dataset, args.dev_sparse_ranking_path)
        dev_dense_ranking = rerank(index, dev_sparse_ranking)
        ff_int = FFInterpolate(alpha=0.5)
        pt.GridSearch(
            to_ir_measures(dev_dense_ranking),
            {ff_int: {"alpha": args.alphas}},
            dev_dataset.queries_iter(),
            dev_dataset.qrels_iter(),
            verbose=True,
        )

    ### Evaluation on test set
    test_dataset, test_sparse_ranking = load_data(args.test_dataset, args.test_sparse_ranking_path)
    test_dense_ranking = rerank(index, test_sparse_ranking)
    results = pt.Experiment(
        [to_ir_measures(test_sparse_ranking), to_ir_measures(test_dense_ranking)],
        test_dataset.get_topics(),
        test_dataset.get_qrels(),
        eval_metrics=eval_metrics,
        names=["Sparse", "Sparse >> FF"],
    )
    print(f"Results on {args.test_dataset}:\n{results}")


if __name__ == '__main__':
    args = parse_args()
    main(args)
