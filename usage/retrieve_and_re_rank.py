import argparse
import cProfile
import pstats
import time
from copy import copy
from pathlib import Path
from typing import List, Tuple

import gspread
import numpy as np
import pandas as pd
import pyterrier as pt
import torch
from gspread_dataframe import set_with_dataframe
from gspread_formatting import (
    Border,
    Borders,
    CellFormat,
    TextFormat,
    format_cell_range,
)
from ir_measures import measures

from fast_forward.encoder.avg import W_METHOD, WeightedAvgEncoder
from fast_forward.encoder.transformer import TCTColBERTQueryEncoder
from fast_forward.index.disk import OnDiskIndex
from fast_forward.ranking import Ranking
from fast_forward.util.pyterrier import FFInterpolate, FFScore
from tqdm import tqdm

PREVIOUS_RESULTS_FILE = Path("results.json")


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
        "--retrieval_depth",
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
        "--remarks",
        type=str,
        default="",
        help="Additional remarks about the experiment. Will be added to the Google Sheets file.",
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
        default=1,
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
        "--dev_dataset",
        type=str,
        default="irds:msmarco-passage/dev/judged",
        help="Dataset to use for validation.",
    )
    parser.add_argument(
        "--dev_eval_metric",
        type=str,
        default="ndcg_cut_10",  # Find official metrics for dataset version on https://ir-datasets.com/msmarco-passage.html
        help="Evaluation metric for pt.GridSearch on dev set.",
    )
    parser.add_argument(
        "--dev_sample_size",
        type=int,
        default=1024,
        help="Number of queries to sample for validation.",
    )
    parser.add_argument(
        "--val_pipelines",
        type=str,
        nargs="+",
        default=[],
        help="List of pipelines to validate, based on exact pipeline names.",
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
    # PROFILING
    parser.add_argument(
        "--profiling",
        action="store_true",
        help="Profile the re-ranking step.",
    )
    return parser.parse_args()


def print_settings() -> None:
    """
    Print general settings used for re-ranking.

    Args:
        pipeline (pt.Transformer): The pipeline used for re-ranking.

    Returns:
        str: A string representation of the settings.
    """
    # General settings
    settings_description: List[str] = [
        f"retrieval_depth={args.retrieval_depth}, in_memory={args.in_memory}, device={args.device}",
        f"WeightedAvgEncoder: w_method={args.w_method.name}, k_avg={args.k_avg}",
    ]
    # Validation settings
    if args.val_pipelines:
        settings_description.append(f"Val: {args.val_pipelines}, α={args.alphas}")

    print("Settings:\n\t" + "\n\t".join(settings_description))
    return "\n".join(settings_description)


def append_to_gsheets(results: pd.DataFrame, settings_str: str) -> None:
    """
    Append the results of an experiment to a Google Sheets document.

    Args:
        results (pd.DataFrame): Results of the experiment to append.
        settings_str (str): Settings used for the experiment
    """
    service_acc = gspread.service_account(
        filename="/home/bvdb9/thesis-gsheets-credentials.json"
    )
    spreadsheet = service_acc.open("Thesis Results")
    worksheet = spreadsheet.sheet1
    print(f"Saving results to Google Sheets file...")

    # Prepend date and time fields to the results
    results["Remarks"] = args.remarks
    results["Date"] = time.strftime("%Y-%m-%d %H:%M")
    results["Settings"] = settings_str
    prepend_cols = ["Remarks", "Date", "Settings"]
    results = results[
        prepend_cols + [col for col in results.columns if col not in prepend_cols]
    ]

    first_row = len(worksheet.get_all_values()) + 1
    last_row = first_row + len(results) - 1

    # Add horizontal line above the data
    format_cell_range(
        worksheet,
        f"A{first_row}:G{first_row}",
        CellFormat(borders=Borders(top=Border("SOLID"))),
    )

    # Append the results
    set_with_dataframe(worksheet, results, row=first_row, include_column_header=False)

    # Merge cells which share the same values
    for col in ["A", "B", "C"]:
        worksheet.merge_cells(f"{col}{first_row}:{col}{last_row}")

    # Highlight the row with the highest nDCG@10 value in bold (excl. baselines)
    non_baselines = results.iloc[2:]
    if not non_baselines.empty:
        max_ndcg10_index = non_baselines["nDCG@10"].idxmax()
        max_ndcg10_row = first_row + max_ndcg10_index
        format_cell_range(
            worksheet,
            f"A{max_ndcg10_row}:G{max_ndcg10_row}",
            CellFormat(textFormat=TextFormat(bold=True)),
        )


# TODO [later]: Further improve efficiency of re-ranking step. Discuss with ChatGPT and Jurek.
# TODO: Split the main function into smaller functions for better readability.
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
        sys_bm25 = pt.BatchRetrieve.from_dataset(
            dataset, "terrier_stemmed", wmodel="BM25", verbose=True
        )
    except:
        indexer = pt.IterDictIndexer(
            str(Path.cwd()),  # ignored but must be a valid path
            type=pt.index.IndexingType.MEMORY,
        )
        index_ref = indexer.index(dataset.get_corpus_iter(), fields=["text"])
        sys_bm25 = pt.BatchRetrieve(index_ref, wmodel="BM25", verbose=True)
    sys_bm25_cut = ~sys_bm25 % args.retrieval_depth

    # Create re-ranking pipeline based on TCTColBERTQueryEncoder (normal FF approach)
    index_tct = OnDiskIndex.load(
        args.index_path,
        TCTColBERTQueryEncoder("castorini/tct_colbert-msmarco", device=args.device),
        verbose=args.verbose,
    )
    if args.in_memory:
        # TODO: somehow --in_memory avg_1 scores higher than OnDiskIndex
        index_tct = index_tct.to_memory(2**14)
    ff_tct = FFScore(index_tct)
    int_tct = FFInterpolate(alpha=0.1)
    sys_tct = sys_bm25_cut >> ff_tct >> int_tct

    # Create re-ranking pipeline based on WeightedAvgEncoder
    index_avg = copy(index_tct)
    index_avg.query_encoder = WeightedAvgEncoder(index_avg, args.w_method, args.k_avg)

    # Create int_avg array of length 4 with each alpha value
    avg_chains = max([1, args.avg_chains])
    avg_int_alphas = args.avg_int_alphas + [0.5] * (
        avg_chains - len(args.avg_int_alphas)
    )
    int_avg = [FFInterpolate(alpha=a) for a in avg_int_alphas[:avg_chains]]
    ff_avg = FFScore(index_avg)
    sys_avg = [sys_bm25_cut]
    for i in range(len(int_avg)):
        sys_avg.append(sys_avg[-1] >> ff_avg >> int_avg[i])
    sys_avg = sys_avg[1:]  # Remove 1st pipeline (bm25) from avg_pipelines

    # TODO [later]: Try using best performing sys_avg in sys_avg_tct rather than the first
    # Re-ranking pipelines based on combining TCTColBERT and WeightedAvgEncoder
    int_avg_tct = FFInterpolate(alpha=0.2)
    sys_avg_tct = sys_avg[0] >> ff_tct >> int_avg_tct

    # View a profile in your webbrowser by running `tuna path/to/profile.prof --port=8000` and opening http://localhost:8000
    if args.profiling:
        profile_dir = Path(__file__).parent.parent / "profiles"
        profile_dir.mkdir(parents=True, exist_ok=True)
        print(f"Creating re-ranking profiles in {profile_dir}...")

        prof_dataset = pt.get_dataset("irds:msmarco-passage/trec-dl-2019/judged")
        prof_queries = prof_dataset.get_topics()

        sparse_df = sys_bm25_cut.transform(prof_queries)
        sparse_ranking = Ranking(sparse_df.rename(columns={"qid": "q_id", "docno": "id"}))
        index_avg.query_encoder.sparse_ranking = sparse_ranking

        print("Creating avg_1 profile...")
        with cProfile.Profile() as profile:
            avg_ranking = index_avg(sparse_ranking)
            int_ranking = sparse_ranking.interpolate(avg_ranking, int_avg[0].alpha)
        pstats.Stats(profile) \
            .sort_stats(pstats.SortKey.TIME) \
            .dump_stats(profile_dir / "avg_1.prof")

        print("Creating tct profile...")
        with cProfile.Profile() as profile:
            tct_ranking = index_tct(int_ranking)
            sparse_ranking.interpolate(tct_ranking, int_tct.alpha)
        pstats.Stats(profile) \
            .sort_stats(pstats.SortKey.TIME) \
            .dump_stats(profile_dir / "tct.prof")

        print("Creating avg_tct profile...")
        with cProfile.Profile() as profile:
            avg_ranking = index_avg(sparse_ranking)
            int_ranking = sparse_ranking.interpolate(avg_ranking, int_avg[0].alpha)
            avg_tct_ranking = index_tct(int_ranking)
            sparse_ranking.interpolate(avg_tct_ranking, int_avg_tct.alpha)
        pstats.Stats(profile) \
            .sort_stats(pstats.SortKey.TIME) \
            .dump_stats(profile_dir / "avg_tct.prof")

    # TODO [maybe]: Improve validation by local optimum search for best alpha
    # Validation and parameter tuning on dev set
    if args.val_pipelines:
        dataset_str = args.dev_dataset
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

        # Validate pipelines in args.val_pipelines.
        # TODO: merge val_pipelines with test_pipelines and prof_pipelines logic
        val_pipelines = [
            # bm25 has no tunable parameters, so it is not included here
            (sys_tct, [int_tct], "tct"),
            (sys_avg[0], [int_avg[0]], "avg_1"),
            (sys_avg_tct, [int_avg_tct], f"avg_tct"),
        ] + [
            (pipeline, [int_avg[i]], f"avg_{i+1}")
            for i, pipeline in enumerate(sys_avg[1:], start=1)
        ]

        for system, tunable_alphas, name in val_pipelines:
            if args.val_pipelines == ["all"] or name in args.val_pipelines:
                print(f"\nValidating pipeline: {name}...")
                pt.GridSearch(
                    system,
                    {tunable: {"alpha": args.alphas} for tunable in tunable_alphas},
                    dev_queries,
                    dev_qrels,
                    metric=args.dev_eval_metric,  # Find official metrics for dataset version on https://ir-datasets.com/msmarco-passage.html
                    verbose=True,
                    batch_size=128,
                )

    if args.test_datasets:
        # Define which pipelines to evaluate on test sets
        test_pipelines = [
            (~sys_bm25, "bm25"),
            (sys_tct, f"tct, α={int_tct.alpha}"),
            (sys_avg[0], f"avg_1, α={int_avg[0].alpha}"),
            (sys_avg_tct, f"avg_tct, α={int_avg_tct.alpha}"),
        ] + [
            (
                sys_avg[i],
                f"avg_{i+1}, α=[{','.join(str(int_avg[j].alpha) for j in range(i+1))}]",
            )
            for i in range(1, len(int_avg))
        ]
        test_pipelines = [(pipeline, desc) for pipeline, desc in test_pipelines]

        # Final evaluation on test sets
        for test_dataset_name in args.test_datasets:
            test_dataset = pt.get_dataset(test_dataset_name)
            test_queries = test_dataset.get_topics()

            print(f"\nRunning final tests on {test_dataset_name}...")
            decimals = 5
            results = pt.Experiment(
                [pipeline for pipeline, _ in test_pipelines],
                test_queries,
                test_dataset.get_qrels(),
                eval_metrics=eval_metrics,
                names=[desc for _, desc in test_pipelines],
                round=decimals,
                verbose=True,
            )
            settings_str = print_settings()
            print(f"\nFinal results on {test_dataset_name}:\n{results}\n")

            # SAVING TO GOOGLE SHEETS
            results_str = results.round(decimals).astype(str)
            prev_results_str = (
                pd.read_json(PREVIOUS_RESULTS_FILE).round(decimals).astype(str)
            )
            if not PREVIOUS_RESULTS_FILE.exists() or not results_str.equals(
                prev_results_str
            ):
                results.to_json(PREVIOUS_RESULTS_FILE, indent=4)
                settings_str += f"\nTest: '{test_dataset_name}'"
                append_to_gsheets(results, settings_str)
            else:
                print(
                    "Results have not changed since the last run. Skipping Google Sheets update."
                )

    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
