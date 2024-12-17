import argparse
import cProfile
import pstats
import time
import warnings
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
from fast_forward.encoder.transformer_embedding import StandaloneEncoder
from fast_forward.index import Index
from fast_forward.index.disk import OnDiskIndex
from fast_forward.ranking import Ranking
from fast_forward.util.pyterrier import FFInterpolate, FFScore

PREV_RESULTS = Path("results.json")
warnings.filterwarnings(
    "ignore", category=FutureWarning, message=".*weights_only=False.*"
)


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
    # TODO [final]: Remove default paths (index_tct_path, index_emb_path, ckpt_path) form the arguments
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
        "--index_tct_path",
        type=Path,
        default="/home/bvdb9/indices/msm-psg/ff_index_msmpsg_TCTColBERT_opq.h5",
        help="Path to the index file.",
    )
    parser.add_argument(
        "--index_emb_path",
        type=Path,
        default="/home/bvdb9/indices/msm-psg/ff_index_L-0_opq.h5",
        help="Path to the index file.",
    )
    parser.add_argument(
        "--ckpt_emb_path",
        type=Path,
        default="/home/bvdb9/models/emb_768.ckpt",
        help="Path to the emb checkpoint file.",
    )
    parser.add_argument(
        "--ckpt_avg_path",
        type=Path,
        default="/home/bvdb9/fast-forward-indexes/lightning_logs/checkpoints/k_avg=10.ckpt",
        help="Path to the avg checkpoint file. Create it by running usage/train.py",
    )
    parser.add_argument(
        "--storage",
        type=str,
        choices=["disk", "mem"],
        default="mem",
        help="Storage type for the index.",
    )
    parser.add_argument(
        "--sparse_cutoff",
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
        default="LEARNED",
        help="Method to estimate query embeddings. Only used for EncodingMethod.WEIGHTED_AVERAGE.",
    )
    parser.add_argument(
        "--k_avg",
        type=int,
        default=10,
        help="Number of top-ranked documents to use. Only used for EncodingMethod.WEIGHTED_AVERAGE.",
    )
    # Chained WeightedAvgEncoder
    parser.add_argument(
        "--avg_chains",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--avg_int_alphas",
        type=float,
        nargs="*",
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
        default="map",  # Find official metrics for dataset version on https://ir-datasets.com/msmarco-passage.html
        help="Evaluation metric for pt.GridSearch on dev set.",
    )
    parser.add_argument(
        "--dev_sample_size",
        type=int,
        default=256,
        help="Number of queries to sample for validation.",
    )
    parser.add_argument(
        "--val_pipelines",
        type=str,
        nargs="*",
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
        nargs="*",
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
    # SAVING TO GOOGLE SHEETS
    parser.add_argument(
        "--gsheets_credentials",
        type=Path,
        default="/home/bvdb9/thesis-gsheets-credentials.json",
        help="Path to the Google Sheets credentials file.",
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
        f"sparse_cutoff={args.sparse_cutoff}, in_memory={args.storage == 'mem'}, device={args.device}",
        f"WeightedAvgEncoder: w_method={args.w_method.name}, k_avg={args.k_avg}",
    ]
    # Validation settings
    if args.val_pipelines:
        settings_description.append(
            f"Val: {args.val_pipelines}, '{args.dev_dataset}', samples={args.dev_sample_size}, α=[{','.join(map(str, args.alphas))}]"
        )

    print("Settings:\n\t" + "\n\t".join(settings_description))
    return "\n".join(settings_description)


def append_to_gsheets(results: pd.DataFrame, settings_str: str) -> None:
    """
    Append the results of an experiment to a Google Sheets document.

    Args:
        results (pd.DataFrame): Results of the experiment to append.
        settings_str (str): Settings used for the experiment
    """
    service_acc = gspread.service_account(filename=args.gsheets_credentials)
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


def profile(pipelines: List[Tuple[str, pt.Transformer, pt.Transformer]]) -> None:
    """
    Profile the re-ranking step to identify bottlenecks.
    View a profile by running `snakeviz path/to/profile.prof` and ctrl-clicking the link in your console.

    Args:
        pipelines (List[Tuple[str, pt.Transformer, pt.Transformer]): List of re-ranking pipelines to profile.
    """
    profile_dir = (
        Path(__file__).parent.parent / "profiles" / f"{args.storage}_{args.device}"
    )
    profile_dir.mkdir(parents=True, exist_ok=True)
    print(f"Creating re-ranking profiles in {profile_dir}...")

    prof_dataset = pt.get_dataset("irds:msmarco-passage/trec-dl-2019/judged")

    for name, system, _ in pipelines:
        with cProfile.Profile() as prof:
            system(prof_dataset.get_topics())

        prof_file = profile_dir / f"{name}.prof"
        prof.dump_stats(prof_file)
        ps = pstats.Stats(prof).sort_stats(pstats.SortKey.TIME)
        print(f"\t...{name} in {ps.total_tt:.2f}s, saved to {prof_file}")


# TODO [later]: Further improve efficiency of re-ranking step. Discuss with ChatGPT and Jurek.
def main(args: argparse.Namespace) -> None:
    """
    Re-ranking Stage: Create query embeddings and re-rank documents based on similarity to query embeddings.

    This script takes the initial ranking of documents and re-ranks them based on the similarity to the query embeddings.
    It uses various encoding methods and evaluation metrics to achieve this.

    See parse_args() for command-line arguments.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
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

    print("\033[96m")  # Prints in this method are cyan
    # Load dataset and create sparse retriever (e.g. BM25)
    dataset = pt.get_dataset(args.dataset)
    print("Creating BM25 retriever via PyTerrier index...")
    try:
        sys_bm25 = pt.BatchRetrieve.from_dataset(
            dataset, "terrier_stemmed", wmodel="BM25", memory=True
        )
    except:
        indexer = pt.IterDictIndexer(
            str(Path.cwd()),  # ignored but must be a valid path
            type=pt.index.IndexingType.MEMORY,
        )
        index_ref = indexer.index(dataset.get_corpus_iter(), fields=["text"])
        sys_bm25 = pt.BatchRetrieve(index_ref, wmodel="BM25", verbose=True, memory=True)
    sys_bm25_cut = ~sys_bm25 % args.sparse_cutoff

    # Create re-ranking pipeline based on TCTColBERTQueryEncoder (normal FF approach)
    index_tct = OnDiskIndex.load(
        args.index_tct_path,
        TCTColBERTQueryEncoder("castorini/tct_colbert-msmarco", device=args.device),
        verbose=args.verbose,
    )
    if args.storage == "mem":
        index_tct = index_tct.to_memory(2**15)
    ff_tct = FFScore(index_tct)
    int_tct = FFInterpolate(alpha=0.1)
    sys_tct = sys_bm25_cut >> ff_tct >> int_tct

    # Create re-ranking pipeline based on WeightedAvgEncoder
    index_avg = copy(index_tct)
    index_avg.query_encoder = WeightedAvgEncoder(
        index_avg,
        args.w_method,
        args.k_avg,
        ckpt_path=args.ckpt_avg_path,
        device=args.device,
    )

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

    query_encoder_emb = StandaloneEncoder(
        "google/bert_uncased_L-12_H-768_A-12",
        ckpt_path=args.ckpt_emb_path,
        device=args.device,
    )
    index_emb = OnDiskIndex.load(
        args.index_emb_path,
        query_encoder_emb,
        verbose=args.verbose,
    )
    if args.storage == "mem":
        index_emb = index_emb.to_memory(2**15)
    ff_emb = FFScore(index_emb)
    int_emb = FFInterpolate(alpha=0.2)
    sys_emb = sys_bm25_cut >> ff_emb >> int_emb

    int_tct_emb = FFInterpolate(alpha=0.5)
    sys_tct_emb = sys_tct >> ff_emb >> int_tct_emb

    int_avg_emb = FFInterpolate(alpha=0.5)
    sys_avg_emb = sys_avg[0] >> ff_emb >> int_avg_emb

    pipelines = [
        ("bm25", ~sys_bm25, None),
        ("tct", sys_tct, int_tct),
        ("avg1", sys_avg[0], int_avg[0]),
        ("emb", sys_emb, int_emb),
        ("tct_emb", sys_tct_emb, int_tct_emb),
        ("avg_emb", sys_avg_emb, int_avg_emb),
    ] + [
        (f"avg{i+1}", system, int_avg[i])
        for i, system in enumerate(sys_avg[1:], start=1)
    ]
    print("\033[0m")  # Reset print color

    if args.profiling:
        profile(pipelines)

    # TODO [maybe]: Improve validation by local optimum search for best alpha
    # Validation and parameter tuning on dev set
    if args.val_pipelines:
        dev_dataset = pt.get_dataset(args.dev_dataset)
        dev_queries = dev_dataset.get_topics()
        dev_qrels = dev_dataset.get_qrels()

        # Sample dev queries if dev_sample_size is set
        if args.dev_sample_size is not None:
            dev_queries = dev_queries.sample(
                n=args.dev_sample_size, random_state=42
            )  # Fixed seed for reproducibility.
            dev_qrels = dev_qrels[dev_qrels["qid"].isin(dev_queries["qid"])]

        # Validate pipelines in args.val_pipelines
        for name, system, tunable in pipelines:
            if tunable is None or (
                args.val_pipelines != ["all"] and name not in args.val_pipelines
            ):
                continue

            print(f"\nValidating pipeline: {name}...")
            pt.GridSearch(
                system,
                {tunable: {"alpha": args.alphas}},
                dev_queries,
                dev_qrels,
                metric=args.dev_eval_metric,
                verbose=True,
                batch_size=128,
            )

    # Evaluate pipelines on args.test_datasets
    if args.test_datasets:
        for test_dataset_name in args.test_datasets:
            test_dataset = pt.get_dataset(test_dataset_name)

            print(f"\nRunning final tests on {test_dataset_name}...")
            decimals = 5
            results = pt.Experiment(
                [pipeline for _, pipeline, _ in pipelines],
                test_dataset.get_topics(),
                test_dataset.get_qrels(),
                eval_metrics=eval_metrics,
                names=[
                    name if not tunable else f"{name}, α=[{tunable.alpha}]"
                    for name, _, tunable in pipelines
                ],
                round=decimals,
                verbose=True,
            )
            settings_str = print_settings()
            print(f"\nFinal results on {test_dataset_name}:\n{results}\n")

            # Save new results to Google Sheets
            if not PREV_RESULTS.exists() or str(results) != str(
                pd.read_json(PREV_RESULTS)
            ):
                results.to_json(PREV_RESULTS, indent=4)
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
