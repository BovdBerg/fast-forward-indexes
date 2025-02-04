import argparse
import os
import time
import warnings
from copy import copy
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyterrier as pt
import torch
from gspread.auth import service_account
from gspread_dataframe import set_with_dataframe
from gspread_formatting import (
    Border,
    Borders,
    CellFormat,
    TextFormat,
    format_cell_range,
)
from ir_measures import measures

from fast_forward.encoder.avg import AvgEmbQueryEstimator
from fast_forward.encoder.transformer import TCTColBERTQueryEncoder
from fast_forward.encoder.transformer_embedding import StandaloneEncoder
from fast_forward.index.disk import OnDiskIndex
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
    # TODO [final]: Remove default paths (index_path, ckpt_path) form the arguments
    parser.add_argument(
        "--index_path",
        type=Path,
        default="/home/bvdb9/indices/msm-psg/ff_index_msmpsg_TCTColBERT_opq.h5",
        help="Path to the index file.",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        default="/home/bvdb9/fast-forward-indexes/lightning_logs/checkpoints/n_docs=10+special_0.00207.ckpt",
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
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for encoding queries.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="Number of queries to sample for validation.",
    )
    # WeightedAvgEncoder
    parser.add_argument(
        "--n_docs",
        type=int,
        default=10,
        help="Number of top-ranked documents to use. Only used for EncodingMethod.WEIGHTED_AVERAGE.",
    )
    return parser.parse_args()


def load_or_generate_df():
    cache_file = Path(f"cache/performance_per_query_cache_{args.n_docs}_{args.samples}.pt")

    if cache_file.exists():
        df = torch.load(cache_file, map_location=args.device)
        print(f"Loaded df from cache file: {cache_file}")
    else:
        dataset = pt.get_dataset("msmarco_passage")
        sys_bm25 = pt.BatchRetrieve.from_dataset(
            dataset, "terrier_stemmed", wmodel="BM25", memory=True
        )
        sys_bm25_cut = ~sys_bm25 % args.n_docs

        # Create re-ranking pipeline based on WeightedAvgEncoder
        index = OnDiskIndex.load(args.index_path)
        if args.storage == "mem":
            index = index.to_memory(2**15)
        index.query_encoder = AvgEmbQueryEstimator(
            index=index,
            n_docs=args.n_docs,
            device=args.device,
            ckpt_path=args.ckpt_path,
        )
        ff_avg = FFScore(index)
        sys_avg = sys_bm25_cut >> ff_avg

        topics = (
            pt.get_dataset("irds:msmarco-passage/dev")
            .get_topics()
            .sample(args.samples, random_state=42)
        )

        df = sys_avg.transform(topics)
        torch.save(df, cache_file)
        print(f"Saved df to cache file: {cache_file}")

    return df


def plot_data(bm25_scores: List[float], avg_scores: List[float]):
    fig, ax = plt.subplots()
    ax.set_xlabel("BM25 scores")
    ax.set_ylabel("AvgEmb score")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.scatter(bm25_scores, avg_scores, alpha=0.5)

    ax.grid(True)

    fig.savefig(
        "plot_data/figures/performance_per_query.png", transparent=True
    )
    plt.show()


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
    print("\033[96m")  # Prints during setup are colored cyan
    pt.init()

    df = load_or_generate_df()
    print("\033[0m")  # Reset color
    print(f"df:\n{df}")

    bm25_scores = df["score_0"]
    avg_scores = df["score"]

    # Normalize the scores
    bm25_scores = (bm25_scores - bm25_scores.min()) / (
        bm25_scores.max() - bm25_scores.min()
    )
    avg_scores = (avg_scores - avg_scores.min()) / (avg_scores.max() - avg_scores.min())

    plot_data(bm25_scores, avg_scores)

    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
