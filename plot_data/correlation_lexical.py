import argparse
from copy import copy
import time
import warnings
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pyterrier as pt
import torch

from fast_forward.encoder.transformer import TCTColBERTQueryEncoder
from fast_forward.encoder.avg import AvgEmbQueryEstimator
from fast_forward.index.disk import OnDiskIndex
from fast_forward.util.pyterrier import FFScore, FFInterpolate

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
    parser.add_argument(
        "--index_path",
        type=Path,
        default="/home/bvdb9/indices/msm-psg/ff_index_msmpsg_TCTColBERT_opq.h5",
        help="Path to the index file.",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        default="/home/bvdb9/fast-forward-indexes/lightning_logs/checkpoints/new_est/10d_tokW+sp+pad_0.00186.ckpt",
        help="Path to the checkpoint file.",
    )
    parser.add_argument(
        "--combine_n_queries",
        type=int,
        default=1,
        help="Number of queries to combine for average performance.",
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
    return parser.parse_args()


def plot_performances(performances: dict):
    """
    Plot the performances of different pipelines for each query.

    Args:
        performances (dict): Dictionary containing performance results for each query.
    """
    plt.figure(figsize=(18, 6))
    
    queries = list(performances.keys())
    pipeline_names = performances[queries[0]]['name'] if performances[queries[0]] is not None else []
    for i, query in enumerate(queries):
        for j, score in enumerate(performances[query]['ndcg_cut_10']):
            markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h']
            plt.scatter(query, score, label=pipeline_names[j] if i == 0 else "", color=f"C{j % 10}", marker=markers[j % len(markers)])

    plt.xlabel(f"Queries sorted on BM25 performance", fontsize=18)
    plt.ylabel("nDCG@10", fontsize=18)
    plt.xlim(-0.25, len(performances) - 1 + 0.25)
    plt.ylim(-0.01, 1.01)
    # plt.title("Correlation of performances between pipelines")
    plt.legend(loc='lower right', fontsize=18)

    plt.xticks(ticks=range(len(performances)), labels=[])  # Keep ticks but remove tick labels

    plt.tight_layout()
    plt.savefig("plot_data/figures/correlation_lexical.png")
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

    cache_file = Path(f"cache/performance_per_query_cache.pt")
    if cache_file.exists():
        performances = torch.load(cache_file, map_location=args.device)
        print(f"Loaded df from cache file: {cache_file}")
    else:
        pt.init()
        test_dataset = pt.get_dataset("irds:msmarco-passage/trec-dl-2019/judged")
        test_topics = test_dataset.get_topics()
        print(f"test_topics: {test_topics}")

        bm25 = pt.BatchRetrieve.from_dataset(
            pt.get_dataset("msmarco_passage"), "terrier_stemmed", wmodel="BM25", memory=True
        )
        bm25 = ~bm25 % 1000

        index = OnDiskIndex.load(args.index_path)
        if args.storage == "mem":
            index = index.to_memory(2**15)
        index.query_encoder = AvgEmbQueryEstimator(
            index=index,
            n_docs=10,
            device=args.device,
            ckpt_path=args.ckpt_path,
        )
        ff_avg = FFScore(index)
        int_avg = FFInterpolate(alpha=0.02)
        avg_0 = ~bm25 >> ff_avg
        avg = ~bm25 >> ff_avg >> int_avg

        index_avgD = copy(index)
        index_avgD.query_encoder = AvgEmbQueryEstimator(
            index=index,
            n_docs=10,
            device=args.device,
            ckpt_path=args.ckpt_path,
            docs_only=True,
        )
        ff_avgD = FFScore(index_avgD)
        int_avgD = FFInterpolate(alpha=0.09)
        avgD_0 = ~bm25 >> ff_avgD
        avgD = ~bm25 >> ff_avgD >> int_avgD

        index_tct = copy(index)
        index_tct.query_encoder = TCTColBERTQueryEncoder(
            "castorini/tct_colbert-msmarco",
            device=args.device,
        )
        ff_tct = FFScore(index_tct)
        int_tct = FFInterpolate(alpha=0.03)
        tct_0 = ~bm25 >> ff_tct
        tct = ~bm25 >> ff_tct >> int_tct

        pipelines = [
            ("bm25", "BM25", ~bm25, None),
            # ("tct_0", "TCT-ColBERT (no int.)", tct_0, None),
            ("tct", "TCT-ColBERT", tct, int_tct),
            # ("avgD_0", "AvgEmbD (no int.)", avgD_0, None),
            ("avgD", "AvgEmbD", avgD, int_avgD),
            # ("avg", "AvgEmb (no int.)", avg_0, None),
            ("avg", "AvgEmb", avg, int_avg),
        ]

        performances = {qno: None for qno in range(len(test_topics))}
        for qno in range(0, len(test_topics)):
            q = test_topics.iloc[qno : qno + 1]
            print(f"q:\n{q}")
            results = pt.Experiment(
                [pipeline for _, _, pipeline, _ in pipelines],
                q,
                test_dataset.get_qrels(),
                eval_metrics=["ndcg_cut_10"],
                names=[
                    name if not tunable else f"{name}, α=[{tunable.alpha}]"
                    for _, name, _, tunable in pipelines
                ],
                round=3,
                verbose=True,
            )
            print(results)
            performances[qno] = results
        torch.save(performances, cache_file)

    # Remove (no int.) pipelines
    for i in performances:
        performances[i] = performances[i][~performances[i]['name'].str.contains(r'\(no int.\)')]
        # performances[i] = performances[i][~performances[i]['name'].str.contains(r'TCT-ColBERT')]
        # performances[i] = performances[i][~performances[i]['name'].str.contains(r'AvgEmb,')]
        performances[i].reset_index(drop=True, inplace=True)

    # Rename AvgEmbD pipeline to AvgEmb_docs
    for i in performances:
        for j in range(len(performances[i])):
            if performances[i].iloc[j, 0] == "AvgEmbD":
                performances[i].iloc[j, 0] = "AvgEmb_docs"

    # Average each pipeline's performance per 3 queries
    combine_n_queries = args.combine_n_queries
    for i in range(0, len(performances), combine_n_queries):
        combined_performance = performances[i].copy()
        for j in range(1, combine_n_queries):
            if i + j < len(performances):
                for k in range(len(combined_performance)):
                    combined_performance.iloc[k, 1] += performances[i + j].iloc[k, 1]
        for k in range(len(combined_performance)):
            combined_performance.iloc[k, 1] /= combine_n_queries
        performances[i] = combined_performance

    # Remove the extra entries
    for i in range(len(performances) - 1, -1, -1):
        if i % combine_n_queries != 0:
            del performances[i]

    # Sort performances on BM25 performance
    performances = dict(sorted(performances.items(), key=lambda item: item[1]['ndcg_cut_10'][0] if item[1] is not None else float('inf')))

    # Reindex
    performances = {i: performances[qno] for i, qno in enumerate(performances)}

    print("\033[0m")  # Reset color
    print(f"performances: {performances}")

    plot_performances(performances)

    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
