import argparse
import cProfile
import os
import pstats
from pathlib import Path

import ir_datasets
import pyterrier as pt
import torch
from ir_measures import AP, RR, calc_aggregate, nDCG

from fast_forward import Mode, OnDiskIndex, Ranking
from fast_forward.encoder.transformer import TCTColBERTQueryEncoder
from fast_forward.util import to_ir_measures


def parse_args():
    """
    Parse command-line arguments for the re-ranking script.

    Returns:
        argparse.Namespace: Parsed command-line arguments.

    Arguments:
        Run the script with --help or -h to see the full list of arguments.
    """
    parser = argparse.ArgumentParser(description="Create profile for re-ranking.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="The device name for the Fast-Forward index.",
    )
    parser.add_argument(
        "--k_s",
        type=int,
        default=1000,
        help="The number of documents to retrieve from the index.",
    )
    parser.add_argument(
        "--in_memory",
        action="store_true",
        help="Whether to load the index into memory.",
    )
    parser.add_argument(
        "--index_path",
        type=Path,
        default="/home/bvdb9/indices/msm-psg/ff_index_msmpsg_TCTColBERT_opq.h5",
        help="The path to the Fast-Forward index.",
    )
    parser.add_argument(
        "--sparse_ranking_path",
        type=Path,
        default=Path(
            "/home/bvdb9/sparse_rankings/msmarco-passage-test2019-sparse10000.txt"
        ),
        help="The path to the sparse ranking.",
    )
    parser.add_argument(
        "--testset",
        type=str,
        default="msmarco-passage/trec-dl-2019",
        help="The dataset to use for evaluation.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    pt.init()
    testset = ir_datasets.load(args.testset)

    # Create profile directory
    mem = "mem" if args.in_memory else "disk"
    profile_dir = f"profiles/{args.index_path}/{args.device}_k{args.k_s}_{mem}/"
    if not os.path.exists(profile_dir):
        os.makedirs(profile_dir)

    q_encoder = TCTColBERTQueryEncoder(
        "castorini/tct_colbert-msmarco", device=args.device
    )
    ff_index = OnDiskIndex.load(
        args.index_path, query_encoder=q_encoder, mode=Mode.MAXP
    )
    if args.in_memory:
        ff_index = ff_index.to_memory()

    sparse_ranking = Ranking.from_file(
        args.sparse_ranking_path,
        {q.query_id: q.text for q in testset.queries_iter()},
    )

    # standard re-ranking, probably takes a few min
    with cProfile.Profile() as profile:
        dense_ranking = ff_index(sparse_ranking.cut(args.k_s))

    stats = pstats.Stats(profile)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.dump_stats(profile_dir + "rerank_ff.prof")

    alpha: float = 0.2
    eval_metrics = [nDCG @ 10, RR(rel=2) @ 10, AP(rel=2) @ 1000]
    print(
        "Sparse ranking: ",
        calc_aggregate(
            eval_metrics, testset.qrels_iter(), to_ir_measures(sparse_ranking)
        ),
        "\nDense ranking:",
        calc_aggregate(
            eval_metrics, testset.qrels_iter(), to_ir_measures(dense_ranking)
        ),
        f"\nFF (alpha={alpha}): ",
        calc_aggregate(
            eval_metrics,
            testset.qrels_iter(),
            to_ir_measures(sparse_ranking.interpolate(dense_ranking, alpha)),
        ),
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
