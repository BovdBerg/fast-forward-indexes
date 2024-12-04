import argparse
import time
from copy import copy
from pathlib import Path

import pyterrier as pt
from tqdm import tqdm

from fast_forward.encoder.avg import WeightedAvgEncoder
from fast_forward.encoder.transformer import TCTColBERTQueryEncoder
from fast_forward.index.disk import OnDiskIndex
from fast_forward.ranking import Ranking
import pandas as pd


def parse_args():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.

    Arguments:
        Run the script with --help or -h to see the full list of arguments.
    """
    parser = argparse.ArgumentParser(description="Plot the re-ranking profiles.")
    parser.add_argument(
        "--profiles",
        type=str,
        nargs="*",
        default=["avg1"],
        help="The names of the profiles to plot.",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default="mem",
        choices=["disk", "mem"],
        help="The storage type of the index.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="The device used for re-ranking.",
    )
    parser.add_argument(
        "--q_dataset",
        type=str,
        default="irds:msmarco-passage/dev",
        help="The name of the dataset.",
    )
    parser.add_argument(
        "--index_path",
        type=Path,
        default="/home/bvdb9/indices/msm-psg/ff_index_msmpsg_TCTColBERT_opq.h5",
        help="Path to the index file.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="The number of runs. The minimum runtime of all runs is taken.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        help="The batch size for encoding queries.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """
    Create plots for CPU re-ranking runtime profiles.

    See parse_args() for command-line arguments.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    pt.init()
    dataset = pt.get_dataset(args.q_dataset)
    topics = dataset.get_topics()
    if args.samples:
        topics = topics.sample(n=args.samples, random_state=42)
    queries = topics["query"]

    print("Creating BM25 retriever via PyTerrier index...")
    sys_bm25 = pt.BatchRetrieve.from_dataset(
        "msmarco_passage", "terrier_stemmed", wmodel="BM25", verbose=True
    )
    sys_bm25_cut = ~sys_bm25 % 1000
    sparse_df = sys_bm25_cut(topics)
    sparse_ranking = Ranking(sparse_df.rename(columns={"qid": "q_id", "docno": "id"}))

    batch_size = 256
    index_tct = OnDiskIndex.load(
        args.index_path,
        TCTColBERTQueryEncoder("castorini/tct_colbert-msmarco", device=args.device),
        verbose=True,
        encoder_batch_size=256,
    )
    if args.storage == "mem":
        index_tct = index_tct.to_memory(2**14)

    index_avg = copy(index_tct)
    index_avg.query_encoder = WeightedAvgEncoder(index_avg)

    pipelines = {
        ("tct", index_tct),
        ("avg1", index_avg),
    }

    profiles = []
    for name, index in pipelines:
        # TODO: Take average runtime over all batches per run.
        runtimes = []
        for _ in tqdm(range(args.runs), desc=f"Profiling {name}", total=args.runs):
            t0 = time.time()
            index.encode_queries(queries, sparse_ranking)
            runtime = round(time.time() - t0, 2)
            runtimes.append(runtime)
        profile = {
            "name": name,
            "runtime": min(runtimes),
            "runtime_batch": round(min(runtimes) / batch_size, 2),
            "runtime_query": round(min(runtimes) / len(queries), 2),
        }
        print(f"Profile:{profile}")
        profiles.append(profile)

    profiles_df = pd.DataFrame(profiles)
    print(f"profiles_df: {profiles_df}")
    profile_dir = Path(f"profiles/{args.storage}_{args.device}")
    profile_dir.mkdir(parents=True, exist_ok=True)
    profiles_df.to_json(profile_dir / "profiles.json", indent=4)


if __name__ == "__main__":
    args = parse_args()
    main(args)
