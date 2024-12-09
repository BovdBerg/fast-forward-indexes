import argparse
import cProfile
import pstats
import time
from copy import copy
from pathlib import Path

import pyterrier as pt
from tqdm import tqdm

from fast_forward.encoder.avg import WeightedAvgEncoder
from fast_forward.encoder.transformer import TCTColBERTQueryEncoder
from fast_forward.encoder.transformer_embedding import StandaloneEncoder
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
        "--runs",
        type=int,
        default=5,
        help="The number of runs. The minimum runtime of all runs is taken.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="The batch size for encoding queries.",
    )
    parser.add_argument(
        "--batches",
        type=int,
        help="The number of batches to process.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print additional information.",
    )
    parser.add_argument(
        "--index_tct_path",
        type=Path,
        default="/home/bvdb9/indices/msm-psg/ff_index_msmpsg_TCTColBERT_opq.h5",
        help="The path to the TCTColBERT index.",
    )
    parser.add_argument(
        "--index_emb_path",
        type=Path,
        default="/home/bvdb9/indices/msm-psg/ff_index_L-0_opq.h5",
        help="The path to the TransformerEmbedding index.",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        default="/home/bvdb9/models/emb_768.ckpt",
        help="The path to the emb checkpoint file.",
    )
    return parser.parse_args()


def print_settings(prof_dir: Path) -> None:
    """
    Print general settings used for re-ranking.
    """
    settings_description = [
        f"prof_dir={prof_dir}",
        f"verbose={args.verbose}",
    ]
    print("Settings:\n\t" + "\n\t".join(settings_description))


# TODO [important]: avg1 sometimes gives an error (easier to detect with many queries). e.g. `python usage/profiling.py --runs=3 --batch_size=256 --batches=2 --storage=disk --verbose`
def main(args: argparse.Namespace) -> None:
    """
    Create plots for CPU re-ranking runtime profiles.

    See parse_args() for command-line arguments.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    start_time = time.time()

    pt.init()
    dataset = pt.get_dataset("irds:msmarco-passage/dev")
    topics = dataset.get_topics()
    if args.batches:
        samples = args.batches * args.batch_size
        topics = topics.sample(n=samples)
    queries = list(topics["query"])

    date = time.strftime("%Y-%m-%d_%H-%M")
    prof_dir = Path("profiles") / f"{args.storage}_{args.device}_batch={args.batch_size}" / f"{args.runs}x{len(queries)}q" / date
    prof_dir.mkdir(parents=True, exist_ok=True)
    prof_file = prof_dir / "_profiles.json"
    print_settings(prof_dir)

    print("Creating BM25 retriever via PyTerrier index...")
    sys_bm25 = pt.BatchRetrieve.from_dataset(
        "msmarco_passage", "terrier_stemmed", wmodel="BM25", verbose=True
    )
    sys_bm25_cut = ~sys_bm25 % 1000
    sparse_df = sys_bm25_cut(topics)
    sparse_ranking = Ranking(sparse_df.rename(columns={"qid": "q_id", "docno": "id"}))

    index_tct = OnDiskIndex.load(
        args.index_tct_path,
        TCTColBERTQueryEncoder("castorini/tct_colbert-msmarco", device=args.device),
        verbose=args.verbose,
        encoder_batch_size=args.batch_size,
    )
    if args.storage == "mem":
        index_tct = index_tct.to_memory(2**14)

    index_avg = copy(index_tct)
    index_avg.query_encoder = WeightedAvgEncoder(index_avg)

    query_encoder_emb = StandaloneEncoder(
        "google/bert_uncased_L-12_H-768_A-12",
        ckpt_path=args.ckpt_path,
        device=args.device,
    )
    index_emb = OnDiskIndex.load(
        args.index_emb_path,
        query_encoder_emb,
        verbose=args.verbose,
    )
    if args.storage == "mem":
        index_emb = index_emb.to_memory(2**14)

    pipelines = [
        ("tct", index_tct),
        ("avg1", index_avg),
        ("emb", index_emb),
    ]

    profiles = []
    runtime_baseline = None
    for name, index in pipelines:
        runtimes = []
        pipeline_dir = prof_dir / name
        pipeline_dir.mkdir(parents=True, exist_ok=True)

        for run in tqdm(range(args.runs), desc=f"Profiling {name}", total=args.runs):
            with cProfile.Profile() as pr:
                index.encode_queries(queries, sparse_ranking)
            pr = pstats.Stats(pr)
            pr.dump_stats(pipeline_dir / f"{name}_{run}.prof")
            runtimes.append(round(pr.total_tt, 2))

        runtime = min(runtimes)
        if runtime_baseline is None:
            runtime_baseline = runtime
        profile = {
            "name": name,
            "runtime": runtime,
            "speedup": round(runtime_baseline / runtime, 2),
            "runtime_batch": round(runtime / (len(queries) / args.batch_size), 2),
            "runtime_query": round(runtime / len(queries), 2),
        }
        print(f"Profile:{profile}")
        profiles.append(profile)

    # Add a profile "avg_emb" that combines the avg1 and emb profiles by summing their runtime.
    runtime_avg_emb = sum(profile["runtime"] for profile in profiles if profile["name"] in ["avg1", "emb"])
    profiles.append({
        "name": "avg_emb",
        "runtime": runtime_avg_emb,
        "speedup": round(runtime_baseline / runtime_avg_emb, 2),
        "runtime_batch": round(runtime_avg_emb / (len(queries) / args.batch_size), 2),
        "runtime_query": round(runtime_avg_emb / len(queries), 2),
    })

    profiles = pd.DataFrame(profiles)
    profiles.to_json(prof_file, indent=4)
    print_settings(prof_dir)
    print(f"profiles:\n{profiles}")

    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
