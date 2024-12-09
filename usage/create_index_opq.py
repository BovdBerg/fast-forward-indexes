import argparse
import logging
import time
from pathlib import Path

import torch

from fast_forward import Indexer, OnDiskIndex
from fast_forward.encoder.transformer_embedding import StandaloneEncoder
from fast_forward.quantizer.nanopq import NanoOPQ

logging.basicConfig(level=logging.DEBUG)


def parse_args():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.

    Arguments:
        Run the script with --help or -h to see the full list of arguments.
    """
    parser = argparse.ArgumentParser(
        description="Create an OPQ index from an existing Fast-Forward index."
    )
    parser.add_argument(
        "--src_path",
        type=Path,
        required=True,
        help="Path to the source index.",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        required=True,
        default="/home/bvdb9/models/emb_768.ckpt",
        help="Path to the encoder checkpoint.",
    )
    parser.add_argument(
        "--src_in_memory",
        action="store_true",
        help="Load the source index in memory.",
    )
    parser.add_argument(
        "--des_in_memory",
        action="store_true",
        help="Load the target index in memory.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print debug information.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """
    Create an OPQ index from an existing Fast-Forward index.

    Args:
        args (argparse.Namespace): Parsed command-line arguments. Run with --help or -h to see the full list of arguments.
    """
    start_time = time.time()

    print("Loading source index...")
    ff_index_source = OnDiskIndex.load(
        args.src_path,
        verbose=args.verbose,
    )
    if args.src_in_memory:
        ff_index_source.to_memory(2**14)

    print("Creating query encoder...")
    query_encoder = StandaloneEncoder(
        "google/bert_uncased_L-12_H-768_A-12",
        ckpt_path=args.ckpt_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    print("Creating target index...")
    ff_index_des = OnDiskIndex(
        Path(str(args.src_path.absolute()).replace(".h5", "_opq.h5")),
        query_encoder,
        overwrite=True,
        init_size=len(ff_index_source),
        verbose=args.verbose,
    )
    if args.des_in_memory:
        ff_index_des.to_memory(2**14)

    print("Indexing des index...")
    Indexer(
        ff_index_des,
        quantizer=NanoOPQ(96, 2048),
        batch_size=100000,
        quantizer_fit_batches=2,
    ).from_index(ff_index_source)

    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
