import argparse
from typing import Any, Dict, List

import numpy as np
from matplotlib import pyplot as plt


def parse_args():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.

    Arguments:
        Run the script with --help or -h to see the full list of arguments.
    """
    parser = argparse.ArgumentParser()
    return parser.parse_args()


def plot_runtimes(profiles: Dict[str, float]):
    total = profiles["total"]
    q_emb_1 = profiles["q_emb_1"]
    _get_top_docs = profiles["_get_top_docs"]
    q_emb_2 = profiles["q_emb_2"]

    fig, ax = plt.subplots()
    ax.set_ylabel("Re-ranking latency (ms)")
    ax.set_ylim(0, total * 1.15)

    bars = [
        ("Query embedding 2", q_emb_2, 0),
        ("Get top docs", _get_top_docs, q_emb_2),
        ("Query embedding 1", q_emb_1, q_emb_2 + _get_top_docs),
    ]
    for label, value, bottom in bars:
        ax.bar(0, value, label=label, bottom=bottom)
        ax.text(
            0,
            bottom + value / 2,
            f"{label:}\n{value}ms ({(value / total) * 100:.1f}%)",
            ha="center",
            va="center",
            color="white",
            fontsize=14,
            fontweight="bold",
        )

    # Above bar 0, add the total runtime
    ax.text(
        0,
        total * 1.05,
        f"Total: {total:.2f} ms",
        ha="center",
        va="center",
        color="black",
        fontsize=14,
        fontweight="bold",
    )

    ax.set_xticks([])

    fig.savefig("plot_data/figures/q_enc_distribution.png", transparent=True)
    plt.show()


def main(args: argparse.Namespace) -> None:
    """
    Create plots for CPU re-ranking runtime profiles.

    See parse_args() for command-line arguments.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    profiles = {
        "total": 0.59,
        "q_emb_1": 0.16,
        "_get_top_docs": 0.27,
        "q_emb_2": 0.16,
    }

    plot_runtimes(profiles)


if __name__ == "__main__":
    args = parse_args()
    main(args)
