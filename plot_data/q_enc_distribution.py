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
    other = total - q_emb_1 - _get_top_docs - q_emb_2

    fig, ax = plt.subplots()
    ax.set_ylabel("Re-ranking latency (ms)")
    ax.set_ylim(0, total * 1.15)

    bars = [
        ("Other", other, 0),
        ("Query embedding 2", q_emb_2, other),
        ("Get top docs", _get_top_docs, other + q_emb_2),
        ("Query embedding 1", q_emb_1, other + q_emb_2 + _get_top_docs),
    ]
    for label, value, bottom in bars:
        ax.bar(0, value, label=label, bottom=bottom)
        ax.text(
            0,
            bottom + value / 2,
            f"{label:}\n{value:.2f}ms ({(value / total) * 100:.1f}%)",
            ha="center",
            va="center",
            color="white",
            fontsize=11,
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

    # handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles[::-1], labels[::-1], bbox_to_anchor=(1, 0.5), loc='center left')
    # fig.tight_layout()

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
        "total": 0.55,
        "q_emb_1": 0.13,
        "_get_top_docs": 0.22,
        "q_emb_2": 0.11,
    }

    plot_runtimes(profiles)


if __name__ == "__main__":
    args = parse_args()
    main(args)
