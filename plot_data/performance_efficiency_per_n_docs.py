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


def plot_runtimes(profiles: List[Dict[str, Any]]):
    def extract_runtimes(key: str) -> np.ndarray:
        return np.array([profile[key] for profile in profiles])

    names = extract_runtimes("name")
    latency = extract_runtimes("latency")
    latency_enc = extract_runtimes("latency_enc")
    nDCG = extract_runtimes("nDCG@10")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot for latency vs nDCG@10
    ax1.set_xlabel('nDCG@10')
    ax1.set_ylabel('Re-ranking latency (ms)')
    for i in range(len(names)):
        ax1.scatter(nDCG[i], latency[i], label=names[i], marker='*' if i < 2 else 's')
    ax1.legend(bbox_to_anchor=(0, 1), loc='upper left')
    ax1.set_title('Full re-ranking')

    # Plot for latency_enc vs nDCG@10
    ax2.set_xlabel('nDCG@10')
    ax2.set_ylabel('Query-encoding latency (ms)')
    for i in range(len(names)):
        ax2.scatter(nDCG[i], latency_enc[i], label=names[i], marker='*' if i < 2 else 's')
    ax2.legend(bbox_to_anchor=(0, 1), loc='upper left')
    ax2.set_title('Query-encoding')

    fig.tight_layout()
    fig.savefig("plot_data/figures/performance_efficiency_per_n_docs.png", transparent=True)
    plt.show()


def main(args: argparse.Namespace) -> None:
    """
    Create plots for CPU re-ranking runtime profiles.

    See parse_args() for command-line arguments.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    queries = 128
    profiles = [
        {
            "name": "TCT-ColBERT",
            "latency": 2609,
            "latency_enc": 1021 / queries,
            "nDCG@10": 0.684,
        },
        {
            "name": "AvgTokEmb",
            "latency": 1634,
            "latency_enc": 7.09 / queries,
            "nDCG@10": 0.673,
        },
        {
            "name": "n_docs=0 (=q_only)",
            "latency": 1636,
            "latency_enc": 15.49 / queries,
            "nDCG@10": 0.650,
        },
        {
            "name": "n_docs=1",
            "latency": 1907,
            "latency_enc": 55.27 / queries,
            "nDCG@10": 0.659,
        },
        {
            "name": "n_docs=10",
            "latency": 1687,
            "latency_enc": 74.77 / queries,
            "nDCG@10": 0.653,
        },
        {
            "name": "n_docs=50",
            "latency": 1591,
            "latency_enc": 79.53 / queries,
            "nDCG@10": 0.656,
        },
    ]

    plot_runtimes(profiles)


if __name__ == "__main__":
    args = parse_args()
    main(args)
