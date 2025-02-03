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
    def extract_runtimes(key: str) -> List[Any]:
        return np.array([profile[key] for profile in profiles])

    names = extract_runtimes("name")
    latency = extract_runtimes("latency")
    nDCG = extract_runtimes("nDCG@10")

    fig, ax = plt.subplots()
    ax.set_xlabel('nDCG@10')
    ax.set_ylabel('Re-ranking latency (ms)')

    # Plot each point and annotate with the name
    for i in range(len(names)):
        ax.scatter(nDCG[i], latency[i], label=names[i])
        # ax.annotate(names[i], (nDCG[i], latency[i]))

    ax.legend(bbox_to_anchor=(1, 0), loc='lower right')
    fig.tight_layout()

    fig.savefig("plot_data/figures/performance_efficiency_graph.png", transparent=True)
    plt.show()


def main(args: argparse.Namespace) -> None:
    """
    Create plots for CPU re-ranking runtime profiles.

    See parse_args() for command-line arguments.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    profiles = [
        {
            "name": "TCT-ColBERT",
            "latency": 2609,
            "nDCG@10": 0.684,
        },
        # TODO: update nDCG
        {
            "name": "AvgTokEmb",
            "latency": 1634,
            "nDCG@10": 0.684,
        },
        {
            "name": "AvgEmb_docs",
            "latency": 1636,
            "nDCG@10": 0.684,
        },
        {
            "name": "AvgEmb_docs + AvgTokEmb",
            "latency": 3350,
            "nDCG@10": 0.684,
        },
        {
            "name": "AvgEmb",
            "latency": 1671,
            "nDCG@10": 0.684,
        },
    ]

    plot_runtimes(profiles)


if __name__ == "__main__":
    args = parse_args()
    main(args)
