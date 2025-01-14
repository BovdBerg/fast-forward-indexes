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
        return [profile[key] for profile in profiles]

    names = extract_runtimes("name")
    index_call = extract_runtimes("total")
    encode_queries = extract_runtimes("encode_queries")
    get_vectors = extract_runtimes("_get_vectors")
    compute_scores = extract_runtimes("_compute_scores")
    other = extract_runtimes("other")

    fig, ax = plt.subplots()

    bar_width = 0.8
    bars = [
        ax.bar(names, encode_queries, color="darkviolet", label="encode queries", width=bar_width),
        ax.bar(names, get_vectors, bottom=encode_queries, label="get vectors", width=bar_width),
        ax.bar(names, compute_scores, bottom=np.array(encode_queries) + np.array(get_vectors), label="compute scores", width=bar_width),
        ax.bar(names, other, bottom=np.array(encode_queries) + np.array(get_vectors) + np.array(compute_scores), label="other", width=bar_width),
    ]

    # Ensure all legend entries are included
    handles, labels = ax.get_legend_handles_labels()
    unique_handles_labels = dict(zip(labels, handles))  # Remove duplicates
    ax.legend(unique_handles_labels.values(), unique_handles_labels.keys())

    # Add percentage text for the first bar of "encode_queries"
    bar = bars[0][0]
    runtime = encode_queries[0]
    total = index_call[0]
    height = bar.get_height()
    percentage = (runtime / total) * 100
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_y() + height / 2,
        f'{percentage:.1f}%',
        ha='center',
        va='center',
        color='white',
        fontsize=12,
        fontweight='bold',
    )

    # Add speedup text for bars except the first
    for i in range(1, len(names)):
        speedup = index_call[0] / index_call[i]
        bar = bars[0][i]
        height = 2.25
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_y() + height + 0.1,
            f'{speedup:.1f}X',
            ha='center',
            va='bottom',
            color='black',
            fontsize=12,
            fontweight='bold',
        )

    ax.set_xlabel('Pipelines')
    ax.set_ylabel('Re-ranking runtime (in seconds)')
    ax.legend()

    fig.savefig("reranking_runtimes.png")  # Save plot as a PNG file
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
            "total": 22.92713,
            "encode_queries": 20.70111,
            "_get_vectors": 0.8245,
            "_compute_scores": 1.18664,
            "other": 0.21488,
        },
        {
            "name": "AvgTokenEmb",
            "total": 2.04237,
            "encode_queries": 0.09653,
            "_get_vectors": 0.67664,
            "_compute_scores": 1.074,
            "other": 0.1952,
        },
        {
            "name": "AvgEmb",
            "total": 1.87285,
            "encode_queries": 0.20171,
            "_get_vectors": 0.56513,
            "_compute_scores": 0.91053,
            "other": 0.19548,
        },
        {
            "name": "AvgEmb + AvgTokEmb",
            "total": 2.16339,
            "encode_queries": 0.23985,
            "_get_vectors": 0.69988,
            "_compute_scores": 1.03983,
            "other": 0.18383,
        },
    ]

    plot_runtimes(profiles)


if __name__ == "__main__":
    args = parse_args()
    main(args)
