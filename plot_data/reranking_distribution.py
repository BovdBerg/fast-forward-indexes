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
    total = extract_runtimes("total")
    encode_queries = extract_runtimes("encode_queries")
    get_vectors = extract_runtimes("_get_vectors")
    compute_scores = extract_runtimes("_compute_scores")
    other = total - encode_queries - get_vectors - compute_scores

    fig, ax = plt.subplots()

    bar_width = 0.9
    bars = [
        ax.bar(names, encode_queries, color="darkviolet", label="encode queries", width=bar_width),
        ax.bar(names, get_vectors, bottom=encode_queries, label="get vectors", width=bar_width),
        ax.bar(names, compute_scores, bottom=encode_queries + get_vectors, label="compute scores", width=bar_width),
        ax.bar(names, other, bottom=encode_queries + get_vectors + compute_scores, label="other", width=bar_width),
    ]

    # Set y-axis limit
    ax.set_ylim(0, (max(total) // 1000 + 1) * 1000)

    # Add percentage text for bars of >X cm
    for i, (bar_group, runtimes) in enumerate(zip(bars, [encode_queries, get_vectors, compute_scores, other])):
        for j, (bar, runtime) in enumerate(zip(bar_group, runtimes)):
            if i == 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y(),
                    f'{(runtime / total[j]) * 100:.1f}%',
                    ha='center',
                    va='bottom',
                    color='white',
                    fontsize=10,
                    fontweight='bold',
                )

    # Add speedup text for bars except the first
    for i in range(0, len(names)):
        bar = bars[0][i]
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            total[i] + 2.5,
            f'{total[0] / total[i]:.1f}X',
            ha='center',
            va='bottom',
            color='black',
            fontsize=12,
            fontweight='bold',
        )

    # ax.set_xlabel('Pipelines')
    ax.set_ylabel('Re-ranking runtime (ms)')
    ax.legend(bbox_to_anchor=(1.05, 0.5), loc='center left')

    # Rotate the names under the X-axis vertically
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=90)
    plt.tight_layout()

    fig.savefig("plot_data/figures/reranking_runtimes_distribution.png", transparent=True)
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
            "total": 2609,
            "encode_queries": 1021,
            "_get_vectors": 543,
            "_compute_scores": 888,
        },
        {
            "name": "AvgTokEmb",
            "total": 1634,
            "encode_queries": 7.09,
            "_get_vectors": 543,
            "_compute_scores": 888,
        },
        {
            "name": "AvgEmb_docs",
            "total": 1636,
            "encode_queries": 64.8,
            "_get_vectors": 543,
            "_compute_scores": 888,
        },
        {
            "name": "AvgEmb_docs + AvgTokEmb",
            "total": 3350,
            "encode_queries": 68.59,
            "_get_vectors": 1087,
            "_compute_scores": 1775,
        },
        {
            "name": "AvgEmb",
            "total": 1671,
            "encode_queries": 75.98,
            "_get_vectors": 543,
            "_compute_scores": 888,
        },
    ]

    plot_runtimes(profiles)


if __name__ == "__main__":
    args = parse_args()
    main(args)
