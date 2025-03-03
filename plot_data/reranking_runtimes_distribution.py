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
    parser.add_argument("--motivation", action="store_true", help="Only plot the pipelines required for the introduction/motivation section")
    return parser.parse_args()


def plot_runtimes(profiles: List[Dict[str, Any]]):
    if args.motivation:
        # Only keep first profile
        profiles = profiles[:1]

    def extract_runtimes(key: str) -> np.ndarray:
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
        ax.bar(names, encode_queries, color="darkviolet", label="Create query vectors", width=bar_width),
        ax.bar(names, get_vectors, bottom=encode_queries, label="Retrieve doc vectors", width=bar_width),
        ax.bar(names, compute_scores, bottom=encode_queries + get_vectors, label="Compute scores", width=bar_width),
        ax.bar(names, other, bottom=encode_queries + get_vectors + compute_scores, label="Other", width=bar_width),
    ]

    # Set y-axis limit
    ax.set_ylim(0, (max(total) // 1000 + 1) * 1000)

    if args.motivation:
        for i, (bar_group, runtimes) in enumerate(zip(bars, [encode_queries, get_vectors, compute_scores, other])):
            for j, (bar, runtime) in enumerate(zip(bar_group, runtimes)):
                ax.text(
                    0,
                    bar.get_y() + bar.get_height() / 2,
                    f"{bar_group.get_label()}: {runtime}ms ({(runtime / total[j]) * 100:.1f}%)",
                    # f'{(runtime / total[j]) * 100:.1f}%',
                    ha='center',
                    va='center',
                    color='white',
                    fontsize=13,
                    fontweight='bold',
                )
    else:
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
                        fontsize=11,
                        fontweight='bold',
                    )

    if not args.motivation:
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
    if args.motivation:
        ax.set_ylabel('Re-ranking runtime (ms)', fontsize=14)
    else:
        ax.set_ylabel('Re-ranking runtime (ms)', fontsize=12.5)
    if not args.motivation:
        ax.legend(bbox_to_anchor=(1, 0.5), loc='center left', fontsize=11)

    # Rotate the names under the X-axis vertically
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names)
    if args.motivation:
        # Bigger font for xticks
        plt.xticks(fontsize=13)
    else:
        plt.xticks(rotation=70, fontsize=13)
    plt.tight_layout()

    if args.motivation:
        fig.savefig("plot_data/figures/reranking_runtimes_distribution_motivation.png", transparent=True)
    else:
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
            "name": "AvgEmb$_{10-docs}$",
            "total": 1636,
            "encode_queries": 64.8,
            "_get_vectors": 543,
            "_compute_scores": 888,
        },
        {
            "name": "AvgEmb$_{10-docs}$ + AvgTokEmb",
            "total": 3350,
            "encode_queries": 68.59,
            "_get_vectors": 1087,
            "_compute_scores": 1775,
        },
        {
            "name": "AvgEmb$_{q,10-docs}$",
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
