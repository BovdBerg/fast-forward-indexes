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


def plot(data: Dict[str, Any]):
    n_docs = data["n_docs"]
    weights = data["weights"]
    rank_scores = data["rank_scores"]

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.set_xlim(0, n_docs)

    ax.set_xlabel('Document rank')
    ax.set_ylabel('Weight distribution (%)')

    # Normalize weights
    weights_normalized = np.array(weights)
    weights_normalized = weights_normalized / np.sum(weights_normalized) * 100

    # Ignore the first element and normalize
    weights = np.array(weights[1:])
    weights = weights / np.sum(weights) * 100
    x_values = np.linspace(0, n_docs, len(weights))

    # Plot the weights as a line plot with dots
    ax.plot(x_values, weights, marker='.', label=f"Learned")

    # Add a horizontal line for uniform weight distribution
    uniform_weight = 100 / len(weights)
    ax.axhline(y=uniform_weight, color='r', linestyle='--', label='Uniform')

    # Add exponential weight distribution
    exp_factor = 0.4
    exponential_approximation = np.array([np.exp(-i**exp_factor) for i in range(len(weights))])
    exponential_approximation = exponential_approximation / np.sum(exponential_approximation) * 100
    ax.plot(x_values, exponential_approximation, linestyle='--', label=f'Exponential decay (-x$^{{{exp_factor}}}$)')

    # Normalize and softmax position scores and plot them as weight distribution
    rank_scores = np.array(rank_scores)
    rank_scores_normalized = rank_scores / np.sum(rank_scores) * 100
    softmax_rank_scores = np.exp(rank_scores_normalized) / np.sum(np.exp(rank_scores_normalized)) * 100
    ax.plot(x_values, softmax_rank_scores, linestyle='--', label='Softmax norm. rank scores')

    ax.legend()

    fig.savefig("plot_data/figures/weight_methods.png", transparent=True)
    plt.show()


def main(args: argparse.Namespace) -> None:
    data = {
        "n_docs": 10,
        "weights": [0.8738, 0.0441, 0.0265, 0.0189, 0.0122, 0.0087, 0.0063, 0.0040, 0.0027, 0.0018, 0.0010],
        "rank_scores": [37.483, 35.713, 34.547, 33.523, 32.566, 31.812, 31.234, 30.64, 30.298, 29.886]
    }

    plot(data)


if __name__ == "__main__":
    args = parse_args()
    main(args)
