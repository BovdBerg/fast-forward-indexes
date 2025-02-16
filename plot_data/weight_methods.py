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


def plot(data: Dict[str, Any], ax: Any):
    n_docs = data["n_docs"]
    weights = data["weights"]
    rank_scores = data["rank_scores"]

    ax.set_title(f"n_docs={n_docs}")
    ax.set_xlim(0, n_docs)
    ax.set_xlabel('Document rank')
    ax.set_ylabel('Weight distribution (%)')

    # Normalize weights
    weights_normalized = np.array(weights)
    weights_normalized = weights_normalized / np.sum(weights_normalized) * 100
    ax.set_ylim(0, 45)

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
    exp_factor = 0.5
    exponential_approximation = np.array([np.exp(-i**exp_factor) for i in range(len(weights))])
    exponential_approximation = exponential_approximation / np.sum(exponential_approximation) * 100
    ax.plot(x_values, exponential_approximation, linestyle='--', label=f'Exponential decay (-x$^{{{exp_factor}}}$)')

    # Normalize and softmax position scores and plot them as weight distribution
    rank_scores = np.array(rank_scores)
    rank_scores_normalized = rank_scores / np.sum(rank_scores) * 100
    softmax_rank_scores = np.exp(rank_scores_normalized) / np.sum(np.exp(rank_scores_normalized)) * 100
    ax.plot(x_values, softmax_rank_scores, linestyle='--', label='Softmax norm. rank scores')

    ax.legend()

def main(args: argparse.Namespace) -> None:
    data_10 = {
        "n_docs": 10,
        "weights": [0.8738, 0.0441, 0.0265, 0.0189, 0.0122, 0.0087, 0.0063, 0.0040, 0.0027, 0.0018, 0.0010],
        "rank_scores": [37.483, 35.713, 34.547, 33.523, 32.566, 31.812, 31.234, 30.64, 30.298, 29.886]
    }

    data_50 = {
        "n_docs": 50,
        "weights": [
            8.9212e-01, 3.6479e-02, 2.2330e-02, 1.6203e-02, 1.1144e-02, 7.5169e-03,
            5.0182e-03, 2.8782e-03, 1.7377e-03, 1.0049e-03, 4.4296e-04, 5.1144e-04,
            2.6757e-04, 2.2374e-04, 1.4806e-04, 1.3727e-04, 1.0585e-04, 1.0608e-04,
            9.6284e-05, 8.7027e-05, 8.9261e-05, 7.7776e-05, 6.8541e-05, 7.1505e-05,
            5.9475e-05, 5.7373e-05, 5.8475e-05, 6.0248e-05, 5.1139e-05, 4.9616e-05,
            4.7762e-05, 4.3286e-05, 4.5835e-05, 4.4199e-05, 4.3274e-05, 3.8881e-05,
            4.1475e-05, 4.0134e-05, 3.8934e-05, 3.7348e-05, 4.1206e-05, 3.6884e-05,
            3.5854e-05, 3.3980e-05, 3.3237e-05, 3.1707e-05, 3.3776e-05, 3.4860e-05,
            3.3675e-05, 3.2397e-05, 3.0572e-05
        ],
        "rank_scores": [37.483, 35.713, 34.547, 33.523, 32.566, 31.812, 31.234, 30.64, 30.298, 29.886, 29.384, 29.012, 28.783, 28.487, 28.303, 28.06, 27.869, 27.642, 27.425, 27.148, 27.023, 26.884, 26.75, 26.606, 26.455, 26.329, 26.236, 26.076, 25.958, 25.835, 25.72, 25.622, 25.528, 25.433, 25.339, 25.277, 25.19, 25.092, 25.013, 24.941, 24.842, 24.75, 24.672, 24.575, 24.514, 24.444, 24.377, 24.315, 24.252, 24.174]
    }

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10))
    plt.subplots_adjust(hspace=0.3)

    plot(data_10, ax1)
    plot(data_50, ax2)

    fig.tight_layout()
    fig.savefig("plot_data/figures/weight_methods.png", transparent=True)
    plt.show()


if __name__ == "__main__":
    args = parse_args()
    main(args)
