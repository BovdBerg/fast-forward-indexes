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
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    ax.set_xlabel('Embeddings position (%)')
    ax.set_ylabel('Weight distribution (%)')
    
    # For each n_docs, plot the weights of the embeddings as a line plot with dots. The first element should be left and the last should be right.
    for i in range(len(data)):
        n_docs = data[i]["n_docs"]
        weights = data[i]["weights"]

        # Normalize weights
        weights_normalized = np.array(weights)
        weights_normalized = weights_normalized / np.sum(weights_normalized) * 100

        # Normalize x-axis values
        x_values = np.linspace(0, n_docs, len(weights_normalized))

        # Ignore the first element and normalize
        weights = np.array(weights[1:])
        weights = weights / np.sum(weights) * 100
        x_values_ignored = np.linspace(0, 10, len(weights))

        # Plot the weights as a line plot with dots
        ax.plot(x_values_ignored, weights, marker='.', label=f"Learned")

        # Add a horizontal line for uniform weight distribution
        uniform_weight = 100 / len(weights)
        ax.axhline(y=uniform_weight, color='r', linestyle='--', label='Uniform')

        # Add exponential weight distribution
        exp_factor = 0.4
        exponential_approximation = np.array([np.exp(-i**exp_factor) for i in range(len(weights))])
        exponential_approximation = exponential_approximation / np.sum(exponential_approximation) * 100
        ax.plot(x_values_ignored, exponential_approximation, linestyle='--', label=f'Exponential decay (-x$^{{{exp_factor}}}$)')

    ax.legend()

    fig.savefig("plot_data/figures/weight_methods.png", transparent=True)
    plt.show()


def main(args: argparse.Namespace) -> None:
    data = [
        {
            "n_docs": 10,
            "weights": [0.8738, 0.0441, 0.0265, 0.0189, 0.0122, 0.0087, 0.0063, 0.0040, 0.0027, 0.0018, 0.0010],
        },
    ]

    plot(data)


if __name__ == "__main__":
    args = parse_args()
    main(args)
