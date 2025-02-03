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
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))
    fig.suptitle('Weights distribution per relative embeddings position')

    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel('Embeddings position (%)')
        ax.set_ylabel('Weight distribution (%)')
        if ax != ax3:
            ax.set_ylim(0, 100)
    ax1.set_title('Including all weights')
    ax2.set_title('Excluding lightweight query encoding')
    ax3.set_title('Approximation for n_docs=10')
    
    plt.subplots_adjust(hspace=0.5)

    # For each n_docs, plot the weights of the embeddings as a line plot with dots. The first element should be left and the last should be right.
    for i in range(len(data)):
        n_docs = data[i]["n_docs"]
        weights = data[i]["weights"]

        # Normalize weights
        weights_normalized = np.array(weights)
        weights_normalized = weights_normalized / np.sum(weights_normalized) * 100

        # Normalize x-axis values
        x_values = np.linspace(0, 100, len(weights_normalized))

        # Plot the weights as a line plot with dots
        ax1.plot(x_values, weights_normalized, marker='.', label=f"n_docs={n_docs}")

        # Ignore the first element and normalize
        weights_ignored = weights[1:]
        weights_ignored = np.array(weights_ignored)
        weights_ignored = weights_ignored / np.sum(weights_ignored) * 100

        # Normalize x-axis values
        x_values_ignored = np.linspace(0, 100, len(weights_ignored))

        # Highlight the dot of n_docs=1 more
        if n_docs == 1:
            ax2.plot(x_values_ignored[0] - 1, weights_ignored[0], marker='o', markersize=10, label="n_docs=1")
            ax2.axhline(y=weights_ignored[0] - 1, linestyle='--')
        else:
            ax2.plot(x_values_ignored, weights_ignored, marker='.', label=f"n_docs={n_docs}")

        if n_docs == 10:
            # Ignore the first element
            weights_ignored = weights[1:]

            # Normalize weights
            weights_ignored = np.array(weights_ignored)
            weights_ignored = weights_ignored / np.sum(weights_ignored) * 100

            # Normalize x-axis values
            x_values_ignored = np.linspace(0, 100, len(weights_ignored))

            # Plot the weights as a line plot with dots
            ax3.plot(x_values_ignored, weights_ignored, marker='.', label=f"Learned, n_docs={n_docs}")

            # Add a horizontal line for uniform weight distribution
            uniform_weight = 100 / len(weights_ignored)
            ax3.axhline(y=uniform_weight, color='r', linestyle='--', label='Uniform')

            # Add exponential weight distribution
            exp_factor = 0.5
            exponential_approximation = np.array([np.exp(-i**exp_factor) for i in range(len(weights_ignored))])
            exponential_approximation = exponential_approximation / np.sum(exponential_approximation) * 100
            ax3.plot(x_values_ignored, exponential_approximation, linestyle='--', label=f'Exponential decay, factor {exp_factor}')

    for ax in [ax1, ax2, ax3]:
        ax.legend()

    fig.savefig("plot_data/figures/weights_per_n_docs.png", transparent=True)
    plt.show()


def main(args: argparse.Namespace) -> None:
    data = [
        {
            "n_docs": 1,
            "weights": [0.9076, 0.0924],
        },
        {
            "n_docs": 10,
            "weights": [0.8738, 0.0441, 0.0265, 0.0189, 0.0122, 0.0087, 0.0063, 0.0040, 0.0027, 0.0018, 0.0010],
        },
        # TODO: weights for n_docs=50 might be wrong, as they were taken from the n_docs=100 ckpt, but the dataset had a cutoff at 50.
        {
            "n_docs": 50,
            "weights": [6.1860e-01, 3.1507e-02, 1.7957e-02, 1.2845e-02, 8.2813e-03, 5.8220e-03, 4.1911e-03, 2.3847e-03, 1.4328e-03, 8.3651e-04, 3.8189e-04, 4.6364e-04, 2.2494e-04, 1.9244e-04, 1.1995e-04, 1.0833e-04, 8.1869e-05, 8.2562e-05, 7.7056e-05, 6.6695e-05, 6.8781e-05, 6.2544e-05, 5.3033e-05, 5.4206e-05, 4.6119e-05, 4.5453e-05, 4.5334e-05, 4.5128e-05, 3.8741e-05, 3.7301e-05, 3.6252e-05, 3.1634e-05, 3.4454e-05, 3.3963e-05, 3.2409e-05, 2.9048e-05, 3.1335e-05, 3.0301e-05, 2.9177e-05, 2.7900e-05, 3.1160e-05, 2.7726e-05, 2.7052e-05, 2.5422e-05, 2.4664e-05, 2.3843e-05, 2.4959e-05, 2.6145e-05, 2.5306e-05, 2.4356e-05, 2.2714e-05],
        },
    ]

    plot(data)


if __name__ == "__main__":
    args = parse_args()
    main(args)
