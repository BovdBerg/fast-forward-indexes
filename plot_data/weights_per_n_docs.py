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
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    for ax in [ax1, ax2]:
        ax.set_xlabel('Embeddings position (%)')
        ax.set_ylabel('Weight distribution (%)')
    ax1.set_title('All weights')
    ax2.set_title('Excluding lightweight query encoding')
    ax1.set_ylim(0, 105)
    
    plt.subplots_adjust(hspace=0.3)

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
            # ax2.plot(x_values_ignored[0], weights_ignored[0], marker='')
            # ax2.axhline(y=weights_ignored[0], linestyle='-', label="n_docs=1")
            pass
        else:
            ax2.plot(x_values_ignored, weights_ignored, marker='.', label=f"n_docs={n_docs}", color=f"C{(i % 9)}")

    for ax in [ax1, ax2]:
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
        {
            "n_docs": 50,
            "weights": [8.9212e-01, 3.6479e-02, 2.2330e-02, 1.6203e-02, 1.1144e-02, 7.5169e-03, 5.0182e-03, 2.8782e-03, 1.7377e-03, 1.0049e-03, 4.4296e-04, 5.1144e-04, 2.6757e-04, 2.2374e-04, 1.4806e-04, 1.3727e-04, 1.0585e-04, 1.0608e-04, 9.6284e-05, 8.7027e-05, 8.9261e-05, 7.7776e-05, 6.8541e-05, 7.1505e-05, 5.9475e-05, 5.7373e-05, 5.8475e-05, 6.0248e-05, 5.1139e-05, 4.9616e-05, 4.7762e-05, 4.3286e-05, 4.5835e-05, 4.4199e-05, 4.3274e-05, 3.8881e-05, 4.1475e-05, 4.0134e-05, 3.8934e-05, 3.7348e-05, 4.1206e-05, 3.6884e-05, 3.5854e-05, 3.3980e-05, 3.3237e-05, 3.1707e-05, 3.3776e-05, 3.4860e-05, 3.3675e-05, 3.2397e-05, 3.0572e-05]
        },
    ]

    plot(data)


if __name__ == "__main__":
    args = parse_args()
    main(args)
