import argparse
import pstats
from pathlib import Path

from matplotlib import pyplot as plt


def parse_args():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.

    Arguments:
        Run the script with --help or -h to see the full list of arguments.
    """
    parser = argparse.ArgumentParser(description="Plot the re-ranking profiles.")
    parser.add_argument(
        "--profiles",
        type=str,
        nargs="*",
        default=["avg1"],
        help="The names of the profiles to plot.",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default="mem",
        choices=["disk", "mem"],
        help="The storage type of the index.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="The device used for re-ranking.",
    )
    parser.add_argument(
        "--print_stats",
        type=int,
        default=0,
        help="The number of stats to print.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="msmarco-passage",
        help="The name of the dataset.",
    )
    parser.add_argument(
        "--dense_approach",
        type=str,
        default="TCT-ColBERT",
        help="The name of the lexical retrieval method.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """
    Create plots for CPU re-ranking runtime profiles.

    See parse_args() for command-line arguments.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    profile_dir = Path(__file__).parent.parent / "profiles" / f"{args.storage}_{args.device}"
    profile_data = {}

    for title in args.profiles:
        ps = pstats.Stats(str(profile_dir / f"{title}.prof")).sort_stats("cumtime")
        ps.print_stats(args.print_stats)

        # Get the total runtime of the profile
        total_time = round(ps.total_tt, 2)
        print(f"\t{total_time}s (100%) re-ranking runtime")
        print("\t" + "-" * 50) # Separator

        def runtime(s: str) -> tuple[float, float]:
            """
            Returns rounded runtime of first class + method match.
            """
            cumtime = next((stat[3] for key, stat in ps.stats.items() if f"{key[0]}:{key[1]}({key[2]})" == s), 0)
            rounded = round(cumtime, 2)
            pct = round(cumtime / total_time * 100, 1)
            return rounded, pct

        # Get the runtime (and re-ranking %) of the 'encode_queries' method
        q_encoder_s = "/home/bvdb9/fast-forward-indexes/fast_forward/index/__init__.py:70(encode_queries)"
        q_encoder_time, q_encoder_p = runtime(
            q_encoder_s
        )
        if q_encoder_time == 0:
            q_encoder_s = "/home/bvdb9/miniconda3/envs/ff/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:996(forward)"
            q_encoder_time, q_encoder_p = runtime(q_encoder_s)
        print(f"\t{q_encoder_time}s ({q_encoder_p}%) {q_encoder_s}")
        
        # Get the runtime (and re-ranking %) of the 'compute_scores' method
        compute_scores_s = "/home/bvdb9/fast-forward-indexes/fast_forward/index/__init__.py:304(_compute_scores)"
        compute_scores_time, compute_scores_p = runtime(compute_scores_s)
        print(f"\t{compute_scores_time}s ({compute_scores_p}%) {compute_scores_s}")

        other_time = round(
            total_time - q_encoder_time - compute_scores_time, 2
        )
        other_p = round(other_time / total_time * 100, 1)
        print(f"\t{other_time}s ({other_p}%) other")
        print("=" * 150) # Separator

        # Save a dict of total_time, q_encoder_time and q_encoder_percentage
        profile_data[title] = {
            "total_time": total_time,
            "q_encoder_time": q_encoder_time,
            "q_encoder_p": q_encoder_p,
            "compute_scores_time": compute_scores_time,
            "compute_scores_p": compute_scores_p,
            "other_time": other_time,
            "other_p": other_p,
        }

    return
    # Create a figure and axis
    fig, ax = plt.subplots()
    ax.set_title(
        f"{args.dataset} with {args.dense_approach}: Re-ranking runtime spent on 'encode_queries' (in %)"
    )

    # Create a bar chart
    ax.bar(
        range(len(profile_data)),
        [data["q_encoder_p"] for data in profile_data.values()],
    )
    # Show percentage above each bar
    for i, data in enumerate(profile_data.values()):
        ax.text(
            i,
            data["q_encoder_p"] + 2,
            f"{data['q_encoder_p']}%\nof {data['total_time']}s",
            ha="center",
        )

    # Set the x-axis
    ax.set_xticks(range(len(profile_data)))
    ax.set_xticklabels(profile_data.keys(), rotation=45, ha="right")

    # Set the y-axis
    ax.set_yticks(range(0, 101, 10))
    ax.set_ylabel("Percentage of encode_queries")

    plt.show()

    # Categorize the data into 4 bins: q_encoder, compute_scores, other
    # Create a bar plot of the 4 bins, with the reranking time as the height
    # Show the percentage of the total reranking time in the middle of each bar

    # Create a figure and axis
    fig, ax = plt.subplots()
    ax.set_title(
        f"{args.dataset} with {args.dense_approach}: Re-ranking runtime breakdown"
    )

    # Create a bar chart
    bar_width = 0.7
    x = range(len(profile_data))
    data = list(profile_data.values())
    ax.bar(
        x,
        [data["q_encoder_time"] for data in profile_data.values()],
        bar_width,
        label="encode_queries",
        color="darkviolet",
    )
    ax.bar(
        x,
        [data["compute_scores_time"] for data in profile_data.values()],
        bar_width,
        label="compute_scores",
        color="seagreen",
        bottom=[
            data["q_encoder_time"]
            for data in profile_data.values()
        ],
    )
    ax.bar(
        x,
        [data["other_time"] for data in profile_data.values()],
        bar_width,
        label="other",
        color="lightgrey",
        bottom=[
            data["q_encoder_time"] + data["compute_scores_time"]
            for data in profile_data.values()
        ],
    )

    # Show amount of seconds for each segment in its middle, except for values < 3
    for i, data in enumerate(profile_data.values()):
        if data["q_encoder_time"] > 3:
            ax.text(
                i,
                data["q_encoder_time"] / 2,
                f"{data['q_encoder_time']}\n({data['q_encoder_p']}%)",
                ha="center",
                va="center",
                color="white",
            )
        if data["compute_scores_time"] > 3:
            ax.text(
                i,
                data["q_encoder_time"]
                + data["compute_scores_time"] / 2,
                f"{data['compute_scores_time']}",
                ha="center",
                va="center",
                color="white",
            )
        if data["other_time"] > 3:
            ax.text(
                i,
                data["q_encoder_time"]
                + data["compute_scores_time"]
                + data["other_time"] / 2,
                f"{data['other_time']}",
                ha="center",
                va="center",
                color="black",
            )

    # Set the x-axis
    ax.set_xticks(range(len(profile_data)))
    ax.set_xticklabels(profile_data.keys(), rotation=45, ha="right")

    # Set the y-axis
    ax.set_yticks(range(0, 101, 10))
    ax.set_ylabel("Distribution of re-ranking time (in s)")
    ax.legend()

    # Give it a transparent background
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    plt.show()


if __name__ == "__main__":
    args = parse_args()
    main(args)
