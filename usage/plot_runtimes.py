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
        nargs="+",
        default=["avg1"],
        help="The names of the profiles to plot.",
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
    project_dir = Path(__file__).parent.parent
    profile_dir = project_dir / "profiles"
    profiles = [
        prof for prof in Path(profile_dir).rglob("*.prof") if prof.stem in args.profiles
    ]

    profile_data = {}
    for profile in profiles:
        title = profile.stem
        print(f"{title}:")

        ps = pstats.Stats(str(profile)).sort_stats("cumtime")

        # Get the total runtime of the profile
        total_time = round(ps.total_tt, 2)
        print(f"\t{total_time}s (100%) re-ranking runtime")
        print("\t" + "-" * 40) # Separator

        def runtime(path):
            """
            Returns rounded runtime of first class + method match.
            """
            parts = path.split("::")
            if parts[-1] == "self":
                cls, _, method = parts[:-1]
            else:
                cls, _, method = parts
            time = next(
                (
                    ps.stats[key][2] if len(parts) == 4 else ps.stats[key][3]
                    for key in ps.stats
                    if key[0] == cls and key[2] == method
                ),
                0,
            )
            rounded = round(time, 2)
            pct = round(time / total_time * 100, 1)
            print(f"\t{rounded}s ({pct}%) {method}")
            return rounded, pct

        # Get the runtime (and re-ranking %) of the 'encode_queries' method
        q_encoder_time, q_encoder_p = runtime(
            "/home/bvdb9/fast-forward-indexes/fast_forward/index/__init__.py::70::encode_queries"
        )

        # Get the runtime (and re-ranking %) of the 'lookup_documents' method
        # Note that the class differs between memory and disk
        d_lookup_time, d_lookup_p = runtime(
            "/home/bvdb9/fast-forward-indexes/fast_forward/index/memory.py::156::_get_vectors"
        )
        if d_lookup_time == 0:
            d_lookup_time, d_lookup_p = runtime(
                "/home/bvdb9/fast-forward-indexes/fast_forward/index/disk.py::254::_get_vectors"
            )

        # Get the runtime (and re-ranking %) of the 'compute_scores' method
        # Note that this only considers the recursive call
        compute_scores_time, compute_scores_p = runtime(
            "/home/bvdb9/fast-forward-indexes/fast_forward/index/__init__.py::304::_compute_scores::self"
        )

        other_time = round(
            total_time - q_encoder_time - d_lookup_time - compute_scores_time, 2
        )
        other_p = round(other_time / total_time * 100, 1)
        print(f"\t{other_time}s ({other_p}%) other")

        sum_p = round(q_encoder_p + d_lookup_p + compute_scores_p + other_p, 0)
        assert sum_p == 100, f"Percentages should sum to 100, but was {sum_p}."

        # Save a dict of total_time, q_encoder_time and q_encoder_percentage
        profile_data[title] = {
            "total_time": total_time,
            "q_encoder_time": q_encoder_time,
            "q_encoder_p": q_encoder_p,
            "d_lookup_time": d_lookup_time,
            "d_lookup_p": d_lookup_p,
            "compute_scores_time": compute_scores_time,
            "compute_scores_p": compute_scores_p,
            "other_time": other_time,
            "other_p": other_p,
        }

    for profile, data in profile_data.items():
        print(f"{profile}:\n{data}")

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

    # Categorize the data into 4 bins: q_encoder, d_lookup, compute_scores, other
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
        [data["d_lookup_time"] for data in profile_data.values()],
        bar_width,
        label="lookup_documents",
        color="cornflowerblue",
        bottom=[data["q_encoder_time"] for data in profile_data.values()],
    )
    ax.bar(
        x,
        [data["compute_scores_time"] for data in profile_data.values()],
        bar_width,
        label="compute_scores",
        color="seagreen",
        bottom=[
            data["q_encoder_time"] + data["d_lookup_time"]
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
            data["q_encoder_time"] + data["d_lookup_time"] + data["compute_scores_time"]
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
        if data["d_lookup_time"] > 3:
            ax.text(
                i,
                data["q_encoder_time"] + data["d_lookup_time"] / 2,
                f"{data['d_lookup_time']}",
                ha="center",
                va="center",
                color="white",
            )
        if data["compute_scores_time"] > 3:
            ax.text(
                i,
                data["q_encoder_time"]
                + data["d_lookup_time"]
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
                + data["d_lookup_time"]
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
