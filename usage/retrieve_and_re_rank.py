import argparse
import time
import warnings
from copy import copy
from pathlib import Path
from typing import List

import numpy as np
import pyterrier as pt
from ir_measures import measures

from fast_forward.encoder.avg import AvgEmbQueryEstimator
from fast_forward.encoder.transformer import TCTColBERTQueryEncoder
from fast_forward.index.disk import OnDiskIndex
from fast_forward.util.pyterrier import FFInterpolate, FFScore

warnings.filterwarnings(
    "ignore", category=FutureWarning, message=".*weights_only=False.*"
)


def parse_args():
    """
    Parse command-line arguments for the re-ranking script.

    Run the script with --help or -h to see the full list of arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--storage",
        type=str,
        choices=["disk", "mem"],
        default="mem",  # all indexes have to fit into RAM
        help="Storage type for the index.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default="cpu",  # "cpu" since we're optimizing for CPU inference efficiency
        help="Device to use for encoding queries.",
    )

    # WeightedAvgEncoder
    parser.add_argument(
        "--index_path",
        type=Path,
        help="Path to the index file.",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to the avg checkpoint file. Create it by running usage/train.py",
    )
    parser.add_argument(
        "--n_docs",
        type=int,
        default=10,
        help="Number of top-ranked documents to use.",
    )
    parser.add_argument(
        "--q_only",
        action="store_true",
        help="Only use the lightweight query embedding for the WeightedAvgEncoder.",
    )

    # VALIDATION
    parser.add_argument(
        "--val_pipelines",
        type=str,
        nargs="*",
        default=["all"],
        help="List of pipelines to validate, based on exact pipeline names.",
    )
    parser.add_argument(
        "--alphas_step",
        type=float,
        default=0.1,
        help="Step size for the alpha values in the validation process.",
    )

    # EVALUATION
    parser.add_argument(
        "--test_datasets",
        type=str,
        nargs="*",
        default=["irds:msmarco-passage/trec-dl-2019/judged"],
        help="Datasets to evaluate the rankings. May never be equal to dev_dataset.",
    )
    parser.add_argument(
        "--eval_metrics",
        type=str,
        nargs="+",
        default=[
            "nDCG@10",
            "RR(rel=2)",  # =MRR
            "AP(rel=2)",  # =MAP
        ],  # Official metrics for TREC '19 according to https://ir-datasets.com/msmarco-passage.html#msmarco-passage/trec-dl-2019/judged
        help="Metrics used for evaluation.",
    )

    return parser.parse_args()


def print_settings() -> None:
    """
    Print general settings used for re-ranking.
    """
    # General settings
    settings_description: List[str] = [
        f"storage={args.storage}, device={args.device}",
        f"WeightedAvgEncoder: n_docs={args.n_docs}",
    ]
    # Validation settings
    if args.val_pipelines:
        settings_description.append(
            f"Val: {args.val_pipelines}, α_step={args.alphas_step}"
        )

    print("\nSettings:\n\t" + "\n\t".join(settings_description))


def main(args: argparse.Namespace) -> None:
    """
    Inference for dual-encoder re-ranking with score interpolation.
    Uses BM25 for 1st-stage retrieval and re-ranks using various methods.

    See parse_args() for command-line arguments.
    """
    start_time = time.time()
    print_settings()
    pt.init()

    # Parse eval_metrics (e.g. "nDCG@10", "RR(rel=2)", "AP") to ir-measures' measure objects.
    eval_metrics = []
    for metric_str in args.eval_metrics:
        if "(" in metric_str:
            metric_name, rest = metric_str.split("(")
            params, at_value = rest.split("@") if "@" in rest else (rest[:-1], None)
            param_dict = {
                k: int(v) for k, v in (param.split("=") for param in params.split(","))
            }
            if at_value:
                eval_metrics.append(
                    getattr(measures, metric_name)(**param_dict) @ int(at_value)
                )
            else:
                eval_metrics.append(getattr(measures, metric_name)(**param_dict))
        else:
            if "@" in metric_str:
                metric_name, at_value = metric_str.split("@")
                eval_metrics.append(getattr(measures, metric_name) @ int(at_value))
            else:
                eval_metrics.append(getattr(measures, metric_str))

    print("\033[96m")  # Prints in this method are cyan
    # Load dataset and create sparse retriever (e.g. BM25)
    dataset = pt.get_dataset("msmarco_passage")
    print("Creating BM25 retriever via PyTerrier index...")
    try:
        bm25 = pt.BatchRetrieve.from_dataset(
            dataset, "terrier_stemmed", wmodel="BM25", memory=True
        )
    except:
        indexer = pt.IterDictIndexer(
            str(Path.cwd()),  # ignored but must be a valid path
            type=pt.index.IndexingType.MEMORY,
        )
        index_ref = indexer.index(dataset.get_corpus_iter(), fields=["text"])
        bm25 = pt.BatchRetrieve(index_ref, wmodel="BM25", verbose=True, memory=True)
    bm25 = ~bm25 % 1000

    # Create re-ranking pipeline based on TCTColBERTQueryEncoder (normal FF approach)
    index_tct = OnDiskIndex.load(
        args.index_path,
        TCTColBERTQueryEncoder("castorini/tct_colbert-msmarco", device=args.device),
    )
    if args.storage == "mem":
        index_tct = index_tct.to_memory(2**15)
    ff_tct = FFScore(index_tct)
    int_tct = FFInterpolate(alpha=0.03)
    tct = bm25 >> ff_tct >> int_tct

    index_avg = copy(index_tct)
    index_avg.query_encoder = AvgEmbQueryEstimator(
        index=index_avg,
        n_docs=args.n_docs,
        device=args.device,
        ckpt_path=args.ckpt_path,
        q_only=args.q_only,
    )
    ff_avg = FFScore(index_avg)
    int_avg = FFInterpolate(alpha=0.02)
    avg = bm25 >> ff_avg >> int_avg

    pipelines = [
        ("bm25", "BM25", ~bm25, None),
        ("tct", "TCT-ColBERT", tct, int_tct),
        ("avg", "AvgEmb$_{q," + str(args.n_docs) + "-docs}$", avg, int_avg),
    ]

    # Validation and parameter tuning on dev set
    if args.val_pipelines:
        print("\033[33m")
        dev_dataset = pt.get_dataset("irds:msmarco-passage/dev/judged")
        dev_queries = dev_dataset.get_topics()
        dev_qrels = dev_dataset.get_qrels()

        # Sample dev queries
        dev_queries = dev_queries.sample(n=512, random_state=42)  # Fixed seed for reproducibility.
        dev_qrels = dev_qrels[dev_qrels["qid"].isin(dev_queries["qid"])]

        # Validate pipelines in args.val_pipelines
        alphas = [
            round(x, 2)
            for x in np.arange(0.0, 1.0 + 1e-5, args.alphas_step)
        ]
        for abbrev, name, system, tunable in pipelines:
            if tunable is None or (
                args.val_pipelines != ["all"]
                and (
                    abbrev not in args.val_pipelines and name not in args.val_pipelines
                )
            ):
                continue

            print(f"\nValidating pipeline: {name}...")
            pt.GridSearch(
                system,
                {tunable: {"alpha": alphas}},
                dev_queries,
                dev_qrels,
                metric="ndcg_cut_10",  # Find official metrics for dataset version on https://ir-datasets.com/msmarco-passage.html
                verbose=True,
                batch_size=128,
            )

    # Evaluate pipelines on args.test_datasets
    if args.test_datasets:
        print("\033[0m")  # Reset print color to black
        for test_dataset_name in args.test_datasets:
            test_dataset = pt.get_dataset(test_dataset_name)

            print(f"\nRunning final tests on {test_dataset_name}...")
            results = pt.Experiment(
                [pipeline for _, _, pipeline, _ in pipelines],
                test_dataset.get_topics(),
                test_dataset.get_qrels(),
                eval_metrics=eval_metrics,
                names=[
                    name if not tunable else f"{name}, α=[{tunable.alpha}]"
                    for _, name, _, tunable in pipelines
                ],
                round=3,
                verbose=True,
                # baseline=1,
                # correction="bonferroni",
            )

            print_settings()
            print(f"\nFinal results on {test_dataset_name}:\n{results}\n")

    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
