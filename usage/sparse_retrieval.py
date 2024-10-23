import argparse
import pyterrier as pt
import os
from tqdm import tqdm


def parse_args():
    """
    Parse command-line arguments for the re-ranking script.

    Returns:
        argparse.Namespace: Parsed command-line arguments.

    Arguments:
        --dataset (str): Dataset for testing (using package ir-datasets).
        --retriever_variant (str): Retriever variant.
        --retriever_wmodel (str): Retriever weighting model.
        --topics_variant (str): Topics variant.
        --k (int): Number of documents to re-rank per query.
        --out_dir (str): Output directory
    """
    parser = argparse.ArgumentParser(
        description="Re-rank documents based on query embeddings."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="msmarco_passage",
        help="Dataset for testing (using package ir-datasets).",
    )
    parser.add_argument(
        "--retriever_variant",
        type=str,
        default="terrier_stemmed",
        help="Retriever variant.",
    )
    parser.add_argument(
        "--retriever_wmodel",
        type=str,
        default="BM25",
        help="Retriever weighting model.",
    )
    parser.add_argument(
        "--pt_variant_topics",
        type=str,
        default=None,
        help="Topics variant. Example: 'test-2019'",
    )
    parser.add_argument(
        "--ir_dataset_topics",
        type=str,
        default=None,
        help="IRDS Dataset for topics (queries). Example: 'irds:msmarco-passage/trec-dl-2019/judged'",
    )
    parser.add_argument(
        "--k", type=int, default=100, help="Number of documents to re-rank per query."
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/home/bvdb9/sparse_rankings",
        help="Output directory.",
    )
    parser.add_argument(
        "--max_queries",
        type=int,
        default=None,
        help="Maximum number of queries to process.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """
    Main function for re-ranking documents based on query embeddings.

    Arguments:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    if not pt.started():
        pt.init()

    output_file_suffix = ""

    # Retrieve BM25 model from the batchretrieve index
    print(f"Retrieving {args.retriever_wmodel} model from {args.dataset} dataset...")
    retriever = pt.BatchRetrieve.from_dataset(
        args.dataset, args.retriever_variant, wmodel=args.retriever_wmodel, verbose=True
    )

    # Get the test topics (dataframe with columns=['qid', 'query'])
    if args.pt_variant_topics is not None:
        topics = pt.get_dataset(args.dataset).get_topics(args.pt_variant_topics)
        topics_str = args.pt_variant_topics
    elif args.ir_dataset_topics is not None:
        topics = pt.get_dataset(args.ir_dataset_topics).get_topics()
        topics_str = args.ir_dataset_topics.split("/", 1)[-1].replace("/", ".")
    else:
        raise ValueError(
            "Either --pt_variant_topics or --ir_dataset_topics must be provided."
        )

    if args.max_queries is not None and args.max_queries < len(topics):
        print(
            f"Limiting the number of queries to {args.max_queries} (out of {len(topics)})."
        )
        topics = topics.head(args.max_queries)
        output_file_suffix = f"-{args.max_queries}queries"
    print(f"topics:\n{topics}")

    top_ranked_docs = (retriever % args.k)(topics)
    print(
        f"top_ranked_docs (Length: {len(topics)} topics * {args.k} docs = {len(top_ranked_docs)}):\n{top_ranked_docs}"
    )

    # Write to the sparse_runfile.tsv
    output_file = os.path.join(
        args.out_dir,
        f"{args.dataset}-{topics_str}-{args.retriever_wmodel}-top{args.k}{output_file_suffix}.tsv",
    )
    with open(output_file, "w") as f:
        # top_ranked_docs.to_csv("sparse_runfile.csv", sep=" ", header=False, index=False)
        for _, row in tqdm(
            top_ranked_docs.iterrows(),
            total=len(top_ranked_docs),
            desc=f"Writing results to {output_file}",
        ):
            f.write(
                f"{row['qid']}\tQ0\t{row['docid']}\t{row['rank']}\t{row['score']}\tsparse\n"
            )


if __name__ == "__main__":
    args = parse_args()
    main(args)
