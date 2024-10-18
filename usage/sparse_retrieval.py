import argparse
import pyterrier as pt
import os


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
    parser = argparse.ArgumentParser(description="Re-rank documents based on query embeddings.")
    parser.add_argument("--dataset", type=str, default="msmarco_passage", help="Dataset for testing (using package ir-datasets).")
    parser.add_argument("--retriever_variant", type=str, default="terrier_stemmed", help="Retriever variant.")
    parser.add_argument("--retriever_wmodel", type=str, default="BM25", help="Retriever weighting model.")
    parser.add_argument("--topics_variant", type=str, default="test-2019", help="Topics variant.")
    parser.add_argument("--k", type=int, default=10000, help="Number of documents to re-rank per query.")
    parser.add_argument("--out_dir", type=str, default="/home/bvdb9/sparse_rankings", help="Output directory.")
    return parser.parse_args()


def main(
        args: argparse.Namespace
    ) -> None:
    """
    Main function for re-ranking documents based on query embeddings.

    Arguments:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    if not pt.started():
        pt.init()

    # Retrieve BM25 model from the batchretrieve index
    print(f"Retrieving {args.retriever_wmodel} model from {args.dataset} dataset...")
    bm25 = pt.BatchRetrieve.from_dataset(args.dataset, args.retriever_variant, wmodel=args.retriever_wmodel, verbose=True)

    # Get the test topics (dataframe with columns=['qid', 'query'])
    if args.topics_variant is None:
        topics = pt.get_dataset(args.dataset).get_topics()
    else:
        topics = pt.get_dataset(args.dataset).get_topics(args.topics_variant)
    print(f"topics:\n{topics}")

    top_ranked_docs = (bm25 % args.k)(topics)
    print(f"top_ranked_docs (Length: {len(topics)} topics * {args.k} docs = {len(top_ranked_docs)}):\n{top_ranked_docs}")

    # Write to the sparse_runfile.tsv
    output_file = os.path.join(args.out_dir, f"{args.dataset}-{args.topics_variant}-{args.retriever_wmodel}-top{args.k}.tsv")
    print("\nWriting to", output_file)
    with open(output_file, "w") as f:
        # top_ranked_docs.to_csv("sparse_runfile.csv", sep=" ", header=False, index=False)
        for i, row in tqdm(top_ranked_docs.iterrows(), total=len(top_ranked_docs), desc=f"Writing results to {output_file}"):
            f.write(f"{row['qid']}\tQ0\t{row['docno']}\t{i + 1}\t{row['score']}\tsparse\n")


if __name__ == "__main__":
    args = parse_args()
    main(args)
