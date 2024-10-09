from types import SimpleNamespace
import pyterrier as pt
import os


config_dict = {
    "dataset": "msmarco_passage",
    "retriever": {
        "variant": "terrier_stemmed",
        "wmodel": "BM25",
    },
    "topics_variant": "test-2019", # e.g. "test-2019", None
    "k": 10000,
    "out_dir": "/home/bvdb9/runs",
}


def dict_to_namespace(d):
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = dict_to_namespace(v)
    return SimpleNamespace(**d)


if __name__ == "__main__":
    if not pt.started():
        pt.init()

    config = dict_to_namespace(config_dict)

    # Retrieve BM25 model from the batchretrieve index
    bm25 = pt.BatchRetrieve.from_dataset(config.dataset, config.retriever.variant, wmodel=config.retriever.wmodel)

    # Get the test topics (dataframe with columns=['qid', 'query'])
    if config.topics_variant is None:
        topics = pt.get_dataset(config.dataset).get_topics()
    else:
        topics = pt.get_dataset(config.dataset).get_topics(config.topics_variant)
    print("topics:")
    print(f"Length: {len(topics)}")
    print(f"Head:\n{topics.head()}")

    topics = pt.get_dataset(config.dataset).get_topics(config.topics_variant)
    top_ranked_docs = (bm25 % config.k)(topics)
    print("\ntop_ranked_docs:")
    print(f"Length: {len(topics)} topics * {config.k} docs = {len(top_ranked_docs)}")
    print(f"Head:\n{top_ranked_docs.head()}")

    # Write to the sparse_runfile.tsv
    output_file = os.path.join(config.out_dir, f"{config.dataset}-{config.topics_variant}-{config.retriever.wmodel}-top{config.k}.tsv")
    print("\nWriting to", output_file)
    with open(output_file, "w") as f:
        # top_ranked_docs.to_csv("sparse_runfile.csv", sep=" ", header=False, index=False)
        for i, row in top_ranked_docs.iterrows():
            f.write(f"{row['qid']}\tQ0\t{row['docno']}\t{i + 1}\t{row['score']}\tsparse\n")
