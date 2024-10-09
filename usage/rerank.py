from enum import Enum
from pathlib import Path
from typing import List
import numpy as np
import torch
from fast_forward.encoder import TCTColBERTQueryEncoder
from fast_forward.index import Index
from fast_forward.index.disk import OnDiskIndex
from fast_forward.ranking import Ranking
from tqdm import tqdm
import ir_datasets
from ir_measures import calc_aggregate, nDCG
from fast_forward.util import to_ir_measures
import pandas as pd


class EncodingMethod(Enum):
    """Enum for encoding method.

    TCTColBERT: Use the TCTColBERT query encoder.
    AVERAGE: Estimate the query embeddings as the average of the top-ranked document embeddings.
    """
    TCTColBERT = 1
    AVERAGE = 2


# TODO: split this code into functions
if __name__ == '__main__':
    """
    Re-ranking Stage: Create query embeddings and re-rank documents based on similarity to query embeddings.

    This script takes the initial ranking of documents and re-ranks them based on the similarity to the query embeddings.
    It uses various encoding methods and evaluation metrics to achieve this.

    Args:
        ranking_path (Path): Path to the first-stage ranking file.
            - Example: Path("/home/bvdb9/sparse_rankings/msmarco-passage-test2019-sparse10000.txt")
        index_path (Path): Path to the index file.
            - Example: Path("/home/bvdb9/indices/msm-psg/ff/ff_index_msmpsg_TCTColBERT_opq.h5")
        ranking_output_path (Path): Path to save the re-ranked ranking.
            - Example: Path("rerank-avg.tsv")
        dataset (Dataset): Dataset to evaluate the re-ranked ranking (provided by ir_datasets package).
            - Example: ir_datasets.load("msmarco-passage/trec-dl-2019")
        rerank_cutoff (int): Number of documents to re-rank per query.
            - Example: 1000
        encoding_method (EncodingMethod): Method to estimate query embeddings.
            - Example: EncodingMethod.AVERAGE
        k_top_docs (int): Number of top-ranked documents to use for EncodingMethod.AVERAGE.
            - Example: 10
        in_memory (bool): Whether to load the index in memory.
            - Allowed: True or False
        device (str): Device to use for encoding queries.
            - Allowed: "cuda" or "cpu"
        eval_metrics (List[str]): Metrics used for evaluation.
            - Example: ["nDCG@10"]
        alphas (List[float]): List of interpolation parameters for evaluation.
            - Example: [0, 0.25, 0.5, 0.75, 1]
            - a = 0: dense score
            - 0 < a < 1: interpolated score
            - a = 1: sparse score

    Input:
        ranking (List[Tuple]): A ranking of documents for each given query.
            - Format: (q_id, q0, d_id, rank, score, name)
        ff_index (Index): Used to retrieve document embeddings.

    Output:
        ranking (List[Tuple]): A re-ranked ranking of documents for each given query.
            - Saved to ranking_output_path

    Example(s):
        ```python
        ranking_path = Path("/home/bvdb9/sparse_rankings/msmarco-passage-test2019-sparse10000.txt")
        index_path = Path("/home/bvdb9/indices/msm-psg/ff/ff_index_msmpsg_TCTColBERT_opq.h5")
        ranking_output_path = Path("rerank-avg.tsv")
        dataset = ir_datasets.load("msmarco-passage/trec-dl-2019")
        rerank_cutoff = 100
        encoding_method = EncodingMethod.AVERAGE
        k_top_docs = 10
        in_memory = True
        device = "cuda"
        eval_metrics = ["nDCG@10"]
        alphas = [0, 0.25, 0.5, 0.75, 1]
        ```
    """
    # Arguments
    ranking_path: Path = Path("/home/bvdb9/sparse_rankings/msmarco-passage-test2019-sparse10000.txt")
    index_path: Path = Path("/home/bvdb9/indices/msm-psg/ff/ff_index_msmpsg_TCTColBERT_opq.h5")
    ranking_output_path: Path = Path("rerank-avg.tsv")
    dataset = ir_datasets.load("msmarco-passage/trec-dl-2019")
    rerank_cutoff: int = 1000
    encoding_method = EncodingMethod.AVERAGE
    k_top_docs: int = 10
    in_memory: bool = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    eval_metrics: list[str] = [nDCG@10]
    alphas: list[float] = [0, 0.25, 0.5, 0.75, 1]

    # load the index
    index: Index = OnDiskIndex.load(index_path)
    if in_memory:
        index = index.to_memory()

    # load the ranking and attach the queries
    sparse_ranking: Ranking = Ranking.from_file(
        ranking_path,
        queries={q.query_id: q.text for q in dataset.queries_iter()},
    ).cut(rerank_cutoff) # Cutoff to top_k docs per query

    # Find unique queries and save their index in q_no column
    uniq_q = sparse_ranking._df[["q_id", "query"]].drop_duplicates().reset_index(drop=True)
    uniq_q["q_no"] = uniq_q.index
    print(f"uniq_q shape: {uniq_q.shape}, head:\n{uniq_q.head()}")

    # Merge q_no into the sparse ranking
    sparse_ranking._df = sparse_ranking._df.merge(uniq_q, on=["q_id", "query"])
    print(f"sparse_ranking._df shape: {sparse_ranking._df.shape}, head:\n{sparse_ranking._df.head()}")

    # Create q_reps as np.ndarray with shape (len(ranking), index.dim) where index.dim is the dimension of the embeddings, often 768.
    q_reps: np.ndarray = np.zeros((len(sparse_ranking), index.dim), dtype=np.float32)
    match encoding_method:
        case EncodingMethod.TCTColBERT:
            # Default approach: encode queries using a query_encoder
            index.query_encoder = TCTColBERTQueryEncoder("castorini/tct_colbert-msmarco", device=device)
            q_reps = index.encode_queries(uniq_q["query"])
        case EncodingMethod.AVERAGE:
            # Estimate the query embeddings as the average of the top-ranked document embeddings
            # TODO: This task can probably be parallelized
            top_docs = sparse_ranking.cut(k_top_docs)
            for q_id, query, q_no in tqdm(
                uniq_q.itertuples(index=False), 
                desc="Estimating query embeddings", 
                total=len(uniq_q)
            ):
                # get the embeddings of the top_docs from the index
                top_docs_ids = top_docs[q_id].keys()
                d_reps: np.ndarray = index._get_vectors(top_docs_ids)[0]
                if index.quantizer is not None:
                    d_reps = index.quantizer.decode(d_reps)

                # calculate the average of the embeddings and save it
                q_reps[q_no] = np.mean(d_reps, axis=0)
    print(f"qreps shape: {q_reps.shape}, head:\n{pd.DataFrame(q_reps).head()}")

    # TODO: Check if only docs up until the cutoff are re-ranked
    result = index._compute_scores(sparse_ranking._df, q_reps)
    result["score"] = result["ff_score"]

    dense_ranking = Ranking(
        result,
        name="fast-forward",
        dtype=sparse_ranking._df.dtypes["score"],
        copy=False,
        is_sorted=False,
    )
    dense_ranking.save(ranking_output_path) # Save dense ranking to file

    sparse_ranking: Ranking = Ranking.from_file(
        ranking_path,
        queries={q.query_id: q.text for q in dataset.queries_iter()},
    )

    # Print settings
    settings_description: List[str] = [
        f"ranking={ranking_path.name}",
        f"index={index_path.name}",
        f"rerank_cutoff={rerank_cutoff}",
        f"encoding_method={encoding_method}",
    ]
    match encoding_method: # Append method-specific settings
        case EncodingMethod.TCTColBERT:
            settings_description.append(f"device={device}")
        case EncodingMethod.AVERAGE:
            settings_description.append(f"k_top_docs={k_top_docs}")
    print("\nSettings:\n\t" + ",\n\t".join(settings_description))

    # Print results
    print('Results:')
    for alpha in alphas:
        interpolated_ranking = sparse_ranking.interpolate(dense_ranking, alpha)
        score = calc_aggregate(eval_metrics, dataset.qrels_iter(), to_ir_measures(interpolated_ranking))
        ranking_type = (
            "Sparse" if alpha == 1 else 
            "Dense" if alpha == 0 else 
            "Interpolated"
        )
        print(f"\t{ranking_type} ranking (alpha={alpha}): {score}")
