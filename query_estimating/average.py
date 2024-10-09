### Description: In the re-ranking stage, we can estimate the query embedding as the average of the top-ranked document embeddings from the first-stage retrieval.
# From first-stage retrieval, we have:
# - ranking: a ranking of documents for a given query
    # - format: (q_id, q0, d_id, rank, score, name)
# - ff_index: used to retrieve document embeddings

from pathlib import Path
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


if __name__ == '__main__':
    ### PARAMETERS
    ranking_path: Path = Path("/home/bvdb9/sparse_rankings/msmarco-passage-test2019-sparse10000.txt")
    index_path: Path = Path("/home/bvdb9/indices/msm-psg/ff/ff_index_msmpsg_TCTColBERT_opq.h5")
    ranking_output_path: Path = Path("rerank-avg.tsv")
    dataset = ir_datasets.load("msmarco-passage/trec-dl-2019")
    top_k: int = 10
    use_traditional_enc: bool = True
    traditional_enc_k_s: int = 1000
    in_memory: bool = False
    device = "cuda" if torch.cuda.is_available() else "cpu"


    # load the ranking and attach the queries
    sparse_ranking: Ranking = Ranking.from_file(
        ranking_path,
        queries={q.query_id: q.text for q in dataset.queries_iter()},
    )

    # load the index
    index: Index = OnDiskIndex.load(index_path)
    if in_memory:
        index = index.to_memory()

    if use_traditional_enc:
        sparse_ranking = sparse_ranking.cut(traditional_enc_k_s)

    # Get each query (q_id, query text) and assign a unique int id q_no
    query_df = (sparse_ranking._df[["q_id", "query"]].drop_duplicates().reset_index(drop=True))
    query_df["q_no"] = query_df.index
    sparse_ranking_df = sparse_ranking._df.merge(query_df, on="q_id", suffixes=[None, "_"])

    # Create q_reps as np.ndarray with shape (len(ranking), index.dim) where index.dim is the dimension of the embeddings, often 768.
    q_reps: np.ndarray = np.zeros((len(sparse_ranking), index.dim), dtype=np.float32)
    if use_traditional_enc:
        # Default approach: encode queries using a query_encoder
        index.query_encoder = TCTColBERTQueryEncoder("castorini/tct_colbert-msmarco", device=device)
        q_reps = index.encode_queries(list(query_df["query"]))
    else:
        # Estimate the query embeddings as the average of the top-ranked document embeddings
        top_sparse_ranking = sparse_ranking.cut(top_k) # keep only the top_k docs per query
        # TODO: Should this for-loop go over the newly indexed query_df instead?
        for i, q_id in enumerate(tqdm(top_sparse_ranking, desc="Estimating query embeddings", total=len(sparse_ranking))):
            # get the embeddings of the top_docs from the index
            top_docs_ids = top_sparse_ranking[q_id].keys()
            d_reps: np.ndarray = index._get_vectors(top_docs_ids)[0]

            # calculate the average of the embeddings and save it
            # TODO: should I use q_id - 1 or i as index? index 451601 is out of bounds for axis 0 with size 43 <-- 451601 is the q_id
            q_reps[int(q_id) - 1] = np.mean(d_reps, axis=0)
    print('q_reps shape', q_reps.shape, 'head:\n', pd.DataFrame(q_reps).head())

    result = index._compute_scores(sparse_ranking_df, q_reps)
    result["score"] = result["ff_score"]

    dense_ranking = Ranking(
        result,
        name="fast-forward",
        dtype=sparse_ranking._df.dtypes["score"],
        copy=False,
        is_sorted=False,
    )

    # Save dense ranking to output file
    dense_ranking.save(ranking_output_path)

    # Compare original [sparse, dense, interpolated] rankings, printing the results
    eval_metrics: list[str] = [nDCG@10]
    alphas: list[float] = [0, 0.1, 0.25, 0.5, 0.75, 1]
    print(f"\nResults (top_k docs={top_k}, ranking={ranking_path.name}, index={index_path.name}, use_default_encoding={use_traditional_enc}):")
    for alpha in alphas:
        interpolated_ranking = sparse_ranking.interpolate(dense_ranking, alpha)
        score = calc_aggregate(eval_metrics, dataset.qrels_iter(), to_ir_measures(interpolated_ranking))
        ranking_type = (
            "Sparse" if alpha == 1 else 
            "Dense" if alpha == 0 else 
            "Interpolated"
        )
        print(f"\t{ranking_type} ranking (alpha={alpha}): {score}")
