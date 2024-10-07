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
    ranking_path: Path = Path("/home/bvdb9/runs/vaswani-None-BM25-top10000.tsv")
    index_path: Path = Path("/home/bvdb9/indices/vaswani/ff_index_TCTColBERT.h5")
    ranking_output_path: Path = Path("rerank-avg.tsv")
    dataset = ir_datasets.load("vaswani")
    top_k: int = 10


    # load the ranking and attach the queries
    sparse_ranking: Ranking = Ranking.from_file(
        ranking_path,
        queries={q.query_id: q.text for q in dataset.queries_iter()},
    )

    # load the index
    index: Index = OnDiskIndex.load(index_path)

    # Create q_reps as np.ndarray with shape (len(ranking), index.dim) where index.dim is the dimension of the embeddings, often 768.
    q_reps: np.ndarray = np.zeros((len(sparse_ranking), index.dim), dtype=np.float32)

    top_sparse_ranking = sparse_ranking.cut(top_k) # keep only the top_k docs per query

    # for each query, get the embeddings of the top_docs
    for i, q_id in enumerate(tqdm(top_sparse_ranking, desc="Estimating query embeddings", total=len(sparse_ranking))):
        # get the embeddings of the top_docs from the index
        top_docs_ids = top_sparse_ranking[q_id].keys()
        d_reps: np.ndarray = index._get_vectors(top_docs_ids)[0]

        # calculate the average of the embeddings and save it
        # TODO: should I use q_id - 1 or i as index?
        q_reps[int(q_id) - 1] = np.mean(d_reps, axis=0)

    q_reps_df = pd.DataFrame(q_reps)
    print('q_reps shape', q_reps.shape, 'head:\n', q_reps_df.head())

    # TODO: understand these next lines (until results)
    query_df = (sparse_ranking._df[["q_id", "query"]].drop_duplicates().reset_index(drop=True))
    query_df["q_no"] = query_df.index

    df = sparse_ranking._df.merge(query_df, on="q_id", suffixes=[None, "_"])

    result = index._compute_scores(df, q_reps)
    result["score"] = result["ff_score"]

    dense_ranking = Ranking(
        result,
        name="fast-forward",
        dtype=sparse_ranking._df.dtypes["score"],
        copy=False,
        is_sorted=False,
    )

    # Compare original [sparse, dense, interpolated] rankings, printing the results
    eval_metrics: list[str] = [nDCG@10]
    print("\nResults:")
    print("\tDense score ranking (a=0): ", calc_aggregate(eval_metrics, dataset.qrels_iter(), to_ir_measures(dense_ranking)))
    for a in [0.1, 0.25, 0.5, 0.75]:
        interpolated_ranking = sparse_ranking.interpolate(dense_ranking, a)
        score = calc_aggregate(eval_metrics, dataset.qrels_iter(), to_ir_measures(interpolated_ranking))
        print(f"\tInterpolated score ranking (a={a}): ", score)
    print("\tSparse score ranking (a=1): ", calc_aggregate(eval_metrics, dataset.qrels_iter(), to_ir_measures(sparse_ranking)))
