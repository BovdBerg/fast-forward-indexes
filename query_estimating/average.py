### Description: In the re-ranking stage, we can estimate the query embedding as the average of the top-ranked document embeddings from the first-stage retrieval.
# From first-stage retrieval, we have:
# - ranking: a ranking of documents for a given query
    # - format: (q_id, q0, d_id, rank, score, name)
# - ff_index: used to retrieve document embeddings

from typing import Dict
from pathlib import Path
import numpy as np
from fast_forward.index import Index
from fast_forward.index.disk import OnDiskIndex
from fast_forward.ranking import Ranking
from tqdm import tqdm
import ir_datasets


if __name__ == '__main__':
    ### PARAMETERS
    ranking_path: Path = Path("/home/bvdb9/runs/vaswani-None-BM25-top1000.tsv")
    index_path: Path = Path("/home/bvdb9/indices/vaswani/ff_index_TCTColBERT.h5")
    ranking_output_path: Path = Path("avg-embeddings.tsv")
    dataset = ir_datasets.load("vaswani")
    top_k: int = 10


    # load the ranking and attach the queries
    ranking: Ranking = Ranking.from_file(
        ranking_path,
        queries={q.query_id: q.text for q in dataset.queries_iter()},
    )
    print('len(ranking.q_ids)', len(ranking))

    # load the index
    index: Index = OnDiskIndex.load(index_path)
    print('len(index)', len(index))

    q_reps: Dict[int, np.ndarray] = {}
    # for each query, get the embeddings of the top_docs
    top_ranking = ranking.cut(top_k) # keep only the top_k docs per query
    for q_id in tqdm(top_ranking, desc="Estimating query embeddings", total=len(ranking)):
        # get the embeddings of the top_docs from the index
        top_docs_ids = top_ranking[q_id].keys()
        d_reps: np.ndarray = index._get_vectors(top_docs_ids)[0]

        # calculate the average of the embeddings and save it
        q_reps[q_id] = d_reps.mean(axis=0)
    print("done")
