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
from ir_measures import calc_aggregate, nDCG
from fast_forward.util import to_ir_measures


if __name__ == '__main__':
    ### PARAMETERS
    ranking_path: Path = Path("/home/bvdb9/runs/vaswani-None-BM25-top1000.tsv")
    index_path: Path = Path("/home/bvdb9/indices/vaswani/ff_index_TCTColBERT.h5")
    ranking_output_path: Path = Path("rerank-avg.tsv")
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

    # For each query
    # - Get the documents from the ranking
    # - Get the embeddings of these documents
    # - re-rank the top documents as nearest neighbors to the estimated query embedding
    # - save the re-ranked documents as the final ranking
    for q_id, q_rep in tqdm(q_reps.items(), desc="Re-ranking on distance to new query embeddings", total=len(q_reps)):
        # Get the top documents from the ranking
        docs_ids = ranking[q_id].keys()
        d_reps: np.ndarray = index._get_vectors(docs_ids)[0]
        
        # Re-rank the nearest neighbors (cosine similarity) to q_vec as top documents
        scores = {}
        for d_id, d_rep in zip(docs_ids, d_reps):
            scores[d_id] = np.dot(q_rep, d_rep) # cosine similarity
        
        # Sort documents by score in descending order
        sorted_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        
        # Save the re-ranked documents as the final ranking
        with open(ranking_output_path, 'a') as f:
            for rank, (d_id, score) in enumerate(sorted_docs, start=1):
                f.write(f"{q_id}\tQ0\t{d_id}\t{rank}\t{score}\tfast-forward\n")

    # print the head of the ranking_output_path
    print(f"Saved reranking to {ranking_output_path}, head:")
    print('\tq_id\titer\td_id\trank\tscore\t\t\ttag')
    with open(ranking_output_path, 'r') as f:
        for i in range(3):
            print("\t" + f.readline().strip())

    ### Compare original sparse ranking to re-ranked ranking and print results
    sparse_ranking = Ranking.from_file(
        ranking_path,
        queries={q.query_id: q.text for q in dataset.queries_iter()},
    )
    reranked_ranking = Ranking.from_file(
        ranking_output_path,
        queries={q.query_id: q.text for q in dataset.queries_iter()},
    )

    print("\nResults:")
    eval_metrics = [nDCG@10]
    print("\tSparse ranking: ", calc_aggregate(eval_metrics, dataset.qrels_iter(), to_ir_measures(sparse_ranking)))
    print("\tRe-ranked ranking: ", calc_aggregate(eval_metrics, dataset.qrels_iter(), to_ir_measures(reranked_ranking)))
