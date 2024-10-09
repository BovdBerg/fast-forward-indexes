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
    """Re-ranking stage: Create query embeddings and re-rank documents based on similarity to queries embeddings.
    
    Input (from first-stage retrieval):
    - ranking: a ranking of documents for each given query
        - format: (q_id, q0, d_id, rank, score, name)
    - ff_index: used to retrieve document embeddings

    Output:
    - ranking: a re-ranked ranking of documents for each given query
    """
    ### PARAMETERS (SETTINGS)
    ranking_path: Path = Path("/home/bvdb9/sparse_rankings/msmarco-passage-test2019-sparse10000.txt")
    index_path: Path = Path("/home/bvdb9/indices/msm-psg/ff/ff_index_msmpsg_TCTColBERT_opq.h5")
    ranking_output_path: Path = Path("rerank-avg.tsv")
    dataset = ir_datasets.load("msmarco-passage/trec-dl-2019")
    rerank_cutoff: int = 1000
    encoding_method = EncodingMethod.AVERAGE
    in_memory: bool = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    eval_metrics: list[str] = [nDCG@10]
    alphas: list[float] = [0, 0.25, 0.5, 0.75, 1] # a=0: dense, 0 < a < 1: interpolated, a=1: sparse


    # load the index
    index: Index = OnDiskIndex.load(index_path)
    if in_memory:
        index = index.to_memory()

    # load the ranking and attach the queries
    sparse_ranking: Ranking = Ranking.from_file(
        ranking_path,
        queries={q.query_id: q.text for q in dataset.queries_iter()},
    ).cut(rerank_cutoff) # Cutoff to top_k docs per query
    sparse_ranking._df["q_no"] = pd.Categorical(sparse_ranking._df["q_id"][::-1]).codes # Map queries to numerical categories in q_no column
    print('sparse_ranking._df shape:', sparse_ranking._df.shape, 'head:\n', sparse_ranking._df.head())

    # Create q_reps as np.ndarray with shape (len(ranking), index.dim) where index.dim is the dimension of the embeddings, often 768.
    q_reps: np.ndarray = np.zeros((len(sparse_ranking), index.dim), dtype=np.float32)
    match encoding_method:
        case EncodingMethod.TCTColBERT:
            # Default approach: encode queries using a query_encoder
            index.query_encoder = TCTColBERTQueryEncoder("castorini/tct_colbert-msmarco", device=device)
            q_reps = index.encode_queries(list(sparse_ranking._df["query"].drop_duplicates()))
        case EncodingMethod.AVERAGE:
            # Estimate the query embeddings as the average of the top-ranked document embeddings
            # TODO: This task can probably be parallelized
            for q_no, q_id in tqdm(
                sparse_ranking._df[["q_no", "q_id"]].drop_duplicates().itertuples(index=False), 
                desc="Estimating query embeddings", 
                total=len(sparse_ranking)
            ):
                # get the embeddings of the top_docs from the index
                top_docs_ids = list(sparse_ranking._df[sparse_ranking._df["q_id"] == q_id]["id"])
                d_reps: np.ndarray = index._get_vectors(top_docs_ids)[0]
                if index.quantizer is not None:
                    d_reps = index.quantizer.decode(d_reps)

                # calculate the average of the embeddings and save it
                q_reps[q_no] = np.mean(d_reps, axis=0)
    print('q_reps shape', q_reps.shape, 'head:\n', pd.DataFrame(q_reps).head())

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

    # TODO: Why does the sparse_ranking score differently depending on the cutoff if I reload the original ranking file here?
    sparse_ranking: Ranking = Ranking.from_file(
        ranking_path,
        queries={q.query_id: q.text for q in dataset.queries_iter()},
    )

    # Save dense ranking to output file
    dense_ranking.save(ranking_output_path)

    # Compare original [sparse, dense, interpolated] rankings, printing the results
    settings_description: List[str] = [
        f"ranking={ranking_path.name}",
        f"index={index_path.name}",
        f"rerank_cutoff={rerank_cutoff}",
        f"encoding_method={encoding_method}",
        f"k_top_docs={k_top_docs}" if encoding_method == EncodingMethod.AVERAGE else "",
    ]
    settings_description = [s for s in settings_description if s]  # Remove empty strings
    print("\nSettings:\n\t" + ",\n\t".join(settings_description) + "\n")
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
