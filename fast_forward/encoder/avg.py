from typing import Sequence

import numpy as np
from tqdm import tqdm
from fast_forward.encoder import Encoder
from fast_forward.index import Index
from fast_forward.ranking import Ranking


# TODO: Add pydoc to class and methods
class AvgEncoder(Encoder):
    """
    Estimate the query embeddings as the average of the top-ranked document embeddings
    """
    def __init__(
        self,
        sparse_ranking: Ranking,
        index: Index,
        k_avg: int,
    ) -> None:
        super().__init__()
        self.sparse_ranking = sparse_ranking
        self.index = index
        self.k_avg = k_avg


    def __call__(self, queries: Sequence[str]) -> np.ndarray:
        q_reps: np.ndarray = np.zeros((len(self.sparse_ranking), self.index.dim), dtype=np.float32)
        top_docs = self.sparse_ranking.cut(self.k_avg)
        for query in tqdm(queries, desc="Estimating query embeddings", total=len(queries)):
            # Find [q_id, q_no] from sparse_ranking filtered on the query
            # TODO: Make line below more readable
            q_id, q_no = self.sparse_ranking._df[self.sparse_ranking._df['query'] == query][['q_id', 'q_no']].iloc[0]
            
            # get the embeddings of the top_docs from the index
            top_docs_ids = top_docs[q_id].keys()
            d_reps: np.ndarray = self.index._get_vectors(top_docs_ids)[0]
            if self.index.quantizer is not None:
                d_reps = self.index.quantizer.decode(d_reps)

            # calculate the average of the embeddings and save it
            q_reps[q_no] = np.mean(d_reps, axis=0)
        return q_reps
