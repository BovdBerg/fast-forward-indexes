from typing import Sequence

import numpy as np
import pandas as pd
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
        self.top_ranking = sparse_ranking.cut(k_avg)
        self.index = index


    def __call__(self, queries: Sequence[str]) -> np.ndarray:
        q_reps: np.ndarray = np.zeros((len(self.top_ranking), self.index.dim), dtype=np.float32)
        
        for query in tqdm(queries, desc="Estimating query embeddings", total=len(queries)):
            # Get the ids of the top-ranked documents for the query
            top_docs: pd.DataFrame = self.top_ranking._df.query("query == @query")
            top_docs_ids: Sequence[int] = top_docs["id"].values

            # Get the embeddings of the top-ranked documents
            d_reps: np.ndarray = self.index._get_vectors(top_docs_ids)[0]            
            if self.index.quantizer is not None:
                d_reps = self.index.quantizer.decode(d_reps)

            # Calculate the average of the embeddings and save it to q_no index in q_reps
            q_no = top_docs["q_no"].iloc[0]
            q_reps[q_no] = np.mean(d_reps, axis=0)
        
        return q_reps
