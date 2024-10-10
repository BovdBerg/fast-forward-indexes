from typing import Sequence

import numpy as np
import pandas as pd
from fast_forward.encoder import Encoder
from fast_forward.index import Index
from fast_forward.ranking import Ranking


class Estimator(Encoder):
    """
    Estimator is an abstract class for estimating query embeddings based on top-ranked documents.

    Attributes:
        index (Index): An index object containing document embeddings.
        top_ranking (TopRanking): An object containing the top-ranked documents.
    """
    def __init__(
        self,
        sparse_ranking: Ranking,
        index: Index,
        k_avg: int,
    ) -> None:
        """
        Initialize the Estimator with the given sparse ranking, index, and number of top documents to average.

        Args:
            sparse_ranking (Ranking): The initial sparse ranking of documents.
            index (Index): The index containing document embeddings.
            k_avg (int): Number of top-ranked documents to use for averaging.
        """
        super().__init__()
        self.top_ranking = sparse_ranking.cut(k_avg)
        self.index = index


class AvgEncoder(Estimator):
    """
    AvgEncoder estimates the query embeddings as the average of the top-ranked document embeddings.

    Attributes:
        top_ranking (Ranking): The top-ranked documents used for averaging.
        index (Index): The index containing document embeddings.
    """
    def __call__(self, queries: Sequence[str]) -> np.ndarray:
        """
        Estimate query embeddings by averaging the embeddings of the top-ranked documents.

        Args:
            queries (Sequence[str]): A sequence of query strings.

        Returns:
            np.ndarray: An array of query embeddings.
        """
        q_reps: np.ndarray = np.zeros((len(queries), self.index.dim), dtype=np.float32)

        for i, query in enumerate(queries):
            # Get the ids of the top-ranked documents for the query
            top_docs: pd.DataFrame = self.top_ranking._df.query("query == @query")
            top_docs_ids: Sequence[int] = top_docs["id"].values

            # Get the embeddings of the top-ranked documents
            d_reps: np.ndarray = self.index._get_vectors(top_docs_ids)[0]            
            if self.index.quantizer is not None:
                d_reps = self.index.quantizer.decode(d_reps)

            # Calculate the average of the embeddings and save it to q_no index in q_reps
            q_reps[i] = np.mean(d_reps, axis=0)

        return q_reps
