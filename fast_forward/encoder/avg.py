from enum import Enum
from typing import Sequence

import numpy as np
import pandas as pd
from fast_forward.encoder import Encoder
from fast_forward.index import Index
from fast_forward.ranking import Ranking


class ProbDist(Enum):
    """
    Enumeration for different types of probability distributions used to assign weights to top-ranked documents in the WeightedAvgEncoder.

    Attributes:
        UNIFORM: all top-ranked documents are weighted equally.
        GEOMETRIC: weights decrease geometrically with rank.
        EXPONENTIAL: weights decrease exponentially with rank.
        HALF_NORMAL: weights decrease with the half-normal distribution.
    """
    UNIFORM = "UNIFORM"
    GEOMETRIC = "GEOMETRIC"
    EXPONENTIAL = "EXPONENTIAL"
    HALF_NORMAL = "HALF_NORMAL"


class WeightedAvgEncoder(Encoder):
    """
    WeightedAvgEncoder estimates the query embeddings as the weighted average of the top-ranked document embeddings.

    Attributes:
        top_ranking (Ranking): The top-ranked documents used for averaging.
        index (Index): The index containing document embeddings.
    """
    def __init__(
        self,
        sparse_ranking: Ranking,
        index: Index,
        k_avg: int,
        prob_dist: ProbDist,
    ) -> None:
        """
        Initialize the WeightedAvgEncoder with the given sparse ranking, index, and number of top documents to average.

        Args:
            sparse_ranking (Ranking): The initial sparse ranking of documents.
            index (Index): The index containing document embeddings.
            k_avg (int): Number of top-ranked documents to use for averaging.
            weights (Sequence[float]): A sequence of interpolation parameters.
        """
        self.top_ranking = sparse_ranking.cut(k_avg)
        self.index = index
        self.prob_dist: ProbDist = prob_dist
        self.k_avg: int = k_avg
        self.weights: Sequence[float] = self._get_weights()
        super().__init__()


    def _get_weights(self) -> Sequence[float]:
        """
        Get the weights for the top-ranked documents based on the probability distribution type.

        Returns:
            Sequence[float]: A sequence of interpolation parameters.
        """
        match self.prob_dist:
            case ProbDist.UNIFORM:
                return np.ones(self.k_avg) / self.k_avg
            case ProbDist.GEOMETRIC:
                return np.geomspace(1, 0.1, self.k_avg)
            case ProbDist.EXPONENTIAL:
                return np.exp(-np.linspace(0, 1, self.k_avg))
            case ProbDist.HALF_NORMAL:
                return np.flip(np.exp(-np.linspace(0, 1, self.k_avg) ** 2))
            case _:
                raise ValueError(f"Unknown probability distribution type: {self.prob_dist}")


    def __call__(self, queries: Sequence[str]) -> np.ndarray:
        """
        Estimate query embeddings by averaging the embeddings of the top-ranked documents with interpolation.

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

            # Calculate the weighted average of the embeddings and save it to q_no index in q_reps
            q_reps[i] = np.average(d_reps, axis=0, weights=self.weights)

        return q_reps
