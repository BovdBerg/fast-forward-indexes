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
    EXPONENTIAL = "EXPONENTIAL"
    HALF_NORMAL = "HALF_NORMAL"
    # TODO: Add LEARNED distribution, with learned model weights based on training/validation data


class WeightedAvgEncoder(Encoder):
    """
    WeightedAvgEncoder estimates the query embeddings as the weighted average of the top-ranked document embeddings.

    Attributes:
        sparse_ranking (Ranking): The top-ranked documents used for averaging.
        index (Index): The index containing document embeddings.
    """

    def __init__(
        self,
        index: Index,
        k_avg: int,
        prob_dist: ProbDist,
        sparse_ranking: Ranking = None,
    ) -> None:
        """
        Initialize the WeightedAvgEncoder with the given sparse ranking, index, and number of top documents to average.

        Args:
            index (Index): The index containing document embeddings.
            k_avg (int): Number of top-ranked documents to use for averaging.
            prob_dist (ProbDist): The probability distribution type used to assign weights to top-ranked documents.
            sparse_ranking (Ranking): The initial sparse ranking of documents.
        """
        self.index = index
        self.prob_dist: ProbDist = prob_dist
        self.k_avg: int = k_avg
        self.sparse_ranking = sparse_ranking if sparse_ranking is not None else None
        super().__init__()

    def _get_weights(self, n_docs: int) -> Sequence[float]:
        """
        Get the weights for the top-ranked documents based on the probability distribution type.

        Args:
            n_docs (int): Number of top-ranked documents

        Returns:
            Sequence[float]: A sequence of interpolation parameters.
        """
        match self.prob_dist:
            case ProbDist.UNIFORM:
                return np.ones(n_docs) / n_docs
            case ProbDist.EXPONENTIAL:
                return np.exp(-np.linspace(0, 1, n_docs))
            case ProbDist.HALF_NORMAL:
                return np.flip(np.exp(-np.linspace(0, 1, n_docs) ** 2))
            case _:
                raise ValueError(
                    f"Unknown probability distribution type: {self.prob_dist}"
                )

    def __call__(self, queries: Sequence[str]) -> np.ndarray:
        """
        Estimate query embeddings by weighted averaging the embeddings of the top-ranked documents.

        Args:
            queries (Sequence[str]): A sequence of query strings.

        Returns:
            np.ndarray: An array of query embeddings.
        """
        assert (
            self.sparse_ranking is not None
        ), "Please set the top_sparse_ranking attribute before calling the encoder."
        top_ranking = self.sparse_ranking

        if self.k_avg is not None:
            top_ranking = top_ranking.cut(self.k_avg)

        q_reps: np.ndarray = np.zeros((len(queries), self.index.dim), dtype=np.float32)

        for i, query in enumerate(queries):
            # Get the ids of the top-ranked documents for the query
            top_docs: pd.DataFrame = top_ranking._df.query("query == @query")
            top_docs_ids: Sequence[int] = top_docs["id"].values

            # Get the embeddings of the top-ranked documents
            d_reps: np.ndarray = self.index._get_vectors(top_docs_ids)[0]
            if self.index.quantizer is not None:
                d_reps = self.index.quantizer.decode(d_reps)

            # Calculate the weighted average of the embeddings and save it to q_no index in q_reps
            q_reps[i] = np.average(
                d_reps, axis=0, weights=self._get_weights(len(d_reps))
            )

        return q_reps
