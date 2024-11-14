from enum import Enum
from typing import Sequence

import numpy as np
import pandas as pd

from fast_forward.encoder import Encoder
from fast_forward.index import Index
from fast_forward.ranking import Ranking


class W_METHOD(Enum):
    """
    Enumeration for different types of probability distributions used to assign weights to top-ranked documents in the WeightedAvgEncoder.

    Attributes:
        UNIFORM: all top-ranked documents are weighted equally.
        EXPONENTIAL: weights decrease exponentially with rank.
        HALF_NORMAL: weights decrease with the half-normal distribution.
        SOFTMAX_SCORES: weights are assigned based on the softmax of the scores.
        LINEAR_DECAY_RANKS: weights decrease linearly with rank.
        LINEAR_DECAY_SCORES: weights decrease linearly with rank, multiplied by the scores.
    """

    UNIFORM = "UNIFORM"
    EXPONENTIAL = "EXPONENTIAL"
    HALF_NORMAL = "HALF_NORMAL"
    SOFTMAX_SCORES = "SOFTMAX_SCORES"
    LINEAR_DECAY_RANKS = "LINEAR_DECAY_RANKS"
    LINEAR_DECAY_SCORES = "LINEAR_DECAY_SCORES"
    # TODO [IMPORTANT]: Add LEARNED distribution, with learned model weights based on training/validation data


# TODO: top_ranking should probably be updated in each chain reranking. Check if only top_docs from original top_ranking are considered?
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
        w_method: W_METHOD,
        sparse_ranking: Ranking = None,
    ) -> None:
        """
        Initialize the WeightedAvgEncoder with the given sparse ranking, index, and number of top documents to average.

        Args:
            index (Index): The index containing document embeddings.
            k_avg (int): Number of top-ranked documents to use for averaging.
            w_method (ProbDist): The probability distribution type used to assign weights to top-ranked documents.
            sparse_ranking (Ranking): The initial sparse ranking of documents.
        """
        self.index = index
        self.w_method: W_METHOD = w_method
        self.k_avg: int = k_avg
        self.sparse_ranking = sparse_ranking if sparse_ranking is not None else None
        super().__init__()

    def _get_weights(self, n_docs: int, scores: Sequence[float]) -> Sequence[float]:
        """
        Get the weights for the top-ranked documents based on the probability distribution type.

        Args:
            n_docs (int): Number of top-ranked documents

        Returns:
            Sequence[float]: A sequence of interpolation parameters.
        """
        match self.w_method:
            # See description of ProbDist for details on each distribution
            case W_METHOD.UNIFORM:
                return np.ones(n_docs) / n_docs
            case W_METHOD.EXPONENTIAL:
                return np.exp(-np.linspace(0, 1, n_docs))
            case W_METHOD.HALF_NORMAL:
                return np.flip(np.exp(-np.linspace(0, 1, n_docs) ** 2))
            case W_METHOD.SOFTMAX_SCORES:
                max_score = np.max(scores)
                exp_scores = np.exp(scores - max_score)
                return exp_scores / np.sum(exp_scores)
            case W_METHOD.LINEAR_DECAY_RANKS:
                return np.linspace(1, 0, n_docs)
            case W_METHOD.LINEAR_DECAY_SCORES:
                return np.linspace(1, 0, n_docs) * scores
            case _:
                raise ValueError(
                    f"Unknown probability distribution type: {self.w_method}"
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
            top_docs_scores: Sequence[float] = top_docs["score"].values

            # Get the embeddings of the top-ranked documents
            d_reps: np.ndarray = self.index._get_vectors(top_docs_ids)[0]
            if self.index.quantizer is not None:
                d_reps = self.index.quantizer.decode(d_reps)

            # Calculate the weighted average of the embeddings and save it to q_no index in q_reps
            q_reps[i] = np.average(
                d_reps, axis=0, weights=self._get_weights(len(d_reps), top_docs_scores)
            )

        return q_reps
