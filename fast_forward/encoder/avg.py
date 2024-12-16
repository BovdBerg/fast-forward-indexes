from enum import Enum
from pathlib import Path
from typing import Sequence

import lightning
import numpy as np
import pandas as pd
import torch

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
    LEARNED = "LEARNED"
    # TODO [later]: After adding LEARNED distribution, should I train different transformers when chaining (per FFScore_i)?


# TODO: Train model with query --> TCT-ColBERT query (KD)
# TODO: Train model with query + top_docs as input
class WeightedAvgEncoder(Encoder):
    """
    WeightedAvgEncoder estimates the query embeddings as the weighted average of the top-ranked document embeddings.
    """

    def __init__(
        self,
        index: Index,
        w_method: W_METHOD = W_METHOD.SOFTMAX_SCORES,
        k_avg: int = 30,
        ckpt_path: Path = None,
        device: str = "cpu",
        **enc_args,
    ) -> None:
        """
        Initialize the WeightedAvgEncoder with the given sparse ranking, index, and number of top documents to average.

        Args:
            index (Index): The index containing document embeddings.
            w_method (W_METHOD): The probability distribution type used to assign weights to top-ranked documents.
            k_avg (int): The number of top-ranked documents to average.
            ranking_in (Ranking): The initial sparse ranking of documents.
        """
        super().__init__()
        self.index = index
        self.w_method = w_method
        self.k_avg = k_avg
        self._ranking_in = None
        self.device = device

        if ckpt_path is not None:
            self.learned_avg_weights = LearnedAvgWeights.load_from_checkpoint(
                ckpt_path, k_avg=k_avg, **enc_args
            )
        else:
            self.learned_avg_weights = LearnedAvgWeights(k_avg=self.k_avg, **enc_args)
        self.learned_avg_weights.to(device)
        self.learned_avg_weights.eval()

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

    def _get_top_docs(self, query: str, top_ranking: Ranking) -> np.ndarray:
        # Get the ids of the top-ranked documents for the query
        top_docs: pd.DataFrame = top_ranking._df.query("query == @query")
        if len(top_docs) == 0:
            return  # Remains encoded as zeros
        top_docs_ids: Sequence[int] = top_docs["id"].values
        top_docs_scores: Sequence[float] = top_docs["score"].values

        # Get the embeddings of the top-ranked documents
        # TODO: Make sure d_reps is only retrieved once throughout full re-ranking pipeline.
        d_reps, d_idxs = self.index._get_vectors(top_docs_ids)
        if self.index.quantizer is not None:
            d_reps = self.index.quantizer.decode(d_reps)

        # TODO: not just flatten, but use mode (e.g. MaxP). Compare to _compute_scores in index. For non-psg datasets.
        order = [x[0] for x in d_idxs]  # [[0], [2], [1]] --> [0, 2, 1]
        d_reps = d_reps[order]  # sort d_reps on d_ids order

        return d_reps, top_docs_scores

    @property
    def ranking_in(self):
        return self._ranking_in

    @ranking_in.setter
    def ranking_in(self, ranking: Ranking):
        self._ranking_in = ranking.cut(self.k_avg)

    def __call__(self, queries: Sequence[str]) -> np.ndarray:
        """
        Estimate query embeddings by weighted averaging the embeddings of the top-ranked documents.

        Args:
            queries (Sequence[str]): A sequence of query strings.

        Returns:
            np.ndarray: An array of query embeddings.
        """
        assert (
            self.ranking_in is not None
        ), "Please set the ranking_in attribute before calling the encoder."

        q_reps: np.ndarray = np.zeros((len(queries), self.index.dim), dtype=np.float32)
        # TODO: could probably be rewritten to handling a batch.
        for i, query in enumerate(queries):
            d_reps, top_docs_scores = self._get_top_docs(query, self.ranking_in)
            if self.w_method == W_METHOD.LEARNED:
                d_reps = np.expand_dims(d_reps, axis=0)  # Add dim on axis 0
                d_reps = torch.from_numpy(d_reps).float().to(self.device)
                if d_reps is None or len(d_reps[0]) < self.k_avg:
                    continue  # TODO [discuss]: Check if I can create a model which accepts <k_avg docs. Padding?
                q_reps[i] = (
                    self.learned_avg_weights(d_reps)[0].detach().cpu().numpy()
                )  # Batch_size == 1, so take 1st element
            else:
                q_reps[i] = np.average(
                    d_reps,
                    axis=0,
                    weights=self._get_weights(len(d_reps), top_docs_scores),
                )

        return q_reps


class LearnedAvgWeights(lightning.LightningModule):
    """
    Watch this short video on PyTorch for this class to make sense: https://youtu.be/ORMx45xqWkA?si=Bvkm9SWi8Hz1n2Sh&t=147
    """

    def __init__(
        self, k_avg: int = 10, hidden_layers: int = 1, hidden_dimensions: int = 10
    ):
        assert hidden_layers >= 1
        assert hidden_dimensions >= 1
        super().__init__()

        self.k_avg = k_avg
        self.hidden_layers = hidden_layers
        self.hidden_dimensions = hidden_dimensions

        self.loss_fn = torch.nn.MSELoss()
        self.flatten = torch.nn.Flatten(0)
        self.softmax = torch.nn.Softmax(dim=0)

        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(k_avg * 768, hidden_dimensions)
        )
        for l in range(hidden_layers - 1):
            self.linear_relu_stack.extend(
                [torch.nn.ReLU(), torch.nn.Linear(hidden_dimensions, hidden_dimensions)]
            )
        self.linear_relu_stack.extend(
            [
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dimensions, k_avg),
            ]
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear_relu_stack(x)
        x = self.softmax(x)
        return x

    def on_train_start(self):
        self.log_dict(
            {
                "k_avg": self.k_avg,
                "hidden_layers": self.hidden_layers,
                "hidden_dimensions": self.hidden_dimensions,
            }
        )

    def step(self, batch, name):
        x, y = batch
        weights = self(x)  # shape (k_avg)
        weights = weights.unsqueeze(0).unsqueeze(-1)  # Add dims to match x for broadcasting -> shape (:, k_avg, :)
        q_rep = torch.sum(x * weights, dim=1)  # Weighted sum along the second dimension
        loss = self.loss_fn(q_rep, y)
        self.log(f"{name}_loss", loss, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
