import json
import warnings
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

warnings.filterwarnings("ignore", message="`training_step` returned `None`.*")


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


# TODO: Train model with query --> TCT-ColBERT query (KD) with custom shape
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

        if w_method == W_METHOD.LEARNED:
            if ckpt_path is not None:
                self.learned_avg_weights = LearnedAvgWeights.load_from_checkpoint(
                    ckpt_path, k_avg=k_avg, **enc_args
                )
            else:
                self.learned_avg_weights = LearnedAvgWeights(
                    k_avg=self.k_avg, **enc_args
                )
            self.learned_avg_weights.to(device)
            self.learned_avg_weights.eval()

    def _get_weights(
        self, d_reps: torch.Tensor, scores: Sequence[float]
    ) -> torch.Tensor:
        """
        Get the weights for the top-ranked documents based on the probability distribution type or learned weights.

        Args:
            d_reps (torch.Tensor): The document representations.
            scores (Sequence[float]): The scores of the top-ranked documents.

        Returns:
            torch.Tensor: A tensor of interpolation parameters (weights).
        """
        n_docs = len(d_reps)
        match self.w_method:
            case W_METHOD.LEARNED:
                padding = max(0, self.k_avg - len(d_reps))
                d_reps_pad = torch.cat(
                    (d_reps, torch.zeros(padding, self.index.dim, device=self.device))
                )
                weights = self.learned_avg_weights(d_reps_pad)[:n_docs]
                return torch.nn.functional.softmax(weights, dim=0)
            case W_METHOD.UNIFORM:
                return torch.ones(n_docs, device=self.device) / n_docs
            case W_METHOD.EXPONENTIAL:
                return torch.exp(-torch.linspace(0, 1, n_docs, device=self.device))
            case W_METHOD.HALF_NORMAL:
                return torch.flip(
                    torch.exp(-torch.linspace(0, 1, n_docs, device=self.device) ** 2),
                    dims=[0],
                )
            case W_METHOD.SOFTMAX_SCORES:
                max_score = torch.max(scores)
                exp_scores = torch.exp(scores - max_score)
                return exp_scores / torch.sum(exp_scores)
            case W_METHOD.LINEAR_DECAY_RANKS:
                return torch.linspace(1, 0, n_docs, device=self.device)
            case W_METHOD.LINEAR_DECAY_SCORES:
                return torch.linspace(1, 0, n_docs, device=self.device) * scores
            case _:
                raise ValueError(
                    f"Unknown probability distribution type: {self.w_method}"
                )

    def _get_top_docs(self, query: str, top_ranking: Ranking) -> torch.Tensor:
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

        # Convert d_reps and top_docs_scores to tensors
        d_reps_tensor = torch.tensor(d_reps, device=self.device)
        top_docs_scores_tensor = torch.tensor(top_docs_scores, device=self.device)

        return d_reps_tensor, top_docs_scores_tensor

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

        # TODO: could probably be rewritten to handle batches at a time.
        q_reps = torch.zeros((len(queries), self.index.dim), device=self.device)
        for i, query in enumerate(queries):
            d_reps, top_docs_scores = self._get_top_docs(query, self.ranking_in)
            if d_reps is None:
                continue
            # Get the weights for the top docs using the selected method
            weights = self._get_weights(
                d_reps, top_docs_scores.clone().detach()
            ).unsqueeze(-1)

            # Calculate the weighted sum of document representations
            weighted_sum = torch.sum(d_reps * weights, dim=0)
            total_weight = torch.sum(weights)
            q_reps[i] = weighted_sum / total_weight

        return q_reps.cpu().detach().numpy()


class LearnedAvgWeights(lightning.LightningModule):
    """
    Watch this short video on PyTorch for this class to make sense: https://youtu.be/ORMx45xqWkA?si=Bvkm9SWi8Hz1n2Sh&t=147
    """

    def __init__(self, k_avg: int = 10):
        super().__init__()

        self.k_avg = k_avg

        self.loss_fn = torch.nn.MSELoss()
        self.flatten = torch.nn.Flatten(0)
        self.softmax = torch.nn.Softmax(dim=0)

        hidden_dim = 100
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(k_avg * 768, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, k_avg),
        )

    def forward(self, x):
        x = self.flatten(x)
        try:
            x = self.linear_relu_stack(x)
        except Exception as e:
            print(f"Batch skipped with exception: {e}")
            return None
        x = self.softmax(x)
        return x

    def on_train_start(self):
        settings_file = Path(self.trainer.log_dir) / "settings.json"
        with open(settings_file, "w") as f:
            json.dump(
                {
                    "Class": self.__class__.__name__,
                    "k_avg": self.k_avg,
                },
                f,
                indent=4,
            )

    def step(self, batch, name):
        x, y = batch
        weights = self(x)  # shape (k_avg)
        if weights is None:  # Skip batch
            return None
        weights = weights.unsqueeze(0).unsqueeze(-1)
        q_rep = torch.sum(x * weights, dim=1)  # Weighted sum along the second dimension
        loss = self.loss_fn(q_rep, y)
        self.log(f"{name}_loss", loss)
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
