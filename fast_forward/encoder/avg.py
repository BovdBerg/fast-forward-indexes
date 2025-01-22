import warnings
from enum import Enum
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from fast_forward.encoder import Encoder
from fast_forward.encoder.transformer_embedding import StandaloneEncoder
from fast_forward.index import Index
from fast_forward.lightning import GeneralModule
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


# TODO: Rename to Encoder names used in ppt.
class WeightedAvgEncoder(Encoder):
    """
    WeightedAvgEncoder estimates the query embeddings as the weighted average of the top-ranked document embeddings.
    """

    def __init__(
        self,
        index: Index,
        ckpt_emb_path: Path,
        w_method: W_METHOD = W_METHOD.LEARNED,
        k_avg: int = 30,
        ckpt_path: Optional[Path] = None,
        device: str = "cpu",
    ) -> None:
        """
        Initialize the WeightedAvgEncoder with the given sparse ranking, index, and number of top documents to average.

        Args:
            index (Index): The index containing document embeddings.
            ckpt_emb_path (Path): The path to the checkpoint file to load the embedding encoder.
            w_method (W_METHOD): The probability distribution type used to assign weights to top-ranked documents.
            k_avg (int): The number of top-ranked documents to average.
            ckpt_path (Optional[Path]): The path to the checkpoint file to load.
            device (str): The device to run the encoder on.
        """
        super().__init__()
        self.index = index
        self.w_method = w_method
        self.k_avg = k_avg
        self._ranking = None
        self.device = device

        if w_method == W_METHOD.LEARNED:
            self.n_weights = k_avg + 1  # +1 for q_emb

            if ckpt_path is not None:
                learned_weights_model = LearnedAvgWeights.load_from_checkpoint(
                    ckpt_path, n_weights=self.n_weights
                )
            else:
                learned_weights_model = LearnedAvgWeights(self.n_weights)
            learned_weights_model.to(device)
            learned_weights_model.eval()

            # TODO: Can I get the weights without feeding mock_embeddings?
            mocked_reps = torch.zeros((self.n_weights, 768), device=device)
            self.learned_weights = learned_weights_model(mocked_reps)
            print(f"WeightedAvgEncoder.learned_weights: {self.learned_weights}")

        self.emb_encoder = StandaloneEncoder(
            ckpt_path=ckpt_emb_path,
            device=device,
        )

    def _get_weights(self, reps: torch.Tensor, scores: Sequence[float]) -> torch.Tensor:
        """
        Get the weights for the top-ranked documents based on the probability distribution type or learned weights.

        Args:
            reps (torch.Tensor): The q_emb + document representations.
            scores (Sequence[float]): The scores of the top-ranked documents.

        Returns:
            torch.Tensor: A tensor of interpolation parameters (weights).
        """
        n_docs = len(reps)
        match self.w_method:
            case W_METHOD.LEARNED:
                weights = self.learned_weights[:n_docs]
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
                scores_tensor = torch.tensor(scores, device=self.device)
                max_score = torch.max(scores_tensor)
                exp_scores = torch.exp(scores_tensor - max_score)
                return exp_scores / torch.sum(exp_scores)
            case W_METHOD.LINEAR_DECAY_RANKS:
                return torch.linspace(1, 0, n_docs, device=self.device)
            case W_METHOD.LINEAR_DECAY_SCORES:
                return torch.linspace(1, 0, n_docs, device=self.device) * scores
            case _:
                raise ValueError(
                    f"Unknown probability distribution type: {self.w_method}"
                )

    def _get_top_docs(self, query: str) -> tuple[torch.Tensor, torch.Tensor]:
        assert (
            self.ranking is not None
        ), "Please set the ranking attribute before calling the encoder."

        # Get the ids of the top-ranked documents for the query
        top_docs: pd.DataFrame = self.ranking._df.query("query == @query")
        if len(top_docs) == 0:
            return  # Remains encoded as zeros # type: ignore
        top_docs_ids: Sequence[str] = top_docs["id"].astype(str).values.tolist()
        top_docs_scores: Sequence[float] = top_docs["score"].values.tolist()

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
    def ranking(self) -> Optional[Ranking]:
        return self._ranking

    @ranking.setter
    def ranking(self, ranking: Ranking):
        self._ranking = ranking.cut(self.k_avg)

    def __call__(self, queries: Sequence[str]) -> np.ndarray:
        """
        Estimate query embeddings by weighted averaging the embeddings of the top-ranked documents.

        Args:
            queries (Sequence[str]): A sequence of query strings.

        Returns:
            np.ndarray: An array of query embeddings.
        """
        assert self.index.dim is not None, "Index dimension cannot be None"

        # TODO [important]: handle in batches.
        q_reps = torch.zeros((len(queries), self.index.dim), device=self.device)
        for i, query in enumerate(queries):
            reps, top_docs_scores = self._get_top_docs(query)
            if reps is None:
                continue

            if self.w_method == W_METHOD.LEARNED:
                q_emb = torch.tensor(
                    self.emb_encoder([query])[0], device=self.device
                ).unsqueeze(0)
                reps = torch.cat((q_emb, reps), dim=0)

            # Get the weights for the top docs using the selected method
            weights = self._get_weights(
                reps, top_docs_scores.clone().detach().tolist()
            ).unsqueeze(-1)

            # Calculate the weighted sum of document representations
            weighted_sum = torch.sum(reps * weights, dim=0)
            total_weight = torch.sum(weights)
            q_reps[i] = weighted_sum / total_weight

        return q_reps.cpu().detach().numpy()


# TODO: Should be similar to BERT Embedding layer, but Should have q_emb and d_reps as input.
# BertEmbeddings(
#   (word_embeddings): Embedding(30522, 768, padding_idx=0) <-- Embedding()
#   (position_embeddings): Embedding(512, 768)
#   (token_type_embeddings): Embedding(2, 768)
#   (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
#   (dropout): Dropout(p=0.1, inplace=False)
# )
class LearnedAvgWeights(GeneralModule):
    def __init__(self, n_weights: int = 10):
        super().__init__()
        # TODO: Experiment with different encoders, e.g. dropout, normalization, attention, activation functions, etc.

        self.n_weights = n_weights

        self.flatten = torch.nn.Flatten(0)

        hidden_dim = 100
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(n_weights * 768, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, n_weights),
        )

    def forward(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        x = self.flatten(x)
        try:
            x = self.linear_relu_stack(x)
        except Exception as e:
            print(f"Batch skipped with exception: {e}")
            return None
        return x

    def step(  # type: ignore
        self, batch: tuple[torch.Tensor, torch.Tensor], name: str
    ) -> Optional[torch.Tensor]:
        x, y = batch
        weights = self(x)  # shape (k_avg)
        if weights is None:  # Skip batch
            return None
        weights = weights.unsqueeze(0).unsqueeze(-1)
        q_rep = torch.sum(x * weights, dim=1)  # Weighted sum along the second dimension
        loss = self.loss_fn(q_rep, y)
        self.log(f"{name}_loss", loss)
        return loss


class AvgEmbQueryEstimator(Encoder, GeneralModule):
    def __init__(
        self,
        index: Index,
        n_docs: int,
        device: str,
        ranking: Optional[Ranking] = None,
        ckpt_path: Optional[Path] = None,
    ) -> None:
        """
        Estimate query embeddings as the weighted average of:
        - its top-ranked document embeddings.
        - lightweight semantic query estimation.
            - based on the weighted average of query's (fine-tuned) token embeddings.

        Note that the optimal values for these values are learned during fine-tuning:
        - `self.tok_embs`: the token embeddings
        - `self.tok_embs_avg_weights`: token embedding weighted averages
        - `self.embs_avg_weights`: embedding weighted averages

        Args:
            index (Index): The index containing document embeddings.
            n_docs (int): The number of top-ranked documents to average.
            device (str): The device to run the encoder on.
            ranking (Optional[Ranking]): The ranking to use for the top-ranked documents.
        """
        super().__init__()
        self.index = index
        self._ranking = ranking
        self.n_docs = n_docs

        doc_encoder_pretrained = "bert-base-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(doc_encoder_pretrained)

        doc_encoder = AutoModel.from_pretrained(doc_encoder_pretrained)
        self.tok_embs = (
            doc_encoder.get_input_embeddings()
        )  # Maps token_id --> embedding, Embedding(vocab_size, embedding_dim)

        self.tok_embs_avg_weights = torch.nn.Parameter(
            torch.randn(self.tokenizer.vocab_size)
        )  # weights for averaging over q's token embedding, shape (vocab_size,)

        n_embs = n_docs + 1
        self.embs_avg_weights = torch.nn.Parameter(
            torch.randn(n_embs)
        )  # weights for averaging over q_emb1 ++ d_embs, shape (n_embs,)

        # TODO: add different w_methods

        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path)
            self.load_state_dict(ckpt["state_dict"])

        self.to(device)
        self.eval()

    @property
    def ranking(self) -> Optional[Ranking]:
        return self._ranking

    @ranking.setter
    def ranking(self, ranking: Ranking):
        self._ranking = ranking.cut(self.n_docs)

    def _get_top_docs(self, queries: Sequence[str]):
        assert self.ranking is not None, "Provide a ranking before encoding."
        assert self.index.dim is not None, "Index dimension cannot be None."

        # Create a tensor of zeros with shape (len(queries), n_docs, 768)
        d_embs_pad = torch.zeros(
            (len(queries), self.n_docs, 768), device=self.device
        )
        n_embs_per_q = torch.ones((len(queries)), dtype=torch.int, device=self.device)

        # Retrieve the top-ranked documents for each query and increment n_docs_per_q
        top_docs = self.ranking._df[self.ranking._df["query"].isin(queries)]
        for i, query in enumerate(queries):  # TODO: can this be done without for-loop?
            query_docs = top_docs[top_docs["query"] == query]
            if len(query_docs) == 0:
                continue

            d_embs, d_idxs = self.index._get_vectors(query_docs["id"].unique())
            if self.index.quantizer is not None:
                d_embs = self.index.quantizer.decode(d_embs)
            d_embs = torch.tensor(d_embs[[x[0] for x in d_idxs]], device=self.device)

            d_embs_pad[i, : len(d_embs)] = d_embs
            n_embs_per_q[i] += len(d_embs)

        return d_embs_pad, n_embs_per_q

    def forward(self, queries: Sequence[str]) -> torch.Tensor:
        # Tokenizer queries using the doc_encoder_pretrained tokenizer
        q_tokens = self.tokenizer(list(queries), return_tensors="pt", padding=True)
        input_ids = q_tokens["input_ids"].to(self.device)

        # lookup q_tokens in self.tok_embs
        q_tok_embs = self.tok_embs(q_tokens["input_ids"].to(self.device))

        # estimate lightweight query as weighted average of q_tok_embs
        # TODO: verify that padding is masked properly before softmax and averaging
        # TODO: try excluding [CLS] and [SEP] token embeddings too (when tokenizing), might improve fine-tuning for query tokens
        q_tok_weights = self.tok_embs_avg_weights[input_ids]  # get weights for q tokens
        q_tok_weights = torch.nn.functional.softmax(q_tok_weights, dim=-1).unsqueeze(
            -1
        )  # softmax over weights
        q_emb_1 = torch.sum(q_tok_embs * q_tok_weights, dim=1).unsqueeze(
            1
        )  # weighted average

        # lookup embeddings of top-ranked documents in (in-memory) self.index
        d_embs_pad, n_embs_per_q = self._get_top_docs(queries)

        # estimate query embedding as weighted average of q_emb and d_embs
        embs = torch.cat((q_emb_1, d_embs_pad), dim=-2)

        # initialize embs_avg_weights as zeros with shape (len(queries), n_embs)
        embs_weights = torch.zeros((len(queries), embs.shape[-2]), device=self.device)
        # assign self.embs_avg_weights to embs_avg_weights, but only up to the number of top-ranked documents per query
        for i, n_embs in enumerate(
            n_embs_per_q
        ):  # TODO: can this be done without for-loop?
            embs_weights[i, :n_embs] = torch.nn.functional.softmax(
                self.embs_avg_weights[:n_embs], dim=0
            )

        q_emb_2 = torch.sum(embs * embs_weights.unsqueeze(-1), dim=-2)
        return q_emb_2

    def __call__(self, queries: Sequence[str]) -> torch.Tensor:
        return self.forward(queries)
