import json
import logging
import warnings
from enum import Enum
from pathlib import Path
from time import perf_counter
from typing import Optional, Sequence

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from fast_forward.encoder import Encoder
from fast_forward.index import Index
from fast_forward.lightning import GeneralModule
from fast_forward.ranking import Ranking

warnings.filterwarnings("ignore", message="`training_step` returned `None`.*")

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class WEIGHT_METHOD(Enum):
    """
    Enumeration for different types of probability distributions used to assign weights to tokens in the WeightedAvgEncoder.

    Attributes:
        UNIFORM: all tokens are weighted equally.
        WEIGHTED: weights are learned during training.
    """

    UNIFORM = "UNIFORM"
    WEIGHTED = "WEIGHTED"


class AvgEmbQueryEstimator(Encoder, GeneralModule):
    def __init__(
        self,
        index: Index,
        n_docs: int,
        device: str,
        ranking: Optional[Ranking] = None,
        ckpt_path: Optional[Path] = None,
        tok_embs_w_method: str = "WEIGHTED",
        embs_w_method: str = "WEIGHTED",
        q_only: bool = False,
        docs_only: bool = False,
        add_special_tokens: bool = False,
        profiling: bool = False,
    ) -> None:
        """
        Estimate query embeddings as the weighted average of:
        - lightweight semantic query estimation.
            - based on the weighted average of query's (fine-tuned) token embeddings.
        - its top-ranked document embeddings.

        Note that the optimal values for these values are learned during fine-tuning:
        - `self.tok_embs`: the token embeddings
        - `self.tok_embs_weights`: token embedding weighted averages
        - `self.embs_weights`: embedding weighted averages

        Args:
            index (Index): The index containing document embeddings.
            n_docs (int): The number of top-ranked documents to average.
            device (str): The device to run the encoder on.
            ranking (Optional[Ranking]): The ranking to use for the top-ranked documents.
            ckpt_path (Optional[Path]): Path to a checkpoint to load.
            tok_embs_w_method (str): The WEIGHT_METHOD name to use for token embedding weighting.
            embs_w_method (str): The WEIGHT_METHOD name to use for embedding weighting.
            q_only (bool): Whether to only use the lightweight query estimation and not the top-ranked documents.
            docs_only (bool): Whether to disable the lightweight query estimation and only use the top-ranked documents.
            add_special_tokens (bool): Whether to add special tokens to the queries.
            profiling (bool): Whether to log profiling information.
        """
        assert not (q_only and docs_only), "Cannot use both q_only and docs_only."

        super().__init__()
        self.index = index
        self._ranking = ranking
        self.n_docs = n_docs
        self.n_embs = n_docs + 1
        self.pretrained_model = "bert-base-uncased"
        self.add_special_tokens = add_special_tokens
        self.tok_embs_w_method = WEIGHT_METHOD(tok_embs_w_method)
        self.embs_w_method = WEIGHT_METHOD(embs_w_method)
        self.q_only = q_only
        self.docs_only = docs_only
        self.profiling = profiling

        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model)

        model = AutoModel.from_pretrained(self.pretrained_model)
        self.tok_embs = (
            model.get_input_embeddings()
        )  # Embedding(vocab_size, embedding_dim)

        vocab_size = self.tokenizer.vocab_size
        self.tok_embs_weights = torch.nn.Parameter(torch.randn(vocab_size) * 0.01)

        # TODO [maybe]: Maybe self.embs_avg_weights should have a dimension for n_embs_per_q too? [[1.0], [0.5, 0.5], [0.33, 0.33, 0.33]] or padded [[1.0, 0.0, 0.0], [0.5, 0.5, 0], [0.33, 0.33, 0.33]] etc... up until n_embs
        self.embs_weights = torch.nn.Parameter(torch.randn(vocab_size) * 0.01)

        # TODO [maybe]: add different WEIGHT_METHODs for d_emb weighting (excluding q_emb_1)

        if ckpt_path is not None:
            self.load_checkpoint(ckpt_path)

        self.to(device)
        self.eval()

        # Print some information about the model
        LOGGER.info(f"embs_weights: {self.embs_weights}")
        LOGGER.info(f"parameters: {self.parameters()}")

    def load_checkpoint(self, ckpt_path: Path) -> None:
        self.ckpt_path = ckpt_path
        ckpt = torch.load(ckpt_path, map_location=self.device)
        state_dict = {}
        for k, v in ckpt["state_dict"].items():
            key = k.replace("query_encoder.", "")
            if key == "embeddings.weight":
                self.tok_embs.weight.data.copy_(v)
                return
            elif key == "tok_embs_avg_weights":
                state_dict["tok_embs_weights"] = v
            elif key == "embs_avg_weights":
                state_dict["embs_weights"] = v
            elif key in self.state_dict():
                state_dict[key] = v
        self.load_state_dict(state_dict)

    def on_train_start(self) -> None:
        super().on_train_start()
        self.train()

        with open(self.settings_file, "r") as f:
            settings = json.load(f)

        settings.update(
            {
                "n_docs": self.n_docs,
                "device": self.device.type,
                "ckpt_path": getattr(self, "ckpt_path", None),
                "tok_embs_w_method": self.tok_embs_w_method.value,
                "embs_w_method": self.embs_w_method.value,
                "add_special_tokens": self.add_special_tokens,
                "q_only": self.q_only,
                "docs_only": self.docs_only,
            }
        )

        with open(self.settings_file, "w") as f:
            json.dump(settings, f, indent=4)

    @property
    def ranking(self) -> Optional[Ranking]:
        return self._ranking

    @ranking.setter
    def ranking(self, ranking: Ranking) -> None:
        self._ranking = ranking.cut(self.n_docs)

    def compute_weighted_average(
        self,
        embs: torch.Tensor,
        init_weights: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        weights = init_weights * mask  # Mask padding
        weights = torch.nn.functional.softmax(weights, dim=-1)  # Positive and sum to 1

        embs = embs * weights.unsqueeze(-1)  # Apply weights
        q_estimation = embs.sum(-2)  # Compute weighted sum
        return q_estimation

    def _get_top_docs_embs(self, queries: Sequence[str]) -> torch.Tensor:
        t0 = perf_counter()
        assert self.ranking is not None, "Provide a ranking before encoding."
        assert self.index.dim is not None, "Index dimension cannot be None."

        # Retrieve the top-ranked documents for all queries
        top_docs = self.ranking._df[self.ranking._df["query"].isin(queries)].copy()
        t1 = perf_counter()
        if self.profiling:
            LOGGER.info(f"1 (top_docs) ranking lookup took: {t1 - t0:.5f}s")

        # Retrieve any needed embeddings from the index
        d_embs, d_idxs = self.index._get_vectors(top_docs["id"])
        if self.index.quantizer is not None:
            d_embs = self.index.quantizer.decode(d_embs)
        d_embs = torch.tensor(
            d_embs[np.array(d_idxs)[:, 0].tolist()], device=self.device
        )
        t2 = perf_counter()
        if self.profiling:
            LOGGER.info(f"2 (d_embs) lookup took: {t2 - t1:.5f}s")

        # Map doc_ids to embeddings
        top_docs_embs = torch.zeros(
            (len(queries), self.n_docs, 768), device=self.device
        )
        q_groups = top_docs.groupby("query")
        q_nos = torch.tensor(q_groups.ngroup().values, device=self.device)
        d_ranks = torch.tensor(q_groups.cumcount().to_numpy(), device=self.device)
        top_docs_embs[q_nos, d_ranks] = d_embs
        t3 = perf_counter()
        if self.profiling:
            LOGGER.info(f"3 (top_docs_embs) mapping took: {t3 - t2:.5f}s")

        return top_docs_embs

    def forward(self, queries: Sequence[str]) -> torch.Tensor:
        t0 = perf_counter()
        batch_size = len(queries)

        if self.docs_only:
            q_emb_1 = torch.zeros((len(queries), 768), device=self.device)
        else:
            # Estimate lightweight query as (weighted) average of q_tok_embs, excluding padding
            q_tokens = self.tokenizer(
                list(queries),
                padding=True,
                return_tensors="pt",
                add_special_tokens=self.add_special_tokens,
            ).to(self.device)
            input_ids = q_tokens["input_ids"].to(self.device)
            max_len = input_ids.size(1)

            q_tok_embs = self.tok_embs(input_ids)
            match self.tok_embs_w_method:
                case WEIGHT_METHOD.UNIFORM:
                    q_tok_weights = (
                        torch.ones_like(input_ids, dtype=torch.float) / max_len
                    )
                case WEIGHT_METHOD.WEIGHTED:
                    q_tok_weights = self.tok_embs_weights[input_ids]
            q_tok_mask = q_tokens["attention_mask"].to(self.device)
            q_emb_1 = self.compute_weighted_average(
                q_tok_embs, q_tok_weights, q_tok_mask
            )
        t1 = perf_counter()
        if self.profiling:
            LOGGER.info(f"Lightweight query estimation (q_emb_1) took: {t1 - t0:.5f}s")

        if self.q_only:
            return q_emb_1

        # Find top-ranked document embeddings
        top_docs_embs = self._get_top_docs_embs(queries)
        t2 = perf_counter()
        if self.profiling:
            LOGGER.info(
                f"Lookup of top-ranked documents (_get_top_docs) took: {t2 - t1:.5f}s"
            )

        # Estimate final query as (weighted) average of q_emb_1 ++ top_docs_embs
        embs = torch.cat((q_emb_1.unsqueeze(1), top_docs_embs), -2)
        match self.embs_w_method:  # embs_weights: (batch_size, n_embs)
            case WEIGHT_METHOD.UNIFORM:
                embs_weights = (
                    torch.ones((batch_size, self.n_embs), device=self.device)
                    / self.n_embs
                )
            case WEIGHT_METHOD.WEIGHTED:
                embs_weights = self.embs_weights.unsqueeze(0).expand(
                    batch_size, -1
                )  # (batch_size, n_embs), repeated values
        if self.docs_only:
            embs_weights[0] = 0.0
        embs_mask = torch.ones(
            (batch_size, self.n_embs), device=self.device
        )  # (batch_size, n_embs), 1 for each non-zero emb
        embs_mask[:, 1:] = torch.any(
            top_docs_embs != 0, dim=-1
        )  # Set empty doc embs to 0
        q_emb_2 = self.compute_weighted_average(embs, embs_weights, embs_mask)
        t3 = perf_counter()
        if self.profiling:
            LOGGER.info(f"Query embedding estimation (q_emb_2) took: {t3 - t2:.5f}s")

        return q_emb_2

    def __call__(self, queries: Sequence[str]) -> np.ndarray:
        return self.forward(queries).cpu().detach().numpy()

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()
        LOGGER.info(f"Current self.embs_weights: {self.embs_weights}")
