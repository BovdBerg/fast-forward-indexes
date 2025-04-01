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
        tok_embs_w_method: str = "WEIGHTED",
        embs_w_method: str = "WEIGHTED",
        ckpt_path: Optional[Path] = None,
        ckpt_path_tok_embs: Optional[Path] = None,
        q_only: bool = False,
        docs_only: bool = False,
        add_special_tokens: bool = True,
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
            ckpt_path_tok_embs (Optional[Path]): Path to a checkpoint to load token embeddings. Overwrites `tok_embs`.
            q_only (bool): Whether to only use the lightweight query estimation and not the top-ranked documents.
            docs_only (bool): Whether to disable the lightweight query estimation and only use the top-ranked documents.
            add_special_tokens (bool): Whether to add special tokens to the queries.
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

        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model)

        model = AutoModel.from_pretrained(self.pretrained_model)
        self.tok_embs = (
            model.get_input_embeddings()
        )  # Embedding(vocab_size, embedding_dim)

        vocab_size = self.tokenizer.vocab_size
        self.tok_embs_weights = torch.nn.Parameter(torch.ones(vocab_size) / vocab_size)

        # TODO [maybe]: Maybe self.embs_avg_weights should have a dimension for n_embs_per_q too? [[1.0], [0.5, 0.5], [0.33, 0.33, 0.33]] or padded [[1.0, 0.0, 0.0], [0.5, 0.5, 0], [0.33, 0.33, 0.33]] etc... up until n_embs
        self._embs_weights = torch.nn.Parameter(torch.ones(self.n_embs) / self.n_embs)

        # TODO [maybe]: add different WEIGHT_METHODs for d_emb weighting (excluding q_light)

        if ckpt_path is not None:
            self.load_checkpoint(ckpt_path)

        if ckpt_path_tok_embs is not None:
            # Load tok_embs checkpoint, use its params, and freeze tok_embs
            ckpt = torch.load(ckpt_path_tok_embs, map_location=device)
            for k, v in ckpt["state_dict"].items():
                if k == "query_encoder.embeddings.weight":
                    self.tok_embs.weight.data.copy_(v)
                    break
            self.tok_embs.requires_grad = False  # IMPORTANT: line only needed for train_avg_emb.py (training tok_embs_weights and embs_weights), NOT FOR dual-encoders!

        self.to(device)

        # Print some information about the model
        # LOGGER.info(f"embs_weights: {self.embs_weights}")
        LOGGER.info(f"embs_avg_weights (softmaxed): {torch.nn.functional.softmax(self._embs_weights, dim=0)}")

    @property
    def ranking(self) -> Optional[Ranking]:
        return self._ranking

    @ranking.setter
    def ranking(self, ranking: Ranking) -> None:
        self._ranking = ranking.cut(self.n_docs)

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
                state_dict["_embs_weights"] = v
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
                "ckpt_path": str(getattr(self, "ckpt_path", None)),
                "tok_embs_w_method": self.tok_embs_w_method.value,
                "embs_w_method": self.embs_w_method.value,
                "add_special_tokens": self.add_special_tokens,
                "q_only": self.q_only,
                "docs_only": self.docs_only,
            }
        )

        print(f"Settings: {settings}")
        with open(self.settings_file, "w") as f:
            json.dump(settings, f, indent=4)

    def __call__(self, queries: Sequence[str]) -> np.ndarray:
        return self.forward(queries).cpu().detach().numpy()

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()
        # LOGGER.info(f"Current self.embs_weights: {self.embs_weights}")
        LOGGER.info(f"embs_avg_weights (softmaxed): {torch.nn.functional.softmax(self._embs_weights, dim=0)}")

    # def _get_top_docs_embs(self, queries: Sequence[str]):
    #     assert self.ranking is not None, "Provide a ranking before encoding."
    #     assert self.index.dim is not None, "Index dimension cannot be None."

    #     # Retrieve the top-ranked documents for all queries
    #     top_docs = self.ranking._df[self.ranking._df["query"].isin(queries)].copy()
    #     top_docs['rank'] = top_docs.groupby('query')['score'].rank(ascending=False, method='first').astype(int) - 1
    #     query_to_idx = {query: idx for idx, query in enumerate(queries)}

    #     # Retrieve any needed embeddings from the index
    #     d_embs, d_idxs = self.index._get_vectors(top_docs["id"].unique())
    #     if self.index.quantizer is not None:
    #         d_embs = self.index.quantizer.decode(d_embs)
    #     d_embs = torch.tensor(d_embs[np.array(d_idxs).flatten()], device=self.device)

    #     # Map doc_ids in top_docs_ids to embeddings
    #     top_docs_embs = torch.zeros((len(queries), self.n_docs, 768), device=self.device)
    #     top_docs_embs[
    #         torch.tensor(top_docs["query"].map(query_to_idx).values, device=self.device),
    #         torch.tensor(top_docs["rank"].values, device=self.device)
    #     ] = d_embs

    #     return top_docs_embs
    def _get_top_docs_embs(self, queries: Sequence[str]):
        assert self.ranking is not None, "Provide a ranking before encoding."
        assert self.index.dim is not None, "Index dimension cannot be None."

        # Create tensors for padding and total embedding counts
        top_docs_embs = torch.zeros((len(queries), self.n_docs, 768), device=self.device)
        q_n_embs = torch.ones((len(queries)), dtype=torch.int, device=self.device)

        # Retrieve the top-ranked documents for all queries
        top_docs = self.ranking._df[self.ranking._df["query"].isin(queries)]
        q_to_idx = {query: idx for idx, query in enumerate(queries)}

        for query, q_top_docs in top_docs.groupby("query"):
            d_embs, d_idxs = self.index._get_vectors(q_top_docs["id"].unique())
            if self.index.quantizer is not None:
                d_embs = self.index.quantizer.decode(d_embs)
            d_embs = torch.tensor(d_embs[np.array(d_idxs).flatten()], device=self.device)

            q_no = q_to_idx[str(query)]
            top_docs_embs[q_no, : len(d_embs)] = d_embs
            q_n_embs[q_no] += len(d_embs)

        return top_docs_embs#, q_n_embs

    def forward(self, queries: Sequence[str]) -> torch.Tensor:
        if self.docs_only:
            q_light = torch.zeros((len(queries), 768), device=self.device)
        else:
            # Tokenizer queries
            q_tokens = self.tokenizer(
                list(queries),
                padding=True,
                return_tensors="pt",
                add_special_tokens=self.add_special_tokens,
            ).to(self.device)
            input_ids = q_tokens["input_ids"].to(self.device)
            attention_mask = q_tokens["attention_mask"].to(self.device)

            # estimate lightweight query as weighted average of q_tok_embs
            q_tok_embs = self.tok_embs(input_ids)
            q_tok_embs = q_tok_embs * attention_mask.unsqueeze(-1)
            # TODO [out of scope]: Probably good use to remove stopwords before averaging.
            match self.tok_embs_w_method:
                case WEIGHT_METHOD.UNIFORM:
                    # q_light = torch.mean(q_tok_embs, 1)
                    n_masked = attention_mask.sum(dim=-1, keepdim=True)
                    q_light = q_tok_embs.sum(dim=-2) / n_masked  # Compute mean, excluding padding
                case WEIGHT_METHOD.WEIGHTED:
                    q_tok_weights = torch.nn.functional.softmax(self.tok_embs_weights[input_ids], -1)
                    q_tok_weights = q_tok_weights * attention_mask  # Mask padding
                    q_tok_weights = q_tok_weights / (q_tok_weights.sum(-1, keepdim=True) + 1e-9)  # Normalize

                    q_light = torch.sum(q_tok_embs * q_tok_weights.unsqueeze(-1), 1)  # Weighted average
        if self.q_only:
            return q_light

        # Find top-ranked document embeddings
        top_docs_embs = self._get_top_docs_embs(queries)

        # Estimate final query as (weighted) average of q_light ++ top_docs_embs
        embs = torch.cat((q_light.unsqueeze(1), top_docs_embs), -2)
        # assign self.embs_avg_weights to embs_avg_weights, but only up to the number of top-ranked documents per query
        # embs_weights = torch.zeros((len(queries), embs.shape[-2]), device=self.device)
        # for q_no, n_embs in enumerate(q_n_embs):
        #     # TODO: what if we use regular normalization to self._embs_weights[:n_embs]? or none at all?
        #     if self.docs_only:
        #         embs_weights[q_no, 0] = 0.0
        #         embs_weights[q_no, 1:n_embs] = torch.nn.functional.softmax(self._embs_weights[1:n_embs], 0)
        #     else:
        #         embs_weights[q_no, :n_embs] = torch.nn.functional.softmax(self._embs_weights[:n_embs], 0)
        embs_weights = torch.zeros((embs.shape[-2]), device=self.device)
        if self.docs_only:
            embs_weights[0] = 0.0
            embs_weights[1:self.n_embs] = torch.nn.functional.softmax(self._embs_weights[1:self.n_embs], 0)
        else:
            # TODO: replace softmax with regular normalization?
            embs_weights[:self.n_embs] = torch.nn.functional.softmax(self._embs_weights[:self.n_embs], 0)
        embs_weights = embs_weights.unsqueeze(0).expand(len(queries), -1)
        embs_weights = embs_weights * (embs.sum(-1) != 0).float()  # Mask padding
        embs_weights = embs_weights / (embs_weights.sum(-1, keepdim=True) + 1e-9)  # Normalize

        q_estimation = torch.sum(embs * embs_weights.unsqueeze(-1), -2)
        return q_estimation
