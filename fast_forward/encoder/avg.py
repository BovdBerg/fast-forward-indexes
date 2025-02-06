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
        LEARNED: weights are learned during training.
    """

    UNIFORM = "UNIFORM"
    LEARNED = "LEARNED"


class AvgEmbQueryEstimator(Encoder, GeneralModule):
    def __init__(
        self,
        index: Index,
        n_docs: int,
        device: str,
        ranking: Optional[Ranking] = None,
        ckpt_path: Optional[Path] = None,
        tok_w_method: str = "LEARNED",
        q_only: bool = False,
        docs_only: bool = False,
        add_special_tokens: bool = True,  # TODO: might make sense to disable special tokens, e.g. [CLS] will learn a generic embedding
        normalize_q_emb_1: bool = False,
        normalize_q_emb_2: bool = False,
        profiling: bool = False,
    ) -> None:
        """
        Estimate query embeddings as the weighted average of:
        - lightweight semantic query estimation.
            - based on the weighted average of query's (fine-tuned) token embeddings.
        - its top-ranked document embeddings.

        Note that the optimal values for these values are learned during fine-tuning:
        - `self.tok_embs`: the token embeddings
        - `self.tok_embs_avg_weights`: token embedding weighted averages
        - `self.embs_avg_weights`: embedding weighted averages

        Args:
            index (Index): The index containing document embeddings.
            n_docs (int): The number of top-ranked documents to average.
            device (str): The device to run the encoder on.
            ranking (Optional[Ranking]): The ranking to use for the top-ranked documents.
            ckpt_path (Optional[Path]): Path to a checkpoint to load.
            tok_w_method (str): The WEIGHT_METHOD name to use for token weighting.
            q_only (bool): Whether to only use the lightweight query estimation and not the top-ranked documents.
            docs_only (bool): Whether to disable the lightweight query estimation and only use the top-ranked documents.
            add_special_tokens (bool): Whether to add special tokens to the queries.
            profiling (bool): Whether to log profiling information.
            normalize_q_emb_1 (bool): Whether to normalize the lightweight query estimation.
            normalize_q_emb_2 (bool): Whether to normalize the final query embedding.
        """
        assert not (q_only and docs_only), "Cannot use both q_only and docs_only."

        super().__init__()
        self.index = index
        self._ranking = ranking
        self.n_docs = n_docs
        self.n_embs = n_docs + 1
        self.pretrained_model = "bert-base-uncased"
        self.add_special_tokens = add_special_tokens
        self.tok_w_method = WEIGHT_METHOD(tok_w_method)
        self.q_only = q_only
        self.docs_only = docs_only
        self.normalize_q_emb_1 = normalize_q_emb_1
        self.normalize_q_emb_2 = normalize_q_emb_2
        self.profiling = profiling

        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model)

        model = AutoModel.from_pretrained(self.pretrained_model)
        self.tok_embs = model.get_input_embeddings()  # Embedding(vocab_size, embedding_dim)

        vocab_size = self.tokenizer.vocab_size
        self.tok_embs_avg_weights = torch.nn.Parameter(torch.ones(vocab_size) / vocab_size)

        # TODO [maybe]: Maybe self.embs_avg_weights should have a dimension for n_embs_per_q too? [[1.0], [0.5, 0.5], [0.33, 0.33, 0.33]] or padded [[1.0, 0.0, 0.0], [0.5, 0.5, 0], [0.33, 0.33, 0.33]] etc... up until n_embs
        self.embs_avg_weights = torch.nn.Parameter(torch.ones(self.n_embs) / self.n_embs)

        # TODO [maybe]: add different WEIGHT_METHODs for d_emb weighting (excluding q_emb_1)

        if ckpt_path is not None:
            self.load_checkpoint(ckpt_path)

        self.to(device)
        self.eval()

        # Print some information about the model
        embs_weights = torch.nn.functional.softmax(self.embs_avg_weights, dim=0)
        print(f"AvgEmbQueryEstimator.embs_weights (softmaxed): {embs_weights}")

    def load_checkpoint(self, ckpt_path: Path) -> None:
        self.ckpt_path = ckpt_path
        ckpt = torch.load(ckpt_path, map_location=self.device)
        state_dict = {}
        for k, v in ckpt["state_dict"].items():
            key = k.replace("query_encoder.", "")
            if key == "embeddings.weight":
                self.tok_embs.weight.data.copy_(v)
                return
            elif key in self.state_dict():
                state_dict[key] = v
        self.load_state_dict(state_dict)

    def on_train_start(self) -> None:
        super().on_train_start()
        with open(self.settings_file, "r") as f:
            settings = json.load(f)

        settings.update(
            {
                "n_docs": self.n_docs,
                "device": self.device.type,
                "ckpt_path": getattr(self, "ckpt_path", None),
                "tok_w_method": self.tok_w_method.value,
                "add_special_tokens": self.add_special_tokens,
                "q_only": self.q_only,
                "docs_only": self.docs_only,
                "normalize_q_emb_1": self.normalize_q_emb_1,
                "normalize_q_emb_2": self.normalize_q_emb_2,
            }
        )

        with open(self.settings_file, "w") as f:
            json.dump(settings, f, indent=4)

    @property
    def ranking(self) -> Optional[Ranking]:
        return self._ranking

    @ranking.setter
    def ranking(self, ranking: Ranking):
        self._ranking = ranking.cut(self.n_docs)

    def _get_top_docs_embs(self, queries: Sequence[str]):
        t0 = perf_counter()
        assert self.ranking is not None, "Provide a ranking before encoding."
        assert self.index.dim is not None, "Index dimension cannot be None."

        # Retrieve the top-ranked documents for all queries
        top_docs = self.ranking._df[self.ranking._df["query"].isin(queries)].copy()
        t1 = perf_counter()
        if self.profiling:
            LOGGER.info(f"1 (top_docs) ranking lookup took: {t1 - t0:.5f}s")

        # Retrieve any needed embeddings from the index
        top_embs, d_idxs = self.index._get_vectors(top_docs["id"].unique())
        if self.index.quantizer is not None:
            top_embs = self.index.quantizer.decode(top_embs)
        top_embs = torch.tensor(
            top_embs[np.array(d_idxs)[:, 0].tolist()], device=self.device
        )
        t2 = perf_counter()
        if self.profiling:
            LOGGER.info(f"4 (top_embs) lookup took: {t2 - t1:.5f}s")

        # Map doc_ids to embeddings
        d_embs = torch.zeros((len(queries), self.n_docs, 768), device=self.device)
        q_groups = top_docs.groupby("query")
        q_nos = torch.tensor(q_groups.ngroup().values, device=self.device)
        d_ranks = torch.tensor(q_groups.cumcount().to_numpy(), device=self.device)
        d_embs[q_nos, d_ranks] = top_embs

        # replace zeros in d_embs with emb at rank 0 (if n_top_docs was < self.n_docs for any queries)
        d_embs[d_embs == 0] = d_embs[:, 0].unsqueeze(1).expand_as(d_embs)[d_embs == 0]
        t3 = perf_counter()
        if self.profiling:
            LOGGER.info(f"5 (d_embs) mapping took: {t3 - t2:.5f}s")

        return d_embs

    def forward(self, queries: Sequence[str]) -> torch.Tensor:
        t0 = perf_counter()

        if self.docs_only:
            q_emb_1 = torch.zeros((len(queries), 768), device=self.device)
        else:
            # Estimate lightweight query as (weighted) average of q_tok_embs, excluding padding
            q_tokens = self.tokenizer(
                list(queries),
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
                add_special_tokens=self.add_special_tokens,
            ).to(self.device)
            input_ids = q_tokens["input_ids"].to(self.device)
            attention_mask = q_tokens["attention_mask"]

            q_tok_embs = self.tok_embs(input_ids)
            q_tok_embs = q_tok_embs * attention_mask.unsqueeze(-1)  # Mask padding tokens

            # Compute the mean of the masked embeddings, excluding padding
            match self.tok_w_method:
                case WEIGHT_METHOD.UNIFORM:
                    n_unmasked = attention_mask.sum(dim=1, keepdim=True)
                    q_emb_1 = q_tok_embs.sum(dim=1) / n_unmasked
                case WEIGHT_METHOD.LEARNED:
                    q_tok_weights = torch.nn.functional.softmax(
                        self.tok_embs_avg_weights[input_ids], -1
                    )
                    q_tok_weights = q_tok_weights * attention_mask  # Mask padding weights
                    q_tok_weights = q_tok_weights / q_tok_weights.sum(dim=1, keepdim=True)  # Normalize to sum to 1
                    q_emb_1 = torch.sum(q_tok_embs * q_tok_weights.unsqueeze(-1), 1)
            if self.normalize_q_emb_1:
                q_emb_1 = torch.nn.functional.normalize(q_emb_1)

        t1 = perf_counter()
        if self.profiling:
            LOGGER.info(f"Lightweight query estimation (q_emb_1) took: {t1 - t0:.5f}s")

        if self.q_only:
            return q_emb_1

        # Find top-ranked document embeddings
        d_embs = self._get_top_docs_embs(queries)
        t2 = perf_counter()
        if self.profiling:
            LOGGER.info(
                f"Lookup of top-ranked documents (_get_top_docs) took: {t2 - t1:.5f}s"
            )

        # Estimate final query embedding as (weighted) average of q_emb ++ d_embs
        embs = torch.cat((q_emb_1.unsqueeze(1), d_embs), -2)
        embs_weights = torch.zeros((self.n_embs), device=self.device)
        if self.docs_only:
            embs_weights[0] = 0.0
            embs_weights[1 : self.n_embs] = torch.nn.functional.softmax(
                self.embs_avg_weights[1 : self.n_embs], 0
            )
        else:
            embs_weights[: self.n_embs] = torch.nn.functional.softmax(
                self.embs_avg_weights[: self.n_embs], 0
            )
        embs_weights = embs_weights.unsqueeze(0).expand(len(queries), -1)

        q_emb_2 = torch.sum(embs * embs_weights.unsqueeze(-1), -2)
        if self.normalize_q_emb_2:
            q_emb_2 = torch.nn.functional.normalize(q_emb_2)

        t3 = perf_counter()
        if self.profiling:
            LOGGER.info(f"Query embedding estimation (q_emb_2) took: {t3 - t2:.5f}s")

        return q_emb_2

    def __call__(self, queries: Sequence[str]) -> np.ndarray:
        return self.forward(queries).cpu().detach().numpy()
