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
        docs_only: bool = False,
        q_only: bool = False,
        add_special_tokens: bool = True,
        normalize_q_emb_1: bool = False,
        normalize_q_emb_2: bool = False,
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
            docs_only (bool): Whether to disable the lightweight query estimation and only use the top-ranked documents.
            q_only (bool): Whether to only use the lightweight query estimation and not the top-ranked documents.
            add_special_tokens (bool): Whether to add special tokens to the queries.
            normalize_q_emb_1 (bool): Whether to normalize the lightweight query estimation.
            normalize_q_emb_2 (bool): Whether to normalize the final query embedding.
        """
        super().__init__()
        self.index = index
        self._ranking = ranking
        self.n_docs = n_docs
        self.n_embs = n_docs + 1
        self.add_special_tokens = add_special_tokens
        self.tok_w_method = WEIGHT_METHOD(tok_w_method)
        self.docs_only = docs_only
        self.q_only = q_only
        self.normalize_q_emb_1 = normalize_q_emb_1
        self.normalize_q_emb_2 = normalize_q_emb_2
        self.rank_scores = np.zeros(n_docs)

        doc_encoder_pretrained = "castorini/tct_colbert-msmarco"
        self.tokenizer = AutoTokenizer.from_pretrained(doc_encoder_pretrained)
        vocab_size = self.tokenizer.vocab_size

        doc_encoder = AutoModel.from_pretrained(doc_encoder_pretrained)
        self.tok_embs = (
            doc_encoder.get_input_embeddings()
        )  # Maps token_id --> embedding, Embedding(vocab_size, embedding_dim)

        self.tok_embs_avg_weights = torch.nn.Parameter(
            torch.ones(vocab_size) / vocab_size
        )  # weights for averaging over q's token embedding, shape (vocab_size,)

        # TODO [maybe]: Maybe self.embs_avg_weights should have a dimension for n_embs_per_q too? [[1.0], [0.5, 0.5], [0.33, 0.33, 0.33]] or padded [[1.0, 0.0, 0.0], [0.5, 0.5, 0], [0.33, 0.33, 0.33]] etc... up until n_embs
        self.embs_avg_weights = torch.nn.Parameter(
            torch.ones(self.n_embs) / self.n_embs
        )  # weights for averaging over q_emb1 ++ d_embs, shape (n_embs,)

        # TODO [maybe]: add different WEIGHT_METHODs for d_emb weighting (excluding q_emb_1)

        if ckpt_path is not None:
            self.ckpt_path = ckpt_path
            ckpt = torch.load(ckpt_path, map_location=device)
            state_dict = {k: v for k, v in ckpt["state_dict"].items() if k in self.state_dict()}
            self.load_state_dict(state_dict)

        self.to(device)
        self.eval()

        ## Print some information about the model
        embs_weights = torch.nn.functional.softmax(self.embs_avg_weights, dim=0)
        print(f"AvgEmbQueryEstimator.embs_weights (softmaxed): {embs_weights}")

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
                "docs_only": self.docs_only,
                "q_only": self.q_only,
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

    def _get_top_docs(self, queries: Sequence[str]):
        assert self.ranking is not None, "Provide a ranking before encoding."
        assert self.index.dim is not None, "Index dimension cannot be None."

        d_embs = torch.zeros((len(queries), self.n_docs, 768), device=self.device)

        # Retrieve the top-ranked documents for all queries
        top_docs = self.ranking._df[self.ranking._df["query"].isin(queries)]
        query_to_idx = {query: idx for idx, query in enumerate(queries)}

        pos_scores = np.zeros(self.n_docs)
        for query, group in top_docs.groupby("query"):
            top_embs, d_idxs = self.index._get_vectors(group["id"].unique())
            if self.index.quantizer is not None:
                top_embs = self.index.quantizer.decode(top_embs)
            top_embs = torch.tensor(top_embs[[x[0] for x in d_idxs]], device=self.device)

            # Repeat d_embs until reaching length n_docs
            if len(top_embs) < self.n_docs:
                top_embs = torch.cat([top_embs] * self.n_docs, dim=0)[: self.n_docs]

            query_idx = query_to_idx[str(query)]
            d_embs[query_idx] = top_embs

            # Update position scores
            for i, idx in enumerate(d_idxs):
                pos_scores[idx[0]] += group.iloc[i]["score"]

        # Update self.rank_scores with new avg and divide by 2
        self.rank_scores = (self.rank_scores + pos_scores / len(queries)) / 2

        return d_embs

    def forward(self, queries: Sequence[str]) -> torch.Tensor:
        t0 = perf_counter()

        if self.docs_only:
            q_emb_1 = torch.zeros((len(queries), 768), device=self.device)
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
            q_tok_embs_masked = q_tok_embs * attention_mask.unsqueeze(-1)
            match self.tok_w_method:
                case WEIGHT_METHOD.UNIFORM:
                    q_emb_1 = torch.mean(q_tok_embs_masked, 1)
                case WEIGHT_METHOD.LEARNED:
                    q_tok_weights = torch.nn.functional.softmax(
                        self.tok_embs_avg_weights[input_ids], -1
                    )
                    q_emb_1 = torch.sum(q_tok_embs_masked * q_tok_weights.unsqueeze(-1), 1)
            if self.normalize_q_emb_1:
                q_emb_1 = torch.nn.functional.normalize(q_emb_1)

        t1 = perf_counter()
        if self.index._profiling:
            LOGGER.info(f"Lightweight query estimation (q_emb_1) took: {t1 - t0:.5f}s")

        if self.q_only:
            return q_emb_1

        # lookup embeddings of top-ranked documents in (in-memory) self.index
        d_embs = self._get_top_docs(queries)

        if self.index._verbose:
            self.rank_scores = np.round(self.rank_scores, 3)
            LOGGER.info(f"AvgEmbQueryEstimator.rank_scores: [{', '.join(map(str, self.rank_scores))}]")

        t2 = perf_counter()
        if self.index._profiling:
            LOGGER.info(f"Lookup of top-ranked documents (_get_top_docs) took: {t2 - t1:.5f}s")

        # estimate query embedding as weighted average of q_emb and d_embs
        embs = torch.cat((q_emb_1.unsqueeze(1), d_embs), -2)
        embs_weights = torch.zeros((self.n_embs), device=self.device)
        if self.docs_only:
            embs_weights[0] = 0.0
            embs_weights[1:self.n_embs] = torch.nn.functional.softmax(self.embs_avg_weights[1:self.n_embs], 0)
        else:
            embs_weights[:self.n_embs] = torch.nn.functional.softmax(self.embs_avg_weights[:self.n_embs], 0)
        embs_weights = embs_weights.unsqueeze(0).expand(len(queries), -1)

        q_emb_2 = torch.sum(embs * embs_weights.unsqueeze(-1), -2)
        if self.normalize_q_emb_2:
            q_emb_2 = torch.nn.functional.normalize(q_emb_2)

        t3 = perf_counter()
        if self.index._profiling:
            LOGGER.info(f"Query embedding estimation (q_emb_2) took: {t3 - t2:.5f}s")

        return q_emb_2

    def __call__(self, queries: Sequence[str]) -> np.ndarray:
        return self.forward(queries).cpu().detach().numpy()
