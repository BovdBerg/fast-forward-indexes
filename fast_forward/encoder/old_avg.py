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
        assert self.ranking is not None, "Provide a ranking before encoding."
        assert self.index.dim is not None, "Index dimension cannot be None."

        # Retrieve the top-ranked documents for all queries
        top_docs = self.ranking._df[self.ranking._df["query"].isin(queries)].copy()
        top_docs['rank'] = top_docs.groupby('query')['score'].rank(ascending=False, method='first').astype(int) - 1
        query_to_idx = {query: idx for idx, query in enumerate(queries)}

        # Retrieve any needed embeddings from the index
        d_embs, d_idxs = self.index._get_vectors(top_docs["id"].unique())
        if self.index.quantizer is not None:
            d_embs = self.index.quantizer.decode(d_embs)
        d_embs = torch.tensor(d_embs[np.array(d_idxs).flatten()], device=self.device)

        # Map doc_ids in top_docs_ids to embeddings
        top_docs_embs = torch.zeros((len(queries), self.n_docs, 768), device=self.device)
        top_docs_embs[
            torch.tensor(top_docs["query"].map(query_to_idx).values, device=self.device),
            torch.tensor(top_docs["rank"].values, device=self.device)
        ] = d_embs

        return top_docs_embs

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
            match self.tok_w_method:
                case WEIGHT_METHOD.UNIFORM:
                    # TODO: /n_masked probably better
                    q_light = torch.mean(q_tok_embs, 1)
                case WEIGHT_METHOD.LEARNED:
                    q_tok_weights = torch.nn.functional.softmax(
                        self.tok_embs_avg_weights[input_ids], -1
                    )
                    q_light = torch.sum(q_tok_embs * q_tok_weights.unsqueeze(-1), 1)
        if self.q_only:
            return q_light

        # lookup embeddings of top-ranked documents in (in-memory) self.index
        top_docs_embs = self._get_top_docs_embs(queries)

        # estimate query embedding as weighted average of q_emb and d_embs
        embs = torch.cat((q_light.unsqueeze(1), top_docs_embs), -2)
        embs_weights = torch.zeros((self.n_embs), device=self.device)
        if self.docs_only:
            embs_weights[0] = 0.0
            embs_weights[1:self.n_embs] = torch.nn.functional.softmax(self.embs_avg_weights[1:self.n_embs], 0)
        else:
            embs_weights[:self.n_embs] = torch.nn.functional.softmax(self.embs_avg_weights[:self.n_embs], 0)
        embs_weights = embs_weights.unsqueeze(0).expand(len(queries), -1)

        q_estimation = torch.sum(embs * embs_weights.unsqueeze(-1), -2)
        return q_estimation

    def __call__(self, queries: Sequence[str]) -> np.ndarray:
        return self.forward(queries).cpu().detach().numpy()
