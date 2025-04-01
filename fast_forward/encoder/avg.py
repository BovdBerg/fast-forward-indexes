import logging
import warnings
from pathlib import Path
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


class AvgEmbQueryEstimator(Encoder, GeneralModule):
    def __init__(
        self,
        index: Index,
        n_docs: int,
        device: str,
        ranking: Optional[Ranking] = None,
        ckpt_path: Optional[Path] = None,
        ckpt_path_tok_embs: Optional[Path] = None,
        q_only: bool = False,
        docs_only: bool = False,
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
            ckpt_path_tok_embs (Optional[Path]): Path to a checkpoint to load token embeddings. Overwrites `tok_embs`.
            q_only (bool): Whether to only use the lightweight query estimation and not the top-ranked documents.
            docs_only (bool): Whether to disable the lightweight query estimation and only use the top-ranked documents.
        """
        assert not (q_only and docs_only), "Cannot use both q_only and docs_only."

        super().__init__()
        self.index = index
        self._ranking = ranking
        self.n_docs = n_docs
        self.n_embs = n_docs + 1
        self.pretrained_model = "bert-base-uncased"
        self.q_only = q_only
        self.docs_only = docs_only

        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model)

        model = AutoModel.from_pretrained(self.pretrained_model)
        self.tok_embs = model.get_input_embeddings()

        vocab_size = self.tokenizer.vocab_size
        self.tok_embs_weights = torch.nn.Parameter(torch.ones(vocab_size) / vocab_size)

        self.embs_weights = torch.nn.Parameter(torch.ones(self.n_embs) / self.n_embs)

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
        LOGGER.info(f"embs_avg_weights (softmaxed): {torch.nn.functional.softmax(self.embs_weights, dim=0)}")

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
            elif key in {"embs_avg_weights", "_embs_weights"}:
                state_dict["embs_weights"] = v
            elif key in self.state_dict():
                state_dict[key] = v
        self.load_state_dict(state_dict)

    def __call__(self, queries: Sequence[str]) -> np.ndarray:
        return self.forward(queries).cpu().detach().numpy()

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()
        LOGGER.info(f"embs_avg_weights (softmaxed): {torch.nn.functional.softmax(self.embs_weights, dim=0)}")

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
            ).to(self.device)
            input_ids = q_tokens["input_ids"].to(self.device)
            attention_mask = q_tokens["attention_mask"].to(self.device)

            # estimate lightweight query as weighted average of q_tok_embs
            q_tok_embs = self.tok_embs(input_ids)
            q_tok_embs = q_tok_embs * attention_mask.unsqueeze(-1)
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
        embs_weights = torch.zeros((embs.shape[-2]), device=self.device)
        if self.docs_only:
            embs_weights[0] = 0.0
            embs_weights[1:self.n_embs] = torch.nn.functional.softmax(self.embs_weights[1:self.n_embs], 0)
        else:
            embs_weights[:self.n_embs] = torch.nn.functional.softmax(self.embs_weights[:self.n_embs], 0)
        embs_weights = embs_weights.unsqueeze(0).expand(len(queries), -1)
        embs_weights = embs_weights * (embs.sum(-1) != 0).float()  # Mask padding
        embs_weights = embs_weights / (embs_weights.sum(-1, keepdim=True) + 1e-9)  # Normalize

        q_estimation = torch.sum(embs * embs_weights.unsqueeze(-1), -2)
        return q_estimation
