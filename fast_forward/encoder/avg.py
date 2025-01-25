import json
import warnings
from enum import Enum
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
        tok_weight_method: WEIGHT_METHOD = WEIGHT_METHOD.LEARNED,
        untrained_tok_weight: float = 1.0,
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
            ckpt_path (Optional[Path]): Path to a checkpoint to load.
            tok_weight_method (TOKEN_WEIGHT_METHOD): The method to use for token weighting.
            untrained_tok_weight (float): The weight to assign to untrained tokens. Use 1.0 to treat them equal to trained tokens.
        """
        super().__init__()
        self.index = index
        self._ranking = ranking
        self.n_docs = n_docs

        doc_encoder_pretrained = "bert-base-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(doc_encoder_pretrained)
        vocab_size = self.tokenizer.vocab_size

        doc_encoder = AutoModel.from_pretrained(doc_encoder_pretrained)
        self.tok_embs = (
            doc_encoder.get_input_embeddings()
        )  # Maps token_id --> embedding, Embedding(vocab_size, embedding_dim)
        self.untrained_tok_weight = untrained_tok_weight
        self.register_buffer(
            "trained_toks", torch.zeros((vocab_size), dtype=torch.bool)
        )

        self.tok_embs_avg_weights = torch.nn.Parameter(
            torch.ones(vocab_size) / vocab_size
        )  # weights for averaging over q's token embedding, shape (vocab_size,)
        self.tok_weight_method = tok_weight_method

        n_embs = n_docs + 1
        # TODO: Maybe self.embs_avg_weights should have a dimension for n_embs_per_q too? [[1.0], [0.5, 0.5], [0.33, 0.33, 0.33]] or padded [[1.0, 0.0, 0.0], [0.5, 0.5, 0], [0.33, 0.33, 0.33]] etc... up until n_embs
        self.embs_avg_weights = torch.nn.Parameter(
            torch.ones(n_embs) / n_embs
        )  # weights for averaging over q_emb1 ++ d_embs, shape (n_embs,)

        # TODO: add different WEIGHT_METHODs for d_emb weighting (excluding q_emb_1)

        if ckpt_path is not None:
            self.ckpt_path = ckpt_path
            ckpt = torch.load(ckpt_path, map_location=device)
            self.load_state_dict(ckpt["state_dict"])

        self.to(device)
        self.eval()

        ## Print some information about the model
        embs_avg_weights = torch.nn.functional.softmax(self.embs_avg_weights, dim=0)
        print(f"AvgEmbQueryEstimator.embs_avg_weights (softmaxed): {embs_avg_weights}")

        trained_tokens_count = int(torch.sum(self.trained_toks).item())
        trained_tokens_percentage = trained_tokens_count / vocab_size * 100
        print(
            f"AvgEmbQueryEstimator.trained_toks: {trained_tokens_count}/{vocab_size} ({trained_tokens_percentage:.2f}%). Ignoring {vocab_size - trained_tokens_count} tokens in averaging."
        )

    def on_train_start(self) -> None:
        super().on_train_start()
        with open(self.settings_file, "r") as f:
            settings = json.load(f)

        settings.update(
            {
                "n_docs": self.n_docs,
                "device": self.device.type,
                "ckpt_path": getattr(self, "ckpt_path", None),
                "untrained_tok_weight": self.untrained_tok_weight,
                "tok_weight_method": self.tok_weight_method.value,
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

        # Create tensors for padding and total embedding counts
        d_embs_pad = torch.zeros((len(queries), self.n_docs, 768), device=self.device)
        n_embs_per_q = torch.ones((len(queries)), dtype=torch.int, device=self.device)

        # Retrieve the top-ranked documents for all queries
        top_docs = self.ranking._df[self.ranking._df["query"].isin(queries)]
        query_to_idx = {query: idx for idx, query in enumerate(queries)}

        for query, group in top_docs.groupby("query"):
            d_embs, d_idxs = self.index._get_vectors(group["id"].unique())
            if self.index.quantizer is not None:
                d_embs = self.index.quantizer.decode(d_embs)
            d_embs = torch.tensor(d_embs[[x[0] for x in d_idxs]], device=self.device)

            # Pad and count embeddings for this query
            query_idx = query_to_idx[str(query)]
            d_embs_pad[query_idx, : len(d_embs)] = d_embs
            n_embs_per_q[query_idx] += len(d_embs)

        return d_embs_pad, n_embs_per_q

    def forward(self, queries: Sequence[str]) -> torch.Tensor:
        # Tokenizer queries similar to TCTColBERTQueryEncoder
        max_length = 36
        q_tokens = self.tokenizer(
            ["[CLS] [Q] " + q + "[MASK]" * max_length for q in queries],
            max_length=max_length,
            truncation=True,
            add_special_tokens=False,
            return_tensors="pt",
            padding=False,
        ).to(self.device)
        input_ids = q_tokens["input_ids"].to(self.device)
        attention_mask = q_tokens["attention_mask"].to(self.device)

        # Remove first 4 tokens from attention mask (similar to TCTQueryEncoder)
        attention_mask[:, :4] = 0
        # # Remove any special tokens from attention mask (similar to TCTQueryEncoder)
        # special_tokens_mask = ~torch.isin(
        #     input_ids, torch.tensor(self.tokenizer.all_special_ids, device=self.device)
        # )
        # attention_mask = attention_mask * special_tokens_mask

        if self._trainer is not None and self.trainer.training:
            # During training, update self.trained_toks with the encountered tokens
            self.trained_toks[torch.unique(input_ids.flatten())] = True
        elif self.untrained_tok_weight != 1.0:
            # During inference, extend attention mask to weigh untrained tokens with untrained_tok_weight
            trained_toks_mask = self.trained_toks[input_ids]
            trained_toks_mask[~trained_toks_mask] = self.untrained_tok_weight
            attention_mask = attention_mask * trained_toks_mask

        # estimate lightweight query as weighted average of q_tok_embs
        q_tok_embs = self.tok_embs(input_ids)
        q_tok_embs_masked = q_tok_embs * attention_mask.unsqueeze(-1)
        match self.tok_weight_method:
            case WEIGHT_METHOD.UNIFORM:
                q_emb_1 = torch.mean(q_tok_embs_masked, 1)
            case WEIGHT_METHOD.LEARNED:
                q_tok_weights = torch.nn.functional.softmax(
                    self.tok_embs_avg_weights[input_ids], -1
                )
                q_emb_1 = torch.sum(q_tok_embs_masked * q_tok_weights.unsqueeze(-1), 1)
        # TODO: What if all (weighted) query tokens are added to doc_embs instead of 1 q_emb_1? Would need different weighting, padding, and masking.

        # lookup embeddings of top-ranked documents in (in-memory) self.index
        d_embs_pad, n_embs_per_q = self._get_top_docs(queries)

        # estimate query embedding as weighted average of q_emb and d_embs
        embs = torch.cat((q_emb_1.unsqueeze(1), d_embs_pad), -2)
        embs_weights = torch.zeros((len(queries), embs.shape[-2]), device=self.device)
        # assign self.embs_avg_weights to embs_avg_weights, but only up to the number of top-ranked documents per query
        for i, n_embs in enumerate(n_embs_per_q):
            embs_weights[i, :n_embs] = torch.nn.functional.softmax(
                self.embs_avg_weights[:n_embs], 0
            )

        q_emb_2 = torch.sum(embs * embs_weights.unsqueeze(-1), -2)
        return q_emb_2

    def __call__(self, queries: Sequence[str]) -> np.ndarray:
        return self.forward(queries).cpu().detach().numpy()
