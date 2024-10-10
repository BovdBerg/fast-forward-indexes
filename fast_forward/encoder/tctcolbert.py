from typing import Sequence

import numpy as np
import torch
from fast_forward.encoder.transformer import TransformerEncoder


class TCTColBERTQueryEncoder(TransformerEncoder):
    """Query encoder for pre-trained TCT-ColBERT models.

    Adapted from Pyserini:
    https://github.com/castorini/pyserini/blob/310c828211bb3b9528cfd59695184c80825684a2/pyserini/encode/_tct_colbert.py#L72
    """

    def __call__(self, queries: Sequence[str]) -> np.ndarray:
        max_length = 36
        inputs = self.tokenizer(
            ["[CLS] [Q] " + q + "[MASK]" * max_length for q in queries],
            max_length=max_length,
            truncation=True,
            add_special_tokens=False,
            return_tensors="pt",
            **self.tokenizer_args,
        )
        inputs.to(self.device)
        with torch.no_grad():
            embeddings = self.model(**inputs).last_hidden_state.detach().cpu().numpy()
            return np.average(embeddings[:, 4:, :], axis=-2)


class TCTColBERTDocumentEncoder(TransformerEncoder):
    """Document encoder for pre-trained TCT-ColBERT models.

    Adapted from Pyserini:
    https://github.com/castorini/pyserini/blob/310c828211bb3b9528cfd59695184c80825684a2/pyserini/encode/_tct_colbert.py#L27
    """

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        max_length = 512
        inputs = self.tokenizer(
            ["[CLS] [D] " + text for text in texts],
            max_length=max_length,
            padding=True,
            truncation=True,
            add_special_tokens=False,
            return_tensors="pt",
            **self.tokenizer_args,
        )
        inputs.to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            token_embeddings = outputs["last_hidden_state"][:, 4:, :]
            input_mask_expanded = (
                inputs["attention_mask"][:, 4:]
                .unsqueeze(-1)
                .expand(token_embeddings.size())
                .float()
            )
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embeddings = sum_embeddings / sum_mask
            return embeddings.detach().cpu().numpy()
