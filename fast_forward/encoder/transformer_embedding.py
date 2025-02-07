from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, BatchEncoding

from fast_forward.encoder import Encoder


# TODO: Rename to Encoder names used in ppt.
class TransformerEmbeddingEncoder(torch.nn.Module):
    """Encodes a string using the average of the embedded tokens.
    Static token embeddings are obtained from a pre-trained Transformer model.
    """

    def __init__(
        self,
        pretrained_model: str,
        ckpt_path: Optional[Path] = None,
        device: str = "cpu",
    ) -> None:
        """Constuctor.

        Args:
            pretrained_model (str): Pre-trained model on the HuggingFace Hub to get the token embeddings from.
            ckpt_path (Path, optional): Path to a checkpoint to load. Defaults to None.
            device (str, optional): Device to use. Defaults to "cpu".
        """
        super().__init__()
        model = AutoModel.from_pretrained(pretrained_model, return_dict=True)
        self.embeddings = model.get_input_embeddings()

        # Load checkpoint and extract encoder weights
        if ckpt_path is not None:
            sd_enc = {}
            prefix = "query_encoder."
            ckpt = torch.load(ckpt_path, map_location=device)
            for k, v in ckpt["state_dict"].items():
                if k.startswith(prefix):
                    sd_enc[k[len(prefix) :]] = v  # remove prefix
            self.load_state_dict(sd_enc)

        self.to(device)
        self.eval()

    def forward(self, batch: BatchEncoding) -> torch.Tensor:
        # Get token embeddings
        tok_embs = self.embeddings(batch["input_ids"])

        # Apply attention mask to remove padding tokens
        mask = batch["attention_mask"]
        masked_emb = tok_embs * mask.unsqueeze(-1)

        # Compute the mean of the masked embeddings
        n_masked = mask.sum(dim=-1, keepdim=True)
        mean_emb = masked_emb.sum(dim=-2) / n_masked

        return mean_emb


class StandaloneEncoder(Encoder):
    """Adapter class to use encoders for indexing, retrieval, or re-ranking.
    Can be used as an encoder for Fast-Forward indexes. Outputs normalized representations.
    """

    def __init__(
        self,
        ckpt_path: Optional[Path] = None,
        device: str = "cpu",
    ) -> None:
        """Instantiate a standalone encoder.

        Args:
            ckpt_path (Path): Checkpoint to load.
            device (str, optional): Device to use. Defaults to "cpu".
        """
        super().__init__()
        pretrained_model = "google/bert_uncased_L-12_H-768_A-12"
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.encoder = TransformerEmbeddingEncoder(
            str(pretrained_model), ckpt_path, device
        )
        self.device = device

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        inputs = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        inputs.to(self.device)

        with torch.no_grad():
            rep = self.encoder(inputs)
            rep_norm = torch.nn.functional.normalize(rep)
            return rep_norm.detach().cpu().numpy()
