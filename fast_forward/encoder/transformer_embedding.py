from pathlib import Path
from typing import Sequence, Union

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, BatchEncoding

from fast_forward.encoder import Encoder


class TransformerEmbeddingEncoder(torch.nn.Module):
    """Encodes a string using the average of the embedded tokens.
    Static token embeddings are obtained from a pre-trained Transformer model.
    """

    def __init__(self, pretrained_model: str) -> None:
        """Constuctor.

        Args:
            pretrained_model (str): Pre-trained model on the HuggingFace Hub to get the token embeddings from.
            dense_layer_dim (int, optional): Add a dense layer to adjust the embedding dimension. Defaults to None.
        """
        super().__init__()
        model = AutoModel.from_pretrained(pretrained_model, return_dict=True)
        self.embeddings = model.get_input_embeddings()

    def forward(self, batch: BatchEncoding) -> torch.Tensor:
        input_ids = batch["input_ids"]
        lengths = (input_ids != 0).sum(dim=1)
        sequences_emb = self.embeddings(input_ids)

        # create a mask corresponding to sequence lengths
        _, max_len, emb_dim = sequences_emb.shape
        mask = torch.arange(max_len, device=lengths.device).unsqueeze(
            0
        ) < lengths.unsqueeze(-1)
        mask = mask.unsqueeze(-1).expand(-1, -1, emb_dim)

        # compute the mean for each sequence
        rep = torch.sum(mask * sequences_emb, dim=1) / lengths.unsqueeze(-1)
        return rep


class StandaloneEncoder(Encoder):
    """Adapter class to use encoders for indexing, retrieval, or re-ranking.
    Can be used as an encoder for Fast-Forward indexes. Outputs normalized representations.
    """

    def __init__(
        self,
        pretrained_model: Union[str, Path],
        ckpt_path: Path = None,
        device: str = "cpu",
    ) -> None:
        """Instantiate a standalone encoder.

        Args:
            pretrained_model (Union[str, Path]): Pre-trained transformer model.
            ckpt_path (Path): Checkpoint to load.
            device (str, optional): Device to use. Defaults to "cpu".
        """
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.device = device

        self.encoder = TransformerEmbeddingEncoder(pretrained_model)
        self.encoder.to(device)

        # Load checkpoint and extract encoder weights
        if ckpt_path is not None:
            sd_enc = {}
            prefix = "query_encoder."
            ckpt = torch.load(ckpt_path, map_location=device)
            for k, v in ckpt["state_dict"].items():
                if k.startswith(prefix):
                    sd_enc[k[len(prefix) :]] = v  # remove prefix
            self.encoder.load_state_dict(sd_enc)
        self.encoder.eval()

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        )
        inputs.to(self.device)

        with torch.no_grad():
            rep = self.encoder(inputs)
            rep_norm = torch.nn.functional.normalize(rep)
            return rep_norm.detach().cpu().numpy()
