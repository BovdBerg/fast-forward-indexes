import abc
from pathlib import Path
from typing import Any, Dict, Sequence, Union

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from fast_forward.encoder import Encoder


class EncoderModel(abc.ABC, torch.nn.Module):
    """Base class for encoders."""

    @abc.abstractmethod
    def forward(self, batch: Any) -> torch.Tensor:
        """Encode a batch of inputs.

        Args:
            batch (EncodingBatch): The encoder model inputs.

        Returns:
            torch.Tensor: The encoded inputs.
        """
        pass

    @property
    @abc.abstractmethod
    def embedding_dimension(self) -> int:
        """Return the embedding dimension.

        Returns:
            int: The dimension of query or document vectors.
        """
        pass


class TransformerEmbeddingEncoder(EncoderModel):
    """Encodes a string using the average of the embedded tokens.
    Static token embeddings are obtained from a pre-trained Transformer model.
    """

    def __init__(
        self,
        pretrained_model: Union[str, Path],
    ) -> None:
        super().__init__()
        model = AutoModel.from_pretrained(pretrained_model, return_dict=True)
        self.embeddings = model.get_input_embeddings()
        self.dense = None # Not sure if needed

    def forward(self, batch: Dict[str, torch.LongTensor]) -> torch.Tensor:
        inputs = batch["input_ids"]
        lengths = (inputs != 0).sum(dim=1)
        sequences_emb = self.embeddings(inputs)

        # create a mask corresponding to sequence lengths
        _, max_len, emb_dim = sequences_emb.shape
        mask = torch.arange(max_len, device=lengths.device).unsqueeze(
            0
        ) < lengths.unsqueeze(-1)
        mask = mask.unsqueeze(-1).expand(-1, -1, emb_dim)

        # compute the mean for each sequence
        rep = torch.sum(mask * sequences_emb, dim=1) / lengths.unsqueeze(-1)
        return rep

    def embedding_dimension(self) -> int:
        return self.embeddings.embedding_dim


class StandaloneEncoder(Encoder):
    """Adapter class to use encoders for indexing, retrieval, or re-ranking.
    Can be used as an encoder for Fast-Forward indexes. Outputs normalized representations.
    """

    def __init__(
        self,
        pretrained_model: Union[str, Path],
        ckpt_path: Path,
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
        self.encoder = TransformerEmbeddingEncoder(pretrained_model)
        self.device = device
        self.encoder.to(device)

        # Load checkpoint and extract encoder weights
        sd_enc = {}
        ckpt = torch.load(ckpt_path, map_location=device)
        for k, v in ckpt["state_dict"].items():
            prefix = "query_encoder."
            if k.startswith(prefix):
                sd_enc[k[len(prefix):]] = v
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
        # TODO: either line probably works. Verify this and then remove one:
        inputs.to(self.device)
        # inputs = {k: v.to(self.device) for k, v in inputs}

        with torch.no_grad():
            # TODO: Can I move the query_encoder.forward call to here? Set self.embeddings in init too.
            rep = self.encoder(inputs)
            rep_norm = torch.nn.functional.normalize(rep)
            return rep_norm.detach().cpu().numpy()
