import abc
from pathlib import Path
from typing import Any, Dict, Sequence, Union

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from fast_forward.encoder import Encoder as FFEncoder


class Tokenizer(abc.ABC):
    """Base class for tokenizers."""

    @abc.abstractmethod
    def __call__(self, batch: Sequence[str]) -> Any:
        """Tokenize a batch of strings.

        Args:
            batch (Sequence[str]): The tokenizer inputs.

        Returns:
            EncodingModelBatch: The tokenized inputs.
        """
        pass


class TransformerTokenizer(Tokenizer):
    """Tokenizer for Transformer models."""

    def __init__(self, pretrained_model: str, max_length: int = None) -> None:
        """Constuctor.

        Args:
            pretrained_model (str): Pre-trained model on the HuggingFace Hub.
            max_length (int, optional): Maximum number of tokens. Defaults to None.
        """
        super().__init__()
        self.tok = AutoTokenizer.from_pretrained(pretrained_model)
        self.max_length = max_length

    def __call__(self, batch: Sequence[str]) -> Any:
        """Tokenize a batch of strings.

        Args:
            batch (Sequence[str]): The tokenizer inputs.

        Returns:
            EncodingModelBatch: The tokenized inputs.
        """
        return self.tok(
            batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length,
        )


class Encoder(abc.ABC, torch.nn.Module):
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


class TransformerEmbeddingEncoder(Encoder):
    """Encodes a string using the average of the embedded tokens.
    Static token embeddings are obtained from a pre-trained Transformer model.
    """

    def __init__(
        self,
        pretrained_model: Union[str, Path],
        device: str = "cpu",
    ) -> None:
        super().__init__()
        model = AutoModel.from_pretrained(pretrained_model, return_dict=True)
        model.to(device)
        self.embeddings = model.get_input_embeddings()
        self.dense = None

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


class StandaloneEncoder(FFEncoder):
    """Adapter class to use encoders for indexing, retrieval, or re-ranking.
    Can be used as an encoder for Fast-Forward indexes. Outputs normalized representations.
    """

    def __init__(
        self,
        model: Union[str, Path],
        ckpt_path: Path,
        device: str = "cpu",
    ) -> None:
        """Instantiate a standalone encoder.

        Args:
            encoder_config (DictConfig): Encoder config.
            ckpt_file (Path): Checkpoint to load.
            weights_prefix (str, optional): Prefix of the keys to be loaded in the state dict of the checkpoint. Defaults to "query_encoder".
            device (str, optional): Device to use. Defaults to "cpu".
        """
        super().__init__()
        self.tokenizer = TransformerTokenizer(model, max_length=512)
        self.encoder = TransformerEmbeddingEncoder(model, device)
        self.device = device
        self.encoder.to(device)

        sd_enc = {}
        ckpt = torch.load(ckpt_path, map_location=device)
        for k, v in ckpt["state_dict"].items():
            if k.startswith("query_encoder."):
                sd_enc[k[14:]] = v
        self.encoder.load_state_dict(sd_enc)
        self.encoder.eval()

    def _encode(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode and normalize tokenized inputs.

        Args:
            inputs (Dict[str, torch.Tensor]): The tokenized inputs.

        Returns:
            torch.Tensor: The normalized representations.
        """
        with torch.no_grad():
            rep = self.encoder({k: v.to(self.device) for k, v in inputs.items()})
            return torch.nn.functional.normalize(rep).detach().cpu().numpy()

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(self.tokenizer(texts))
