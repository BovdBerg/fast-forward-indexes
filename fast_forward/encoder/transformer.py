from pathlib import Path
from typing import Sequence, Union

import numpy as np
import torch
from fast_forward.encoder import Encoder
from transformers import AutoModel, AutoTokenizer


class TransformerEncoder(Encoder):
    """Uses a pre-trained transformer model for encoding. Returns the pooler output."""

    def __init__(
        self, model: Union[str, Path], device: str = "cpu", **tokenizer_args
    ) -> None:
        """Create a transformer encoder.

        Args:
            model (Union[str, Path]): Pre-trained transformer model (name or path).
            device (str, optional): PyTorch device. Defaults to "cpu".
            **tokenizer_args: Additional tokenizer arguments.
        """
        super().__init__()
        self.model = AutoModel.from_pretrained(model)
        self.model.to(device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.device = device
        self.tokenizer_args = tokenizer_args

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        inputs = self.tokenizer(texts, return_tensors="pt", **self.tokenizer_args)
        inputs.to(self.device)
        with torch.no_grad():
            return self.model(**inputs).pooler_output.detach().cpu().numpy()
