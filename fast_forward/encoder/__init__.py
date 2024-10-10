"""
.. include:: docs/encoder.md
"""
import abc
from typing import Sequence

import numpy as np


class Encoder(abc.ABC):
    """Base class for encoders."""

    @abc.abstractmethod
    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        """Encode a list of texts.

        Args:
            texts (Sequence[str]): The texts to encode.

        Returns:
            np.ndarray: The resulting vector representations.
        """
        pass
