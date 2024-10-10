from typing import Callable, Sequence

import numpy as np
from fast_forward.encoder import Encoder


class LambdaEncoder(Encoder):
    """Encoder adapter class for arbitrary encoding functions."""

    def __init__(self, f: Callable[[str], np.ndarray]) -> None:
        """Create a lambda encoder.

        Args:
            f (Callable[[str], np.ndarray]): Function to encode a single piece of text.
        """
        super().__init__()
        self._f = f

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        return np.array(list(map(self._f, texts)))
