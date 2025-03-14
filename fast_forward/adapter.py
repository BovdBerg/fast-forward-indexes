from pathlib import Path
from typing import Optional
import torch

from fast_forward.lightning import GeneralModule


class Adapter(GeneralModule):
    def __init__(self, ckpt_path: Optional[Path] = None, device: str = "cpu") -> None:
        super().__init__()

        encoding_dim = 768
        hidden_dim = 100
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(encoding_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, encoding_dim),
        )

        if ckpt_path is not None:
            sd_enc = {}
            prefix = "linear_relu_stack."
            ckpt = torch.load(ckpt_path, map_location=device)
            for k, v in ckpt["state_dict"].items():
                if k.startswith(prefix):
                    sd_enc[k[len(prefix) :]] = v  # remove prefix
            self.linear_relu_stack.load_state_dict(sd_enc)

        self.to(device)
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_relu_stack(x)
        return x
