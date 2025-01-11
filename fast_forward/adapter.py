from typing import Optional
import torch

from fast_forward.lightning import GeneralModule


class Adapter(GeneralModule):
    def __init__(self, ckpt_path: Optional[str] = None, device: str = "cpu") -> None:
        super().__init__()
        self.flatten = torch.nn.Flatten(0)

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
        self.linear_relu_stack.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.linear_relu_stack(x)
        x = x.unsqueeze(0).unsqueeze(0)
        return x
