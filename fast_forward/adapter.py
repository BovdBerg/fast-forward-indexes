import torch

from fast_forward.lightning import GeneralModule


class Adapter(GeneralModule):
    def __init__(self):
        super().__init__()
        self.flatten = torch.nn.Flatten(0)

        encoding_dim = 768
        hidden_dim = 100
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(encoding_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, encoding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.linear_relu_stack(x)
        x = x.unsqueeze(0).unsqueeze(0)
        return x
