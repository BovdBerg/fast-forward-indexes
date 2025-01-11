import torch
from fast_forward.util.lightning import GeneralModule


class Adapter(GeneralModule):
    def __init__(self):
        super().__init__()
        self.flatten = torch.nn.Flatten(0)

        hidden_dim = 1000
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(768, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 768)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.linear_relu_stack(x)
        return x
