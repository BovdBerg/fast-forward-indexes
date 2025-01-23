import json
from pathlib import Path

import lightning
import torch


class GeneralModule(lightning.LightningModule):
    """
    Watch this short video on PyTorch for this class to make sense: https://youtu.be/ORMx45xqWkA?si=Bvkm9SWi8Hz1n2Sh

    Override __init__ to define the model architecture. Make sure to set self.loss_fn to a loss function.
    Override forward to define the forward pass.
    Override configure_optimizers to define the optimizer.
    """

    def __init__(self) -> None:
        super().__init__()
        self.loss_fn = torch.nn.MSELoss()

    def on_train_start(self) -> None:
        if self.trainer.log_dir is None:
            raise ValueError("Trainer log directory is None")

        settings_file = Path(self.trainer.log_dir) / "settings.json"
        with open(settings_file, "w") as f:
            json.dump(
                {"Class": self.__class__.__name__},
                f,
                indent=4,
            )

    def step(self, batch: tuple[torch.Tensor, torch.Tensor], name: str) -> torch.Tensor:
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        self.log(f"{name}_loss", loss)
        return loss

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        return self.step(batch, "train")

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        return self.step(batch, "val")

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        return self.step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
