import argparse
import os
import time
import warnings
from math import ceil
from pathlib import Path
from typing import Tuple

import lightning
import numpy as np
import pyterrier as pt
import torch
from lightning.pytorch import callbacks
from torch.utils.data import DataLoader
from tqdm import tqdm

from fast_forward.encoder.avg import LearnedAvgWeights, WeightedAvgEncoder
from fast_forward.encoder.transformer import TCTColBERTQueryEncoder, TransformerEncoder
from fast_forward.index.disk import OnDiskIndex
from fast_forward.ranking import Ranking


class SimpleModel(lightning.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(1, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 1),
        )

    def forward(self, x):
        return self.encoder(x)

    def step(self, batch, name):
        x, y = batch
        print(f"x: {x}, y: {y}")
        logits = self(x)
        loss = torch.nn.MSELoss()(logits, y)
        self.log(f"{name}_loss", loss, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


def main() -> None:
    """
    Train a model using PyTorch Lightning.
    """
    dataset = []
    inputs = range(10)
    for inp in inputs:
        inp = torch.tensor([inp], dtype=torch.float32)
        target = inp
        dataset.append((inp, target))
    print(f"dataset: {dataset}")

    train_loader = DataLoader(
        dataset,
        sampler=torch.utils.data.RandomSampler(
            dataset, replacement=False, num_samples=3
        ),
        num_workers=11,
    )
    val_loader = DataLoader(
        dataset,
        shuffle=False,
        num_workers=11,
    )

    model = SimpleModel()
    trainer = lightning.Trainer(
        deterministic="warn",
        max_epochs=1,
        callbacks=[
            callbacks.EarlyStopping(
                monitor="val_loss", min_delta=0.001, patience=2, verbose=True
            ),
            callbacks.ModelCheckpoint(monitor="val_loss", verbose=True),
            callbacks.ModelSummary(max_depth=2),
        ],
    )
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )


if __name__ == "__main__":
    main()
