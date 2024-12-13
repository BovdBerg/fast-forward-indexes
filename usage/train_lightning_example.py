"""
Copied from PyTorch Lightning tutorial: https://lightning.ai/docs/pytorch/stable/starter/introduction.html
Example of training a simple autoencoder on the MNIST dataset (images (28x28 pix) of numbers 0-9).

Date: 12-12-2024.
"""

import os

import lightning as L
from torch import nn, optim, utils
import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


### 2: Define a LightningModule
class LitAutoEncoder(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(28 * 28, 10))
        self.decoder = nn.Sequential(nn.Linear(10, 28 * 28))

    def training_step(self, batch, batch_idx):
        x, _ = (
            batch  # autoencoder does not need y (target) as loss is calculated using x
        )
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


### 3: Define a dataset
dataset = MNIST(os.getcwd() + "/data", download=True, transform=ToTensor())
train_loader = utils.data.DataLoader(dataset, num_workers=11)


### 4: Train the model
autoencoder = LitAutoEncoder()
trainer = L.Trainer(limit_train_batches=500, max_epochs=3)  # See Trainer docs for more options
trainer.fit(model=autoencoder, train_dataloaders=train_loader)


### 5: Use the model
# # load checkpoint
# checkpoint = "./lightning_logs/version_0/checkpoints/epoch=2-step=1500.ckpt"
# autoencoder = LitAutoEncoder.load_from_checkpoint(checkpoint)

# # choose your trained nn.Module
# encoder = autoencoder.encoder
# encoder.eval()

# # embed 4 fake images!
# fake_image_batch = torch.rand(4, 28 * 28, device=autoencoder.device)
# embeddings = encoder(fake_image_batch)
# print("⚡" * 20, "\nPredictions (4 image embeddings):\n", embeddings, "\n", "⚡" * 20)
