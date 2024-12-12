"""
Copied from PyTorch Lightning tutorial: https://lightning.ai/docs/pytorch/stable/starter/introduction.html

Date: 12-12-2024.
"""

import os

import lightning as L
import torch
from torch import Tensor, nn, optim, utils
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

### 2: Define a LightningModule
# define any number of nn.Modules (or use your current ones)
encoder = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))
decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))


# define the LightningModule
class LitAutoEncoder(L.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, _ = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    # def predict_step(self, *args, **kwargs):
    #     return super().predict_step(*args, **kwargs)

    # def validation_step(self, *args, **kwargs):
    #     return super().validation_step(*args, **kwargs)

    # def test_step(self, *args, **kwargs):
    #     return super().test_step(*args, **kwargs)


# init the autoencoder
autoencoder = LitAutoEncoder(encoder, decoder)


### 3: Define a dataset
# setup data
dataset = MNIST(os.getcwd(), download=True, transform=ToTensor())
train_loader = utils.data.DataLoader(dataset)


### 4: Train the model
# train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
trainer = L.Trainer(limit_train_batches=100, max_epochs=5)
trainer.fit(model=autoencoder, train_dataloaders=train_loader)


### 5: Use the model
# load checkpoint
checkpoint = "./lightning_logs/version_0/checkpoints/epoch=0-step=100.ckpt"
autoencoder = LitAutoEncoder.load_from_checkpoint(
    checkpoint, encoder=encoder, decoder=decoder
)

# choose your trained nn.Module
encoder = autoencoder.encoder
encoder.eval()

# embed 4 fake images!
fake_image_batch = torch.rand(4, 28 * 28, device=autoencoder.device)
embeddings = encoder(fake_image_batch)
print("⚡" * 20, "\nPredictions (4 image embeddings):\n", embeddings, "\n", "⚡" * 20)
