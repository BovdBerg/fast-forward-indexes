"""
Copied from PyTorch Lightning tutorial: https://lightning.ai/docs/pytorch/stable/starter/introduction.html

Date: 12-12-2024.
"""

import os

import lightning as L
from torch import nn, optim, utils
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


### 2: Define a LightningModule
class LitAutoEncoder(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(28 * 28, 10))
        self.decoder = nn.Sequential(nn.Linear(10, 28 * 28))

    # def forward(self, *args, **kwargs):
    #     return super().forward(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        print(f"batch shape: {batch[0].shape}")
        x, _ = batch
        print(f"x shape: {x.shape}")
        x = x.view(x.size(0), -1)
        print(f"x shape 2: {x.shape}")
        z = self.encoder(x)
        print(f"z shape: {z.shape}")
        x_hat = self.decoder(z)
        print(f"x_hat shape: {x_hat.shape}")
        loss = nn.functional.mse_loss(x_hat, x)
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
autoencoder = LitAutoEncoder()


### 3: Define a dataset
# setup data
dataset = MNIST(os.getcwd(), download=True, transform=ToTensor())
print(f"dataset.data.shape: {dataset.data.shape}")
train_loader = utils.data.DataLoader(dataset, num_workers=11, batch_size=2)


### 4: Train the model
# train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
trainer = L.Trainer(limit_train_batches=500, max_epochs=1)
trainer.fit(model=autoencoder, train_dataloaders=train_loader)


### 5: Use the model
# # load checkpoint
# checkpoint = "./lightning_logs/version_0/checkpoints/epoch=0-step=100.ckpt"
# autoencoder = LitAutoEncoder.load_from_checkpoint(
#     checkpoint, encoder=encoder, decoder=decoder
# )

# # choose your trained nn.Module
# encoder = autoencoder.encoder
# encoder.eval()

# # embed 4 fake images!
# fake_image_batch = torch.rand(4, 28 * 28, device=autoencoder.device)
# embeddings = encoder(fake_image_batch)
# print("⚡" * 20, "\nPredictions (4 image embeddings):\n", embeddings, "\n", "⚡" * 20)
