import lightning as L
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch import nn


class SAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.GELU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return latent, reconstructed


class LitSAE(L.LightningModule):
    def __init__(self, sae: SAE):
        super().__init__()
        self.sae = sae

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        input = batch[0]
        latent, reconstructed = self.sae(input)
        mse_loss = nn.functional.mse_loss(input, reconstructed)
        l1_loss = 0.1 * torch.mean(torch.abs(latent))
        loss = mse_loss + l1_loss
        self.log("mse_loss", mse_loss, prog_bar=True)
        self.log("l1_loss", l1_loss, prog_bar=True)
        self.log("total_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)