import lightning as L
import torch
import torchmetrics
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
    def __init__(self, sae: nn.Module):
        super().__init__()
        self.sae = sae

        self.metrics = {
            "mse": torchmetrics.MeanSquaredError(),
            "mae": torchmetrics.MeanAbsoluteError(),
        }

    def setup(self, stage=None):

        for metric in self.metrics.values():
            metric.to(self.device)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        input = batch
        latent, reconstructed = self.sae(input)
        l2_loss = nn.functional.mse_loss(input, reconstructed)
        l1_loss = 0.1 * torch.mean(torch.abs(latent))
        loss = l2_loss + l1_loss

        self.log("train_l2", l2_loss, prog_bar=True)
        self.log("train_l1", l1_loss, prog_bar=True)
        self.log("train_total", loss)

        return loss

    def validation_step(self, batch, batch_idx) -> None:
        input = batch
        latent, reconstructed = self.sae(input)

        for metric_name, metric in self.metrics.items():
            metric.update(input, reconstructed)

    def on_validation_epoch_end(self) -> None:

        for metric_name, metric in self.metrics.items():
            value = metric.compute()
            self.log(f"val_{metric_name}", value, prog_bar=True)
            metric.reset()

    def test_step(self, batch, batch_idx) -> None:
        input = batch
        latent, reconstructed = self.sae(input)

        for metric_name, metric in self.metrics.items():
            metric.update(input, reconstructed)

    def on_test_epoch_end(self) -> None:

        for metric_name, metric in self.metrics.items():
            value = metric.compute()
            self.log(f"test_{metric_name}", value, prog_bar=True)
            metric.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
