import lightning as L
import torch
import torchmetrics
from torch import nn as nn, Tensor


class SimpleNetwork(L.LightningModule):
    def __init__(self, input_dim, lr):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(input_dim, 1), nn.Sigmoid())
        self.ap = torchmetrics.classification.BinaryAveragePrecision()
        self.acc = torchmetrics.classification.BinaryAccuracy()
        self.auc = torchmetrics.classification.BinaryAUROC()
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        x, y = batch["values"].nan_to_num(), batch["subcellular"]
        y_hat = self.model(x)
        loss = nn.functional.binary_cross_entropy(y_hat, y.unsqueeze(1))
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["values"].nan_to_num(), batch["subcellular"]
        y_hat = self.model(x)
        loss = nn.functional.binary_cross_entropy(y_hat, y.unsqueeze(1))
        self.ap.update(y_hat, y.unsqueeze(1).int())
        self.acc.update(y_hat, y.unsqueeze(1).int())
        self.auc.update(y_hat, y.unsqueeze(1).int())
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def on_validation_epoch_end(self) -> None:
        self.log("val_ap", self.ap.compute(), prog_bar=True)
        self.log("val_acc", self.acc.compute(), prog_bar=True)
        self.log("val_auc", self.auc.compute(), prog_bar=True)
        self.ap.reset()
        self.acc.reset()
        self.auc.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def get_feature_importance(self) -> Tensor:
        return self.model[0].weight
