import os

import wandb
from lightning import Trainer
from lightning.pytorch.loggers import CSVLogger
from pytorch_lightning.loggers import WandbLogger

from rfdiffusion.sae.sae import SAE, LitSAE
from argparse import ArgumentParser
from rfdiffusion.sae.dataloader import ActivationsDataModule
from dotenv import load_dotenv


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("activations_dir")
    parser.add_argument("--batch_size", type=int, default=20048)
    parser.add_argument("--dataset_name", default="block4_pair")
    parser.add_argument("--env_path", default="/home/wzarzecki/envs/.env_sae")
    parser.add_argument("--disable_wandb", "-d", action="store_true")
    args = parser.parse_args()
    activations_dir = args.activations_dir
    batch_size = args.batch_size
    dataset_name = args.dataset_name
    dot_env_path = args.env_path
    disable_wandb = args.disable_wandb

    model = LitSAE(SAE(128, 128*3))
    datamodule = ActivationsDataModule(activations_dir, dataset_name, batch_size)
    if not disable_wandb:
        load_dotenv(dot_env_path)
        wandb.login(key=os.environ["WANDB_TOKEN"])
        logger = WandbLogger(project="RFDiffSAE", save_dir="wandb_logs")
    else:
        logger = CSVLogger("csv_logs")

    trainer = Trainer(max_epochs=1, logger=logger, log_every_n_steps=1, accelerator="gpu", devices=1)
    trainer.fit(model, datamodule=datamodule)
    test_result = trainer.test(model, datamodule=datamodule)
    print(f"Test result: {test_result}")
