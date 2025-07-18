import argparse
import os

import lightning as L
import torch
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

from src.probes.probes import SimpleNetwork
from src.tools.dataset import get_dataset_latents_target_label, load_ds_from_dirs_flattening_timesteps


def get_data_loaders_using_vanilla_ds(path, test_size, feature_type, target_label, train_batch_size, test_batch_size,
                                      n_shard_per_timestep):
    raw_ds = load_ds_from_dirs_flattening_timesteps(path, columns=["values", feature_type], dtype=torch.float32,
                                                    n_shards_per_timestep=n_shard_per_timestep)
    ds = get_dataset_latents_target_label(raw_ds, feature_type, target_label)
    ds_dict = ds.train_test_split(test_size=test_size)
    train_dataloader = DataLoader(ds_dict["train"], batch_size=train_batch_size)
    test_dataloader = DataLoader(ds_dict["test"], batch_size=test_batch_size)
    return train_dataloader, test_dataloader


def main(
        base_dir,
        sae_type,
        feature_type,
        target_label,
        train_batch_size,
        test_batch_size,
        test_size,
        cuda_devices,
        wandb_project,
        max_epochs,
        log_every_n_steps,
        lr,
        n_shard_per_timestep,
):
    path = os.path.join(base_dir, "latents", sae_type)
    print(f"n_shard_per_timestep given {n_shard_per_timestep} - using vanilla Dataset")
    train_dataloader, test_dataloader = get_data_loaders_using_vanilla_ds(path, test_size, feature_type,
                                                                          target_label, train_batch_size,
                                                                          test_batch_size, n_shard_per_timestep)

    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices

    logger = WandbLogger(project=wandb_project)
    model_input_dim = next(iter(train_dataloader))["values"].shape[1]
    model = SimpleNetwork(model_input_dim, lr)
    trainer = L.Trainer(max_epochs=max_epochs, logger=logger, log_every_n_steps=log_every_n_steps)
    trainer.fit(model, train_dataloader, test_dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SimpleNetwork on SAE latents")
    parser.add_argument("--base_dir", type=str, default="/home/wzarzecki/ds_sae_latents_1600x")
    parser.add_argument("--sae_type", type=str, default="pair")
    parser.add_argument("--feature_type", type=str, default="subcellular")
    parser.add_argument("--target_label", type=str, default="Cytoplasm")
    parser.add_argument("--train_batch_size", type=int, default=4096)
    parser.add_argument("--test_batch_size", type=int, default=4096)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--cuda_devices", type=str, default="1")
    parser.add_argument("--wandb_project", type=str, default="SAE_latents")
    parser.add_argument("--max_epochs", type=int, default=5)
    parser.add_argument("--log_every_n_steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--n_shards_per_timestep", type=int, default=None)
    args = parser.parse_args()
    main(
        args.base_dir,
        args.sae_type,
        args.feature_type,
        args.target_label,
        args.train_batch_size,
        args.test_batch_size,
        args.test_size,
        args.cuda_devices,
        args.wandb_project,
        args.max_epochs,
        args.log_every_n_steps,
        args.lr,
        args.n_shards_per_timestep,
    )
