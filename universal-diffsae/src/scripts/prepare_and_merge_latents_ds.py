import argparse
import os

import torch

from src.tools.dataset import (
    load_ds_from_dirs_flattening_timesteps,
    get_dataset_latents_target_label,
    load_ds_from_dirs_independent_timesteps
)


def build_path(base_dir: str, sae_type: str) -> str:
    print(f"base dir {base_dir}")
    return os.path.join(base_dir, "latents", sae_type)


def save_label_specific_datasets(raw_ds, feature_type, output_dir, extra_dir=None):
    for label in set(raw_ds[feature_type]):
        print(f"processing {label}")
        labels_ds = get_dataset_latents_target_label(raw_ds, feature_type, label)
        labels_ds.reset_format()
        label = label.replace("/", " ")
        path = os.path.join(output_dir, label)
        if extra_dir:
            path = os.path.join(path, extra_dir)
        labels_ds.save_to_disk(path)


def create_dataset_flattening_timesteps(input_path, output_dir, feature_type) -> None:
    raw_ds = load_ds_from_dirs_flattening_timesteps(
        input_path, [feature_type, "values"], torch.float32
    )
    save_label_specific_datasets(raw_ds, feature_type, output_dir)


def create_datasets_per_timestep(input_path, feature_type, output_dir):
    print(input_path)
    timestep_dataset = load_ds_from_dirs_independent_timesteps(
        input_path, [feature_type, "values"], torch.float32
    )
    for timestep in timestep_dataset:
        ds = timestep_dataset[timestep]
        save_label_specific_datasets(ds, feature_type, output_dir, str(timestep))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_shards_path", type=str, required=True)
    parser.add_argument("--output_datasets_dir", type=str, required=True)
    parser.add_argument("--sae_type", choices=["pair", "non_pair"], default="pair")
    parser.add_argument("--feature_type", type=str, default="subcellular")
    parser.add_argument("--ds_per_timestep", action="store_true", default=False)
    args = parser.parse_args()

    if args.ds_per_timestep:
        create_datasets_per_timestep(
            build_path(args.dataset_shards_path, args.sae_type),
            args.feature_type,
            args.output_datasets_dir,
        )
    else:
        create_dataset_flattening_timesteps(
            build_path(args.dataset_shards_path, args.sae_type),
            args.output_datasets_dir,
            args.feature_type,
        )
