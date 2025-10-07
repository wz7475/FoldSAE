#!/usr/bin/env python
import argparse
import os

import torch
from datasets import Dataset
import numpy as np
import pandas as pd

from universaldiffsae.src.tools.dataset import get_normalized_latents


def process_shard(input_path: str, output_path: str, block_names: list):
    """
    Process a dataset shard by normalizing latents for specified block names.

    Args:
        input_path: Path to input dataset shard
        output_path: Path to save processed dataset shard
        block_names: List of block names to process
    """

    ds = Dataset.load_from_disk(input_path)
    df = ds.to_pandas()

    processed_dfs = []

    for block_name in block_names:
        df_block = df[df["key"] == block_name].copy()

        if len(df_block) == 0:
            print(f"Warning: No data found for block '{block_name}'")
            continue

        latents = np.vstack(df_block["latents"].values)

        transformed_latents = get_normalized_latents(torch.from_numpy(latents).float())

        df_block["latents"] = transformed_latents.tolist()

        processed_dfs.append(df_block)
        print(f"Processed {len(df_block)} samples for block '{block_name}'")

    if not processed_dfs:
        print("Error: No data was processed")
        return

    df_processed = pd.concat(processed_dfs, ignore_index=True)

    ds_new = Dataset.from_pandas(df_processed)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    ds_new.save_to_disk(output_path)
    print(f"Saved processed dataset to {output_path}")
    print(f"Final dataset shape: {ds_new.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process dataset shards by normalizing latents"
    )
    parser.add_argument(
        "--input_shard",
        default="/home/wzarzecki/ds_10000x/very_tiny_ds/shard_0",
        help="Path to input dataset shard",
    )
    parser.add_argument(
        "--output_shard",
        default="/home/wzarzecki/ds_10000x/very_tiny_normalized_ds/shard_0",
        help="Path to save processed dataset shard",
    )
    parser.add_argument(
        "--block_names",
        nargs="+",
        default=["block4_non_pair", "block4_pair"],
        help="List of block names to process",
    )
    args = parser.parse_args()

    process_shard(args.input_shard, args.output_shard, args.block_names)
