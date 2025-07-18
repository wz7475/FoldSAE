import argparse
import os

import torch

from src.tools.dataset import load_ds_from_dirs_flattening_timesteps

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets_dir", help="dir of dir with raw latents datasets")
    parser.add_argument("--output_path", help="output dir")
    args = parser.parse_args()
    input_path = args.datasets_dir
    output_path = args.output_path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)


    ds = load_ds_from_dirs_flattening_timesteps(input_path, ["values"], torch.float32)
    ds.set_format("torch")
    latents_tensor = ds["values"]
    max_tensor = torch.max(latents_tensor, dim=0).values
    torch.save(max_tensor, output_path)
