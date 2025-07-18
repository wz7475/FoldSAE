import argparse
import os

import torch
from datasets import Dataset

from src.sae.sae import Sae


def process_main_dir(base_dir: str, sae_for_pair: Sae, sae_for_non_pair: Sae, device: torch.device):
    """
    ├── latents
        ├── non_pair
            ├── 49
                ├── xxx_0_4fa31532-5f5f-4c68-946e-cec879956d0d
                └── xxx_1_e47e7a10-3851-4b63-9149-206c673acc22
            └── 50
                ├── xxx_0_4fa31532-5f5f-4c68-946e-cec879956d0d
                └── xxx_1_e47e7a10-3851-4b63-9149-206c673acc22
        └── pair
            ├── 49
                ├── xxx_0_4fa31532-5f5f-4c68-946e-cec879956d0d
                └── xxx_1_e47e7a10-3851-4b63-9149-206c673acc22
            └── 50
                ├── xxx_0_4fa31532-5f5f-4c68-946e-cec879956d0d
                └── xxx_1_e47e7a10-3851-4b63-9149-206c673acc22
    ├── seqs
    └── structures
    └── classifiers.csv
    """
    for sae, activations_dir, latents_dir, in zip((sae_for_non_pair, sae_for_pair), ("activations/block4_non_pair", "activations/block4_pair"),("latents/non_pair", "latents/pair")):
        activations_path = os.path.join(base_dir, activations_dir)
        process_block_dir(activations_path, activations_dir, latents_dir, sae, device)


def process_block_dir(activations_dir_path: str, activations_dir_name: str, latents_dir_name: str, sae: Sae, device: torch.device) -> None:
    all_directories = os.listdir(activations_dir_path)
    for timestep_dir_name in sorted(all_directories):
        timestrep_dir = os.path.join(activations_dir_path, timestep_dir_name)
        for design_der_name in os.listdir(timestrep_dir):
            design_dir = os.path.join(timestrep_dir, design_der_name)
            activations_ds = Dataset.load_from_disk(
                design_dir, keep_in_memory=False
            )
            activations_ds.set_format("torch", columns=["values"], dtype=torch.float32)
            activations = activations_ds["values"].to(device)
            with torch.no_grad():
                sae_input, _, _ = sae.preprocess_input(activations.unsqueeze(1))
                pre_acts = sae.pre_acts(sae_input)
                top_acts, top_indices = sae.select_topk(pre_acts)
                buf = top_acts.new_zeros(top_acts.shape[:-1] + (sae.W_dec.mT.shape[-1],))
                latents = buf.scatter_(dim=-1, index=top_indices, src=top_acts)
            latents_path = design_dir.replace(activations_dir_name, latents_dir_name)
            ds = Dataset.from_dict({"values": latents})
            ds.save_to_disk(latents_path)
            print(f"saved to {latents_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--sae_pair_path", type=str, default="sae-ckpts/picked/patch_topk_expansion_factor16_k32_multi_topkFalse_auxk_alpha0.0lr0.0005_..activations_1200_block4_pair/block4_pair")
    parser.add_argument("--sae_non_pair_path", type=str, default="sae-ckpts/picked/patch_topk_expansion_factor16_k64_multi_topkFalse_auxk_alpha0.0lr0.0001_..activations_1200_block4_non_pair/block4_non_pair")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    sae_for_pair = Sae.load_from_disk(args.sae_pair_path, device=args.device).to(device)
    sae_for_non_pair = Sae.load_from_disk(args.sae_non_pair_path, device=args.device).to(device)
    process_main_dir(args.base_dir, sae_for_pair, sae_for_non_pair, device)
