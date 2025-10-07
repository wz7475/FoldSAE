import argparse
import os

import torch
from datasets import Dataset, concatenate_datasets
from torch.utils.data import DataLoader

from src.sae.sae import Sae


def process_dataset_part(dataset: Dataset, sae: Sae, device: torch.device, batch_size: int):
    """Process a part of the dataset using the specified SAE model.
    
    Args:
        dataset: Dataset to process
        sae: SAE model to use for processing
        device: Device to run computations on
        batch_size: Batch size for processing
        
    Returns:
        torch.Tensor: Processed latents
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_latents = []
    
    for batch in dataloader:
        activations = batch["activations"].to(device)
        with torch.no_grad():
            sae_input, _, _ = sae.preprocess_input(activations.unsqueeze(1))
            pre_acts = sae.pre_acts(sae_input)
            top_acts, top_indices = sae.select_topk(pre_acts)
            buf = top_acts.new_zeros(top_acts.shape[:-1] + (sae.W_dec.mT.shape[-1],))
            latents = buf.scatter_(dim=-1, index=top_indices, src=top_acts)
            all_latents.append(latents.cpu())
            
    return torch.cat(all_latents, dim=0)

def process_and_save(dataset: Dataset, sae: Sae, device: torch.device, batch_size: int, output_path: str):
    """Process a dataset part and save the latents along with metadata."""
    if len(dataset) == 0:
        print(f"No data to process for {output_path}")
        return
    latents = process_dataset_part(dataset, sae, device, batch_size)
    output_dict = {k: dataset[k] for k in dataset.features.keys() if k != 'activations'}
    output_dict['latents'] = latents
    ds = Dataset.from_dict(output_dict)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ds.save_to_disk(output_path)
    print(f"Saved latents to {output_path}")


def process_dataset(dataset_path: str, output_path: str, sae_pair: Sae, sae_non_pair: Sae, device: torch.device, batch_size: int):
    """Process a dataset, splitting it by key type and applying appropriate SAE models.
    Saves one concatenated dataset.
    """
    # Load dataset
    dataset = Dataset.load_from_disk(dataset_path)
    
    # Split dataset based on key column
    non_pair_mask = [k.endswith('non_pair') for k in dataset['key']]
    pair_mask = [not k.endswith('non_pair') for k in dataset['key']]
    
    non_pair_ds = dataset.filter(lambda _, idx: non_pair_mask[idx], with_indices=True)
    pair_ds = dataset.filter(lambda _, idx: pair_mask[idx], with_indices=True)
    
    # Set format for both datasets
    for ds in [non_pair_ds, pair_ds]:
        ds.set_format("torch", columns=["activations"], dtype=torch.float32)
    
    # Process datasets and collect output dicts
    datasets_to_concat = []
    if len(non_pair_ds) > 0:
        non_pair_latents = process_dataset_part(non_pair_ds, sae_non_pair, device, batch_size)
        non_pair_dict = {k: non_pair_ds[k] for k in non_pair_ds.features.keys() if k != 'activations'}
        non_pair_dict['latents'] = non_pair_latents
        datasets_to_concat.append(Dataset.from_dict(non_pair_dict))
    if len(pair_ds) > 0:
        pair_latents = process_dataset_part(pair_ds, sae_pair, device, batch_size)
        pair_dict = {k: pair_ds[k] for k in pair_ds.features.keys() if k != 'activations'}
        pair_dict['latents'] = pair_latents
        datasets_to_concat.append(Dataset.from_dict(pair_dict))
    
    if datasets_to_concat:
        combined_ds = concatenate_datasets(datasets_to_concat)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        combined_ds.save_to_disk(output_path)
        print(f"Saved concatenated latents to {output_path}")
    else:
        print("No data to process")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--activations_path", type=str, required=True,
                      help="Path to the input dataset containing activations")
    parser.add_argument("--output_path", type=str, required=True,
                      help="Path where to save the output dataset with latents")
    parser.add_argument("--sae_pair_path", type=str, required=True,
                      help="Path to the SAE model checkpoint for pair data")
    parser.add_argument("--sae_non_pair_path", type=str, required=True,
                      help="Path to the SAE model checkpoint for non-pair data")
    parser.add_argument("--batch_size", type=int, default=1024,
                      help="Batch size for processing (default: 1024)")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda",
                      help="Device to run computations on (default: cuda)")
    
    args = parser.parse_args()
    
    # Set up device
    device = torch.device(args.device)
    
    # Load both SAE models
    sae_pair = Sae.load_from_disk(args.sae_pair_path, device=args.device).to(device)
    sae_non_pair = Sae.load_from_disk(args.sae_non_pair_path, device=args.device).to(device)
    
    # Process the dataset
    process_dataset(args.activations_path, args.output_path, sae_pair, sae_non_pair, device, args.batch_size)
