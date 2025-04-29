import os

from datasets import Dataset, DatasetDict, load_from_disk, concatenate_datasets
from uuid import uuid4


def save_activations_incrementally(activations_per_design, timesteps, output_dir="activation_datasets"):
    """
    Save activations for a single design incrementally. Adds random identifier not to other write design with same
    index.
    """
    os.makedirs(output_dir, exist_ok=True)

    # TODO: save activations per timestep

    num_timesteps = len(timesteps)
    for key in activations_per_design:
        activations = activations_per_design[key]
        total_num_of_activations = len(activations)
        num_of_activation_per_timestep = total_num_of_activations // num_timesteps
        for idx, activations_start_idx in enumerate(range(0, total_num_of_activations, num_of_activation_per_timestep)):
            activations_per_timestep = activations[activations_start_idx:activations_start_idx + num_of_activation_per_timestep]
            ds = Dataset.from_dict({
                "values": activations_per_timestep
            })
            ds_output_dir = os.path.join(output_dir, key, f"{timesteps[idx]}")
            ds.save_to_disk(os.path.join(ds_output_dir, f"{uuid4()}"))
