import os
import random
from uuid import uuid4

from datasets import Dataset


def save_activations_incrementally(activations_per_design, timesteps, output_dir="activation_datasets"):
    """
    Save activations for a single design incrementally. Adds random identifier not to other write design with same
    index.
    """
    os.makedirs(output_dir, exist_ok=True)
    for key in activations_per_design:
        for timestep in activations_per_design[key]: # all
            activations_per_timestep = activations_per_design[key][timestep]
            if "non" not in key:
                # pair generates seq_len more activations(seq_len x seq_len) instead of (seq_len)
                activations_per_timestep = random.sample(activations_per_timestep, int(len(activations_per_timestep) ** .5))
            ds = Dataset.from_dict({
                "values": activations_per_timestep
            })
            ds_output_dir = os.path.join(output_dir, key, f"{timestep}")
            ds.save_to_disk(os.path.join(ds_output_dir, f"{uuid4()}"))
