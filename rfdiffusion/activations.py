import os
import random
from uuid import uuid4

from datasets import Dataset


def save_activations_incrementally(activations_per_design, output_dir, structure_id: str):
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
            ds.save_to_disk(os.path.join(ds_output_dir, structure_id))


def append_timestep_activations(activations_per_design: dict, timestep_activations: dict, timestep: int, keep_every_n_timestep: int = 1,
        keep_every_n_token: int = 1):
    if random.randint(0, keep_every_n_timestep - 1) == 0:
        for key in timestep_activations:
            tokens_to_save = len(timestep_activations[key]) // keep_every_n_token
            reduced_activations = random.choices(timestep_activations[key], k=tokens_to_save)
            if activations_per_design.get(key):
                activations_per_design[key][timestep] = reduced_activations
            else:
                activations_per_design[key] = {}
                activations_per_design[key][timestep] = reduced_activations
