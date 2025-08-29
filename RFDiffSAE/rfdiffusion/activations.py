import random


def add_meta_data_and_reduce_activations(

    activations_per_design: dict,
    timestep_activations: dict,
    timestep: int,
    keep_every_n_timestep: int = 1,
    keep_every_n_token: int = 1,
):
    """
    Processes and reduces activations for a given timestep, adding metadata and subsampling tokens and timesteps.

    This function selectively processes activations for a given timestep, reducing the number of tokens and timesteps
    saved according to the specified sampling intervals. For each activation key, it samples a subset of tokens,
    adds metadata (such as amino acid ID), and stores the reduced activations in the provided activations_per_design dictionary.

    Args:
        activations_per_design (dict): Dictionary to store reduced activations per design, keyed by activation name and timestep.
        timestep_activations (dict): Dictionary containing activations for the current timestep, keyed by activation name.
        timestep (int): The current timestep index.
        keep_every_n_timestep (int, optional): Interval for keeping timesteps. Only process if the current timestep is selected. Defaults to 1.
        keep_every_n_token (int, optional): Interval for keeping tokens. Only a subset of tokens is sampled and saved. Defaults to 1.

    Notes:
        - The function uses random sampling to select which timesteps and tokens to keep.
        - For each activation, the function determines whether the activation is pairwise or single-token and processes accordingly.
        - The reduced activations are stored with metadata including the amino acid ID and the activation value.
    """
    if random.randint(0, keep_every_n_timestep - 1) == 0:
        seq_len = None
        for key in timestep_activations:
            if key.endswith("non_pair"):
                seq_len = len(timestep_activations[key])
        print(f"seq_len: {seq_len}")
        for key in timestep_activations:
            base_tokens_to_save = seq_len // keep_every_n_token
            sampled_indices = random.choices(range(seq_len), k=base_tokens_to_save)
            reduced_activations = []
            for i in sampled_indices:
                if seq_len * seq_len == len(timestep_activations[key]):
                    row = i // seq_len
                    amino_acid_id = int(row)
                else:
                    amino_acid_id = int(i)
                reduced_activations.append(
                    {
                        "amino_acid_id": amino_acid_id,
                        "value": timestep_activations[key][i],
                    }
                )
            if activations_per_design.get(key):
                activations_per_design[key][timestep] = reduced_activations
            else:
                activations_per_design[key] = {}
                activations_per_design[key][timestep] = reduced_activations


def save_activations_shard(
    activations_for_all_designs: dict,
    dataset_path: str,
    shard_idx: int = 0,
):
    """
    Save activations for a batch of designs as a dataset shard to disk.
    Each call creates a new subfolder/shard in dataset_path.
    """
    from datasets import Dataset
    import os

    all_records = []
    for structure_id, activations_per_design in activations_for_all_designs.items():
        for key in activations_per_design:
            for timestep in activations_per_design[key]:
                activations_per_timestep = activations_per_design[key][timestep]
                for idx, item in enumerate(activations_per_timestep):
                    if isinstance(item, dict) and "amino_acid_id" in item:
                        amino_acid_id = item.get("amino_acid_id")
                        value = item.get("value")
                    else:
                        amino_acid_id = None
                        value = item
                    all_records.append(
                        {
                            "structure_id": structure_id,
                            "key": key,
                            "timestep": timestep,
                            "idx": idx,
                            "amino_acid_id": amino_acid_id,
                            "activations": value,
                        }
                    )
    if all_records:
        ds = Dataset.from_list(all_records)
        os.makedirs(dataset_path, exist_ok=True)
        shard_path = os.path.join(dataset_path, f"shard_{shard_idx}")
        ds.save_to_disk(shard_path)
