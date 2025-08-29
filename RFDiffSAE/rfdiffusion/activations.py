import random


def add_meta_data_and_reduce_activations(
    activations_per_design: dict,
    timestep_activations: dict,
    timestep: int,
    keep_every_n_timestep: int = 1,
    keep_every_n_token: int = 1,
):
    if random.randint(0, keep_every_n_timestep - 1) == 0:
        for key in timestep_activations:
            # number of tokens to save (may be zero)
            num_tokens_to_save = len(timestep_activations[key]) // keep_every_n_token
            # sample indices (allowing repeats, to preserve existing behavior)
            if num_tokens_to_save <= 0:
                reduced_activations = []
            else:
                sampled_indices = random.choices(
                    range(len(timestep_activations[key])), k=num_tokens_to_save
                )
                # save amino_acid id(s) and value so downstream can record residue indices
                reduced_activations = []
                total_len = len(timestep_activations[key])
                # try to infer seq_len for pair tensors (flattened seq_len * seq_len)
                try:
                    seq_len = int(round(total_len ** 0.5))
                except Exception:
                    seq_len = None
                for i in sampled_indices:
                    if "non_pair" in key:
                        # non-pair entries correspond to per-residue features
                        amino_acid_id = int(i)
                    else:
                        # pair entries were flattened as (seq_len * seq_len, -1)
                        if seq_len and seq_len * seq_len == total_len:
                            row = i // seq_len
                            # col = i % seq_len
                            amino_acid_id = int(row)
                        else:
                            # fallback: keep flat index if we can't infer seq_len
                            amino_acid_id = int(i)
                    reduced_activations.append(
                        {"amino_acid_id": amino_acid_id, "value": timestep_activations[key][i]}
                    )
            if activations_per_design.get(key):
                activations_per_design[key][timestep] = reduced_activations
            else:
                activations_per_design[key] = {}
                activations_per_design[key][timestep] = reduced_activations
            # activations_per_design["structure_id"] = structure_id
