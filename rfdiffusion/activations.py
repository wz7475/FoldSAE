import os

from datasets import Dataset, DatasetDict, load_from_disk, concatenate_datasets


def save_activations_incrementally(activations_per_design, design_num, output_dir="activation_datasets"):
    """Save activations for a single design incrementally."""
    os.makedirs(output_dir, exist_ok=True)

    processed_dict = {}
    for key in activations_per_design:
        processed_dict[key] = [tensor.tolist() for tensor in activations_per_design[key]]

    design_datasets = {}
    for key in processed_dict:
        design_datasets[key] = Dataset.from_dict({
            "design_id": [design_num] * len(processed_dict[key]),
            "values": processed_dict[key]
        })

    design_dataset = DatasetDict(design_datasets)

    dataset_path = os.path.join(output_dir, f"design_{design_num}")
    design_dataset.save_to_disk(dataset_path)

    print(f"Saved activations for design {design_num}")
    return dataset_path


def merge_datasets(dataset_paths, output_path="merged_activation_dataset"):
    """Merge multiple datasets into one.
    DatasetDict({
        block4_non_pair: Dataset({
            features: ['design_id', 'values'],
            num_rows: 1243
        })
        block4_pair: Dataset({
            features: ['design_id', 'values'],
            num_rows: 140459
        })
    })
    dataset["block4_pair"]["values"].shape
    > torch.Size([140459, 128])
    train_dataset["block4_non_pair"]["values"].shape
    > torch.Size([1243, 296])
    """
    merged_dict = {}

    for path in dataset_paths:
        dataset = load_from_disk(path)
        for key in dataset:
            if key not in merged_dict:
                merged_dict[key] = dataset[key]
            else:
                merged_dict[key] = concatenate_datasets([merged_dict[key], dataset[key]])

    merged_dataset = DatasetDict(merged_dict)
    merged_dataset.save_to_disk(output_path)
    print(f"Saved merged dataset to {output_path}")
    return merged_dataset
