import argparse
from glob import glob

from datasets import Dataset, concatenate_datasets


def merge_all_datasets(base_dir: str, target_path: str) -> None:
    """
    Load all datasets from base directory, merge them, and save to target path.

    Args:
        base_dir: Path to the directory containing dataset subdirectories
        target_path: Path where the merged dataset will be saved
    """
    # Load datasets
    datasets = []
    paths = glob(f"{base_dir}/*")

    if not paths:
        raise ValueError(f"No datasets found in {base_dir}")

    for path in paths:
        print(f"Processing {path}")
        datasets.append(Dataset.load_from_disk(path))

    # Merge datasets
    print(f"Merging {len(datasets)} datasets...")
    merged_ds = concatenate_datasets(datasets)
    print(f"Merged dataset size: {len(merged_ds)}")

    # Save merged dataset
    print(f"Saving merged dataset to {target_path}")
    merged_ds.save_to_disk(target_path)
    print("Dataset saved successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load, merge, and save multiple datasets"
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="/home/wzarzecki/ds_10000x_normalized/structures_datasets",
        help="Base directory containing dataset subdirectories to load",
    )
    parser.add_argument(
        "--target_path",
        type=str,
        default="/home/wzarzecki/ds_10000x_normalized/structures_datasets",
        help="Target path where merged dataset will be saved",
    )

    args = parser.parse_args()
    merge_all_datasets(args.base_dir, args.target_path)
