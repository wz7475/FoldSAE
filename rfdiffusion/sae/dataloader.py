import os
import argparse
from datasets import load_from_disk, concatenate_datasets
from torch.utils.data import DataLoader


def load_sharded_dataset_collection(base_dir):
    shards = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    datasets_by_split = {}

    for shard in shards:
        shard_path = os.path.join(base_dir, shard)
        dataset_dict = load_from_disk(shard_path)

        for split_name, dataset in dataset_dict.items():
            if split_name not in datasets_by_split:
                datasets_by_split[split_name] = []
            datasets_by_split[split_name].append(dataset)

    result = {}
    for split_name, datasets in datasets_by_split.items():
        result[split_name] = concatenate_datasets(datasets)

    return result

def get_dataloader(base_dir: str, dataset_name: str, batch_size: int):
    combined_dataset = load_sharded_dataset_collection(base_dir)
    ds_torch_pair = combined_dataset[dataset_name].with_format("torch")
    return DataLoader(ds_torch_pair, batch_size=batch_size, shuffle=True)

def main():
    parser = argparse.ArgumentParser(description="Load sharded dataset collection and create a DataLoader.")
    parser.add_argument("--base_dir", type=str, required=True, help="Base directory containing the sharded datasets.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for the DataLoader (default: 4).")
    args = parser.parse_args()

    dataloader = get_dataloader(args.base_dir, args.dataset_name, args.batch_size)
    for batch in dataloader:
        print(batch)
        break


if __name__ == "__main__":
    main()
