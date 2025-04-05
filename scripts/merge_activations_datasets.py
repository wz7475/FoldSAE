import os
from argparse import ArgumentParser
from rfdiffusion.activations import merge_datasets

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("dir_with_datasets")
    parser.add_argument("target_dir")
    args = parser.parse_args()
    dir_with_datasets = args.dir_with_datasets
    datasets_paths = [os.path.join(dir_with_datasets, dirname) for dirname in os.listdir(dir_with_datasets)]
    merge_datasets(datasets_paths, args.target_dir)
    print(f"merged datasets from {dir_with_datasets}")
