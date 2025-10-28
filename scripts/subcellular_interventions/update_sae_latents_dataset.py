import argparse
import os
import shutil
from typing import Tuple

import pandas as pd
from datasets import Dataset


def process_main_dir(base_dir: str):
    """
    ├── latents
        ├── non_pair
            ├── 49
                ├── xxx_0_4fa31532-5f5f-4c68-946e-cec879956d0d
                └── xxx_1_e47e7a10-3851-4b63-9149-206c673acc22
            └── 50
                ├── xxx_0_4fa31532-5f5f-4c68-946e-cec879956d0d
                └── xxx_1_e47e7a10-3851-4b63-9149-206c673acc22
        └── pair
            ├── 49
                ├── xxx_0_4fa31532-5f5f-4c68-946e-cec879956d0d
                └── xxx_1_e47e7a10-3851-4b63-9149-206c673acc22
            └── 50
                ├── xxx_0_4fa31532-5f5f-4c68-946e-cec879956d0d
                └── xxx_1_e47e7a10-3851-4b63-9149-206c673acc22
    ├── seqs
    └── structures
    └── classifiers.csv
    """
    classifiers_file = os.path.join(base_dir, "classifiers.csv")
    process_block_dir(os.path.join(base_dir, "latents", "non_pair"), classifiers_file)
    process_block_dir(os.path.join(base_dir, "latents", "pair"), classifiers_file)


def get_labels_for_structure(structure_id: str, classifiers_file: str) -> Tuple[str, str]:
    df = pd.read_csv(classifiers_file)
    subcellular = df.loc[df["Sequence_Id"] == structure_id]["Subcellular Localization"].iloc[0]
    solubility = df.loc[df["Sequence_Id"] == structure_id]["Solubility/Membrane-boundness"].iloc[0]
    return subcellular, solubility


def process_block_dir(block_dir: str, labels_file: str):
    all_directories = os.listdir(block_dir)
    for timestep_dir_name in sorted(all_directories):
        timestrep_dir = os.path.join(block_dir, timestep_dir_name)
        for design_der_name in os.listdir(timestrep_dir):
            design_dir = os.path.join(timestrep_dir, design_der_name)
            dataset = Dataset.load_from_disk(
                design_dir, keep_in_memory=False
            )
            try:
                subcellular, solubility = get_labels_for_structure(design_der_name, labels_file)
                # in case of second iteration over this dataset / script execution
                if "subcellular" not  in dataset.column_names:
                    dataset = dataset.add_column("subcellular", [subcellular] * len(dataset))
                if "solubility" not in dataset.column_names:
                    dataset = dataset.add_column("solubility", [solubility] * len(dataset))
                temp_path = f"{design_dir}_temp"
                dataset.save_to_disk(temp_path)
                shutil.rmtree(design_dir)
                os.rename(temp_path, design_dir)
            except IndexError:
                # some generations are halted at before final generation, so activation for initial stage
                # might not have associated and classified structure
                shutil.rmtree(design_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=str, required=True)
    args = parser.parse_args()
    process_main_dir(args.base_dir)
