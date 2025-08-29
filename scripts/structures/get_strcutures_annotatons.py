import os
from time import sleep
from datasets import Dataset, disable_caching
import re

from simple_parsing import ArgumentParser


def parse_stride_file(file_path: str) -> str:
    """
    Parses a STRIDE file to extract secondary structure strings based on
    the amino acid sequence location in corresponding 'SEQ' lines.
    """
    structure_parts = []
    try:
        with open(file_path, "r") as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            if line.startswith("SEQ"):
                match = re.search(r"SEQ\s+\d+\s+(\S)", line)
                if not match:
                    continue

                start_index = match.start(1)

                match = re.search(r"(\S)\s+\d+\s*~*$", line)
                if not match:
                    continue

                end_index = match.start(1) + 1

                if i + 1 < len(lines) and lines[i + 1].startswith("STR"):
                    str_line = lines[i + 1]

                    structure_part = str_line[start_index:end_index]
                    structure_parts.append(structure_part)

    except FileNotFoundError:
        print(f"Warning: Stride file not found: {file_path}")
        return ""
    return "".join(structure_parts)


def add_secondary_struct_column(ds: Dataset, stride_dir: str) -> Dataset:
    """
    Adds a 'secondary_struct' column to the dataset by mapping amino_acid_id to
    secondary structure information from STRIDE files.
    """
    stride_cache = {}

    def get_secondary_structure(example):
        structure_id = example["structure_id"]
        amino_acid_id = example["amino_acid_id"]

        if structure_id not in stride_cache:
            stride_file_path = os.path.join(stride_dir, f"{structure_id}.stride")
            stride_cache[structure_id] = parse_stride_file(stride_file_path)

        full_ss_string = stride_cache[structure_id]

        if amino_acid_id < len(full_ss_string):
            label = full_ss_string[amino_acid_id]
            return label if label.strip() != "" else None
        return None

    return ds.map(
        lambda example: {"secondary_struct": get_secondary_structure(example)}
    )


def add_helix_and_beta_columns(ds: Dataset) -> Dataset:
    """
    Adds 'helix' and 'beta' columns to the dataset based on the 'secondary_struct' column.
    The 'helix' column is True if 'secondary_struct' is 'G', 'H', or 'I', otherwise False.
    The 'beta' column is True if 'secondary_struct' is 'E' or 'B', otherwise False.
    """
    helix_letters = {"G", "H", "I"}
    beta_letters = {"E", "B"}

    def add_helix_beta(example):
        ss = example.get("secondary_struct")
        return {
            "helix": ss is not None and ss in helix_letters,
            "beta": ss is not None and ss in beta_letters,
        }

    ds = ds.map(add_helix_beta)
    return ds


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--stride_dir")
    parser.add_argument("--input_dataset_path")
    parser.add_argument("--output_dataset_path")
    args = parser.parse_args()

    disable_caching()

    ds = Dataset.load_from_disk(args.input_dataset_path)
    ds = add_secondary_struct_column(ds, args.stride_dir)
    ds = add_helix_and_beta_columns(ds)

    ds.save_to_disk(args.output_dataset_path)