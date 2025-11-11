#!/usr/bin/env python3
"""
Analyze helix to beta sheet ratios for all PDB files in the intervention sweep directory structure.
"""

import glob
import argparse

import pandas as pd


def extract_helix_beta_counts_from_stride(stride_path_file: str, counter):
    # count occurrences of G/H/I in STR lines as helix indicator and E/B as beta indicator
    with open(stride_path_file) as f:
        counted_residues = 0
        for line in f:
            if line.startswith("STR"):
                for key in counter:
                    count = line.count(key)
                    counted_residues += count
                    counter[key] += count
            elif line.startswith("SEQ"):
                counter["None"] += line.count("G") - counted_residues
                counted_residues = 0
    return counter


def process_directory(dir_path: str, results_file: str) -> None:
    counters = []
    for path in glob.glob(f"{dir_path}/*"):
        counter = {"E": 0, "H": 0, "T": 0, "None": 0, "G": 0, "B": 0, "b": 0, "C": 0}
        counter = extract_helix_beta_counts_from_stride(path, counter)
        counters.append(counter)
        print(path, counter)
    df = pd.DataFrame(counters)
    df.to_csv(results_file, index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze helix to beta sheet ratios from intervention sweep"
    )
    parser.add_argument(
        "--output_file", required=True, help="Output JSON file to save results"
    )
    parser.add_argument(
        "--stride_dir",
        help="Directory containing STRIDE annotations (if different from base_dir)",
    )

    args = parser.parse_args()

    # Parse the directory structure and calculate ratios
    process_directory(args.stride_dir, args.output_file)


if __name__ == "__main__":
    main()
