#!/usr/bin/env python3
"""
Generate lambda x seed tables of alpha-helix to beta-sheet ratios.

For each directory matching: lambda_{value}_thr_{threshold}_{class}/
we read per-design (seed) PDB/STRIDE pairs, compute ratio = helix_count / beta_count,
and create one CSV per (threshold, class) with:
  - rows: seeds (PDB basenames)
  - columns: lambda values
  - values: helix/beta ratio (NaN if beta_count == 0 or missing)

STRIDE files are searched in --stride_dir mirroring the structure under --base_dir.
If --stride_dir is omitted, STRIDE files are expected next to PDBs in --base_dir.
"""

import os
import argparse
import re
from collections import defaultdict
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def extract_helix_beta_counts_from_stride(stride_path_file: str) -> Tuple[int, int, int]:
    # count occurrences of G/H/I in STR lines as helix indicator and E/B as beta indicator
    ghi = 0
    be = 0
    all_residues = 0
    if not os.path.exists(stride_path_file):
        return 0, 0, 0
    with open(stride_path_file) as f:
        for line in f:
            if line.startswith("STR"):
                # Count helix residues (G, H, I)
                ghi += line.count("G")
                ghi += line.count("H")
                ghi += line.count("I")
                # Count beta sheet residues (E, B)
                be += line.count("E")
                be += line.count("B")
                # Count all residues in STR line (total structure)
                all_residues += len([c for c in line[5:] if c.isalpha()])  # Skip "STR  " prefix
            elif line.startswith("SEQ"):
                # Count all residues in SEQ line as backup
                seq_residues = len([c for c in line[5:] if c.isalpha()])  # Skip "SEQ  " prefix
                if all_residues == 0:  # Only use SEQ if STR didn't provide count
                    all_residues = seq_residues
    return ghi, be, all_residues


def collect_ratios(base_dir: str, stride_dir: str = None) -> Dict[float, Dict[str, Dict[str, Dict[float, float]]]]:
    """
    Walk the base_dir, parse directories of the form `lambda_{val}_thr_{thr}_{class}`.

    Returns nested dict:
      ratios[threshold][class_name][seed_name][lambda_val] = ratio
    """
    ratios: Dict[float, Dict[str, Dict[str, Dict[float, float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    if not os.path.exists(base_dir):
        print(f"Error: Base directory {base_dir} does not exist")
        return ratios

    for root, dirs, files in os.walk(base_dir):
        for dir_name in dirs:
            match = re.match(r"lambda_([^_]+)_thr_([^_]+)_([^_]+)$", dir_name)
            if not match:
                continue

            try:
                lambda_val = float(match.group(1))
                threshold = float(match.group(2))
            except ValueError:
                # Skip directories with non-numeric lambda/threshold
                continue
            class_name = match.group(3)

            dir_path = os.path.join(root, dir_name)
            pdb_files = [f for f in os.listdir(dir_path) if f.endswith(".pdb")]
            if not pdb_files:
                print(f"Warning: No PDB files found in {dir_path}")
                continue

            for pdb_file in pdb_files:
                seed_name = os.path.splitext(pdb_file)[0]
                stride_file = pdb_file.replace(".pdb", ".stride")

                rel_path = os.path.relpath(dir_path, base_dir)
                if rel_path == ".":
                    stride_path = os.path.join(stride_dir, stride_file)
                else:
                    stride_path = os.path.join(stride_dir, rel_path, stride_file)

                helix_count, beta_count, _ = extract_helix_beta_counts_from_stride(stride_path)
                ratio = 1.0
                if beta_count > 0:
                    ratio = helix_count / beta_count
                ratios[threshold][class_name][seed_name][lambda_val] = ratio

            print(f"Processed lambda={lambda_val}, thr={threshold}, class={class_name} ({len(pdb_files)} designs)")

    return ratios


def save_lambda_seed_tables(ratios: Dict[float, Dict[str, Dict[str, Dict[float, float]]]], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    count_saved = 0
    for threshold in sorted(ratios.keys()):
        classes = ratios[threshold]
        for class_name in sorted(classes.keys()):
            seeds_map = classes[class_name]  # seed -> {lambda: ratio}

            # Collect full set of seeds and lambdas
            all_seeds = sorted(seeds_map.keys())
            all_lambdas = sorted({lambda_key for seed in seeds_map.values() for lambda_key in seed.keys()})

            # Build DataFrame with NaNs for missing entries
            data = []
            for seed in all_seeds:
                row = [seeds_map[seed].get(lmb, np.nan) for lmb in all_lambdas]
                data.append(row)

            df = pd.DataFrame(data=data, index=all_seeds, columns=all_lambdas)

            # Create deterministic, concise filename
            thr_str = ("%g" % threshold).replace(".", "p")
            out_name = f"lambda_x_seed_thr_{thr_str}_{class_name}.csv"
            out_path = os.path.join(output_dir, out_name)
            df.to_csv(out_path, index=True)
            count_saved += 1
            print(f"Saved: {out_path}  (rows={len(all_seeds)}, cols={len(all_lambdas)})")

    if count_saved == 0:
        print("No tables were saved (no data found).")


def main():
    parser = argparse.ArgumentParser(description="Generate lambda x seed helix/beta ratio tables")
    parser.add_argument("--base_dir", required=True, help="Base directory containing intervention sweep results")
    parser.add_argument("--output_dir", required=True, help="Directory to save CSV tables")
    parser.add_argument("--stride_dir", help="Directory containing STRIDE annotations (mirrors base_dir)")

    args = parser.parse_args()

    print(f"Scanning base directory: {args.base_dir}")
    if args.stride_dir:
        print(f"Using STRIDE directory: {args.stride_dir}")

    ratios = collect_ratios(args.base_dir, args.stride_dir)
    save_lambda_seed_tables(ratios, args.output_dir)


if __name__ == "__main__":
    main()


