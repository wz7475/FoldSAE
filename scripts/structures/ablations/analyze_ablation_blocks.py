#!/usr/bin/env python3
"""
Analyze secondary structure composition per block for ablation studies.

Directory layout example (base_dir):
  base_dir/
    main_1_4/
      main_1/pdb/*.pdb
      main_2/pdb/*.pdb
      extra_-1/pdb/*.pdb
      reference/pdb/*.pdb
    reference/
      reference/pdb/*.pdb

We aggregate counts across all PDBs for each block name (e.g., "main_1", "extra_-1", "reference").
We use STRIDE annotations mirrored under stride_dir with identical relative paths and .stride extension.
"""

import os
import argparse
import json
from collections import defaultdict


def extract_helix_beta_counts_from_stride(stride_path_file: str):
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


def find_block_pdb_dirs(base_dir: str):
    """Yield tuples of (block_name, pdb_dir_path). A valid block dir contains a child 'pdb' directory."""
    for root, dirs, files in os.walk(base_dir):
        if 'pdb' in dirs:
            block_name = os.path.basename(root)
            pdb_path = os.path.join(root, 'pdb')
            yield block_name, pdb_path


def analyze_blocks(base_dir: str, stride_dir: str | None):
    # If stride_dir is not provided, assume STRIDE files reside alongside PDBs in base_dir tree
    if stride_dir is None:
        stride_dir = base_dir

    # block_name -> dict of counts
    results: dict[str, dict[str, int]] = defaultdict(lambda: {
        'helix': 0,
        'beta': 0,
        'total': 0,
        'num_pdb': 0,
    })

    for block_name, pdb_path in find_block_pdb_dirs(base_dir):
        pdb_files = [f for f in os.listdir(pdb_path) if f.endswith('.pdb')]
        if not pdb_files:
            continue

        helix_sum = 0
        beta_sum = 0
        total_sum = 0
        num_pdb = 0

        for pdb_file in pdb_files:
            abs_pdb = os.path.join(pdb_path, pdb_file)
            rel = os.path.relpath(abs_pdb, base_dir)
            stride_rel = os.path.splitext(rel)[0] + '.stride'
            stride_path = os.path.join(stride_dir, stride_rel)

            h, b, t = extract_helix_beta_counts_from_stride(stride_path)
            helix_sum += h
            beta_sum += b
            total_sum += t
            num_pdb += 1

        results[block_name]['helix'] += helix_sum
        results[block_name]['beta'] += beta_sum
        results[block_name]['total'] += total_sum
        results[block_name]['num_pdb'] += num_pdb

    # compute other = total - helix - beta (not stored in-place to avoid negatives)
    final_results: dict[str, dict[str, int]] = {}
    for block, counts in results.items():
        total = counts['total']
        helix = counts['helix']
        beta = counts['beta']
        other = max(0, total - helix - beta)
        final_results[block] = {
            'helix': int(helix),
            'beta': int(beta),
            'other': int(other),
            'total': int(total),
            'num_pdb': int(counts['num_pdb']),
        }

    return final_results


def main():
    parser = argparse.ArgumentParser(description='Analyze per-block secondary structure ratios for ablation study')
    parser.add_argument('--base_dir', required=True, help='Base directory containing ablation results with block/pdb structure')
    parser.add_argument('--output_file', required=True, help='Output JSON file to save results')
    parser.add_argument('--stride_dir', help='Directory containing STRIDE annotations mirrored to base_dir')
    args = parser.parse_args()

    print(f"Analyzing ablation blocks in: {args.base_dir}")
    if args.stride_dir:
        print(f"Looking for STRIDE files in: {args.stride_dir}")

    results = analyze_blocks(args.base_dir, args.stride_dir)
    if not results:
        print("No valid data found. Exiting.")
        return

    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {args.output_file}")
    print("\nSummary (blocks):")
    for block in sorted(results.keys()):
        r = results[block]
        print(f"  {block}: total={r['total']} (H={r['helix']}, B={r['beta']}, O={r['other']}), n={r['num_pdb']}")


if __name__ == "__main__":
    main()


