#!/usr/bin/env python3
"""
Analyze helix to beta sheet ratios for all PDB files in the intervention sweep directory structure.
"""

import os
import argparse
import json
import re
from collections import defaultdict
import numpy as np


def parse_directory_structure(base_dir, stride_dir=None):
    """
    Parse the directory structure to extract lambda values, thresholds, and classes.
    
    Expected structure: lambda_{value}_thr_{threshold}_{class}/
    """
    results = defaultdict(lambda: defaultdict(dict))  # threshold -> class -> lambda -> ratio
    
    if not os.path.exists(base_dir):
        print(f"Error: Base directory {base_dir} does not exist")
        return results
    
    # If stride_dir is not provided, assume STRIDE files are in the same location as PDB files
    if stride_dir is None:
        stride_dir = base_dir
    
    # Walk through the directory structure
    for root, dirs, files in os.walk(base_dir):
        # Look for directories matching the pattern lambda_{value}_thr_{threshold}_{class}
        for dir_name in dirs:
            match = re.match(r'lambda_([^_]+)_thr_([^_]+)_([^_]+)$', dir_name)
            if match:
                lambda_val = float(match.group(1))
                threshold = float(match.group(2))
                class_name = match.group(3)
                
                dir_path = os.path.join(root, dir_name)
                
                # Process all PDB files in this directory
                pdb_files = [f for f in os.listdir(dir_path) if f.endswith('.pdb')]
                
                if not pdb_files:
                    print(f"Warning: No PDB files found in {dir_path}")
                    continue
                
                # Calculate helix/beta ratios for all PDB files in this directory
                helices, betas, all_residues = [], [], []
                
                for pdb_file in pdb_files:
                    # Generate corresponding stride file path
                    stride_file = pdb_file.replace('.pdb', '.stride')
                    
                    # Look for STRIDE file in the corresponding location in stride_dir
                    rel_path = os.path.relpath(dir_path, base_dir)
                    if rel_path == ".":
                        stride_path = os.path.join(stride_dir, stride_file)
                    else:
                        stride_path = os.path.join(stride_dir, rel_path, stride_file)
                    
                    # Extract helix and beta counts
                    helix_count, beta_count, total_residues = extract_helix_beta_counts_from_stride(stride_path)
                    helices.append(helix_count)
                    betas.append(beta_count)
                    all_residues.append(total_residues)
                

                results[threshold][class_name][lambda_val] = [sum(helices), sum(betas), sum(all_residues)]

                print(f"Processed files for lambda={lambda_val}, thr={threshold}, class={class_name}")

    return results


def save_results(results, output_file):
    """Save results to JSON file."""
    # Convert defaultdict to regular dict for JSON serialization
    json_results = {}
    for threshold, classes in results.items():
        json_results[str(threshold)] = {}
        for class_name, lambdas in classes.items():
            json_results[str(threshold)][class_name] = dict(lambdas)
    
    with open(output_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Analyze helix to beta sheet ratios from intervention sweep')
    parser.add_argument('--base_dir', required=True, help='Base directory containing the intervention sweep results')
    parser.add_argument('--output_file', required=True, help='Output JSON file to save results')
    parser.add_argument('--stride_dir', help='Directory containing STRIDE annotations (if different from base_dir)')
    
    args = parser.parse_args()
    
    print(f"Analyzing directory structure in: {args.base_dir}")
    if args.stride_dir:
        print(f"Looking for STRIDE files in: {args.stride_dir}")
    
    # Parse the directory structure and calculate ratios
    results = parse_directory_structure(args.base_dir, args.stride_dir)
    
    if not results:
        print("No valid data found. Exiting.")
        return
    
    # Save results
    save_results(results, args.output_file)
    
    # Print summary
    print("\nSummary:")
    for threshold in sorted(results.keys()):
        print(f"  Threshold {threshold}:")
        for class_name in sorted(results[threshold].keys()):
            lambda_values = sorted(results[threshold][class_name].keys())
            print(f"    Class {class_name}: {len(lambda_values)} lambda values")


if __name__ == "__main__":
    main()


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
