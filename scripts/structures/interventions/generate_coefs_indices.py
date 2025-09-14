#!/usr/bin/env python3
"""
Script to process coefficients for protein secondary structure analysis.
Compares helix and beta coefficients and extracts significant differences.
input
- coefs for class a (a vs rest (including b)
- coefs for class b (b vs rest (including a)))
- threshold
output
- indices for coefs which after a - b are bigger than threshold (EXTRACT INDICES OF FEATURES WHICH FOR BOTH CLASSIFIERS
HAVE BIG MAGNITUDE BUT OPPOSITE SIGN)
- values - 1 for feature positively correlated via a and negatively correlated via b, -1 analogically
"""

import argparse
import numpy as np
import torch
from typing import Tuple
import os


def get_indices_and_values_for_class_a(
    coefs_for_class_a: torch.Tensor, coefs_for_class_b: torch.Tensor, threshold: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract indices and values where absolute difference between coefficients exceeds threshold.

    Args:
        coefs_for_class_a: Coefficients for first class (e.g., helix)
        coefs_for_class_b: Coefficients for second class (e.g., beta)
        threshold: Minimum absolute difference threshold
        use_sign_instead_of_values: If True, return signs instead of actual values

    Returns:
        Tuple of (indices, class_a_values, class_b_values)
    """
    diff = coefs_for_class_a - coefs_for_class_b
    abs_diff = torch.abs(diff)
    indices = torch.nonzero(abs_diff >= threshold)[:, 0]
    return indices, torch.sign(diff[indices])


def read_npy_coefs_to_torch(path_to_npy_file: str) -> torch.Tensor:
    """
    Load numpy coefficients file and convert to PyTorch tensor.

    Args:
        path_to_npy_file: Path to .npy file containing coefficients

    Returns:
        PyTorch tensor with coefficients
    """
    return torch.from_numpy(np.load(path_to_npy_file)[0])


def main():
    parser = argparse.ArgumentParser(
        description="Process protein secondary structure coefficients and extract significant differences",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--coef_class_a",
        type=str,
        default="/home/wzarzecki/ds_10000x/coefs/non_pair_helix_no_timestep/coef.npy",
        help="Path to first class coefficients .npy file",
    )

    parser.add_argument(
        "--coef_class_b",
        type=str,
        default="/home/wzarzecki/ds_10000x/coefs/non_pair_beta_no_timestep/coef.npy",
        help="Path to second class coefficients .npy file",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for absolute difference between coefficients",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="/home/wzarzecki/ds_secondary_struct/coefs_processed/thr_{threshold}_{first_class}.pt",
        help="Output path for processed coefficients (use {threshold} and {first_class} placeholders for automatic naming)",
    )

    parser.add_argument(
        "--first_class",
        type=str,
        default="a",
        help="First class to steer towards",
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    # Check if input files exist
    if not os.path.exists(args.coef_class_a):
        raise FileNotFoundError(f"First class coefficients file not found: {args.coef_class_a}")
    if not os.path.exists(args.coef_class_b):
        raise FileNotFoundError(f"Second class coefficients file not found: {args.coef_class_b}")

    # Format output path with threshold and first_class if placeholders are used
    output_path = args.output_path.format(threshold=args.threshold, first_class=args.first_class)

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if args.verbose:
        print(f"Loading first class coefficients from: {args.coef_class_a}")
        print(f"Loading second class coefficients from: {args.coef_class_b}")
        print(f"Using threshold: {args.threshold}")
        print(f"Steering towards first class: {args.first_class}")
        print(f"Output path: {output_path}")

    # Load coefficients
    coefs_class_a = read_npy_coefs_to_torch(args.coef_class_a)
    coefs_class_b = read_npy_coefs_to_torch(args.coef_class_b)

    if args.verbose:
        print(f"First class coefficients shape: {coefs_class_a.shape}")
        print(f"Second class coefficients shape: {coefs_class_b.shape}")

    # Process coefficients - use loaded coefficients directly
    coefs_for_class_a = coefs_class_a
    coefs_for_class_b = coefs_class_b

    indices, values = get_indices_and_values_for_class_a(
        coefs_for_class_a, coefs_for_class_b, args.threshold
    )

    if args.verbose:
        print(f"Found {len(indices)} significant differences")
        print(f"Saving results to: {output_path}")

    # Save results
    torch.save((values, indices), output_path)

    print(f"Successfully processed coefficients and saved to {output_path}")
    print(
        f"Results: {len(indices)} significant differences found with threshold {args.threshold}"
    )


if __name__ == "__main__":
    main()
