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
        "--coef_helix",
        type=str,
        default="/home/wzarzecki/ds_10000x/coefs/non_pair_helix_no_timestep/coef.npy",
        help="Path to helix coefficients .npy file",
    )

    parser.add_argument(
        "--coef_beta",
        type=str,
        default="/home/wzarzecki/ds_10000x/coefs/non_pair_beta_no_timestep/coef.npy",
        help="Path to beta coefficients .npy file",
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
        default="/home/wzarzecki/ds_secondary_struct/coefs_processed/thr_{threshold}.pt",
        help="Output path for processed coefficients (use {threshold} placeholder for automatic naming)",
    )

    parser.add_argument(
        "--use_values",
        action="store_true",
        help="Use actual coefficient values instead of signs",
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    # Check if input files exist
    if not os.path.exists(args.coef_helix):
        raise FileNotFoundError(f"Helix coefficients file not found: {args.coef_helix}")
    if not os.path.exists(args.coef_beta):
        raise FileNotFoundError(f"Beta coefficients file not found: {args.coef_beta}")

    # Format output path with threshold if placeholder is used
    output_path = args.output_path.format(threshold=args.threshold)

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if args.verbose:
        print(f"Loading helix coefficients from: {args.coef_helix}")
        print(f"Loading beta coefficients from: {args.coef_beta}")
        print(f"Using threshold: {args.threshold}")
        print(f"Using {'actual values' if args.use_values else 'signs'}")
        print(f"Output path: {output_path}")

    # Load coefficients
    helix_coefs = read_npy_coefs_to_torch(args.coef_helix)
    beta_coefs = read_npy_coefs_to_torch(args.coef_beta)

    if args.verbose:
        print(f"Helix coefficients shape: {helix_coefs.shape}")
        print(f"Beta coefficients shape: {beta_coefs.shape}")

    # Process coefficients
    use_sign = not args.use_values  # Invert because default behavior is to use signs
    indices, values = get_indices_and_values_for_class_a(
        helix_coefs, beta_coefs, args.threshold
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
