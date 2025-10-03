import argparse
import numpy as np
import torch
from typing import Tuple
import os


def get_big_coefs_mask(
    helix: torch.Tensor, beta: torch.Tensor, threshold: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """positive mean positively correlated with first class"""
    positive_indices = torch.logical_and(helix > threshold, beta < -threshold)
    negative_indices = torch.logical_and(helix < -threshold, beta > threshold)
    return positive_indices, negative_indices


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
        "--bias_class_a",
        type=str,
        default="/home/wzarzecki/ds_10000x/coefs/non_pair_helix_no_timestep/bias.npy",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Threshold for absolute difference between coefficients",
    )

    parser.add_argument(
        "--first_class",
        type=str,
        help="name to append to config",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="/home/wzarzecki/ds_secondary_struct/coefs_processed/thr_{threshold}_{first_class}.pt",
        help="Output path for processed coefficients (use {threshold} and {first_class} placeholders for automatic naming)",
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    # Check if input files exist
    if not os.path.exists(args.coef_class_a):
        raise FileNotFoundError(
            f"First class coefficients file not found: {args.coef_class_a}"
        )
    if not os.path.exists(args.coef_class_b):
        raise FileNotFoundError(
            f"Second class coefficients file not found: {args.coef_class_b}"
        )

    output_path = args.output_path.format(threshold=args.threshold, first_class=args.first_class)

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if args.verbose:
        print(f"Loading first class coefficients from: {args.coef_class_a}")
        print(f"Loading second class coefficients from: {args.coef_class_b}")
        print(f"Using threshold: {args.threshold}")
        print(f"Output path: {output_path}")

    # Load coefficients
    coefs_for_class_a = read_npy_coefs_to_torch(args.coef_class_a)
    coefs_for_class_b = read_npy_coefs_to_torch(args.coef_class_b)

    if args.verbose:
        print(f"First class coefficients shape: {coefs_for_class_a.shape}")
        print(f"Second class coefficients shape: {coefs_for_class_b.shape}")

    positive_coefs_mask, negative_coefs_mask = get_big_coefs_mask(
        coefs_for_class_a, coefs_for_class_b, args.threshold
    )
    bias_a = torch.from_numpy(np.load(args.bias_class_a))

    if args.verbose:
        print(f"Saving results to: {output_path}")

    # Save results
    torch.save(
        (positive_coefs_mask, negative_coefs_mask, coefs_for_class_a.float(), bias_a),
        output_path,
    )


if __name__ == "__main__":
    main()
