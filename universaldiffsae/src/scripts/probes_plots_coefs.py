import os
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse


def visualize_logistic_coefficients(input_dir, output_png, label, sae_type):
    # List all files in the directory
    files = os.listdir(input_dir)

    # Filter files by label and extract timestep id
    pattern = re.compile(rf"{label}_(\d+)_coef\.npy")
    filtered_files = []
    timesteps = []
    for f in files:
        match = pattern.search(f)
        if match:
            filtered_files.append(f)
            timesteps.append(int(match.group(1)))

    # Sort files by timestep descending
    sorted_files = [
        x
        for _, x in sorted(
            zip(timesteps, filtered_files), key=lambda pair: pair, reverse=True
        )
    ]
    sorted_timesteps = sorted(timesteps, reverse=True)

    # Load coefficients from files
    coef_list = []
    for f in sorted_files:
        coef = np.load(os.path.join(input_dir, f))
        coef_list.append(coef)

    # Stack coefficients into 2D array (timesteps x coefficients)
    coef_array = np.vstack(coef_list)

    # Plotting
    plt.figure(figsize=(12, 4))  # width 3 times height
    sns.heatmap(
        np.abs(coef_array).T,
        cmap="Reds",
        cbar=True,
        xticklabels=sorted_timesteps,
        yticklabels=False,
    )
    plt.xlabel("Timestep ID (descending)")
    plt.ylabel("Coefficient ID")
    plt.title(f"Logistic Regression Coefficients for {label}, {sae_type}")
    plt.tight_layout()
    plt.savefig(output_png)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize logistic regression coefficients"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory containing coefficient files",
    )
    parser.add_argument(
        "--output_png", type=str, required=True, help="Output PNG file path"
    )
    parser.add_argument(
        "--label",
        type=str,
        required=True,
        help="Label to filter files (e.g. Cytoplasm)",
    )
    parser.add_argument(
        "--sae_type",
        type=str,
        default="pair",
    )
    args = parser.parse_args()
    visualize_logistic_coefficients(args.input_dir, args.output_png, args.label, args.sae_type)
