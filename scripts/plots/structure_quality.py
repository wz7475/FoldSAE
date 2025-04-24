import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_with_files", default="ablation_test", help="Directory to search for files")
    parser.add_argument("--filename", default="results.csv", help="Name of files to search for")
    parser.add_argument("--rmsd_path", default="rmsd_boxplot_comparison.png", help="Output path for RMSD boxplot")
    parser.add_argument("--tm_path", default="tm_boxplot_comparison.png", help="Output path for TM score boxplot")
    args = parser.parse_args()

    input_dir = args.dir_with_files
    filename = args.filename

    # Find all matching files
    matching_files = glob.glob(f"{input_dir}/**/{filename}", recursive=True)
    print(f"Found {len(matching_files)} matching files")

    # Read data from each file separately and prepare for boxplot comparison
    rmsd_data = []
    tm_data = []
    labels = []

    # Process each file
    for file_path in matching_files:
        print(f"Processing: {file_path}")
        try:
            df = pd.read_csv(file_path)
            # Use directory name as the label for this boxplot
            block_name = os.path.basename(os.path.dirname(file_path))

            # Check if required columns exist
            if 'rmsd' in df.columns and 'tm' in df.columns:
                # Filter out NaN values
                rmsd_values = df['rmsd'].dropna().tolist()
                tm_values = df['tm'].dropna().tolist()

                # Only add if we have data
                if len(rmsd_values) > 0 and len(tm_values) > 0:
                    labels.append(block_name)
                    rmsd_data.append(rmsd_values)
                    tm_data.append(tm_values)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    # Ensure we have data to plot
    if not labels:
        print("No valid data found. Check if files exist and contain required columns.")
        exit(1)

    # Output the found files and data sizes for debugging
    for i, label in enumerate(labels):
        print(f"{label}: RMSD samples={len(rmsd_data[i])}, TM samples={len(tm_data[i])}")

    # sort labels
    labels, rmsd_data, tm_data = zip(*sorted(zip(labels, rmsd_data, tm_data)))
    # Create figure for RMSD boxplot
    plt.figure(figsize=(max(8, len(labels) * 1.2), 6))
    box = plt.boxplot(rmsd_data, patch_artist=True, notch=True)

    # Customize appearance
    for patch in box['boxes']:
        patch.set_facecolor('lightblue')

    plt.title('Comparison of RMSD Scores Across Files')
    plt.ylabel('RMSD')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Set x-axis tick labels to the file/directory names
    plt.xticks(range(1, len(labels) + 1), labels, rotation=45, ha='right')
    plt.tight_layout()

    # Add statistics for reference
    rmsd_means = [np.mean(data) for data in rmsd_data]
    plt.axhline(y=np.mean(rmsd_means), color='r', linestyle='-', alpha=0.3, label='Overall Mean')

    plt.savefig(args.rmsd_path, dpi=300)
    print(f"Saved RMSD boxplot to {args.rmsd_path}")

    # Create figure for TM score boxplot
    plt.figure(figsize=(max(8, len(labels) * 1.2), 6))
    box = plt.boxplot(tm_data, patch_artist=True, notch=True)

    # Customize appearance
    for patch in box['boxes']:
        patch.set_facecolor('lightgreen')

    plt.title('Comparison of TM Scores Across Files')
    plt.ylabel('TM score')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Set x-axis tick labels to the file/directory names
    plt.xticks(range(1, len(labels) + 1), labels, rotation=45, ha='right')
    plt.tight_layout()

    # Add statistics for reference
    tm_means = [np.mean(data) for data in tm_data]
    plt.axhline(y=np.mean(tm_means), color='r', linestyle='-', alpha=0.3, label='Overall Mean')

    plt.savefig(args.tm_path, dpi=300)
    print(f"Saved TM score boxplot to {args.tm_path}")

    # Print summary statistics
    print("\nRMSD Statistics by file:")
    for i, label in enumerate(labels):
        print(f"{label}: Mean={np.mean(rmsd_data[i]):.4f}, Median={np.median(rmsd_data[i]):.4f}")

    print("\nTM Score Statistics by file:")
    for i, label in enumerate(labels):
        print(f"{label}: Mean={np.mean(tm_data[i]):.4f}, Median={np.median(tm_data[i]):.4f}")
