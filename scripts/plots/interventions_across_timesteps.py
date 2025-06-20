import argparse
import os
import re

import matplotlib.pyplot as plt
import pandas as pd


def get_label_to_other_ratio(df: pd.DataFrame, label: str, column: str) -> float:
    df_label = df[df[column] == label]
    return len(df_label) / len(df) if len(df) > 0 else 0.0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_png", type=str, required=True)
    parser.add_argument("--dir_prefixes", nargs="+", type=str, default=["k_15", "k_45"])
    parser.add_argument("--label", type=str, default="Cytoplasm")
    parser.add_argument("--column", type=str, default="Subcellular Localization")
    args = parser.parse_args()


    prefix_to_points = {prefix: [] for prefix in args.dir_prefixes}

    for entry in os.listdir(args.input_dir):

        # _, num, lambda_ = entry.split("_")
        splits = entry.split("_")
        prefix = "_".join(splits[:-1])
        if prefix not in prefix_to_points:
            continue
        if entry == "80_1_30_.14":
            continue
        lambda_ = splits[-1]
        lambda_ = float(lambda_)
        csv_path = os.path.join(args.input_dir, entry, "classifiers.csv")
        df = pd.read_csv(csv_path)
        ratio = get_label_to_other_ratio(df, args.label, args.column)
        prefix_to_points[prefix].append((lambda_, ratio))

    plt.figure(figsize=(8, 6))
    all_multipliers = []
    for prefix, points in prefix_to_points.items():
        if not points:
            continue
        points.sort()  # sort by lambda_
        multipliers, ratios = zip(*points)
        all_multipliers.extend(multipliers)
        plt.plot(multipliers, ratios, marker='o', label=prefix[2:])
    # Set more x-ticks
    if all_multipliers:
        import numpy as np
        min_x, max_x = min(all_multipliers), max(all_multipliers)
        xticks = np.array([x /1 for x in range(-3, 4, 1)])
        plt.xticks(xticks)
    plt.xlabel("lambda")
    plt.ylabel(f"Ratio of '{args.label}' to all rows")
    plt.title(f"Label ratio across lambdas for {args.label}")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.minorticks_on()
    plt.tight_layout()
    plt.savefig(args.output_png)