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
    parser.add_argument("--dir_prefixes", nargs="+", type=str, default=["2_1_50"])
    parser.add_argument("--label", type=str, default="Cytoplasm")
    parser.add_argument("--column", type=str, default="Subcellular Localization")
    args = parser.parse_args()

    # (prefix, num_blocked_neurons) -> list of (lambda, ratio)
    results = {}

    for entry in os.listdir(args.input_dir):
        entry_path = os.path.join(args.input_dir, entry)
        if not os.path.isdir(entry_path):
            continue
        # Check if entry starts with any prefix
        matched_prefix = None
        for prefix in args.dir_prefixes:
            if entry.startswith(prefix + "_") or entry == prefix:
                matched_prefix = prefix
                break
        if not matched_prefix:
            continue
        # Parse lambda (after last underscore)
        try:
            lambda_ = float(entry.split("_")[-1])
        except Exception:
            continue
        classifiers_dir = os.path.join(entry_path, "classifiers")
        if not os.path.isdir(classifiers_dir):
            continue
        for blocked_neurons in os.listdir(classifiers_dir):
            blocked_dir = os.path.join(classifiers_dir, blocked_neurons)
            if not os.path.isdir(blocked_dir):
                continue
            csv_path = os.path.join(blocked_dir, "classifiers.csv")
            if not os.path.isfile(csv_path):
                continue
            try:
                df = pd.read_csv(csv_path)
                if len(df) == 0:
                    continue
                ratio = get_label_to_other_ratio(df, args.label, args.column)
            except Exception:
                continue
            key = (matched_prefix, blocked_neurons)
            if key not in results:
                results[key] = []
            results[key].append((lambda_, ratio))

    plt.figure(figsize=(10, 7))
    all_lambdas = []
    for (prefix, blocked_neurons), points in results.items():
        if not points:
            continue
        points.sort()  # sort by lambda
        lambdas, ratios = zip(*points)
        all_lambdas.extend(lambdas)
        label = f"{prefix} | blocked={blocked_neurons}"
        plt.plot(lambdas, ratios, marker='o', label=label)
    if all_lambdas:
        import numpy as np
        min_x, max_x = min(all_lambdas), max(all_lambdas)
        xticks = np.linspace(min_x, max_x, num=10)
        plt.xticks(xticks)
    plt.xlabel("lambda")
    plt.ylabel(f"Ratio of '{args.label}' to all rows")
    plt.title(f"Label ratio across lambdas for {args.label}")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.minorticks_on()
    plt.tight_layout()
    plt.savefig(args.output_png)