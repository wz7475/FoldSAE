import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Plot probe metrics for a given label from CSV.")
    parser.add_argument("csv_path", type=str, help="Path to CSV file")
    parser.add_argument("label", type=str, help="Label to plot")
    parser.add_argument("output_png")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    df = df[df['label_dir'] == args.label]

    # Remove 'all' from top_k for plotting, keep only numeric
    df = df[df['top_k'] != 'all']
    df['top_k'] = df['top_k'].astype(int)

    top_ks = sorted(df['top_k'].unique()) + ["all"]
    colors = plt.cm.viridis_r(np.linspace(0, 1, len(top_ks)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    metrics = ['auc_roc', 'average_precision']
    titles = ['AUC ROC', 'Average Precision']

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        for color, k in zip(colors, top_ks):
            sub = df[df['top_k'] == k]
            sub["timestep"] = sub["timestep"].astype(int)
            sub = sub.sort_values("timestep", ascending=False)
            ax.plot(
                sub['timestep'], sub[metric],
                label=f'top_k={k}', color=color
            )
        ax.set_title(titles[idx])
        ax.set_xlabel('Timestep')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.invert_xaxis()
        ax.legend()

    plt.suptitle(f"Probes for label: {args.label}")
    plt.tight_layout()
    plt.savefig(args.output_png)

if __name__ == "__main__":
    main()
