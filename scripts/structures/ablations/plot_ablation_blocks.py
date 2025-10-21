#!/usr/bin/env python3
"""
Plot stacked bar chart of secondary structure composition per block for ablations.

Input JSON format (from analyze_ablation_blocks.py):
{
  "main_1": {"helix": int, "beta": int, "other": int, "total": int, "num_pdb": int},
  ...
}
"""

import os
import argparse
import json
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_results(path: str):
    with open(path) as f:
        return json.load(f)


def _format_block_label(block_name: str) -> str:
    """Zero-pad all numeric parts to two digits for consistent sorting/labels."""
    return re.sub(r"(\d+)", lambda m: m.group(0).zfill(2), block_name)


def plot_stacked_bars(results: dict, output_dir: str, title: str | None = None):
    os.makedirs(output_dir, exist_ok=True)

    # Pair original keys with formatted labels, sort by formatted labels
    block_pairs = [(b, _format_block_label(b)) for b in results.keys()]
    block_pairs.sort(key=lambda t: t[1].lower())

    blocks = [orig for orig, _ in block_pairs]
    block_labels = [label for _, label in block_pairs]

    helix = [results[b]['helix'] for b in blocks]
    beta = [results[b]['beta'] for b in blocks]
    other = [results[b]['other'] for b in blocks]

    x = range(len(blocks))
    width = 0.8

    # Compute doubled font size based on matplotlib default font size
    base_font = matplotlib.rcParams.get('font.size', 10.0)
    big_font = float(base_font) * 2.0

    # Temporarily apply larger font sizes (titles, labels, ticks, legend, suptitle)
    with matplotlib.rc_context({
        'axes.titlesize': big_font,
        'axes.labelsize': big_font,
        'xtick.labelsize': big_font,
        'ytick.labelsize': big_font,
        'legend.fontsize': big_font,
        'figure.titlesize': big_font
    }):
        plt.figure(figsize=(max(8, len(blocks) * 0.6), 6))
        p1 = plt.bar(x, helix, width, label='Alpha helix', color='#1f77b4')
        p2 = plt.bar(x, beta, width, bottom=helix, label='Beta sheet', color='#ff7f0e')
        bottom_other = [h + b for h, b in zip(helix, beta)]
        p3 = plt.bar(x, other, width, bottom=bottom_other, label='Other', color='#2ca02c')

        plt.xticks(list(x), block_labels, rotation=45, ha='right')
        plt.ylabel('Residue count')
        plt.xlabel('Block')
        if title:
            plt.title(title)
        else:
            plt.title('Secondary structure composition per block')
        plt.legend()
        plt.tight_layout()

        out_path = os.path.join(output_dir, 'ablation_blocks_stacked_bars.png')
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"Saved plot: {out_path}")


def main():
    parser = argparse.ArgumentParser(description='Plot stacked bars for ablation blocks secondary structure composition')
    parser.add_argument('--results_file', required=True, help='Path to JSON results from analyze_ablation_blocks.py')
    parser.add_argument('--output_dir', required=True, help='Directory to save plots')
    parser.add_argument('--title', help='Optional plot title')
    args = parser.parse_args()

    results = load_results(args.results_file)
    if not results:
        print('Empty results; nothing to plot.')
        return

    plot_stacked_bars(results, args.output_dir, args.title)


if __name__ == '__main__':
    main()


