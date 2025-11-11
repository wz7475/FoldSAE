#!/usr/bin/env python3
import argparse
import os
from typing import Tuple

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


ORDER = ["E", "H", "T", "None"]


def read_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path.lower())[1]
    if ext in (".csv", ".tsv"):
        sep = "\t" if ext == ".tsv" else ","
        return pd.read_csv(path, sep=sep)
    if ext in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if ext in (".json",):
        return pd.read_json(path)
    raise ValueError(f"Unsupported file type: {ext}")


def prepare(df: pd.DataFrame, category_col: str, value_col: str) -> pd.DataFrame:
    if category_col not in df.columns or value_col not in df.columns:
        raise KeyError(f"Missing required columns: '{category_col}', '{value_col}'")
    out = df[[category_col, value_col]].copy()
    # Normalize category labels as strings and align to expected order
    out[category_col] = out[category_col].astype(str).str.strip()
    out[category_col] = out[category_col].replace({"NONE": "None", "none": "None", "nan": "None"})
    out = out[out[category_col].isin(ORDER)]
    out[category_col] = pd.Categorical(out[category_col], categories=ORDER, ordered=True)
    # Ensure values are numeric
    out[value_col] = pd.to_numeric(out[value_col], errors="coerce")
    out = out.dropna(subset=[value_col])
    return out


def add_bar_labels(ax):
    for p in ax.patches:
        height = p.get_height()
        if height is None or height == 0:
            continue
        ax.annotate(
            f"{int(height)}",
            (p.get_x() + p.get_width() / 2.0, height),
            ha="center",
            va="bottom",
            fontsize=9,
            xytext=(0, 3),
            textcoords="offset points",
        )


def make_plot(df: pd.DataFrame, category_col: str, value_col: str, title: str = None, log_value: bool = False) -> plt.Figure:
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(11, 7), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 2])

    # Frequency bar plot (top, spans both columns)
    ax_count = fig.add_subplot(gs[0, :])
    sns.countplot(
        data=df,
        x=category_col,
        order=ORDER,
        palette="Set2",
        ax=ax_count,
    )
    ax_count.set_title("Frequency per category")
    ax_count.set_xlabel("")
    ax_count.set_ylabel("Count")
    add_bar_labels(ax_count)

    # Violin + jitter (bottom-left)
    ax_violin = fig.add_subplot(gs[1, 0])
    sns.violinplot(
        data=df,
        x=category_col,
        y=value_col,
        order=ORDER,
        palette="Set2",
        inner=None,
        cut=0,
        ax=ax_violin,
    )
    sns.stripplot(
        data=df,
        x=category_col,
        y=value_col,
        order=ORDER,
        color="k",
        size=2,
        alpha=0.35,
        ax=ax_violin,
    )
    ax_violin.set_title("Value distribution (violin + points)")
    ax_violin.set_xlabel("")
    ax_violin.set_ylabel(value_col)
    if log_value:
        ax_violin.set_yscale("log")

    # Boxplot (bottom-right)
    ax_box = fig.add_subplot(gs[1, 1])
    sns.boxplot(
        data=df,
        x=category_col,
        y=value_col,
        order=ORDER,
        palette="Set2",
        showfliers=False,
        ax=ax_box,
    )
    sns.stripplot(
        data=df,
        x=category_col,
        y=value_col,
        order=ORDER,
        color="k",
        size=2,
        alpha=0.25,
        ax=ax_box,
    )
    ax_box.set_title("Value range summary (boxplot)")
    ax_box.set_xlabel("")
    ax_box.set_ylabel(value_col)
    if log_value:
        ax_box.set_yscale("log")

    if title:
        fig.suptitle(title, y=1.02)

    return fig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualize frequency and value range for categories E, H, T, None."
    )
    p.add_argument("--input", "-i", required=True, help="Path to CSV/TSV/Parquet/JSON file.")
    p.add_argument("--category-col", "-c", default="category", help="Column with categories (E, H, T, None).")
    p.add_argument("--value-col", "-v", default="value", help="Numeric value column to plot range for.")
    p.add_argument("--output", "-o", default="ehtnone_plot.png", help="Output image path (png/pdf/svg).")
    p.add_argument("--title", "-t", default=None, help="Optional figure title.")
    p.add_argument("--log-value", action="store_true", help="Use log scale for value axis.")
    return p.parse_args()


def main():
    args = parse_args()
    df = read_table(args.input)
    df = prepare(df, args.category_col, args.value_col)
    if df.empty:
        raise SystemExit("No data after filtering to categories E, H, T, None and numeric values.")
    fig = make_plot(df, args.category_col, args.value_col, args.title, args.log_value)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
