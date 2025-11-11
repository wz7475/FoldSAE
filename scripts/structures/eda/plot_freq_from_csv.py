import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main(input_csv, output_img):
    # Read CSV
    df = pd.read_csv(input_csv)
    # Define categories and check presence
    ORDER = ["E", "H", "T", "None"]
    cat_cols = [c for c in ORDER if c in df.columns]
    if not cat_cols:
        raise ValueError(f"No category columns from {ORDER} found in df.columns")
    # Ensure numeric counts and keep only target columns
    df_counts = df[cat_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    # Normalize per-file counts to [0, 1] (max=150)
    MAX_VALUE = 150
    df_counts_norm = df_counts / MAX_VALUE
    # Long format for per-file distribution
    df_counts_norm = df_counts_norm.copy()
    df_counts_norm["file"] = df.index.astype(str)
    df_long = df_counts_norm.melt(id_vars="file", value_vars=cat_cols, var_name="category", value_name="value")
    df_long["category"] = pd.Categorical(df_long["category"], categories=ORDER, ordered=True)

    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(10,5), constrained_layout=True)
    gs = fig.add_gridspec(1,2)
    # Violin + points: per-file distributions (normalized)
    ax1 = fig.add_subplot(gs[0,0])
    sns.violinplot(data=df_long, x='category', y='value', order=ORDER, palette='Set2', inner=None, cut=0, ax=ax1)
    sns.stripplot(data=df_long, x='category', y='value', order=ORDER, color='k', size=2, alpha=0.35, ax=ax1)
    ax1.set_xlabel(''); ax1.set_ylabel('per-file count (normalized)'); ax1.set_title('Distribution (violin)')
    ax1.set_ylim(0, 1)
    # Boxplot + points: range summary (normalized)
    ax2 = fig.add_subplot(gs[0,1])
    sns.boxplot(data=df_long, x='category', y='value', order=ORDER, palette='Set2', showfliers=False, ax=ax2)
    sns.stripplot(data=df_long, x='category', y='value', order=ORDER, color='k', size=2, alpha=0.25, ax=ax2)
    ax2.set_xlabel(''); ax2.set_ylabel('per-file count (normalized)'); ax2.set_title('Range (boxplot)')
    ax2.set_ylim(0, 1)
    fig.suptitle("Secondary structure kinds per generated protein")
    plt.savefig(output_img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot secondary structure frequencies from CSV.")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--output_img", type=str, required=True, help="Path to save output image")
    args = parser.parse_args()
    main(args.input_csv, args.output_img)
