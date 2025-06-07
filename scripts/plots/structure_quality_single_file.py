import argparse
import pandas as pd
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Create box plot for rmds and tm from CSV.")
    parser.add_argument("input_csv", help="Path to input CSV file")
    parser.add_argument("output_png", help="Path to output PNG file")
    parser.add_argument("--title", default="Box Plot of rmsd and tm", help="Title of the plot")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(args.title)

    axes[0].boxplot(df["rmsd"].dropna())
    axes[0].set_title("rmsd")
    axes[0].set_ylabel("Value")

    axes[1].boxplot(df["tm"].dropna())
    axes[1].set_title("tm")
    axes[1].set_ylabel("Value")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(args.output_png)
    plt.close()

if __name__ == "__main__":
    main()
