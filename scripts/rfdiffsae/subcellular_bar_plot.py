from argparse import ArgumentParser

import matplotlib.pyplot as plt
import pandas as pd


def make_bar_plot(df: pd.DataFrame, target_column: str, path, k):
    df[target_column].value_counts().plot(kind='bar', title=f"k: {k}")
    plt.xticks(rotation=0)
    plt.savefig(path)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--column", default="Subcellular Localization")
    parser.add_argument("-k", "--k_param", required=True)
    args = parser.parse_args()
    path_input = args.input
    path_output = args.output

    df = pd.read_csv(path_input)
    make_bar_plot(df, target_column=args.column, path=path_output, k=args.k_param)