from argparse import ArgumentParser

import pandas as pd
import matplotlib.pyplot as plt


def plot_model_comparison(pair_file_path, non_pair_file_path, img_path, label_filter):
    # Load the CSV files
    pair_df = pd.read_csv(pair_file_path)
    non_pair_df = pd.read_csv(non_pair_file_path)

    # Filter by label_dir_path endswith label_filter
    if label_filter is not None:
        pair_df = pair_df[pair_df["label_dir_path"].astype(str).str.endswith(label_filter)]
        non_pair_df = non_pair_df[non_pair_df["label_dir_path"].astype(str).str.endswith(label_filter)]

    # Sort dataframes by timestep descending
    pair_df_sorted = pair_df.sort_values(by="timestep", ascending=False)
    non_pair_df_sorted = non_pair_df.sort_values(by="timestep", ascending=False)

    # Create subplots: 1 row, 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Colors
    pair_color = "tab:blue"
    non_pair_color = "tab:orange"

    # Left pane: AUC ROC
    axes[0].scatter(
        pair_df_sorted["timestep"],
        pair_df_sorted["auc_roc"],
        color=pair_color,
        label="Pair",
        s=20,  # smaller dots
        zorder=3,
    )
    axes[0].plot(
        pair_df_sorted["timestep"],
        pair_df_sorted["auc_roc"],
        color=pair_color,
        linewidth=1,
        alpha=0.7,
        zorder=2,
    )
    axes[0].scatter(
        non_pair_df_sorted["timestep"],
        non_pair_df_sorted["auc_roc"],
        color=non_pair_color,
        label="Non_pair",
        s=20,  # smaller dots
        zorder=3,
    )
    axes[0].plot(
        non_pair_df_sorted["timestep"],
        non_pair_df_sorted["auc_roc"],
        color=non_pair_color,
        linewidth=1,
        alpha=0.7,
        zorder=2,
    )
    axes[0].set_xlabel("Timestep (descending)")
    axes[0].set_ylabel("AUC ROC")
    axes[0].legend()
    axes[0].grid(True)
    axes[0].set_title("AUC ROC")
    # Set x-axis to descending
    axes[0].invert_xaxis()

    # Right pane: AP
    axes[1].scatter(
        pair_df_sorted["timestep"],
        pair_df_sorted["ap"],
        color=pair_color,
        label="Pair",
        s=20,  # smaller dots
        zorder=3,
    )
    axes[1].plot(
        pair_df_sorted["timestep"],
        pair_df_sorted["ap"],
        color=pair_color,
        linewidth=1,
        alpha=0.7,
        zorder=2,
    )
    axes[1].scatter(
        non_pair_df_sorted["timestep"],
        non_pair_df_sorted["ap"],
        color=non_pair_color,
        label="Non_pair",
        s=20,  # smaller dots
        zorder=3,
    )
    axes[1].plot(
        non_pair_df_sorted["timestep"],
        non_pair_df_sorted["ap"],
        color=non_pair_color,
        linewidth=1,
        alpha=0.7,
        zorder=2,
    )
    axes[1].set_xlabel("Timestep (descending)")
    axes[1].set_ylabel("AP")
    axes[1].legend()
    axes[1].grid(True)
    axes[1].set_title("AP")
    # Set x-axis to descending
    axes[1].invert_xaxis()

    # Set the overall figure title (not axes title)
    if label_filter is not None:
        fig.suptitle(str(label_filter))
    plt.tight_layout()
    plt.savefig(img_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--pair_csv", required=True)
    parser.add_argument("--non_pair_csv", required=True)
    parser.add_argument("--img_path", required=True)
    parser.add_argument("--label", required=True, help="Label to filter for (e.g., Cytoplams)")
    args = parser.parse_args()
    pair_file_path = args.pair_csv
    non_pair_file_path = args.non_pair_csv
    plot_model_comparison(pair_file_path, non_pair_file_path, args.img_path, args.label)
