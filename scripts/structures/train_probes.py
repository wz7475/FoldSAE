import os
from typing import Tuple

import argparse
import numpy as np
import pandas as pd
from datasets import Dataset, concatenate_datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def load_and_concat_datasets(datasets_dir: str) -> Dataset:
    """Load all datasets found in subdirectories of datasets_dir and concatenate them."""
    datasets = []
    for entry in sorted(os.listdir(datasets_dir)):
        path = os.path.join(datasets_dir, entry)
        if not os.path.isdir(path):
            continue
        try:
            ds = Dataset.load_from_disk(path)
            ds.set_format("torch")
            datasets.append(ds)
            print(f"Loaded dataset from: {path} (n={len(ds)})")
        except Exception:
            continue
    if not datasets:
        raise RuntimeError(f"No datasets loaded from {datasets_dir}")
    if len(datasets) == 1:
        return datasets[0]
    return concatenate_datasets(datasets)


def prepare_train_test(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split by unique structure_id while preserving helix stratification."""
    group_df = df[["structure_id", "helix"]].drop_duplicates("structure_id")
    train_ids, test_ids = train_test_split(
        group_df["structure_id"],
        test_size=test_size,
        stratify=group_df["helix"],
        random_state=random_state,
    )
    train_df = df[df["structure_id"].isin(train_ids)]
    test_df = df[df["structure_id"].isin(test_ids)]
    return train_df, test_df


def train_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_weight="balanced",
    plot: bool = True,
):
    model = LogisticRegression(class_weight=class_weight, max_iter=1000)
    print("Fitting logistic regression...")
    model.fit(X_train, y_train)
    print("Model fitted.")

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
    print(f"ROC AUC Score: {roc_auc:.4f}")
    print(f"AUPR: {average_precision_score(y_test, y_pred_proba):.4f}")

    if plot:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.figure()
        plt.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=2,
            label=f"ROC curve (area = {roc_auc:.2f})",
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC) Curve")
        plt.legend(loc="lower right")
        plt.show()

        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        plt.figure()
        plt.plot(recall, precision)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.show()

    return model


def save_coeffs(
    model: LogisticRegression, coefs_dir: str, coefs_filename: str, bias_filename: str
):
    os.makedirs(coefs_dir, exist_ok=True)
    coef = model.coef_
    bias = model.intercept_
    np.save(os.path.join(coefs_dir, coefs_filename), coef)
    np.save(os.path.join(coefs_dir, bias_filename), bias)
    print(f"Saved coef -> {os.path.join(coefs_dir, coefs_filename)}")
    print(f"Saved bias -> {os.path.join(coefs_dir, bias_filename)}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Train logistic regression probes on concatenated datasets."
    )
    p.add_argument(
        "--datasets_dir",
        type=str,
        default="/home/wzarzecki/ds_10000x/very_tiny_ds",
        help="Directory containing one or more dataset subdirectories to load and concatenate.",
    )
    p.add_argument(
        "--coefs_dir",
        type=str,
        default="/home/wzarzecki/ds_10000x/coefs/",
        help="Directory to save coefficients.",
    )
    p.add_argument("--coefs_filename", type=str, default="baseline_coef.npy")
    p.add_argument("--bias_filename", type=str, default="baseline_bias.npy")
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument(
        "--no_plot", action="store_true", help="Disable plotting of ROC and PR curves."
    )
    p.add_argument(
        "--pairing",
        choices=["pair", "non_pair"],
        default="pair",
        help=(
            "Select rows by key suffix: 'non_pair' keeps rows where key endswith 'non_pair',"
            " 'pair' keeps the other rows. Requires a 'key' column in the dataset."
        ),
    )
    p.add_argument("--target", choices=["helix", "beta"], default="helix", help="Target helix type for OVR classification.")
    return p.parse_args()


def main():
    args = parse_args()

    ds = load_and_concat_datasets(args.datasets_dir)
    df = ds.to_pandas()

    # Filter dataset by pairing option using the 'key' column
    if args.pairing:
        if "key" not in df.columns:
            raise RuntimeError("Dataset does not contain required 'key' column for pairing filter.")
        if args.pairing == "non_pair":
            df = df[df["key"].astype(str).str.endswith("non_pair")]
        else:
            df = df[~df["key"].astype(str).str.endswith("non_pair")]
    train_df, test_df = prepare_train_test(
        df, test_size=args.test_size, random_state=args.random_state
    )

    train_df_small = train_df.sample(frac=1, random_state=args.random_state)
    test_df_small = test_df.sample(frac=1, random_state=args.random_state)

    target_column = args.target
    X_train = np.vstack(np.array(train_df_small["latents"]))
    y_train = train_df_small[target_column].to_numpy()
    X_test = np.vstack(np.array(test_df_small["latents"]))
    y_test = test_df_small[target_column].to_numpy()

    model = train_and_evaluate(X_train, y_train, X_test, y_test, plot=not args.no_plot)

    save_coeffs(model, args.coefs_dir, args.coefs_filename, args.bias_filename)


if __name__ == "__main__":
    main()
