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
import json


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
    max_iter: int = 100
):
    model = LogisticRegression(class_weight=class_weight, max_iter=max_iter)
    print("Fitting logistic regression...")
    try:
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
        print(f"avg precision: {average_precision_score(y_test, y_pred_proba):.4f}")

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

        metrics = {
            "accuracy": float(accuracy),
            "balanced_accuracy": float(balanced_accuracy),
            "roc_auc": float(roc_auc),
            "average_precision": float(average_precision_score(y_test, y_pred_proba)),
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
        }

        return model, metrics
    except ValueError as e:
        print(f"Error during model training or evaluation: {e}")
        return None, {}


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
    p.add_argument("--multiple_datasets", action="store_true", help="Load and concatenate all datasets in datasets_dir.", default=False)
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
        "--plot", action="store_true", help="Disable plotting of ROC and PR curves."
    )
    p.add_argument(
        "--pairing",
        choices=["pair", "non_pair", "concat", "loose_concat"],
        default="pair",
        help=(
            "Select rows by key suffix: 'non_pair' keeps rows where key endswith 'non_pair',"
            " 'pair' keeps the other rows. 'concat' will group by (structure_id, amino_acid_id)"
            " and concatenate their latent vectors. Requires 'structure_id','amino_acid_id'"
            " and 'latents' columns. 'loose_concat' is like 'concat' but discards any 'timestep'"
            " information and produces every possible combination of a 'non_pair' and a 'pair'"
            " entry for the same (structure_id, amino_acid_id)."
        ),
    )
    p.add_argument("--target", choices=["helix", "beta"], default="helix", help="Target helix type for OVR classification.")
    p.add_argument("--max_iter", type=int, default=100, help="Maximum iterations for logistic regression.")
    p.add_argument("--timestep", type=int, default=None, help="Timestep to filter on if applicable.")
    p.add_argument(
        "--results_json",
        type=str,
        default=None,
        help="Path to write a JSON file with evaluation metrics (optional).",
    )
    return p.parse_args()


def concat_latents_by_residue(df: pd.DataFrame) -> pd.DataFrame:
    """For each (structure_id, amino_acid_id) pair, find one row whose 'key' endswith 'non_pair'
    and one row whose 'key' does not; concatenate their 'latents' (non_pair first, then pair)
    and return a DataFrame containing one row per matched pair. Groups missing either side are
    skipped. Other metadata is taken from the 'pair' (non 'non_pair') row.
    """
    # validate required columns
    for col in ("structure_id", "amino_acid_id", "latents", "key"):
        if col not in df.columns:
            raise RuntimeError(f"Dataset must contain '{col}' column to use pairing='concat'.")

    rows = []
    grouped = df.groupby(["structure_id", "amino_acid_id"])
    for (sid, aaid), g in grouped:
        # ensure keys are strings
        keys = g["key"].astype(str)
        non_pair_idx = keys[keys.str.endswith("non_pair")].index
        pair_idx = keys[~keys.str.endswith("non_pair")].index
        # require one of each
        if len(non_pair_idx) == 0 or len(pair_idx) == 0:
            continue
        # pick the first occurrence of each
        non_pair_row = g.loc[non_pair_idx[0]]
        pair_row = g.loc[pair_idx[0]]

        latent_non = np.asarray(non_pair_row["latents"])
        latent_pair = np.asarray(pair_row["latents"])
        concat_latent = np.concatenate([latent_non, latent_pair], axis=0)

        example = pair_row.to_dict()
        example["latents"] = concat_latent
        rows.append(example)

    if not rows:
        # return empty df with same columns to avoid downstream failures
        return pd.DataFrame(columns=df.columns)
    return pd.DataFrame(rows)


def loose_concat_latents_by_residue(df: pd.DataFrame) -> pd.DataFrame:
    """For each (structure_id, amino_acid_id) pair, find all rows whose 'key' endswith
    'non_pair' and all rows whose 'key' does not; create every possible combination
    (cartesian product) of one non_pair and one pair, concatenate their 'latents'
    (non_pair first, then pair) and return a DataFrame containing one row per combination.
    The returned examples will have timestep information removed (if present).
    Groups missing either side are skipped.
    """
    # validate required columns
    for col in ("structure_id", "amino_acid_id", "latents", "key"):
        if col not in df.columns:
            raise RuntimeError(f"Dataset must contain '{col}' column to use pairing='loose_concat'.")

    rows = []
    grouped = df.groupby(["structure_id", "amino_acid_id"])
    for (sid, aaid), g in grouped:
        # ensure keys are strings
        keys = g["key"].astype(str)
        non_pair_idx = keys[keys.str.endswith("non_pair")].index
        pair_idx = keys[~keys.str.endswith("non_pair")].index
        # require at least one of each
        if len(non_pair_idx) == 0 or len(pair_idx) == 0:
            continue
        non_pairs = g.loc[non_pair_idx]
        pairs = g.loc[pair_idx]

        # produce every combination
        for _, non_row in non_pairs.iterrows():
            for _, pair_row in pairs.iterrows():
                latent_non = np.asarray(non_row["latents"])
                latent_pair = np.asarray(pair_row["latents"])
                concat_latent = np.concatenate([latent_non, latent_pair], axis=0)

                example = pair_row.to_dict()
                example["latents"] = concat_latent
                # discard timestep info as requested
                if "timestep" in example:
                    example.pop("timestep", None)
                rows.append(example)

    if not rows:
        # return empty df with same columns minus 'timestep' to avoid downstream failures
        cols = [c for c in df.columns if c != "timestep"]
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows)


def main():
    args = parse_args()

    if args.multiple_datasets:
        print(f"Loading and concatenating all datasets in {args.datasets_dir}...")
        ds = load_and_concat_datasets(args.datasets_dir)
    else:
        print(f"Loading single dataset from {args.datasets_dir}...")
        ds = Dataset.load_from_disk(args.datasets_dir)
        ds.set_format("torch")
        print(f"Loaded dataset from: {args.datasets_dir}")
    df = ds.to_pandas()

    if args.timestep is not None:
        if "timestep" not in df.columns:
            raise RuntimeError("Dataset does not contain 'timestep' column for filtering.")
        df = df[df["timestep"] == args.timestep]
        print(f"Filtered dataset to timestep={args.timestep}, new size: {len(df)}")

    # pairing handling: support 'pair', 'non_pair' (key-based) and 'concat' (group & concat latents)
    if args.pairing == "concat":
        df = concat_latents_by_residue(df)
        print(f"Concatenated latents by residue, new dataset size: {len(df)}")
    elif args.pairing == "loose_concat":
        df = loose_concat_latents_by_residue(df)
        print(f"Loose-concatenated latents by residue (all combinations, timestep removed), new dataset size: {len(df)}")
    else:
        # existing key-based filtering
        if "key" not in df.columns:
            raise RuntimeError("Dataset does not contain required 'key' column for pairing filter.")
        if args.pairing == "non_pair":
            df = df[df["key"].astype(str).str.endswith("non_pair")]
        else:
            df = df[~df["key"].astype(str).str.endswith("non_pair")]
        print(f"Filtered dataset by pairing='{args.pairing}', new size: {len(df)}")

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

    model, metrics = train_and_evaluate(X_train, y_train, X_test, y_test, plot=args.plot, max_iter=args.max_iter)

    if model is not None:
        save_coeffs(model, args.coefs_dir, args.coefs_filename, args.bias_filename)

    if args.results_json is not None and metrics:
        out_dir = os.path.dirname(args.results_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.results_json, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Wrote results JSON -> {args.results_json}")


if __name__ == "__main__":
    main()
