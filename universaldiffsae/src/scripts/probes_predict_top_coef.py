import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score


def load_dataset(dataset_path):
    """Load HuggingFace dataset from disk"""
    try:
        ds = Dataset.load_from_disk(dataset_path)
        ds.set_format("torch")

        X = ds["values"]

        columns = ds.column_names
        columns.remove("values")

        if not columns:
            raise ValueError(f"No label column found in dataset at {dataset_path}")

        label_col = columns[0]
        y = ds[label_col]

        X = X.numpy() if isinstance(X, torch.Tensor) else np.array(X)
        y = y.numpy() if isinstance(y, torch.Tensor) else np.array(y)

        return X, y, label_col

    except Exception as e:
        raise ValueError(f"Error loading dataset from {dataset_path}: {e}")


def load_model_params(coefs_dir, biases_dir):
    """Load coefficients and biases from directories"""
    coefs = {}
    biases = {}

    if os.path.exists(coefs_dir):
        for file in os.listdir(coefs_dir):
            if file.endswith("_coef.npy"):
                key = file.replace("_coef.npy", "")
                coef = np.load(os.path.join(coefs_dir, file))

                if coef.ndim == 2 and coef.shape[0] == 1:
                    coef = coef.flatten()
                coefs[key] = coef

    if os.path.exists(biases_dir):
        for file in os.listdir(biases_dir):
            if file.endswith("_bias.npy"):
                key = file.replace("_bias.npy", "")
                bias_val = np.load(os.path.join(biases_dir, file))

                if bias_val.ndim == 0:
                    biases[key] = bias_val.item()
                elif bias_val.ndim == 1 and len(bias_val) == 1:
                    biases[key] = bias_val[0]
                else:
                    biases[key] = bias_val[0] if len(bias_val) > 0 else 0.0

    return coefs, biases


def evaluate_model(X, y, coef, bias, top_ks):
    """Evaluate model with different numbers of top coefficients"""
    results = {}

    if X.shape[1] != len(coef):
        raise ValueError(
            f"Feature dimension mismatch: X has {X.shape[1]} features, coef has {len(coef)}"
        )

    abs_coef = np.abs(coef)
    sorted_indices = np.argsort(abs_coef)[::-1]

    for k in top_ks:
        if k > len(coef):
            k = len(coef)

        top_indices = sorted_indices[:k]
        X_top = X[:, top_indices]
        coef_top = coef[top_indices]

        logits = np.dot(X_top, coef_top) + bias
        preds = (logits > 0).astype(int)
        probs = 1 / (1 + np.exp(-logits))

        acc = accuracy_score(y, preds)
        try:
            auc_roc = roc_auc_score(y, probs)
        except ValueError:
            auc_roc = np.nan

        try:
            ap = average_precision_score(y, probs)
        except ValueError:
            ap = np.nan

        results[k] = {
            "accuracy": acc,
            "auc_roc": auc_roc,
            "average_precision": ap,
            "n_features": k,
        }

    logits_all = np.dot(X, coef) + bias
    preds_all = (logits_all > 0).astype(int)
    probs_all = 1 / (1 + np.exp(-logits_all))

    acc_all = accuracy_score(y, preds_all)
    try:
        auc_roc_all = roc_auc_score(y, probs_all)
    except ValueError:
        auc_roc_all = np.nan

    try:
        ap_all = average_precision_score(y, probs_all)
    except ValueError:
        ap_all = np.nan

    results["all"] = {
        "accuracy": acc_all,
        "auc_roc": auc_roc_all,
        "average_precision": ap_all,
        "n_features": len(coef),
    }

    return results


def main(coefs_dir, biases_dir, dataset_path, output_csv, top_ks):

    coefs, biases = load_model_params(coefs_dir, biases_dir)

    results_list = []

    for label_dir_name in os.listdir(dataset_path):
        label_dir_path = os.path.join(dataset_path, label_dir_name)
        if os.path.isdir(label_dir_path):
            print(f"Processing label directory: {label_dir_name}")

            for timestep_dir in os.listdir(label_dir_path):
                timestep_dir_path = os.path.join(label_dir_path, timestep_dir)
                if os.path.isdir(timestep_dir_path):
                    try:

                        X, y, label_col = load_dataset(timestep_dir_path)

                        model_key = f"{label_dir_name}_{timestep_dir}"

                        if model_key in coefs:
                            coef = coefs[model_key]

                            bias = biases.get(model_key, 0.0)

                            print(
                                f"  Processing {model_key}: X shape {X.shape}, coef shape {coef.shape}"
                            )

                            results = evaluate_model(X, y, coef, bias, top_ks)

                            for k, metrics in results.items():
                                for metric_name, value in metrics.items():
                                    results_list.append(
                                        {
                                            "label_dir": label_dir_name,
                                            "timestep": timestep_dir,
                                            "model_key": model_key,
                                            "label_column": label_col,
                                            "top_k": k,
                                            "metric": metric_name,
                                            "value": value,
                                            "n_samples": len(y),
                                            "positive_ratio": np.mean(y),
                                        }
                                    )
                        else:
                            print(f"  Model coefficients not found for {model_key}")

                    except Exception as e:
                        print(
                            f"  Error processing {label_dir_name}/{timestep_dir}: {e}"
                        )

    if results_list:
        df = pd.DataFrame(results_list)

        summary_df = df.pivot_table(
            index=[
                "label_dir",
                "timestep",
                "model_key",
                "label_column",
                "top_k",
                "n_samples",
                "positive_ratio",
            ],
            columns="metric",
            values="value",
            aggfunc="first",
        ).reset_index()

        df.to_csv(output_csv.replace(".csv", "_detailed.csv"), index=False)
        summary_df.to_csv(output_csv, index=False)

        print(
            f"Detailed results saved to {output_csv.replace('.csv', '_detailed.csv')}"
        )
        print(f"Summary results saved to {output_csv}")

        print("\n=== SUMMARY STATISTICS ===")
        for k in ["all"] + top_ks:
            k_data = summary_df[summary_df["top_k"] == k]
            if not k_data.empty:
                print(f"\nTop-{k} features:")
                print(
                    f"  Mean Accuracy: {k_data['accuracy'].mean():.4f} ± {k_data['accuracy'].std():.4f}"
                )
                print(
                    f"  Mean AUC-ROC: {k_data['auc_roc'].mean():.4f} ± {k_data['auc_roc'].std():.4f}"
                )
                print(
                    f"  Mean AP: {k_data['average_precision'].mean():.4f} ± {k_data['average_precision'].std():.4f}"
                )
    else:
        print("No results to save")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--datasets_dir", required=True)
    parser.add_argument(
        "--csv_path", type=str, default=None, help="Path to output CSV file"
    )
    parser.add_argument(
        "--coefs_dir",
        type=str,
        default=None,
        help="Directory to save model coefficients",
    )
    parser.add_argument(
        "--biases_dir", type=str, default=None, help="Directory to save model biases"
    )
    args = parser.parse_args()
    main(args.coefs_dir, args.biases_dir, args.datasets_dir, args.csv_path, [2,5,10,50,100,150,200])
