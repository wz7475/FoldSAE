import argparse
import os
import csv

import torch
from datasets import Dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

def train_model(ds_path):
    ds = Dataset.load_from_disk(ds_path)
    ds.set_format("torch")
    X = ds["values"]
    columns = ds.column_names
    columns.remove("values")
    label = columns[0]
    y = ds[label]
    model = LogisticRegression(solver="newton-cholesky", fit_intercept=False)
    model.fit(X, y)
    probs = model.predict_proba(X)
    auc_roc = roc_auc_score(y, probs[:, 1])
    ap = average_precision_score(y, probs[:, 1])
    bin_count = torch.bincount(ds[label].int())
    positive_class_ratio = bin_count[1] / bin_count.sum()
    return positive_class_ratio, auc_roc, ap, model

def train_regressions_flat_timesteps(base_dir: str, csv_path: str, weights_dir: str = None, biases_dir: str = None):
    report = "REPORT\n"
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["label_dir_path", "positive_class_ratio", "auc_roc", "ap"])
        for label_dir_name in os.listdir(base_dir):
            label_dir_path = os.path.join(base_dir, label_dir_name)
            postive_class_ratio, auc, ap, model = train_model(label_dir_path)
            results = f"label {label_dir_path}; ratio {postive_class_ratio} auc_roc: {auc}; ap: {ap}\n"
            report += results
            print(results)
            writer.writerow([label_dir_path, float(postive_class_ratio), float(auc), float(ap)])
            # Save model coefficients
            if weights_dir:
                os.makedirs(weights_dir, exist_ok=True)
                weights_path = os.path.join(weights_dir, f"{label_dir_name}_coef.npy")
                import numpy as np
                np.save(weights_path, model.coef_)
            # Save model biases
            if biases_dir:
                os.makedirs(biases_dir, exist_ok=True)
                biases_path = os.path.join(biases_dir, f"{label_dir_name}_bias.npy")
                import numpy as np
                np.save(biases_path, model.intercept_)
    print(report)

def train_regressions_per_timestep(base_dir: str, csv_path: str, weights_dir: str = None, biases_dir: str = None):
    report = "REPORT\n"
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["label_dir_path", "timestep", "positive_class_ratio", "auc_roc", "ap"])
        for label_dir_name in os.listdir(base_dir):
            label_dir_path = os.path.join(base_dir, label_dir_name)
            for timestep_dir in os.listdir(label_dir_path):
                timestep_dir_path = os.path.join(label_dir_path, timestep_dir)
                postive_class_ratio, auc, ap, model = train_model(timestep_dir_path)
                results = f"label {label_dir_path}; timestep {timestep_dir}; ratio {postive_class_ratio} auc_roc: {auc}; ap: {ap}\n"
                report += results
                print(results)
                writer.writerow([label_dir_path, timestep_dir, float(postive_class_ratio), float(auc), float(ap)])
                # Save model coefficients
                if weights_dir:
                    os.makedirs(weights_dir, exist_ok=True)
                    weights_path = os.path.join(weights_dir, f"{label_dir_name}_{timestep_dir}_coef.npy")
                    import numpy as np
                    np.save(weights_path, model.coef_)
                # Save model biases
                if biases_dir:
                    os.makedirs(biases_dir, exist_ok=True)
                    biases_path = os.path.join(biases_dir, f"{label_dir_name}_{timestep_dir}_bias.npy")
                    import numpy as np
                    np.save(biases_path, model.intercept_)
    print(report)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="input dir")
    parser.add_argument("--per_timestep", action="store_true", default=False)
    parser.add_argument("--csv_path", type=str, default=None, help="Path to output CSV file")
    parser.add_argument("--weights_dir", type=str, default=None, help="Directory to save model coefficients")
    parser.add_argument("--biases_dir", type=str, default=None, help="Directory to save model biases")
    args = parser.parse_args()
    csv_path = args.csv_path if args.csv_path else os.path.join(args.input, "probes_results.csv")
    if args.per_timestep:
        train_regressions_per_timestep(args.input, csv_path, args.weights_dir, args.biases_dir)
    else:
        train_regressions_flat_timesteps(args.input, csv_path, args.weights_dir, args.biases_dir)
