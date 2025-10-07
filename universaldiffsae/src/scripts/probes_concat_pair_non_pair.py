#!/usr/bin/env python


import json
import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, concatenate_datasets
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def get_train_test_split(df):

    X = np.stack(df["concat_activations"].apply(lambda x: x.flatten()).values)
    y = df["Cytoplasm"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


def train_and_eval_regression(X_train, X_test, y_train, y_test) -> float:
    clf = LogisticRegression(
        max_iter=100, solver="newton-cholesky", class_weight="balanced", random_state=42
    )
    # clf = SGDClassifier(loss="log_loss", max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    # clf.partial_fit(X_train, y_train)
    y_pred = clf.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, y_pred)


def load_ds_from_dirs_flattening_timesteps(
    path: str, columns, dtype, activations_col_name: str = "values"
) -> Dataset:
    datasets = []
    for timestep_dir_name in os.listdir(path):
        timestep_dir_path = os.path.join(path, timestep_dir_name)
        ds_dir_names = os.listdir(timestep_dir_path)
        for example_dir_name in ds_dir_names:
            example_dir_path = os.path.join(timestep_dir_path, example_dir_name)
            ds = Dataset.load_from_disk(example_dir_path, keep_in_memory=False)
            ds.set_format(type="torch", columns=columns, dtype=dtype)
            structure_id = [example_dir_name] * len(ds)
            timestep = [timestep_dir_name] * len(ds)
            ds = ds.add_column("Sequence_Id", structure_id)
            ds = ds.add_column("Timestep", timestep)
            ds = ds.rename_column("values", activations_col_name)
            datasets.append(ds)
        print(f"processed {timestep_dir_name}")
    return concatenate_datasets(datasets)


def concat_activations(row):
    return np.concat((row["activations_non_pair"], row["activations_pair"]))


def ovr_label_row_cytoplasm(row):
    return row["Subcellular Localization"] == "Cytoplasm"


def ovr_label_row_nucleus(row):
    return row["Subcellular Localization"] == "Nucleus"


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--block4_pair_path",
        type=str,
        default="/home/wzarzecki/ds_sae_latents_1600x/latents/pair",
        help="Path to block4_pair activations directory.",
    )
    parser.add_argument(
        "--classifiers_csv_path",
        type=str,
        default="/home/wzarzecki/ds_sae_latents_1600x/classifiers.csv",
        help="Path to classifiers.csv file.",
    )
    parser.add_argument(
        "--block4_non_pair_path",
        type=str,
        default="/home/wzarzecki/ds_sae_latents_1600x/latents/non_pair",
        help="Path to block4_non_pair activations directory.",
    )

    parser.add_argument(
        "--output_json",
        type=str,
        default="temp.json",
        help="Output JSON file for results.",
    )
    args = parser.parse_args()

    ds = load_ds_from_dirs_flattening_timesteps(
        args.block4_pair_path,
        ["values"],
        torch.float32,
        "activations_pair",
    )
    df = ds.to_pandas()

    labels_df = pd.read_csv(args.classifiers_csv_path)

    ds1 = load_ds_from_dirs_flattening_timesteps(
        args.block4_non_pair_path,
        ["values"],
        torch.float32,
        "activations_non_pair",
    )
    df1 = ds1.to_pandas()



    merged_df = pd.merge(df, labels_df, on=["Sequence_Id"], how="inner")
    merged_df = pd.merge(
        merged_df, df1, on=["Sequence_Id", "Timestep"], how="inner"
    )

    merged_df["concat_activations"] = merged_df.apply(
        concat_activations, axis=1
    )

    subcellular_localization_values = merged_df[
        "Subcellular Localization"
    ].unique()

    merged_df["Cytoplasm"] = merged_df.apply(ovr_label_row_cytoplasm, axis=1)
    merged_df["Nucleus"] = merged_df.apply(ovr_label_row_nucleus, axis=1)

    found_timesteps = np.unique(merged_df["Timestep"].values)
    timestep_datasets = []
    for timestep in found_timesteps:
        timestep_datasets.append(
            merged_df[merged_df["Timestep"] == str(timestep)][
                [
                    "concat_activations",
                    "Subcellular Localization",
                    "Cytoplasm",
                    "Nucleus",
                ]
            ]
        )

    X_train, X_test, y_train, y_test = get_train_test_split(timestep_datasets[0])
    train_and_eval_regression(X_train, X_test, y_train, y_test)

    results = {}
    for idx, timestep_ds in enumerate(timestep_datasets):
        X_train, X_test, y_train, y_test = get_train_test_split(timestep_ds)
        res = train_and_eval_regression(X_train, X_test, y_train, y_test)
        print(f"{idx}: {res}")
        results[idx] = res

    with open(args.output_json, "w") as f:
        json.dump(results, f)
