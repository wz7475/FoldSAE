#!/usr/bin/env python


import json
import os
import random
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, concatenate_datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def load_ds_from_dirs(
    path: str, columns, dtype
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
            datasets.append(ds)
        print(f"processed {timestep_dir_name}")
    return concatenate_datasets(datasets)


def get_train_test_split(df):

    X = np.stack(df["values"].apply(lambda x: x.flatten()).values)
    y = df["Cytoplasm"].values
    return train_test_split(
        X, y, test_size=0.2, random_state=42
    )


def ovr_label_row_cytoplasm(row):
    return row["Subcellular Localization"] == "Cytoplasm"


def ovr_label_row_nucleus(row):
    return row["Subcellular Localization"] == "Nucleus"


def train_and_eval_regression(X_train, X_test, y_train, y_test) -> float:
    clf = LogisticRegression(
      solver="saga", random_state=42, max_iter=1000
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, y_pred)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--activations_path", default="/home/wzarzecki/ds_sae_latents_1600x/activations/block4_non_pair")
    parser.add_argument("--labels_csv", default="/home/wzarzecki/ds_sae_latents_1600x/classifiers.csv")
    parser.add_argument("--output_json", default="latents_probes.json")
    args = parser.parse_args()
    ds = load_ds_from_dirs(
        args.activations_path,
        ["values"],
        torch.float32,
    )
    labels_df = pd.read_csv(args.labels_csv)
    df = ds.to_pandas()
    merged_df = pd.merge(df, labels_df, on="Sequence_Id", how="inner")
    merged_df["Cytoplasm"] = merged_df.apply(ovr_label_row_cytoplasm, axis=1)
    merged_df["Nucleus"] = merged_df.apply(ovr_label_row_nucleus, axis=1)

    found_timesteps = np.unique(merged_df["Timestep"].values)
    timestep_datasets = []
    for timestep in found_timesteps:
        timestep_datasets.append(
            merged_df[merged_df["Timestep"] == str(timestep)][
                ["values", "Subcellular Localization", "Cytoplasm", "Nucleus"]
            ]
        )

    print(timestep_datasets[0].shape)
    results = {}
    for idx, timestep_ds in enumerate(timestep_datasets):
        X_train, X_test, y_train, y_test = get_train_test_split(timestep_ds)
        res = train_and_eval_regression(X_train, X_test, y_train, y_test)
        print(f"{idx}: {res}")
        results[idx] = res

    with open(args.output_json, "w") as f:
        json.dump(results, f)
    print(results)
