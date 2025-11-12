#!/usr/bin/env python3
"""
Analyze helix to beta sheet ratios for all PDB files in the intervention sweep directory structure.
"""

import glob
import os

import pandas as pd
from scipy.stats import norm


def extract_helix_beta_counts_from_stride(stride_path_file: str):
    be_counter = 0
    with open(stride_path_file) as f:
        for line in f:
            if line.startswith("STR"):
                be_counter += line.count("B")
                be_counter += line.count("E")
                be_counter += line.count("b")
    design_id = stride_path_file.split("_")[-1].split(".stride")[0]
    return design_id, be_counter


def get_betas_counts_df(stride_dir: str) -> pd.DataFrame:
    ids, betas = [], []
    for path in sorted(glob.glob(f"{stride_dir}/*")):
        design_ids, helix_count = extract_helix_beta_counts_from_stride(path)
        ids.append(design_ids)
        betas.append(helix_count)
    df = pd.DataFrame({"ids": ids, "betas": betas})
    return df


def read_sequences_from_dir(
    ids: list[int] | None, fasta_dir: str, prefix: str = "design_", ext: str = "fa"
) -> list[str]:
    seqs = []
    if ids is None:
        for fasta_path in glob.glob(f"{fasta_dir}/*"):
            with open(fasta_path) as f:
                seqs.append(f.readlines()[1].strip())
    else:
        for id_ in ids:
            fasta_path = os.path.join(fasta_dir, f"{prefix}{id_}.{ext}")
            with open(fasta_path) as f:
                seqs.append(f.readlines()[1].strip())
    return seqs


def get_ref_seqs(val_stride, ref_seqs, ref_stride, n_ref_seqs):
    val_df = get_betas_counts_df(val_stride)
    val_mean, val_sigma = val_df["betas"].mean(), val_df["betas"].std()

    ref_df = get_betas_counts_df(ref_stride)
    ref_df["weights"] = norm.pdf(ref_df["betas"], val_mean, val_sigma)

    sampled_ref_df = ref_df.sample(
        n=n_ref_seqs, weights="weights", replace=True, random_state=42
    )
    return read_sequences_from_dir(sampled_ref_df["ids"], ref_seqs)
