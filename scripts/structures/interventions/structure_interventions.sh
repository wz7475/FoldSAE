#!/usr/bin/env bash

set -euo pipefail

# Parse command line arguments with defaults
output_config_name=${1:-"test_struct.yaml"}
indices_path_non_pair=${2:-"/home/wzarzecki/ds_secondary_struct/coefs/baseline_top_100.pt"}
indices_path_pair=${3:-""}
lambda_=${4:-0.0}
sae_non_pair=${5:-"sae-ckpts/RFDiffSAE/patch_topk_expansion_factor16_k32_multi_topkFalse_auxk_alpha0.0lr1e-05_datawzarzeckiactivations_block4_non_pair/block4_non_pair"}
sae_pair=${6:-""}
base_dir_for_config=${7:-"RFDiffSAE/config/saeinterventions/"}

# RFDiffusion arguments
python=${8:-"/home/wzarzecki/miniforge3/envs/rf/bin/python"}
input_dir=${9:-"./temp_interventions"}
prefix=${10:-"design"}
length=${11:-150}
num_designs=${1:-2}

# Coefficient processing arguments
threshold=${13:-0.7}
first_class=${14:-"beta"}
coef_helix=${15:-"/home/wzarzecki/ds_10000x/coefs/non_pair_helix_no_timestep/coef.npy"}
coef_beta=${16:-"/home/wzarzecki/ds_10000x/coefs/non_pair_beta_no_timestep/coef.npy"}
coefs_output_dir=${17:-"/home/wzarzecki/ds_10000x/coefs_processed"}

# 0) generate indices for non-pair from coefficients
generated_indices_path="$coefs_output_dir/thr_${threshold}_${first_class}.pt"

# Determine coefficient order based on first class choice
if [ "$first_class" = "helix" ]; then
  coef_class_a="$coef_helix"
  coef_class_b="$coef_beta"
else  # beta
  coef_class_a="$coef_beta"
  coef_class_b="$coef_helix"
fi

$python scripts/structures/interventions/generate_coefs_indices.py \
  --coef_class_a "$coef_class_a" \
  --coef_class_b "$coef_class_b" \
  --threshold "$threshold" \
  --first_class "$first_class" \
  --output_path "$coefs_output_dir/thr_{threshold}_{first_class}.pt" \
  --verbose

# 1) generate config for RFdiffusion
$python scripts/structures/interventions/generate_config_structures.py \
  --output_config_name "$output_config_name" \
  --indices_path_non_pair "$generated_indices_path" \
  --indices_path_pair "$indices_path_pair" \
  --lambda_ "$lambda_" \
  --sae_non_pair "$sae_non_pair" \
  --sae_pair "$sae_pair" \
  --base_dir_for_config "$base_dir_for_config"

# 2) run RFdiffusion
$python RFDiffSAE/scripts/run_inference.py \
  "inference.output_prefix=$input_dir/$prefix" \
  "contigmap.contigs=[$length-$length]" \
  "inference.num_designs=$num_designs" \
  "inference.final_step=1" \
  "inference.use_random_suffix_for_new_design=False" \
  "inference.seed=1" \
  "saeinterventions=$output_config_name"

# 3) run stride for evaluation
