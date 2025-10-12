#!/usr/bin/env bash

set -euo pipefail

# Normalize numeric strings to a canonical format (e.g., 0.10 -> 0.1)
normalize_num() {
  awk -v x="$1" 'BEGIN{printf "%.15g\n", x}'
}

# Parse command line arguments with defaults (ordered from top to bottom)
num_designs=${1:-2}
indices_path_pair=${2:-""}
lambda_=${3:-0.0}
threshold=${4:-0.7}
first_class=${5:-"beta"}
sae_non_pair=${6:-"sae-ckpts/RFDiffSAE/patch_topk_expansion_factor16_k32_multi_topkFalse_auxk_alpha0.0lr1e-05_datawzarzeckiactivations_block4_non_pair/block4_non_pair"}
sae_pair=${7:-""}
base_dir_for_config=${8:-"RFDiffSAE/config/saeinterventions/"}
python=${9:-"/home/wzarzecki/miniforge3/envs/rf/bin/python"}
input_dir=${10:-"./temp_interventions"}
prefix=${11:-"design"}
length=${12:-150}
coef_helix=${13:-"/home/wzarzecki/ds_10000x/coefs/non_pair_helix_no_timestep/coef.npy"}
coef_beta=${14:-"/home/wzarzecki/ds_10000x/coefs/non_pair_beta_no_timestep/coef.npy"}
coefs_output_dir=${15:-"/home/wzarzecki/ds_10000x/coefs_processed"}
seed=${16:-1}

# Generate config name from other args
normalized_threshold=$(normalize_num "$threshold")
normalized_lambda=$(normalize_num "$lambda_")
output_config_name="thr_${normalized_threshold}_${first_class}_lambda_${normalized_lambda}.yaml"

# 0) generate indices for non-pair from coefficients
generated_indices_path="$coefs_output_dir/thr_${normalized_threshold}_${first_class}.pt"

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
  --threshold "$normalized_threshold" \
  --first_class "$first_class" \
  --output_path "$generated_indices_path" \
  --verbose

# 1) generate config for RFdiffusion
$python scripts/structures/interventions/generate_config_structures.py \
  --output_config_name "$output_config_name" \
  --indices_path_non_pair "$generated_indices_path" \
  --indices_path_pair "$indices_path_pair" \
  --lambda_ "$normalized_lambda" \
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
  "inference.seed=$seed" \
  "saeinterventions=$output_config_name"

# 3) run stride for evaluation
