#!/bin/bash

input_dir=${1:-/home/wzarzecki/ds_10000x_block_2}
merged_activations_dir=${2:-/home/wzarzecki/ds_10000x_block_2/merged_activations}
log_file=${3:-/home/wzarzecki/logs/probes_ds.log}
SAE_PAIR_PATH=${4:-sae-ckpts/picked/patch_topk_expansion_factor32_k128_multi_topkFalse_auxk_alpha0.0lr0.005_homewzarzeckids_10000x_block_2merged_activations_block2_pair/block2_pair}
SAE_NON_PAIR_PATH=${5:-sae-ckpts/picked/patch_topk_expansion_factor32_k32_multi_topkFalse_auxk_alpha0.0lr0.005_homewzarzeckids_10000x_block_2merged_activations_block2_non_pair/block2_non_pair}
stride_binary=${6:-/data/wzarzecki/SAEtoRuleRFDiffusion/stride/stride}
PYTHON_SAE=${7:-/home/wzarzecki/miniforge3/envs/diffsae/bin/python}

# paths variables
pdb_dir="$input_dir/pdb"
latents_path="$input_dir/latents"
stride_dir="$input_dir/stride"
merged_datasets_dir="$input_dir/structures_ds_merged"

set -euo pipefail

# 0) set up log
touch "$log_file";
echo "" > "$log_file";

# 1) generate latents
cmd="$PYTHON_SAE universaldiffsae/src/scripts/sae_latents_from_activations.py \
  --activations_path \"$merged_activations_dir\" \
  --output_path \"$latents_path\" \
  --sae_pair_path \"$SAE_PAIR_PATH\" \
  --sae_non_pair_path \"$SAE_NON_PAIR_PATH\" \
  --batch_size \"${BATCH_SIZE:-4096}\" \
  --device \"${DEVICE:-cuda}\""
echo "$cmd"
eval $cmd
echo "generated latents" >> "$log_file";

# 2) run stride annotations
cmd="$PYTHON_SAE scripts/structures/utils/run_stride.py \
--pdb_dir \"$pdb_dir\" \
--stride_dir \"$stride_dir\" \
--stride_binary \"$stride_binary\""
echo "$cmd"
eval $cmd
echo "generated stride annotations" >> "$log_file";

# 3) add stride column to datasetsw
cmd="$PYTHON_SAE scripts/structures/create_ds/get_strcutures_annotatons.py \
  --stride_dir \"$stride_dir\" \
  --input_dataset_path \"$latents_path\" \
  --output_dataset_path \"$merged_datasets_dir\""
echo "$cmd"
eval $cmd
echo "genertated datasets with updates" >> "$log_file";
