#!/bin/bash
# This script generates protein structures and collects block activations using RFDiffusion.
#
# Usage:
#   ./collect_activations.sh [num_designs] [input_dir] [protein_length] [final_step] [log_file] [PYTHON_RFDIFFUSION]
#
# Parameters:
#   num_designs        Number of protein designs to generate (default: 2)
#   input_dir         Input directory path (default: /home/wzarzecki/ds_10000x_block_2)
#   protein_length    Length of the protein sequence (default: 150)
#   final_step        Final step for inference (default: 1)  
#   config_name         name of yaml config of RFdiffusion storing params of which block to store activations
#   log_file          Path to log file (default: /home/wzarzecki/logs/probes_ds.log)
#   PYTHON_RFDIFFUSION Path to Python for RFDiffusion (default: /home/wzarzecki/miniforge3/envs/rf/bin/python)
#   PYTHON_SAE Path to Python used for SAE training (default: /home/wzarzecki/miniforge3/envs/diffsae/bin/python)
#
# The script:
# 1. Sets up directory paths for PDB files and activations
# 2. Creates/clears a log file
# 3. Runs RFDiffusion inference to generate structures and collect block activations
# 4. Logs completion status
#
# Note: Script uses set -euo pipefail for strict error handling


num_designs=${1:-2}
input_dir=${2:-/home/wzarzecki/ds_10000x_block_2}
protein_length=${3:-150}
final_step=${4:-1}
config_name=${5:-block2_10_token_10th_timestep}
log_file=${14:-/home/wzarzecki/logs/activations.log}
PYTHON_RFDIFFUSION=${14:-/home/wzarzecki/miniforge3/envs/rf/bin/python}
PYTHON_SAE=${14:-/home/wzarzecki/miniforge3/envs/diffsae/bin/python}

# paths variables
pdb_dir="$input_dir/pdb"
activations_dir="$input_dir/activations"
merged_activations_dir="$input_dir/merged_activations"

set -euo pipefail

# 0) set up log
touch "$log_file";
echo "" > "$log_file";

# 1) generate structures and collect block activations
$PYTHON_RFDIFFUSION RFDiffSAE/scripts/run_inference.py \
 "inference.output_prefix=$pdb_dir/design" \
 "contigmap.contigs=[$protein_length-$protein_length]" \
 "inference.num_designs=$num_designs" \
 "inference.final_step=$final_step" \
 "inference.use_random_suffix_for_new_design=False" \
" activations=$config_name" \
 "activations.dataset_path=$activations_dir" ;
echo "generated structures and collected activations from $num_designs proteins" >> log_file;

# 2) merge them into single dataset
cmd="$PYTHON_SAE scripts/tools/merge_datasets.py \
   --base_dir $activations_dir \
   --target_path $merged_activations_dir "
 echo "$cmd"
 eval "$cmd"
 echo "merged datasets" >> $log_file;