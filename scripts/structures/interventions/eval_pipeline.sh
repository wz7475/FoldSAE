#!/bin/bash



structures_source_dir=${1:-"temp_interventions_sweep"}
target_dir_name=${2:-"pdb_and_results"}
stride_binary=${3:-"/data/wzarzecki/SAEtoRuleRFDiffusion/stride/stride"}
python=${4:-"/home/wzarzecki/miniforge3/envs/diffsae/bin/python"}



stride_dir="$target_dir_name/stride"
plot_dir="$target_dir_name/helix_beta_plots"
results_file="$target_dir_name/helix_beta_analysis_results.json"
pdb_dir="$target_dir_name/pdb"

echo "Starting evaluation pipeline..."
mkdir -p "$stride_dir"
mkdir -p "$plot_dir"
mkdir -p "$(dirname "$results_file")"


# 0) set directories
mkdir -p "$target_dir_name";

 # 1) create stride annotations in stride dir with same dir structure as input dir
echo "Step 1: Creating STRIDE annotations..."
$python scripts/structures/utils/run_stride.py \
  --pdb_dir "$structures_source_dir" \
  --stride_dir "$stride_dir" \
  --stride_binary "$stride_binary"

# 2) analyze helix beta ratios for all combinations
echo "Step 2: Analyzing helix to beta sheet ratios..."
$python scripts/structures/interventions/analyze_helix_beta_ratios.py \
  --base_dir "$structures_source_dir" \
  --stride_dir "$stride_dir" \
  --output_file "$results_file"

# 3) generate plots
echo "Step 3: Generating plots..."
$python scripts/structures/interventions/plot_helix_beta_ratios.py \
  --results_file "$results_file" \
  --output_dir "$plot_dir" \
  --summary


# 4) copy pdbs
cp -r "$structures_source_dir" "$pdb_dir"

echo "Evaluation pipeline finished."