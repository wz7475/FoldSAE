#!/usr/bin/env bash

set -euo pipefail

# Grid search parameters
lambda_start=${1:-0.0}
lambda_stop=${2:-1.0}
lambda_step=${3:-2}
threshold_start=${4:-0.3}
threshold_stop=${5:-0.6}
threshold_step=${6:-0.3}
first_classes=${7:-"helix beta"}

# Fixed parameters for all runs
input_dir=${8:-"./temp_interventions_sweep"}
num_designs=${9:-1}

# Additional parameters (rarely changed)
indices_path_pair=${10:-""}
sae_non_pair=${11:-"sae-ckpts/RFDiffSAE/patch_topk_expansion_factor16_k32_multi_topkFalse_auxk_alpha0.0lr1e-05_datawzarzeckiactivations_block4_non_pair/block4_non_pair"}
sae_pair=${12:-""}
base_dir_for_config=${13:-"RFDiffSAE/config/saeinterventions/"}
python=${14:-"/home/wzarzecki/miniforge3/envs/rf/bin/python"}
prefix=${15:-"design"}
length=${16:-150}
coef_helix=${17:-"/home/wzarzecki/ds_10000x/coefs/non_pair_helix_no_timestep/coef.npy"}
coef_beta=${18:-"/home/wzarzecki/ds_10000x/coefs/non_pair_beta_no_timestep/coef.npy"}
coefs_output_dir=${19:-"/home/wzarzecki/ds_10000x/coefs_processed"}

# Create main output directory
mkdir -p "$input_dir"

# Log file for the sweep
log_file="$input_dir/sweep_log.txt"
echo "Starting structure interventions sweep at $(date)" > "$log_file"
echo "Parameters:" >> "$log_file"
echo "  Lambda range: $lambda_start to $lambda_stop (step $lambda_step)" >> "$log_file"
echo "  Threshold range: $threshold_start to $threshold_stop (step $threshold_step)" >> "$log_file"
echo "  First classes: $first_classes" >> "$log_file"
echo "  Input dir: $input_dir" >> "$log_file"
echo "  Num designs: $num_designs" >> "$log_file"
echo "" >> "$log_file"

# Counter for total runs
total_runs=0
completed_runs=0

# Calculate total number of runs
lambda_count=$(python3 -c "import math; print(int((($lambda_stop - $lambda_start) / $lambda_step) + 1))")
threshold_count=$(python3 -c "import math; print(int((($threshold_stop - $threshold_start) / $threshold_step) + 1))")
class_count=$(echo "$first_classes" | wc -w)
total_runs=$((lambda_count * threshold_count * class_count))

echo "Total runs planned: $total_runs"
echo "Total runs planned: $total_runs" >> "$log_file"

# Function to run a single experiment
run_experiment() {
    local lambda=$1
    local threshold=$2
    local first_class=$3
    
    echo "Running experiment: lambda=$lambda, threshold=$threshold, first_class=$first_class"
    echo "Running experiment: lambda=$lambda, threshold=$threshold, first_class=$first_class" >> "$log_file"
    
    # Create experiment-specific output directory
    local exp_dir="$input_dir/lambda_${lambda}_thr_${threshold}_${first_class}"
    mkdir -p "$exp_dir"
    
    # Run the structure intervention script
    ./scripts/structures/interventions/structure_interventions.sh \
        "$num_designs" \
        "$indices_path_pair" \
        "$lambda" \
        "$threshold" \
        "$first_class" \
        "$sae_non_pair" \
        "$sae_pair" \
        "$base_dir_for_config" \
        "$python" \
        "$exp_dir" \
        "$prefix" \
        "$length" \
        "$coef_helix" \
        "$coef_beta" \
        "$coefs_output_dir";
    ((completed_runs+=1))
    echo "Progress: $completed_runs/$total_runs completed"
}

# Generate lambda values
lambda_values=()
for lambda in $(seq $lambda_start $lambda_step $lambda_stop); do
    lambda_values+=($lambda)
done

# Generate threshold values
threshold_values=()
for threshold in $(seq $threshold_start $threshold_step $threshold_stop); do
    threshold_values+=($threshold)
done

# Run experiments
for first_class in $first_classes; do
    for lambda in "${lambda_values[@]}"; do
        for threshold in "${threshold_values[@]}"; do
            run_experiment "$lambda" "$threshold" "$first_class"
        done
    done
done

echo ""
echo "Sweep completed at $(date)"
echo "Sweep completed at $(date)" >> "$log_file"
echo "Completed runs: $completed_runs/$total_runs"
echo "Completed runs: $completed_runs/$total_runs" >> "$log_file"

# Summary
echo ""
echo "Summary:"
echo "  Total experiments: $total_runs"
echo "  Successful: $completed_runs"
echo "  Failed: $((total_runs - completed_runs))"
echo "  Results saved in: $input_dir"
echo ""
echo "Summary:" >> "$log_file"
echo "  Total experiments: $total_runs" >> "$log_file"
echo "  Successful: $completed_runs" >> "$log_file"
echo "  Failed: $((total_runs - completed_runs))" >> "$log_file"
echo "  Results saved in: $input_dir" >> "$log_file"
