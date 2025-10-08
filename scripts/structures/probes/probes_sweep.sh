#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="/data/wzarzecki/SAEtoRuleRFDiffusion"
PY_SCRIPT="$PROJECT_ROOT/scripts/structures/probes/train_probes.py"
DATASETS_DIR="/home/wzarzecki/ds_10000x_normalized/structures_ds_merged"
BASE_COEFS_DIR="/home/wzarzecki/ds_10000x_normalized/coefs"
BASE_RESULTS_DIR="/home/wzarzecki/ds_10000x_normalized/results/probes"
PYTHON="/home/wzarzecki/miniforge3/envs/diffsae/bin/python"

# pairings=("pair" "non_pair" "concat")
#pairings=("loose_concat")
pairings=("non_pair" "pair")

# add target sweep
targets=("helix" "beta")
# targets=("beta")

timesteps=("none") # include case with no --timestep argument
#for i in $(seq 1 50); do
#  timesteps+=("$i")
#done



# iterate pairing outer, target middle, timesteps inner
for pairing in "${pairings[@]}"; do
  for target in "${targets[@]}"; do
    for ts in "${timesteps[@]}"; do
      if [ "$ts" = "none" ]; then
        ts_label="no_timestep"
        ts_args=()
      else
        ts_label="t${ts}"
        ts_args=(--timestep "$ts")
      fi

      run_name="${pairing}_${target}_${ts_label}"

      coefs_dir="$BASE_COEFS_DIR/$run_name"
      results_json="$BASE_RESULTS_DIR/$run_name.json"

      # Skip this run if results JSON already exists
      if [ -f "$results_json" ]; then
        echo "Skipping run: $run_name (results already exist at $results_json)"
        continue
      fi

      mkdir -p "$coefs_dir"
      mkdir -p "$(dirname "$results_json")"

      echo "============================================================"
      echo "Running: pairing=$pairing, target=$target, timestep=${ts} -> run_name=${run_name}"
      echo "Coefs dir: $coefs_dir"
      echo "Results JSON: $results_json"
      echo "Datasets dir: $DATASETS_DIR"
      echo ""

      # invoke training script
      # build the command and print it before running
      cmd=("$PYTHON" "$PY_SCRIPT" \
        --datasets_dir "$DATASETS_DIR" \
        --pairing "$pairing" \
        --target "$target" \
        --coefs_dir "$coefs_dir" \
        --coefs_filename "coef.npy" \
        --bias_filename "bias.npy" \
        --results_json "$results_json" \
        "${ts_args[@]:-}")

      # safely print the command with shell-escaped args, skip empty arguments to avoid printing '' quotes
      cmd_str=""
      for a in "${cmd[@]}"; do
        if [ -n "$a" ]; then
          printf -v esc '%q' "$a"
          cmd_str+="$esc "
        fi
      done
      # remove trailing space
      cmd_str=${cmd_str% }
      echo "+ $cmd_str"

      # execute the command
      eval "$cmd_str"

      echo "Finished run: $run_name"
      echo ""
    done
  done
done