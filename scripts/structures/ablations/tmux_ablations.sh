#!/usr/bin/env bash

set -euo pipefail

# tmux_ablations.sh
# Launches ablation runs for (main, extra) block combinations in a tmux session.
# Creates 37 combinations: (-1,-1), (-1,1..4), (1..32,-1) and runs 50 designs each.

SESSION_NAME="ablations"
BASE_DIR="/data/wzarzecki/SAEtoRuleRFDiffusion"
OUT_BASE="/data/wzarzecki/ablations_50x"
LOG_DIR="$BASE_DIR/logs"
mkdir -p "$LOG_DIR"
mkdir -p "$OUT_BASE"

# Parameters for run
NUM_DESIGNS=50
FINAL_STEP=1
PYTHON_EXEC="/home/wzarzecki/miniforge3/envs/rf/bin/python"


# Function to create tmux window and run ablation
run_ablation() {
    local start_main=$1
    local end_main=$2
    local start_extra=$3
    local end_extra=$4
    local cuda_idx=$5
    local window_name=$6
    
    local output_dir="$OUT_BASE/${window_name}"
    
    tmux new-window -t "$SESSION_NAME" -n "$window_name"
    tmux send-keys -t "$SESSION_NAME:$window_name" "cd $BASE_DIR" Enter
    tmux send-keys -t "$SESSION_NAME:$window_name" "export CUDA_VISIBLE_DEVICES=$cuda_idx" Enter
    tmux send-keys -t "$SESSION_NAME:$window_name" "bash $BASE_DIR/scripts/ablations/ablations.sh $start_main $end_main $start_extra $end_extra $NUM_DESIGNS $FINAL_STEP $output_dir $PYTHON_EXEC false" Enter
}

# Check if tmux session exists, create if not
if ! tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    tmux new-session -d -s "$SESSION_NAME" -n "reference"
    # Use the first window for reference, then create additional windows
    FIRST_WINDOW_CREATED=true
else
    FIRST_WINDOW_CREATED=false
fi

# Run ablations in tmux windows
# Reference run (no ablation)
if [ "$FIRST_WINDOW_CREATED" = "true" ]; then
    # Use the existing reference window
    tmux send-keys -t "$SESSION_NAME:reference" "cd $BASE_DIR" Enter
    tmux send-keys -t "$SESSION_NAME:reference" "export CUDA_VISIBLE_DEVICES=0" Enter
    tmux send-keys -t "$SESSION_NAME:reference" "bash $BASE_DIR/scripts/ablations/ablations.sh -1 -1 -1 -1 $NUM_DESIGNS $FINAL_STEP $OUT_BASE/reference/reference $PYTHON_EXEC" Enter
else
    # Create new reference window
    # Create new reference window, run explicit reference with nested output dir
    tmux new-window -t "$SESSION_NAME" -n "reference"
    tmux send-keys -t "$SESSION_NAME:reference" "cd $BASE_DIR" Enter
    tmux send-keys -t "$SESSION_NAME:reference" "export CUDA_VISIBLE_DEVICES=0" Enter
    tmux send-keys -t "$SESSION_NAME:reference" "bash $BASE_DIR/scripts/ablations/ablations.sh -1 -1 -1 -1 $NUM_DESIGNS $FINAL_STEP $OUT_BASE/reference/reference $PYTHON_EXEC" Enter
fi

# Main block ablations (1-32, -1)
for i in {1..8}; do
    start=$((4*i-3))
    end=$((4*i))
    cuda_idx=$((i <= 3 ? 1 : (i <= 6 ? 2 : 0)))
    run_ablation $start $end -1 -1 $cuda_idx "main_${start}_${end}"
done
#
# Extra block ablations (-1, 1-4)
run_ablation -1 -1 1 4 0 "extra_1_4"