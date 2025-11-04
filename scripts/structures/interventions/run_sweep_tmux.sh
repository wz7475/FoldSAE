#!/usr/bin/env bash

set -euo pipefail

# Script to run structure interventions sweep in tmux session with 3 tabs
# Each tab runs with different CUDA device and lambda parameters

# --- Configuration with Default Values ---

# Positional arguments:
# $1: seed (default 1)
# $2: lambda_start (default -3)
# $3: lambda_end (default 3)
# $4: lambda_step (default 1)
# $5: threshold_a (default 0.1)
# $6: threshold_b (default 0.1)
# $7: threshold_c (default 0.5)
# $8: num_designs (default 20)
# $9: classes_string (default 'beta')

seed=${1:-1}
lambda_start=${2:--3}
lambda_end=${3:-3}
lambda_step=${4:-1}
threshold_a=${5:-0.1} # Corresponds to argument 4 in sweep_structure_interventions.sh
threshold_b=${6:-0.1} # Corresponds to argument 5 in sweep_structure_interventions.sh
threshold_c=${7:-0.5} # Corresponds to argument 6 in sweep_structure_interventions.sh
num_designs=${8:-20}  # Corresponds to argument 8 in sweep_structure_interventions.sh
classes_string=${9:-'beta'} # Corresponds to argument 7 in sweep_structure_interventions.sh

# Session name
SESSION_NAME="sweep_interventions"

# Base directory
BASE_DIR="/data/wzarzecki/SAEtoRuleRFDiffusion"

# Logs directory and timestamp
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
LOG_DIR="$BASE_DIR/logs"
mkdir -p "$LOG_DIR"

# Check if tmux session already exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Session $SESSION_NAME already exists. Killing it first..."
    tmux kill-session -t "$SESSION_NAME"
fi

# Create new tmux session
echo "Creating new tmux session: $SESSION_NAME"
tmux new-session -d -s "$SESSION_NAME" -c "$BASE_DIR"

# Generate lambda list; ensure normalized formatting
# Uses the new configurable lambda_start, lambda_end, lambda_step
LAMBDAS=$(python3 - <<PY
import numpy as np
start = float('${lambda_start}')
end = float('${lambda_end}')
step = float('${lambda_step}')
vals = np.arange(start, end + 1e-9, step)
def norm(x: float) -> str:
    s = ("%.*g" % (15, float(x)))
    return s
print(' '.join(norm(v) for v in vals))
PY
)

echo "Setting up windows for lambdas: $LAMBDAS"
echo "Configuration:"
echo "  Seed: ${seed}"
echo "  Lambda Range: ${lambda_start} to ${lambda_end} step ${lambda_step}"
echo "  Thresholds: ${threshold_a}, ${threshold_b}, ${threshold_c}"
echo "  Num Designs: ${num_designs}"
echo "  Classes: ${classes_string}"
echo "---"

idx=0
win_idx=0
for LAMBDA in $LAMBDAS; do
    # Assign CUDA device based on index
    if [[ $idx -le 6 ]]; then
        CUDA_IDX=0
    elif [[ $idx -le 8 ]]; then
        CUDA_IDX=1
    else
        CUDA_IDX=2
    fi

    # Window target: first command goes to existing window 0, then create new ones
    if [[ $win_idx -eq 0 ]]; then
        TMUX_TARGET="$SESSION_NAME"
    else
        tmux new-window -t "$SESSION_NAME" -c "$BASE_DIR"
        TMUX_TARGET="$SESSION_NAME:$win_idx"
    fi

    echo "Setting up Tab $win_idx: CUDA_VISIBLE_DEVICES=$CUDA_IDX, lambda=$LAMBDA, thresholds=${threshold_a}-${threshold_b}"
    tmux send-keys -t "$TMUX_TARGET" "cd $BASE_DIR" Enter

    # Per-lambda output dir and log file
    # Uses threshold_a and threshold_b in the log file name for context
    OUT_DIR="./temp_interventions_sweep/lambda_${LAMBDA}"
    LOG_FILE="$LOG_DIR/sweep_interventions_cuda${CUDA_IDX}_lm${LAMBDA}_${LAMBDA}_s${lambda_step}_thr${threshold_a}_${threshold_b}_c${threshold_c}_classes-${classes_string}_${TIMESTAMP}.log"

    # Launch command in the window
    # The LAMBDA value is used twice as arguments 1 and 2
    # The thresholds, num_designs, classes_string, and seed are now configurable
    tmux send-keys -t "$TMUX_TARGET" "CUDA_VISIBLE_DEVICES=${CUDA_IDX} ./scripts/structures/interventions/sweep_structure_interventions.sh ${LAMBDA} ${LAMBDA} ${lambda_step} ${threshold_a} ${threshold_b} ${threshold_c} '${classes_string}' '${OUT_DIR}' ${num_designs} ${seed} 2>&1 | tee -a ${LOG_FILE}" Enter

    idx=$((idx+1))
    win_idx=$((win_idx+1))
done

echo ""
echo "Tmux session '$SESSION_NAME' created successfully!"
echo "${win_idx} tabs are running, CUDA assignment: first 7 -> cuda0, next 2 -> cuda1, last 2 -> cuda2"
echo ""
echo "To attach to the session: tmux attach -t $SESSION_NAME"
echo "To switch between tabs: Ctrl+b then window number"
echo "To detach: Ctrl+b then d"