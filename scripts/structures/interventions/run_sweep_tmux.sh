#!/usr/bin/env bash

set -euo pipefail

# Script to run structure interventions sweep in tmux session with 3 tabs
# Each tab runs with different CUDA device and lambda parameters

# Parse command line arguments
seed=${1:-1}

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
LAMBDAS=$(python3 - <<'PY'
import numpy as np
vals = np.arange(-5, 5 + 1e-9, 1)
def norm(x: float) -> str:
    s = ("%.*g" % (15, float(x)))
    return s
print(' '.join(norm(v) for v in vals))
PY
)

echo "Setting up windows for lambdas: $LAMBDAS"

idx=0
win_idx=0
for LAMBDA in $LAMBDAS; do
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

    echo "Setting up Tab $win_idx: CUDA_VISIBLE_DEVICES=$CUDA_IDX, lambda=$LAMBDA, threshold=0.5-3.5"
    tmux send-keys -t "$TMUX_TARGET" "cd $BASE_DIR" Enter

    # Per-lambda output dir and log file
    OUT_DIR="./temp_interventions_sweep/lambda_${LAMBDA}"
    LOG_FILE="$LOG_DIR/sweep_interventions_cuda${CUDA_IDX}_lm${LAMBDA}_${LAMBDA}_s1_thr0.5_3.5_s0.5_classes-helix-beta_${TIMESTAMP}.log"

    # Launch command in the window
    tmux send-keys -t "$TMUX_TARGET" "CUDA_VISIBLE_DEVICES=${CUDA_IDX} ./scripts/structures/interventions/sweep_structure_interventions.sh ${LAMBDA} ${LAMBDA} 1 0.5 3.5 0.5 'beta' '${OUT_DIR}' 20 ${seed} 2>&1 | tee -a ${LOG_FILE}" Enter

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
