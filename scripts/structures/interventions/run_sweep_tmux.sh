#!/usr/bin/env bash

set -euo pipefail

# Script to run structure interventions sweep in tmux session with 3 tabs
# Each tab runs with different CUDA device and lambda parameters

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

# Generate lambda list from -2.5 to 2.5 inclusive with step 0.5
LAMBDAS=$(python3 -c "import numpy as np; print(' '.join([str(x) for x in np.arange(-2.5, 2.5 + 1e-9, 0.5)]))")

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

    echo "Setting up Tab $win_idx: CUDA_VISIBLE_DEVICES=$CUDA_IDX, lambda=$LAMBDA, threshold=0.3-1.0"
    tmux send-keys -t "$TMUX_TARGET" "cd $BASE_DIR" Enter

    # Per-lambda output dir and log file
    OUT_DIR="./temp_interventions_sweep/lambda_${LAMBDA}"
    LOG_FILE="$LOG_DIR/sweep_interventions_cuda${CUDA_IDX}_lm${LAMBDA}_${LAMBDA}_s1_thr0.3_1.0_s0.1_classes-helix-beta_${TIMESTAMP}.log"

    # Launch command in the window
    tmux send-keys -t "$TMUX_TARGET" "CUDA_VISIBLE_DEVICES=${CUDA_IDX} ./scripts/structures/interventions/sweep_structure_interventions.sh ${LAMBDA} ${LAMBDA} 1 0.3 1.0 0.1 'helix beta' '${OUT_DIR}' 45 2>&1 | tee -a ${LOG_FILE}" Enter

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
