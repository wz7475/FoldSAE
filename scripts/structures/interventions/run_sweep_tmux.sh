#!/usr/bin/env bash

set -euo pipefail

# Script to run structure interventions sweep in tmux session with 3 tabs
# Each tab runs with different CUDA device and lambda parameters

# Session name
SESSION_NAME="sweep_interventions"

# Base directory
BASE_DIR="/data/wzarzecki/SAEtoRuleRFDiffusion"

# Check if tmux session already exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Session $SESSION_NAME already exists. Killing it first..."
    tmux kill-session -t "$SESSION_NAME"
fi

# Create new tmux session
echo "Creating new tmux session: $SESSION_NAME"
tmux new-session -d -s "$SESSION_NAME" -c "$BASE_DIR"

# Tab 1: CUDA_VISIBLE_DEVICES=0, lambda -0.5 to -0.5, threshold 0.3 to 1.0 step 0.1
echo "Setting up Tab 1: CUDA_VISIBLE_DEVICES=0, lambda=-0.5, threshold=0.3-1.0"
tmux send-keys -t "$SESSION_NAME" "cd $BASE_DIR" Enter
tmux send-keys -t "$SESSION_NAME" "CUDA_VISIBLE_DEVICES=0 ./scripts/structures/interventions/sweep_structure_interventions.sh -0.5 -0.5 1 0.3 1.0 0.1 'helix beta' './temp_interventions_sweep' 45" Enter

# Create new window for Tab 2
tmux new-window -t "$SESSION_NAME" -c "$BASE_DIR"

# Tab 2: CUDA_VISIBLE_DEVICES=1, lambda 0 to 0, threshold 0.3 to 1.0 step 0.1
echo "Setting up Tab 2: CUDA_VISIBLE_DEVICES=1, lambda=0, threshold=0.3-1.0"
tmux send-keys -t "$SESSION_NAME:1" "cd $BASE_DIR" Enter
tmux send-keys -t "$SESSION_NAME:1" "CUDA_VISIBLE_DEVICES=1 ./scripts/structures/interventions/sweep_structure_interventions.sh 0 0 1 0.3 1.0 0.1 'helix beta' './temp_interventions_sweep' 45" Enter

# Create new window for Tab 3
tmux new-window -t "$SESSION_NAME" -c "$BASE_DIR"

# Tab 3: CUDA_VISIBLE_DEVICES=2, lambda 0.5 to 0.5, threshold 0.3 to 1.0 step 0.1
echo "Setting up Tab 3: CUDA_VISIBLE_DEVICES=2, lambda=0.5, threshold=0.3-1.0"
tmux send-keys -t "$SESSION_NAME:2" "cd $BASE_DIR" Enter
tmux send-keys -t "$SESSION_NAME:2" "CUDA_VISIBLE_DEVICES=2 ./scripts/structures/interventions/sweep_structure_interventions.sh 0.5 0.5 1 0.3 1.0 0.1 'helix beta' './temp_interventions_sweep' 45" Enter

echo ""
echo "Tmux session '$SESSION_NAME' created successfully!"
echo "3 tabs are running with the following configurations:"
echo "  Tab 0: CUDA_VISIBLE_DEVICES=0, lambda=-0.5, threshold=0.3-1.0 (step 0.1)"
echo "  Tab 1: CUDA_VISIBLE_DEVICES=1, lambda=0, threshold=0.3-1.0 (step 0.1)"
echo "  Tab 2: CUDA_VISIBLE_DEVICES=2, lambda=0.5, threshold=0.3-1.0 (step 0.1)"
echo ""
echo "To attach to the session: tmux attach -t $SESSION_NAME"
echo "To switch between tabs: Ctrl+b then 0, 1, or 2"
echo "To detach: Ctrl+b then d"
echo ""
echo "Output directories:"
echo "  Tab 0: ./temp_interventions_sweep_cuda0"
echo "  Tab 1: ./temp_interventions_sweep_cuda1"
echo "  Tab 2: ./temp_interventions_sweep_cuda2"
