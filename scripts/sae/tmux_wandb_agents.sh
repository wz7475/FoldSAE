#!/usr/bin/env bash

set -euo pipefail

agent_arg=${1:-"RFDiffSAE/RFDiffSAE/1oytl5as"}
cuda_idx=${2:-0}
num_of_tabs=${3-4}
SESSION_NAME=${4:-"wandb_agents"}
wandb_bin=${5:-"/home/wzarzecki/miniforge3/envs/diffsae/bin/wandb"}
base_dir=${6:-"/data/wzarzecki/SAEtoRuleRFDiffusion"}

tmux new-session -d -s "$SESSION_NAME"

for i in $(seq 1 "$num_of_tabs"); do
    tmux new-window -t "$SESSION_NAME" -n "agent_$i"
    tmux send-keys -t "$SESSION_NAME" "cd $base_dir" Enter
    tmux send-keys -t "$SESSION_NAME" "export CUDA_VISIBLE_DEVICES=$cuda_idx" Enter
    tmux send-keys -t "$SESSION_NAME" "$wandb_bin agent $agent_arg" Enter
done

