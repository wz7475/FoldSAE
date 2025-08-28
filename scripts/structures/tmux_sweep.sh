#!/bin/bash

# Create tmux session and start first command
tmux new-session -d -s sweep_generate -c /data/wzarzecki/SAEtoRuleRFDiffusion
tmux send-keys -t sweep_generate:0 'CUDA_VISIBLE_DEVICES=0 bash scripts/structures/sweep_generate_structures.sh structures_sweep -15 -5 5 design 100 /home/wzarzecki/ds_secondary_struct/coefs/baseline_top_100.pt' C-m

# Create additional windows for each command
tmux new-window -t sweep_generate:1 -c /data/wzarzecki/SAEtoRuleRFDiffusion
tmux send-keys -t sweep_generate:1 'CUDA_VISIBLE_DEVICES=0 bash scripts/structures/sweep_generate_structures.sh structures_sweep 0 15 5 design 100 /home/wzarzecki/ds_secondary_struct/coefs/baseline_top_100.pt' C-m

tmux new-window -t sweep_generate:2 -c /data/wzarzecki/SAEtoRuleRFDiffusion
tmux send-keys -t sweep_generate:2 'CUDA_VISIBLE_DEVICES=0 bash scripts/structures/sweep_generate_structures.sh structures_sweep -4 0 2 design 100 /home/wzarzecki/ds_secondary_struct/coefs/baseline_top_250.pt' C-m

tmux new-window -t sweep_generate:3 -c /data/wzarzecki/SAEtoRuleRFDiffusion
tmux send-keys -t sweep_generate:3 'CUDA_VISIBLE_DEVICES=0 bash scripts/structures/sweep_generate_structures.sh structures_sweep 2 4 2 design 100 /home/wzarzecki/ds_secondary_struct/coefs/baseline_top_250.pt' C-m

tmux new-window -t sweep_generate:4 -c /data/wzarzecki/SAEtoRuleRFDiffusion
tmux send-keys -t sweep_generate:4 'CUDA_VISIBLE_DEVICES=0 bash scripts/structures/sweep_generate_structures.sh structures_sweep 0.2 0.6 0.2 design 100 /home/wzarzecki/ds_secondary_struct/coefs/baseline_top_500.pt' C-m

tmux new-window -t sweep_generate:5 -c /data/wzarzecki/SAEtoRuleRFDiffusion
tmux send-keys -t sweep_generate:5 'CUDA_VISIBLE_DEVICES=0 bash scripts/structures/sweep_generate_structures.sh structures_sweep -0.6 0 0.2 design 100 /home/wzarzecki/ds_secondary_struct/coefs/baseline_top_500.pt' C-m

tmux new-window -t sweep_generate:6 -c /data/wzarzecki/SAEtoRuleRFDiffusion
tmux send-keys -t sweep_generate:6 'CUDA_VISIBLE_DEVICES=1 bash scripts/structures/sweep_generate_structures.sh structures_sweep 0.1 0.3 0.1 design 100 /home/wzarzecki/ds_secondary_struct/coefs/baseline_top_1000.pt' C-m

tmux new-window -t sweep_generate:7 -c /data/wzarzecki/SAEtoRuleRFDiffusion
tmux send-keys -t sweep_generate:7 'CUDA_VISIBLE_DEVICES=1 bash scripts/structures/sweep_generate_structures.sh structures_sweep -0.3 0 0.1 design 100 /home/wzarzecki/ds_secondary_struct/coefs/baseline_top_1000.pt' C-m

tmux new-window -t sweep_generate:8 -c /data/wzarzecki/SAEtoRuleRFDiffusion
tmux send-keys -t sweep_generate:8 'CUDA_VISIBLE_DEVICES=1 bash scripts/structures/sweep_generate_structures.sh structures_sweep 0.05 0.15 0.05 design 100 /home/wzarzecki/ds_secondary_struct/coefs/baseline_top_1000.pt' C-m

tmux new-window -t sweep_generate:9 -c /data/wzarzecki/SAEtoRuleRFDiffusion
tmux send-keys -t sweep_generate:9 'CUDA_VISIBLE_DEVICES=2 bash scripts/structures/sweep_generate_structures.sh structures_sweep -0.15 0 0.05 design 100 /home/wzarzecki/ds_secondary_struct/coefs/baseline_top_1000.pt' C-m

# Attach to the session
tmux attach -t sweep_generate
