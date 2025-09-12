#!/bin/bash

SESSION="sae"
WORKDIR="/data/wzarzecki/SAEtoRuleRFDiffusion"
CMD_CUDA_0="CUDA_VISIBLE_DEVICES=0 bash scripts/sae/sae_latents_subcelluar_ds.sh /home/wzarzecki/ds_sae_latents_1600x 200"
CMD_CUDA_1="CUDA_VISIBLE_DEVICES=1 bash scripts/sae/sae_latents_subcelluar_ds.sh /home/wzarzecki/ds_sae_latents_1600x 200"
CMD_CUDA_2="CUDA_VISIBLE_DEVICES=2 bash scripts/sae/sae_latents_subcelluar_ds.sh /home/wzarzecki/ds_sae_latents_1600x 200"
CMD_NVIDIA_SMI="watch -n 1 nvidia-smi"
# Start tmux session in detached mode
tmux new-session -s "$SESSION" -d -c $WORKDIR "$CMD_NVIDIA_SMI"

# Create 5 more windows (since the first is already created)
for i in {1..4}
do
    tmux new-window -t $SESSION: -c $WORKDIR "$CMD_CUDA_0"
done

for i in {1..2}
do
    tmux new-window -t $SESSION: -c $WORKDIR "$CMD_CUDA_1"
done

for i in {1..2}
do
    tmux new-window -t $SESSION: -c $WORKDIR "$CMD_CUDA_2"
done



# Attach to the session
tmux attach-session -t $SESSION
