#!/bin/bash

#SBATCH --partition=common
#SBATCH --qos=wzarzecki
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/home/wzarzecki/blocks_-1_14_v2.txt
#SBATCH --cpus-per-task=4

output_dir="/raid/wzarzecki/ablated_outputs";
log_file="$output_dir/log.txt";

mkdir -p "$output_dir";

touch "$log_file";

for x in {-1..-1} {1..14}; do
    /home/wzarzecki/miniforge3/envs/rf124/bin/python ./scripts/run_inference.py \
        "inference.output_prefix=$output_dir/ablations_main_iter_block_$x" \
        "inference.num_designs=1000" \
        "contigmap.contigs=[100-200]" \
        "model.skipped_main_block=$x"

    echo "block no $x done" >> "$log_file";
done
