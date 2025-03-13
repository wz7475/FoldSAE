#!/bin/bash

#SBATCH --partition=common
#SBATCH --qos=wzarzecki
#SBATCH --time=08:00:00
#SBATCH --output=/home/wzarzecki/test_job_n.txt

for x in {-1..-1} {1..2}; do
    /home/wzarzecki/miniforge3/envs/rf124/bin/python ./scripts/run_inference.py \
        "inference.output_prefix=ablations_main_iter_block_$x" \
        "inference.num_designs=1" \
        "model.skipped_main_block=$x" \
        "inference.final_step=50"
done
