#!/bin/bash

# scritpt to generate ablations of main IterBlock for given range

start=$1
end=$2
output_dir=${3:-/raid/rfdiffsae/ablated_outputs}
PYTHON_EXEC=${4:-/home/wzarzecki/miniforge3/envs/rf124/bin/python}
log_file="$output_dir/log.txt";


mkdir -p "$output_dir";
touch "$log_file";

for x in $(seq $start $end); do
    $PYTHON_EXEC ./RFDiffSAE/scripts/run_inference.py \
        "inference.output_prefix=$output_dir/ablations_main_iter_block_$x" \
        "inference.num_designs=1" \
        "inference.final_step=49" \
        "contigmap.contigs=[100-200]" \
        "model.skipped_main_block=$x"

    echo "block no $x done" >> "$log_file";
done


