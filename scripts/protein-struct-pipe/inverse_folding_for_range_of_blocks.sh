#!/bin/bash

# script to run inverse folding for range of block eg. for directory of blocks created scripts from ./scripts/management/

start=$1
end=$2
input_dir=$3;
PYTHON_EXEC=${4:-/home/wzarzecki/miniforge3/envs/uncond38/bin/python}

for i in $(seq $start $end); do
  echo "$i";
  bash ./run_inverse_folding.sh "$input_dir/output_blocks_$i/pdb" "$input_dir/output_blocks_$i" $PYTHON_EXEC;
done
