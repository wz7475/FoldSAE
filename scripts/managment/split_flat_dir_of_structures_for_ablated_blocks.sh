#!/bin/bash

input_dir=$1;
output_dir=$2;

for i in {-1..-1} {1..32}; do
  echo "$i";
  current_block_out_dir="$output_dir/output_blocks_$i/pdb";
  mkdir -p "$current_block_out_dir";
  cp "$input_dir/ablations_main_iter_block_${i}_"*.pdb "$current_block_out_dir";
done
