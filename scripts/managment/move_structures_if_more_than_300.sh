#!/bin/bash


for i in {-1..-1} {1..31}; do
  echo "$i";

  num_of_generations=$(ls "per_block_outputs/output_blocks_$i/pdb" | wc -l);
  # move pdbs

  if [ $num_of_generations -gt 300  ]; then
    target_dir_pdb="per_block_outputs_above_300/output_blocks_$i/pdb/";
    mkdir -p "$target_dir_pdb";
    mv "per_block_outputs/output_blocks_$i/pdb/ablations_main_iter_block_${i}_"[3-9][0-9][0-9].pdb "$target_dir_pdb";

    if [ -d "per_block_outputs/output_blocks_$i/seqs" ]; then
      target_dir_seqs="per_block_outputs_above_300/output_blocks_$i/seqs";
      mkdir -p "$target_dir_seqs";
      mv "per_block_outputs/output_blocks_$i/seqs/ablations_main_iter_block_${i}_"[3-9][0-9][0-9].fa "$target_dir_seqs";
    fi
  fi


done
