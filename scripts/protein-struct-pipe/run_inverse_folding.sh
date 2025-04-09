#!/bin/bash

# script to run inverse folding of pdb files (in flat dir) to fasta sequences
# writes output to $output_dir/seqs

pdb_dir="$1"
output_dir="$2"
PYTHON_EXEC=${3:-/home/wzarzecki/miniforge3/envs/uncond38/bin/python}

if [ ! -d $output_dir ]
then
    mkdir -p $output_dir
fi

for pdb_file in $pdb_dir/*.pdb;
do
    echo "$pdb_file";
    $PYTHON_EXEC protein-struct-pipe/protein_mpnn_run.py \
            --pdb_path $pdb_file \
            --out_folder $output_dir \
            --num_seq_per_target 2 \
            --sampling_temp "0.1" \
            --seed 37 \
            --batch_size 1
done
