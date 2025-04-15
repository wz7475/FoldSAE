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
            --num_seq_per_target 1 \
            --sampling_temp "0.1" \
            --seed 37 \
            --batch_size 1
done

for file in $output_dir/seqs/*.fa; do
  id=$(basename "$file" | sed -E 's/.*_([0-9]+)\.fa/\1/')
  tmpfile=$(mktemp)

  # Extract lines 3 and 4, update the header line
  sed -n '3,4p' "$file" | sed "1s/.*/> $id.fa/" > "$tmpfile"

  # Move and rename the processed file
  mv "$tmpfile" "$output_dir/seqs/$id.fa"

  [ "$file" != "$output_dir/seqs/$id.fa" ] && rm -f "$file"

done
