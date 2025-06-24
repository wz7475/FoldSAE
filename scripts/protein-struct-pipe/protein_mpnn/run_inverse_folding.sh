#!/bin/bash

pdb_dir="$1"
output_dir="$2"
PYTHON_EXEC=${3:-/home/wzarzecki/miniforge3/envs/uncond38/bin/python}

mkdir -p $output_dir ;

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
  id=$(basename "$file" .fa)
  tmpfile=$(mktemp)

  # Extract lines 3 and 4, update the header line
  sed -n '3,4p' "$file" | sed "1s/.*/>$id/" > "$tmpfile"

  # Move and rename the processed file
  mv "$tmpfile" "$output_dir/$id.fa"

  [ "$file" != "$output_dir/$id.fa" ] && rm -f "$file"

done

rm "$output_dir/seqs" -r
