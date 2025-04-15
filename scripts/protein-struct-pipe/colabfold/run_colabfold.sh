#!/bin/bash

# script to run inverse folding of pdb files (in flat dir) to fasta sequences
# writes output to $output_dir/seqs

seq_dir="$1"
output_dir="$2"

export PATH="localcolabfold/colabfold-conda/bin:$PATH"

if [ ! -d $output_dir ]
then
    mkdir -p $output_dir
fi

for fa_file in $seq_dir/*.fa;
do
    echo "$fa_file";
    $PYTHON_EXEC colabfold_batch $fa_file $output_dir
done

# Clean up: keep only *.pdb files containing ptm_model_4, delete all else
find "$output_dir" -type f ! -name '*ptm_model_4*.pdb' -delete

# Remove any empty directories (including those without ptm_model_4)
find "$output_dir" -type d -empty -delete