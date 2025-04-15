#!/bin/bash

experiment_dir="$1"
rf_dir="$2"

# Inverse folding
echo 'Running Protein-MPNN'
sh scripts/protein-struct-pipe/protein_mpnn/run_inverse_folding.sh \
    $rf_dir \
    $experiment_dir &&

# AF
echo 'Running AF2'
sh scripts/protein-struct-pipe/colabfold/run_colabfold.sh \
    $experiment_dir/seqs \
    $experiment_dir/alphafold &&

# openstructures
echo 'Running OpenStructures eval'
sh scripts/protein-struct-pipe/openstructures/evaluate_structures.sh \
    $rf_dir \
    $experiment_dir/alphafold \
    $experiment_dir/structure_evaluation.csv

echo 'Done'