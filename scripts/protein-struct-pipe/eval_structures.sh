#!/bin/bash

experiment_dir="$1"
rf_dir="$2"
DEVICE_IDX_FOR_PYTHON_PROTEINMPNN=${3:-0}
PYTHON_PROTEINMPNN=${4:-/home/wzarzecki/miniforge3/envs/bio_emb/bin/python}
PYTHON_OPENSTRCUTERS=${5:-/home/wzarzecki/miniforge3/envs/openstruct/bin/python}

# Inverse folding
echo 'Running Protein-MPNN'
CUDA_VISIBLE_DEVICES="$DEVICE_IDX_FOR_PYTHON_PROTEINMPNN" sh scripts/protein-struct-pipe/protein_mpnn/run_inverse_folding.sh \
    $rf_dir \
    $experiment_dir \
    "$PYTHON_PROTEINMPNN" &&

# AF
echo 'Running AF2'
sh scripts/protein-struct-pipe/colabfold/run_colabfold.sh \
    $experiment_dir/seqs \
    $experiment_dir/alphafold  &&

# openstructures
echo 'Running OpenStructures eval'
sh scripts/protein-struct-pipe/openstructures/evaluate_structures.sh \
    $rf_dir \
    $experiment_dir/alphafold \
    $experiment_dir/structure_evaluation.csv \
    $PYTHON_OPENSTRCUTERS

echo 'Done'