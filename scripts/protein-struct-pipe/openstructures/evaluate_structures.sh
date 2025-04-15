#!/bin/bash

rf_dir="$1"
af_dir="$2"
output_path="$3"
PYTHON_EXEC=${4:-/raid/battleamp_root/miniforge3/envs/openstructure/bin/python}

$PYTHON_EXEC scripts/protein-struct-pipe/openstructures/evaluate_structures.py \
        --rf_path $rf_dir \
        --af2_path $af_dir \
        --output_path $output_path \
