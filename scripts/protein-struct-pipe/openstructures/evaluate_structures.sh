#!/usr/bin/env bash

rf_dir=$1
af_dir=$2
output_path=$3
PYTHON_EXEC=${4:-/home/wzarzecki/miniforge3/envs/openstruct/bin/python}

# TMSCORE in python is just a wrapper for unix package installed through conda, it has to be added to path
TMSCORE_EXEC_ROOT=$(dirname $PYTHON_EXEC)
export PATH="$TMSCORE_EXEC_ROOT:$PATH";

$PYTHON_EXEC scripts/protein-struct-pipe/openstructures/evaluate_structures.py \
    --rf_path $rf_dir \
    --af2_path $af_dir \
    --output_path $output_path
