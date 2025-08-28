#!/usr/bin/env bash
# sweep_generate_structures.sh

# Usage: sweep_generate_structures.sh <input_dir> <lambda_start> <lambda_end> <lambda_step> [prefix] [num_designs] [indices_path] [python_executable] [length] [seed]
# Example: ./sweep_generate_structures.sh outputs/designs -2 2 0.5 design 1 indices.pt /home/wzarzecki/miniforge3/envs/rf/bin/python 150 1

set -euo pipefail


if [[ $# -lt 4 ]]; then
    echo "Usage: $0 <input_dir> <lambda_start> <lambda_end> <lambda_step> [prefix] [num_designs] [indices_path] [python_executable] [length] [seed]"
    exit 1
fi

INPUT_DIR=$1
LAMBDA_START=$2
LAMBDA_END=$3
LAMBDA_STEP=$4
PREFIX=${5:-design}
NUM_DESIGNS=${6:-1}
INDICES_PATH=${7:-/home/wzarzecki/ds_secondary_struct/coefs/baseline_top_2000.pt}
PYTHON_EXECUTABLE=${8:-/home/wzarzecki/miniforge3/envs/rf/bin/python}
LENGTH=${9:-150}
SEED=${10:-1}


# Use Python for floating point range
LAMBDAS=$(python3 -c "import numpy as np; print(' '.join([str(round(x, 6)) for x in np.arange($LAMBDA_START, $LAMBDA_END + $LAMBDA_STEP/2, $LAMBDA_STEP)]))")

echo "Sweeping lambda values: $LAMBDAS"


for LAMBDA in $LAMBDAS; do
    echo "\n=== Running for lambda: $LAMBDA ==="
    CMD=("$(dirname "$0")/generate_structures.sh" \
        "$INPUT_DIR" \
        "$PREFIX" \
        "$NUM_DESIGNS" \
        "$LAMBDA" \
        "$INDICES_PATH" \
        "$PYTHON_EXECUTABLE" \
        "$LENGTH" \
        "$SEED")
    echo "Executing: bash ${CMD[*]}"
    bash "${CMD[@]}"
done
