#!/bin/bash

input_dir="$1" # path to flat dir with seqs
output_csv="$2"
PYTHON_EXEC=${3:-/home/wzarzecki/miniforge3/envs/uncond38/bin/python}
subcellular_location_checkpoint="./protein-struct-pipe/classifiers/la_bert_subcellular_localization.pt"
membrane_checkpoint="./protein-struct-pipe/classifiers/la_bert_solubility.pt"

# initialization of embedder takes looong time, but inference time later is more bearable
$PYTHON_EXEC ./protein-struct-pipe/classifiers/classify.py \
        --input_dir "$input_dir" \
        --output_csv "$output_csv" \
        --subcellular_location_checkpoint "$subcellular_location_checkpoint" \
        --membrane_checkpoint "$membrane_checkpoint"
