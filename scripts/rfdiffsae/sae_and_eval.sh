#!/usr/bin/env bash

num_designs=${1:-1}
input_dir=${2:-./temp}
final_step=${3:-49}

PYTHON_RFDIFFUSION=${4:-/home/wzarzecki/miniforge3/envs/rf/bin/python}
PYTHON_PROTEINMPNN=${5:-/home/wzarzecki/miniforge3/envs/bio_emb/bin/python}

# if you have only one gpu change value of CUDA_VISIBLE_DEVICES to 0

# 1)
echo "generation of structures ..." ;
structures_dir="$input_dir/pdb" ;
# generate structure by RfDiffusion with SAE intervention
CUDA_VISIBLE_DEVICES=0 $PYTHON_RFDIFFUSION ./RFDiffSAE/scripts/run_inference.py inference.output_prefix="$structures_dir/$input_dir" \
 'contigmap.contigs=[100-200]' inference.num_designs="$num_designs" inference.final_step="$final_step" saeinterventions=block4 ;
# keep only pdb
rm "$structures_dir"/*.trb ;

# 2)
# inverse-folding with protein-mpnn sequences from structures - 1 sequence per 1 structure (checkout run_inverse_folding.sh)
echo "inverse-folding to sequences ..."
CUDA_VISIBLE_DEVICES=1 bash ./scripts/protein-struct-pipe/protein_mpnn/run_inverse_folding.sh "$structures_dir"  \
"$input_dir" "$PYTHON_PROTEINMPNN" ;
# this script^^^ saves that to dir_given_as_second_arg/seqs
sequences_dir="$input_dir/seqs"
# for some strange reason protein mpnn always produces extra sequence with G amino-acid only, let's remove it
sed -i '1,2d' "$sequences_dir"/*

# template for accessing GPUS
#CUDA_VISIBLE_DEVICES=0 python  RFDiffSAE/cuda_checker.py ;
#
#CUDA_VISIBLE_DEVICES=1 python  RFDiffSAE/cuda_checker.py ;
#
#CUDA_VISIBLE_DEVICES=1,2,3 python  RFDiffSAE/cuda_checker.py ;