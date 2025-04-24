#!/usr/bin/env bash

# if you have only one gpu change value of CUDA_VISIBLE_DEVICES to 0
# structure of input dir
#inpu_dir
#├── af2
#|   └── temp11_0_5b71bf18-3cd7-4bc3-8b58-2ec0bbf59939_unrelaxed_rank_002_alphafold2_ptm_model_4_seed_000.pdb
#├── classifiers.csv
#├── pdb
#|   └── temp11_0_5b71bf18-3cd7-4bc3-8b58-2ec0bbf59939.pdb
#├── seqs
#|   └── temp11_0_5b71bf18-3cd7-4bc3-8b58-2ec0bbf59939.fa
#└── structure_evaluation.csv


num_designs=${1:-1}
input_dir=${2:-./temp}
final_step=${3:-49}

PYTHON_RFDIFFUSION=${4:-/home/wzarzecki/miniforge3/envs/rf/bin/python}
PYTHON_PROTEINMPNN=${5:-/home/wzarzecki/miniforge3/envs/bio_emb/bin/python}
PYTHON_OPENSTRCUTERS=${6:-/home/wzarzecki/miniforge3/envs/openstruct/bin/python}
PYTHON_BIOEMB=${7:-/home/wzarzecki/miniforge3/envs/bio_emb/bin/python}



# 1)
echo "generation of structures ..." ;
structures_dir="$input_dir/pdb" ;
# generate structure by RfDiffusion with SAE intervention
#CUDA_VISIBLE_DEVICES=0 $PYTHON_RFDIFFUSION ./RFDiffSAE/scripts/run_inference.py inference.output_prefix="$structures_dir/$input_dir" \
# 'contigmap.contigs=[100-200]' inference.num_designs="$num_designs" inference.final_step="$final_step"  \
# saeinterventions=block4 ;
# keep only pdb
rm "$structures_dir"/*.trb ;

# 2)
# inverse-folding with protein-mpnn sequences from structures - 1 sequence per 1 structure (checkout run_inverse_folding.sh)
echo "inverse-folding to sequences ..."
CUDA_VISIBLE_DEVICES=1 bash ./scripts/protein-struct-pipe/protein_mpnn/run_inverse_folding.sh "$structures_dir"  \
  "$input_dir" "$PYTHON_PROTEINMPNN" ;
# this script^^^ saves that to dir_given_as_second_arg/seqs
sequences_dir="$input_dir/seqs"

# 3)
# new structures from sequences with AlphaFold2
echo "new structures from sequences with AlphaFold2 ..."
af2_dir="$input_dir/af2"
CUDA_VISIBLE_DEVICES=1 bash scripts/protein-struct-pipe/colabfold/run_colabfold.sh \
  $sequences_dir \
  $af2_dir

# 4)
# comparison of RFDiff and AF2 structures
echo "comparison of RFDiff and AF2 structures ..."
results_file=$input_dir/structure_evaluation.csv
CUDA_VISIBLE_DEVICES=1 bash scripts/protein-struct-pipe/openstructures/evaluate_structures.sh \
  $structures_dir \
  $af2_dir \
  $results_file \
  $PYTHON_OPENSTRCUTERS
echo "saved metrics to $results_file"

# 5)
echo "running bio_embeddings classifiers"
classifiers_file="$input_dir/classifiers.csv"
CUDA_VISIBLE_DEVICES=1 bash scripts/protein-struct-pipe/bio_emb/run_classifiers.sh \
  $sequences_dir \
  $classifiers_file \
  $PYTHON_BIOEMB
