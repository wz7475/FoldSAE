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
probes_multiplier=${4:-3}
cuda_idx_for_generation=${4:-0}
PYTHON_RFDIFFUSION=${9:-/home/wzarzecki/miniforge3/envs/rf/bin/python}
PYTHON_PROTEINMPNN=${6:-/home/wzarzecki/miniforge3/envs/bio_emb/bin/python}
PYTHON_OPENSTRCUTERS=${7:-/home/wzarzecki/miniforge3/envs/openstruct/bin/python}
PYTHON_BIOEMB=${8:-/home/wzarzecki/miniforge3/envs/bio_emb/bin/python}



# 1)
echo "generation of structures ..." ;
structures_dir="$input_dir/pdb" ;
# generate structure by RfDiffusion with SAE intervention
CUDA_VISIBLE_DEVICES=2 SAE_DISABLE_TRITON=1 $PYTHON_RFDIFFUSION ./RFDiffSAE/scripts/run_inference.py \
	inference.output_prefix="$structures_dir/" \
 'contigmap.contigs=[100-200]' \
 inference.num_designs="$num_designs" \
 inference.final_step="$final_step"  \
 saeinterventions=block4 \
 saeinterventions.probes_multiplier=$probes_multiplier;
# keep only pdb
rm "$structures_dir"/*.trb ;

# 2)
# inverse-folding with protein-mpnn sequences from structures - 1 sequence per 1 structure (checkout run_inverse_folding.sh)
echo "inverse-folding to sequences ..."
CUDA_VISIBLE_DEVICES=2 bash ./scripts/protein-struct-pipe/protein_mpnn/run_inverse_folding.sh "$structures_dir"  \
  "$input_dir" "$PYTHON_PROTEINMPNN" ;
# this script^^^ saves that to dir_given_as_second_arg/seqs
sequences_dir="$input_dir/seqs"

# 3)
echo "running bio_embeddings classifiers"
classifiers_file="$input_dir/classifiers.csv"
CUDA_VISIBLE_DEVICES=2 bash scripts/protein-struct-pipe/bio_emb/run_classifiers.sh \
  $sequences_dir \
  $classifiers_file \
  $PYTHON_BIOEMB

# 4) make plot
plot_file="$input_dir/subcellular.png"
$PYTHON_BIOEMB scripts/rfdiffsae/subcellular_bar_plot.py -i  $classifiers_file -o $plot_file -k $probes_multiplier