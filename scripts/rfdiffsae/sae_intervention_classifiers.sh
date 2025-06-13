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
probes_lambda_=${4:-3}
lowest_timestep=${5:-2}
highest_timestep=${6:-2}
cuda_idx=${7:-1}
PYTHON_OPENSTRCUTERS=${7:-/home/wzarzecki/miniforge3/envs/openstruct/bin/python}
PYTHON_BIOEMB=${8:-/home/wzarzecki/miniforge3/envs/bio_emb/bin/python}
PYTHON_RFDIFFUSION=${9:-/home/wzarzecki/miniforge3/envs/rf/bin/python}
PYTHON_PROTEINMPNN=${10:-/home/wzarzecki/miniforge3/envs/bio_emb/bin/python}

# 1)
echo "running on cuda: ${cuda_idx}"
echo "generation of structures ..." ;
structures_dir="$input_dir/pdb" ;
# generate config
config_name_no_ext="${lowest_timestep}_${highest_timestep}_${probes_lambda_}"
config_name_with_ext="${config_name_no_ext}.yaml"
$PYTHON_RFDIFFUSION RFDiffSAE/scripts/generate_config.py --lowest_timestep $lowest_timestep \
  --highest_timestep $highest_timestep \
  --lambda_ $probes_lambda_ \
  --output_config_name $config_name_with_ext;
# generate structure by RfDiffusion with SAE intervention
CUDA_VISIBLE_DEVICES="${cuda_idx}" SAE_DISABLE_TRITON=1 $PYTHON_RFDIFFUSION ./RFDiffSAE/scripts/run_inference.py \
	inference.output_prefix="$structures_dir/" \
 'contigmap.contigs=[100-200]' \
 inference.num_designs="$num_designs" \
 inference.final_step="$final_step"  \
 saeinterventions="$config_name_no_ext" \
 inference.seed=0
# keep only pdb
rm "$structures_dir"/*.trb ;

# 2)
# inverse-folding with protein-mpnn sequences from structures - 1 sequence per 1 structure (checkout run_inverse_folding.sh)
echo "inverse-folding to sequences ..."
CUDA_VISIBLE_DEVICES="${cuda_idx}" bash ./scripts/protein-struct-pipe/protein_mpnn/run_inverse_folding.sh "$structures_dir"  \
  "$input_dir" "$PYTHON_PROTEINMPNN" ;
# this script^^^ saves that to dir_given_as_second_arg/seqs
sequences_dir="$input_dir/seqs"

# 3)
echo "running bio_embeddings classifiers"
classifiers_file="$input_dir/classifiers.csv"
CUDA_VISIBLE_DEVICES="${cuda_idx}" bash scripts/protein-struct-pipe/bio_emb/run_classifiers.sh \
  $sequences_dir \
  $classifiers_file \
  $PYTHON_BIOEMB

# 4) make plot
plot_file="$input_dir/subcellular.png"
$PYTHON_BIOEMB scripts/rfdiffsae/subcellular_bar_plot.py -i  $classifiers_file -o $plot_file -k $probes_lambda_

# 5)
# new structures from sequences with AlphaFold2
# echo "new structures from sequences with AlphaFold2 ..."
# af2_dir="$input_dir/af2"
# CUDA_VISIBLE_DEVICES="${cuda_idx bash scripts/protein-struct-pipe/colabfold/run_colabfold.sh \
#   $sequences_dir \
#   $af2_dir

# # 6)
# # comparison of RFDiff and AF2 structures
# echo "comparison of RFDiff and AF2 structures ..."
# results_file=$input_dir/structure_evaluation.csv
# CUDA_VISIBLE_DEVICES="${cuda_idx bash scripts/protein-struct-pipe/openstructures/evaluate_structures.sh \
#   $structures_dir \
#   $af2_dir \
#   $results_file \
#   "$PYTHON_OPENSTRCUTERS"
# echo "saved metrics to $results_file"

# # 7)
# echo "structure quality plots ..."
# plot_file_quality="$input_dir/strcuture_quality.png"
# $PYTHON_BIOEMB /data/wzarzecki/SAEtoRuleRFDiffusion/scripts/plots/structure_quality_single_file.py \
#   /data/wzarzecki/sae_interventions/sae_30_04_10x/structure_evaluation.csv \
#   $plot_file_quality \
#   --title $probes_lambda_
