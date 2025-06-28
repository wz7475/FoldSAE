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
label=${7:-"Cytoplasm"}
cuda_idx=${8:-1}
num_of_coefs=${9:-10}
coefs_src_dir=${10:-"/home/wzarzecki/ds_sae_latents_1600x/coefs/non_pair"}
PYTHON_OPENSTRCUTERS=${11:-/home/wzarzecki/miniforge3/envs/openstruct/bin/python}
PYTHON_BIOEMB=${12:-/home/wzarzecki/miniforge3/envs/bio_emb/bin/python}
PYTHON_RFDIFFUSION=${14:-/home/wzarzecki/miniforge3/envs/rf124/bin/python}
PYTHON_PROTEINMPNN=${13:-/home/wzarzecki/miniforge3/envs/bio_emb/bin/python}

# 1)
echo "running on cuda: ${cuda_idx}"

structures_dir="$input_dir/pdb/$num_of_coefs" ;
coefs_dest_dir="$input_dir/coefs/$num_of_coefs" ;
mkdir -p $coefs_dest_dir ;
$PYTHON_RFDIFFUSION universal-diffsae/src/scripts/probes_per_intervention_indices.py \
  --dir_with_coefs $coefs_src_dir \
  --output_dir $coefs_dest_dir \
  --top_k $num_of_coefs ;

# generate config
config_name_no_ext="conf_${lowest_timestep}_${highest_timestep}_${probes_lambda_}_${num_of_coefs}"
# config_name_no_ext="temp-.-for-.-testing"
config_name_with_ext="${config_name_no_ext}.yaml"
$PYTHON_RFDIFFUSION RFDiffSAE/scripts/generate_config.py --lowest_timestep $lowest_timestep \
  --highest_timestep $highest_timestep \
  --lambda_ $probes_lambda_ \
  --output_config_name $config_name_with_ext \
  --label "${label}" \
  --base_dir_non_pair $coefs_dest_dir;
echo "generated config for RF $label lambda $probes_lambda_ range of timestep $highest_timestep  $lowest_timestep" ;
# generate structure by RfDiffusion with SAE intervention
echo "generation of structures ..." ;
echo "$PYTHON_RFDIFFUSION ./RFDiffSAE/scripts/run_inference.py inference.output_prefix=\"$structures_dir/\" contigmap.contigs=[100-200] inference.num_designs=\"$num_designs\" inference.final_step=\"$final_step\" saeinterventions=\"$config_name_no_ext\" inference.seed=0"

CUDA_VISIBLE_DEVICES="${cuda_idx}" SAE_DISABLE_TRITON=1 $PYTHON_RFDIFFUSION ./RFDiffSAE/scripts/run_inference.py \
	inference.output_prefix="$structures_dir/" \
 'contigmap.contigs=[100-200]' \
 inference.num_designs="$num_designs" \
 inference.final_step="$final_step"  \
 saeinterventions="$config_name_no_ext" \
 inference.seed=0
# keep only pdb
rm "$structures_dir"/*.trb ;

# # 2)
# # inverse-folding with protein-mpnn sequences from structures - 1 sequence per 1 structure (checkout run_inverse_folding.sh)
# echo "inverse-folding to sequences ..."
# sequences_dir="$input_dir/seqs/$num_of_coefs"
# CUDA_VISIBLE_DEVICES="${cuda_idx}" bash ./scripts/protein-struct-pipe/protein_mpnn/run_inverse_folding.sh "$structures_dir"  \
#   "$sequences_dir" \
#   "$PYTHON_PROTEINMPNN" ;


# # 3)
# echo "running bio_embeddings classifiers"
# classifiers_dir="$input_dir/classifiers/$num_of_coefs"
# mkdir -p $classifiers_dir ;
# classifiers_file="$classifiers_dir/classifiers.csv"
# CUDA_VISIBLE_DEVICES="${cuda_idx}" bash scripts/protein-struct-pipe/bio_emb/run_classifiers.sh \
#   $sequences_dir \
#   $classifiers_file \
#   $PYTHON_BIOEMB

# # 4) make plot
# plot_file="$classifiers_dir/subcellular.png"
# $PYTHON_BIOEMB scripts/rfdiffsae/subcellular_bar_plot.py -i  $classifiers_file -o $plot_file -k $probes_lambda_

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
