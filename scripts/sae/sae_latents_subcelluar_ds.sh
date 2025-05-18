#!usr/bin/env bash

input_dir=$1
num_of_structures=$2
sae_pair_path=${3:-./sae-ckpts/RFDiffSAE/patch_topk_expansion_factor16_k32_multi_topkFalse_auxk_alpha0.0lr1e-05_datawzarzeckiactivations_block4_pair/block4_pair}
sae_non_pair_path=${4:-./sae-ckpts/RFDiffSAE/patch_topk_expansion_factor16_k32_multi_topkFalse_auxk_alpha0.0lr1e-05_datawzarzeckiactivations_block4_non_pair/block4_non_pair}
PYTHON_BIOEMB=${5:-/home/wzarzecki/miniforge3/envs/bio_emb/bin/python}
DEVICE_IDX_FOR_OLD_GPU=${6:-1}
PYTHON_RFDIFFUSION=${7:-/home/wzarzecki/miniforge3/envs/rf/bin/python}
PYTHON_PROTEINMPNN=${8:-/home/wzarzecki/miniforge3/envs/bio_emb/bin/python}

dir_for_latents="$input_dir/latents"
dir_for_structures="$input_dir/structures"
classifiers_file="$input_dir/classifiers.csv"

# 1) generate structures and save SAEs' latents
echo "generation of structures ..." ;
$PYTHON_RFDIFFUSION ./RFDiffSAE/scripts/run_inference.py \
    inference.output_prefix="$dir_for_structures/xxx" \
    'contigmap.contigs=[100-200]' \
    inference.num_designs=$num_of_structures \
    inference.final_step=1 \
    saeinterventions=block4 \
    saeinterventions.sae_pair_path=$sae_pair_path \
 		saeinterventions.sae_non_pair_path=$sae_non_pair_path \
    saeinterventions.sae_latents_base_dir=$dir_for_latents

echo "used sae: $sae_pair_path\n $sae_non_pair_path" > "$input_dir/sae_paths.txt"

# 2) inverse folding to sequences
echo "inverse-folding to sequences ..."
sequences_dir="$input_dir/seqs"
CUDA_VISIBLE_DEVICES="$DEVICE_IDX_FOR_OLD_GPU" bash scripts/protein-struct-pipe/protein_mpnn/run_inverse_folding.sh \
    $dir_for_structures \
    $input_dir \
    "$PYTHON_PROTEINMPNN"

# 5) running bio_embeddings classifiers
echo "running bio_embeddings classifiers ..."
CUDA_VISIBLE_DEVICES="$DEVICE_IDX_FOR_OLD_GPU" CUDA_VISIBLE_DEVICES=2 bash scripts/protein-struct-pipe/bio_emb/run_classifiers.sh \
  $sequences_dir \
  $classifiers_file \
  $PYTHON_BIOEMB

# 6 "update datasets with labels from classifiers"
echo "adding labels to HF datasets ..."
$PYTHON_RFDIFFUSION scripts/sae/update_sae_latents_dataset.py \
	--base-dir $input_dir

echo "Done."

