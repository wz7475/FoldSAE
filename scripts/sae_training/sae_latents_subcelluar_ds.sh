#!/usr/bin/env bash

input_dir=$1
num_of_structures=$2
sae_pair_path=${3:-./sae-ckpts/RFDiffSAE/patch_topk_expansion_factor16_k32_multi_topkFalse_auxk_alpha0.0lr1e-05_datawzarzeckiactivations_block4_pair/block4_pair}
sae_non_pair_path=${4:-./sae-ckpts/RFDiffSAE/patch_topk_expansion_factor16_k32_multi_topkFalse_auxk_alpha0.0lr1e-05_datawzarzeckiactivations_block4_non_pair/block4_non_pair}
PYTHON_BIOEMB=${5:-/home/wzarzecki/miniforge3/envs/bio_emb/bin/python}
DEVICE_IDX_FOR_OLD_GPU=${6:-1}
PYTHON_RFDIFFUSION=${7:-/home/wzarzecki/miniforge3/envs/rf/bin/python}
PYTHON_PROTEINMPNN=${8:-/home/wzarzecki/miniforge3/envs/bio_emb/bin/python}
PYTHON_SAE=${9:-/home/wzarzecki/miniforge3/envs/diffsae/bin/python}

dir_for_latents="$input_dir/latents"
dir_for_activations="$input_dir/activations"
dir_for_structures="$input_dir/structures"
classifiers_file="$input_dir/classifiers.csv"
dir_for_ovr_datasets="$input_dir/ovr_datasets"
dir_for_ovr_datasets_per_timestep="$input_dir/ovr_datasets_per_timestep"

# 1) generate structures and save RFDiffusion activations
echo "generation of structures ..." ;
$PYTHON_RFDIFFUSION ./RFDiffSAE/scripts/run_inference.py \
   inference.output_prefix="$dir_for_structures/xxx" \
   'contigmap.contigs=[100-200]' \
   inference.num_designs=$num_of_structures \
   inference.final_step=1 \
		activations=block4 \
		activations.dataset_path=$dir_for_activations \
		activations.keep_every_n_timestep=10 \
		activations.keep_every_n_token=10

# 2) generate latent with SAE
echo "generating latents..."
$PYTHON_SAE ./universal-diffsae/src/scripts/sae_latents_from_activations.py \
		--base_dir $input_dir \
		--sae_pair_path=$sae_pair_path \
		--sae_non_pair_path=$sae_non_pair_path

echo "used sae: $sae_pair_path\n $sae_non_pair_path" > "$input_dir/sae_paths.txt"

# 3) inverse folding to sequences
echo "inverse-folding to sequences ..."
sequences_dir="$input_dir/seqs"
CUDA_VISIBLE_DEVICES="$DEVICE_IDX_FOR_OLD_GPU" bash scripts/protein-struct-pipe/protein_mpnn/run_inverse_folding.sh \
   $dir_for_structures \
   $input_dir \
   "$PYTHON_PROTEINMPNN"

# 4) running bio_embeddings classifiers
echo "running bio_embeddings classifiers ..."
CUDA_VISIBLE_DEVICES="$DEVICE_IDX_FOR_OLD_GPU" CUDA_VISIBLE_DEVICES=2 bash scripts/protein-struct-pipe/bio_emb/run_classifiers.sh \
 $sequences_dir \
 $classifiers_file \
 $PYTHON_BIOEMB

# 5 "update datasets with labels from classifiers"
echo "adding labels to HF datasets ..."
$PYTHON_RFDIFFUSION scripts/sae/update_sae_latents_dataset.py \
#	--base-dir $input_dir

# 6) prepare specific datasets
for sae_type in "pair" "non_pair"; do
 $PYTHON_SAE universal-diffsae/src/scripts/prepare_and_merge_latents_ds.py \
   --dataset_shards_path "$input_dir" \
   --output_datasets_dir "$dir_for_ovr_datasets/$sae_type" \
   --sae_type "$sae_type"
done
for sae_type in "pair" "non_pair"; do
  $PYTHON_SAE universal-diffsae/src/scripts/prepare_and_merge_latents_ds.py \
    --dataset_shards_path "$input_dir" \
    --output_datasets_dir "$dir_for_ovr_datasets_per_timestep/$sae_type" \
    --sae_type "$sae_type" \
    --ds_per_timestep
done

echo "Done."

