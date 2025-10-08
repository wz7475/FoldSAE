num_designs=${1:-2}
input_dir=${2:-/home/wzarzecki/ds_10000x_normalized}
protein_length=${3:-150}
final_step=${3:-1}
log_file=${14:-/home/wzarzecki/logs/probes_ds.log}
SAE_PAIR_PATH=${14:-sae-ckpts/picked/patch_topk_expansion_factor16_k32_multi_topkFalse_auxk_alpha0.0lr0.0005_..activations_1200_block4_pair/block4_pair}
SAE_NON_PAIR_PATH=${14:-sae-ckpts/picked/patch_topk_expansion_factor16_k64_multi_topkFalse_auxk_alpha0.0lr0.0001_..activations_1200_block4_non_pair/block4_non_pair}
stride_binary=${14:-/data/wzarzecki/SAEtoRuleRFDiffusion/stride/stride}
PYTHON_RFDIFFUSION=${14:-/home/wzarzecki/miniforge3/envs/rf/bin/python}
PYTHON_SAE=${14:-/home/wzarzecki/miniforge3/envs/diffsae/bin/python}

# paths variables
pdb_dir="$input_dir/pdb"
latents_path="$input_dir/latents"
normalized_latents_path="$input_dir/normalized_latents"
activations_dir="$input_dir/activations"
stride_dir="$input_dir/stride"
structure_datasets_dir="$input_dir/structures_datasets"
merged_datasets_dir="$input_dir/structures_ds_merged"

set -euo pipefail

# 0) set up log
touch "$log_file";
echo "" > "$log_file";

## 1) generate structures and collect block activations
#$PYTHON_RFDIFFUSION RFDiffSAE/scripts/run_inference.py \
#  "inference.output_prefix=$pdb_dir/design" \
#  "contigmap.contigs=[$protein_length-$protein_length]" \
#  "inference.num_designs=$num_designs" \
#  "inference.final_step=$final_step" \
#  "inference.use_random_suffix_for_new_design=False" \
#  activations=block4_10_token_10th_timestep \
#  "activations.dataset_path=$activations_dir" ;
#echo "generated structures and collected activations from $num_designs proteins" >> log_file;
#
## 2) generate latents
#for shard in "$activations_dir"/shard*; do
#  shard_name=$(basename "$shard")
#  output_shard="$latents_path/$shard_name"
#  cmd="$PYTHON_SAE universal-diffsae/src/scripts/sae_latents_from_activations.py \
#    --activations_path \"$shard\" \
#    --output_path \"$output_shard\" \
#    --sae_pair_path \"$SAE_PAIR_PATH\" \
#    --sae_non_pair_path \"$SAE_NON_PAIR_PATH\" \
#    --batch_size \"${BATCH_SIZE:-1024}\" \
#    --device \"${DEVICE:-cuda}\""
#  echo "$cmd"
#  eval $cmd
#done
#echo "generated latents" >> "$log_file";

## 3) normalize latents
#for shard in "$latents_path"/shard*; do
#  shard_name=$(basename "$shard")
#  output_shard="$normalized_latents_path/$shard_name"
#  echo "output shard $output_shard"
#  cmd="$PYTHON_SAE -m scripts.structures.create_ds.normalize_latents \
#    --input_shard \"$shard\" \
#    --output_shard \"$output_shard\""
#  echo "$cmd"
#  eval $cmd
#done
#echo "generated normalized_latents" >> "$log_file";

## 4) run stride annotations
#cmd="$PYTHON_SAE scripts/structures/utils/run_stride.py \
#  --pdb_dir \"$pdb_dir\" \
#  --stride_dir \"$stride_dir\" \
#  --stride_binary \"$stride_binary\""
#echo "$cmd"
#eval $cmd
#echo "generated stride annotations" >> "$log_file";

## 5) add stride column to datasetsw
#for shard in "$normalized_latents_path"/shard*; do
#  shard_name=$(basename "$shard")
#  output_shard="$structure_datasets_dir/$shard_name"
#  echo "output shard $output_shard"
#  cmd="$PYTHON_SAE scripts/structures/create_ds/get_strcutures_annotatons.py \
#    --stride_dir \"$stride_dir\" \
#    --input_dataset_path \"$shard\" \
#    --output_dataset_path \"$output_shard\""
#  echo "$cmd"
#  eval $cmd
#done
#echo "genertated datasets with updates" >> "$log_file";

# 6) merge datasets
cmd="$PYTHON_SAE scripts/structures/create_ds/merge_datasets.py \
  --base_dir \"$structure_datasets_dir\" \
  --target_path \"$merged_datasets_dir\""
echo "$cmd"
eval "$cmd"
echo "merged datasets" >> "$log_file";
