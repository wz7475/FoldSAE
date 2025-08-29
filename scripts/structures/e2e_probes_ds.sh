num_designs=${1:-2}
input_dir=${2:-./temp_probes_ds}
protein_length=${3:-50}
final_step=${3:-49}
log_file=${14:-/home/wzarzecki/logs/probes_ds.log}
SAE_PAIR_PATH=${14:-sae-ckpts/picked/patch_topk_expansion_factor16_k32_multi_topkFalse_auxk_alpha0.0lr0.0005_..activations_1200_block4_pair/block4_pair}
SAE_NON_PAIR_PATH=${14:-sae-ckpts/picked/patch_topk_expansion_factor16_k64_multi_topkFalse_auxk_alpha0.0lr0.0001_..activations_1200_block4_non_pair/block4_non_pair}
PYTHON_RFDIFFUSION=${14:-/home/wzarzecki/miniforge3/envs/rf/bin/python}
PYTHON_SAE=${14:-/home/wzarzecki/miniforge3/envs/diffsae/bin/python}

# paths variables
pdb_path="$input_dir/pdb"
latents_path="$input_dir/latents"
activations_dir="$input_dir/activations"

# 0) set up log
touch "$log_file";
echo "" > "$log_file";

# # 1) generate structures and collect block activations
# $PYTHON_RFDIFFUSION RFDiffSAE/scripts/run_inference.py \
#   "inference.output_prefix=$pdb_path/design_" \
#   "contigmap.contigs=[$protein_length-$protein_length]" \
#   "inference.num_designs=$num_designs" \
#   "inference.final_step=$final_step" \
#   activations=block4_10_token_10th_timestep \
#   "activations.dataset_path=$activations_dir" ;
# echo "generated structures and collected activations from $num_designs proteins" >> log_file;

# 2) generate latents
for shard in "$activations_dir"/shard*; do
  shard_name=$(basename "$shard")
  output_shard="$latents_path/$shard_name"
  $PYTHON_SAE universal-diffsae/src/scripts/sae_latents_from_activations.py \
    --activations_path "$shard" \
    --output_path "$output_shard" \
    --sae_pair_path "$SAE_PAIR_PATH" \
    --sae_non_pair_path "$SAE_NON_PAIR_PATH" \
    --batch_size "${BATCH_SIZE:-1024}" \
    --device "${DEVICE:-cuda}"
done
