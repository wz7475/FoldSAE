
pdb_dir="temp_interventions_sweep"
stride_dir="temp_results/temp_interventions_sweep_stride"
plot_dir="temp_results/helix_beta_plots"
results_file="temp_results/helix_beta_analysis_results.json"
stride_binary="/data/wzarzecki/SAEtoRuleRFDiffusion/stride/stride"
python="/home/wzarzecki/miniforge3/envs/diffsae/bin/python"

echo "Starting evaluation pipeline..."
mkdir -p "$stride_dir"
mkdir -p "$plot_dir"
mkdir -p "$(dirname "$results_file")"

# # 1) create stride annotations in stride dir with same dir structure as input dir
# echo "Step 1: Creating STRIDE annotations..."
# $python scripts/structures/utils/run_stride.py \
#   --pdb_dir "$pdb_dir" \
#   --stride_dir "$stride_dir" \
#   --stride_binary "$stride_binary"

# 2) analyze helix beta ratios for all combinations
echo "Step 2: Analyzing helix to beta sheet ratios..."
$python scripts/structures/interventions/analyze_helix_beta_ratios.py \
  --base_dir "$pdb_dir" \
  --stride_dir "$stride_dir" \
  --output_file "$results_file"

# 3) generate plots
echo "Step 3: Generating plots..."
$python scripts/structures/interventions/plot_helix_beta_ratios.py \
  --results_file "$results_file" \
  --output_dir "$plot_dir" \
  --summary

