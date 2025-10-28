#!/usr/bin/env bash


# Evaluation pipeline for ablation study. Mirrors the interventions pipeline but
# expects a different directory structure where PDBs are nested under block
# folders that each contain a `pdb/` directory, e.g.:
#   <base_dir>/main_1_4/main_1/pdb/*.pdb
#   <base_dir>/main_1_4/main_2/pdb/*.pdb
#   <base_dir>/main_1_4/extra_-1/pdb/*.pdb
#   <base_dir>/reference/reference/pdb/*.pdb

# - Traverses a base PDB directory organized by block folders (each with a pdb/ subdirectory).
# - Generates STRIDE secondary-structure annotations for all PDBs.
# - Computes per-block secondary-structure ratios (e.g., alpha, beta, other) and writes results to a JSON file.
# - Produces stacked-bar plots summarizing the per-block structure distributions.

# Configurable inputs/outputs (adjustable in the script or via environment variables):
# - pdb_dir: base directory with PDBs
# - stride_dir: directory to store STRIDE annotations
# - plot_dir: directory to write plots
# - results_file: JSON output with analysis results
# - stride_binary: path to STRIDE executable
# - python: Python interpreter used to run the analysis steps

# Outputs:
# - A JSON results file with per-block secondary structure statistics.
# - Plots (stacked bars) saved into the specified plot directory.

set -euo pipefail

# User-configurable paths (can be overridden via env)
: "${pdb_dir:=/data/wzarzecki/ablations_50x}"
: "${stride_dir:=temp_results/temp_ablations_sweep_stride}"
: "${plot_dir:=temp_results/ablations_plots}"
: "${results_file:=temp_results/ablations_analysis_results.json}"
: "${stride_binary:=/data/wzarzecki/SAEtoRuleRFDiffusion/stride/stride}"
: "${python:=/home/wzarzecki/miniforge3/envs/diffsae/bin/python}"

echo "Starting ablations evaluation pipeline..."
mkdir -p "$stride_dir"
mkdir -p "$plot_dir"
mkdir -p "$(dirname "$results_file")"

# 1) Create STRIDE annotations in stride dir with same dir structure as input dir
echo "Step 1: Creating STRIDE annotations..."
"$python" scripts/structures/utils/run_stride.py \
  --pdb_dir "$pdb_dir" \
  --stride_dir "$stride_dir" \
  --stride_binary "$stride_binary"

# 2) Analyze secondary structure ratios per block (alpha, beta, other)
echo "Step 2: Analyzing secondary structure ratios per block..."
"$python" scripts/structures/ablations/analyze_ablation_blocks.py \
  --base_dir "$pdb_dir" \
  --stride_dir "$stride_dir" \
  --output_file "$results_file"

# 3) Generate stacked bar plots ordered alphabetically by block
echo "Step 3: Generating plots..."
"$python" scripts/structures/ablations/plot_ablation_blocks.py \
  --results_file "$results_file" \
  --output_dir "$plot_dir"

echo "Done. Results: $results_file; Plots in: $plot_dir"


