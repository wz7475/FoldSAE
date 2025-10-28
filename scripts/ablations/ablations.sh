#!/bin/bash

# Positional arguments (defaults shown in parentheses):
# - start_main, end_main: range of main blocks to ablate (use -1 -1 to skip main ablations)
# - start_extra, end_extra: range of extra blocks to ablate (use -1 -1 to skip extra ablations)
# - num_designs (1): number of designs to generate
# - final_step (1): final step index for inference
# - output_dir (/raid/rfdiffsae/ablated_outputs): directory for outputs
# - PYTHON_EXEC (/home/wzarzecki/miniforge3/envs/rf/bin/python): Python executable to run the script
# - reference_dir (true): if true, also run the reference (no-ablations) job

# Notes:
# - A reference run (no ablations) is executed when reference_dir is true (default), or when all four block-range args are -1 -1 -1 -1.
# - To ablate only main blocks, set start_extra and end_extra to -1 -1.
# - To ablate only extra blocks, set start_main and end_main to -1 -1.

start_main=$1
end_main=$2
start_extra=$3
end_extra=$4
num_designs=${5:-1}
final_step=${6:-1}
output_dir=${7:-/raid/rfdiffsae/ablated_outputs}
PYTHON_EXEC=${8:-/home/wzarzecki/miniforge3/envs/rf/bin/python}
reference_dir=${9:-true}

mkdir -p "$output_dir"


# no block ablated - reference
# Run if reference_dir is true, or if all ranges are -1 (special-case reference run)
if { [ "$reference_dir" = true ] || [ "$reference_dir" = "True" ]; } || { [ "$start_main" -eq -1 ] && [ "$end_main" -eq -1 ] && [ "$start_extra" -eq -1 ] && [ "$end_extra" -eq -1 ]; }; then
$PYTHON_EXEC ./RFDiffSAE/scripts/run_inference.py \
	"inference.output_prefix=$output_dir/reference/pdb/ablations" \
	"inference.num_designs=$num_designs" \
	"inference.final_step=$final_step" \
	"inference.use_random_suffix_for_new_design=False" \
	"contigmap.contigs=[150-150]"
fi


# main blocks ablated
if ! { [ "$start_main" -eq -1 ] || [ "$end_main" -eq -1 ]; }; then
for x in $(seq $start_main $end_main); do
	$PYTHON_EXEC ./RFDiffSAE/scripts/run_inference.py \
		"inference.output_prefix=$output_dir/main_$x/pdb/ablations" \
		"inference.num_designs=$num_designs" \
		"inference.final_step=$final_step" \
		"inference.use_random_suffix_for_new_design=False" \
		"contigmap.contigs=[150-150]" \
		"model.skipped_main_block=$x"
done
fi

# extra blocks ablated
if ! { [ "$start_extra" -eq -1 ] || [ "$end_extra" -eq -1 ]; }; then
for x in $(seq $start_extra $end_extra); do
	$PYTHON_EXEC ./RFDiffSAE/scripts/run_inference.py \
		"inference.output_prefix=$output_dir/extra_$x/pdb/ablations" \
		"inference.num_designs=$num_designs" \
		"inference.final_step=$final_step" \
		"inference.use_random_suffix_for_new_design=False" \
		"contigmap.contigs=[150-150]" \
		"model.skipped_extra_block=$x"
done
fi


# clean trb files
find "$output_dir" -name "*.trb" -type f -delete
