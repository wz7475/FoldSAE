#!/bin/bash

start_main=$1
end_main=$2
start_extra=$3
end_extra=$4
num_designs=${5:-1}
final_step=${6:-1}
output_dir=${7:-/raid/rfdiffsae/ablated_outputs}
PYTHON_EXEC=${8:-/home/wzarzecki/miniforge3/envs/rf124/bin/python}
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
