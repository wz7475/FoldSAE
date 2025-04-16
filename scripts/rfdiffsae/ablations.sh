#!/bin/bash

start_main=$1
end_main=$2
start_extra=$3
end_extra=$4
output_dir=${5:-/raid/rfdiffsae/ablated_outputs}
PYTHON_EXEC=${6:-/home/wzarzecki/miniforge3/envs/rf124/bin/python}

mkdir -p "$output_dir"

# main blocks ablated
for x in $(seq $start_main $end_main); do
	$PYTHON_EXEC ./RFDiffSAE/scripts/run_inference.py \
		"inference.output_prefix=$output_dir/main_$x/pdb/ablations" \
		"inference.num_designs=4" \
		"inference.final_step=49" \
		"contigmap.contigs=[100-200]" \
		"model.skipped_main_block=$x"
done

# extra blocks ablated
for x in $(seq $start_extra $end_main); do
	$PYTHON_EXEC ./RFDiffSAE/scripts/run_inference.py \
		"inference.output_prefix=$output_dir/extra_$x/pdb/ablations" \
		"inference.num_designs=4" \
		"inference.final_step=49" \
		"contigmap.contigs=[100-200]" \
		"model.skipped_extra_block=$x"
done

# no block ablated - reference
$PYTHON_EXEC ./RFDiffSAE/scripts/run_inference.py \
	"inference.output_prefix=$output_dir/reference/pdb/ablations" \
	"inference.num_designs=4" \
	"inference.final_step=49" \
	"contigmap.contigs=[100-200]"

# clean trb files
find "$output_dir" -name "*.trb" -type f -delete
