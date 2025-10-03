#!/bin/bash

start_main=$1
end_main=$2
start_extra=$3
end_extra=$4
output_dir=${5:-/data/wzarzecki/ablation_test}
PYTHON_RFDIFFUSION=${6:-/home/wzarzecki/miniforge3/envs/rf/bin/python}
PYTHON_PROTEINMPNN=${7:-/home/wzarzecki/miniforge3/envs/bio_emb/bin/python}
PYTHON_OPENSTRCUTERS=${8:-/home/wzarzecki/miniforge3/envs/openstruct/bin/python}
PYTHON_BIOEMB=${9:-/home/wzarzecki/miniforge3/envs/bio_emb/bin/python}

# 1)
# generate structures with RFDiff with ablated blocks
# creates dir $output_dir/block_XXX/pdb for each block with pdb files
bash scripts/ablations/ablations.sh $start_main $end_main $start_extra $end_extra 10 1 $output_dir $PYTHON_RFDIFFUSION

# 2)
# inverse-folding
for pdb_dir in $(find $output_dir -name "*pdb*" -type d); do
	block_dir=$(dirname $pdb_dir)
	CUDA_VISIBLE_DEVICES=1 bash scripts/protein-struct-pipe/protein_mpnn/run_inverse_folding.sh \
		$pdb_dir \
		$block_dir \
		$PYTHON_PROTEINMPNN
done

# 3)
# structures with AF2
for seq_dir in $(find $output_dir -name "*seqs*" -type d); do
	block_dir=$(dirname $seq_dir)
	CUDA_VISIBLE_DEVICES=0 bash scripts/protein-struct-pipe/colabfold/run_colabfold.sh \
		$seq_dir \
		"$block_dir/af2"
done

# 4)
# metrics with openstructures
for RF_dir in $(find $output_dir -name "*pdb*" -type d); do
	block_dir=$(dirname $RF_dir)
	CUDA_VISIBLE_DEVICES=0 bash scripts/protein-struct-pipe/openstructures/evaluate_structures.sh \
		$RF_dir \
		"$block_dir/af2" \
		$block_dir/structure_evaluation.csv \
		$PYTHON_OPENSTRCUTERS
done

# 5)
# solubility and subcellular localization
for seq_dir in $(find $output_dir -name "*seqs*" -type d); do
	block_dir=$(dirname $seq_dir)
	CUDA_VISIBLE_DEVICES=1 bash scripts/protein-struct-pipe/bio_emb/run_classifiers.sh \
		$seq_dir \
		"$block_dir/classifiers_bio_emb.csv" \
		$PYTHON_BIOEMB
done

# 6)
# plot for classifiers
$PYTHON_BIOEMB scripts/plots/bio_emb_classifiers.py \
	--dir_with_csvs $output_dir \
	--filename classifiers_bio_emb.csv \
	--subcellular_path $output_dir/subcellular.png \
	--solubility_path $output_dir/solubility.png

$PYTHON_BIOEMB scripts/plots/structure_quality.py \
	--dir_with_files $output_dir \
	--filename structure_evaluation.csv \
	--tm_path $output_dir/tm.png \
	--rmsd_path $output_dir/rmsd.png

