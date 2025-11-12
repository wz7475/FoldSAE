
val_input_pdb=${1:-"temp_pdb"}
results_dir=${2:-"temp_results"}
n_ref=${3:-20}

ref_stride="/home/wzarzecki/ds_10000x/stride"
ref_fasta="/home/wzarzecki/ds_10000x/fasta"
stride_bin="./stride/stride"
python="/home/wzarzecki/miniforge3/envs/seqme/bin/python"
cuda_idx=2


val_fasta="$results_dir/val_fasta"
val_pdb="$results_dir/val_pdb"
val_stride="$results_dir/val_stride"
results_file="$results_dir/results.csv"

# 1) convert pdb to fasta
cp -r $val_input_pdb $val_pdb ;
CUDA_VISIBLE_DEVICES=$cuda_idx bash scripts/protein-struct-pipe/protein_mpnn/run_inverse_folding.sh $val_pdb $val_fasta ;

# 2) make stride annotations
$python scripts/structures/utils/run_stride.py \
  --pdb_dir "$val_pdb" \
  --stride_dir "$val_stride" \
  --stride_binary "$stride_bin" ;

# 3) validation with seqme
$python scripts/structures/validation/validate_seqs.py \
  --val_seqs_dir $val_fasta \
  --val_stride_dir $val_stride \
  --ref_seqs_dir $ref_fasta \
  --ref_stride_dir $ref_stride \
  --n_ref $n_ref \
  --results_file $results_file ;