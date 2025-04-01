
dir_with_fastas=$1;
output_csv=$2;
fasta_ext=${3-"fa"};
cuda=${4-"-1"};

temp_fasta="temp.fasta";
find "$dir_with_fastas" -name "*.$fasta_ext" -exec cat {} + > "$temp_fasta";

conda run -n clean python CLEAN/app/rename_fasta_id.py "$temp_fasta" "$temp_fasta";
bash scripts/clean/preds-for-single-fasta.sh "$temp_fasta" "$output_csv" "$cuda" ;
rm "$temp_fasta";