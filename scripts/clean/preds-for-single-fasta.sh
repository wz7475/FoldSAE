
input_fasta=$1;
output_csv="$2";
cuda=${3-"-1"}


filename=$(basename "$input_fasta" | sed 's/\.[^.]*$//');
input_intermediate_path="CLEAN/app/data/inputs/${filename}.fasta";

cp "$input_fasta" "$input_intermediate_path";

cd CLEAN/app || exit ;
conda run -n clean python  CLEAN_infer_fasta.py --fasta_data "$filename" -c "$cuda";
cd - || exit ;

mv "CLEAN/app/results/inputs/${filename}_maxsep.csv" "$output_csv";
rm "$input_intermediate_path";

