
input_dir=$1;

if [[ ! "$input_dir" =~ /$ ]]; then
  input_dir="$input_dir/"
fi

for block in "$input_dir"*; do
    bash scripts/clean/preds-for-dir-with-fastas.sh "$block" "${block}/enzymes.csv" "fa" 2 ;
    echo "$block";
done