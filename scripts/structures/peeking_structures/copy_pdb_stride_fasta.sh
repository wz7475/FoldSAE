#!/usr/bin/env bash

set -euo pipefail

# Positional args (mirroring style of structure_interventions.sh)
# 1: base_dir (required) - output root for pdb/ stride/ fasta/
# 2: peek_base_dir (required) - source root containing lambda_* dirs
# 3: ids (required; space or comma separated list)
# 4: id_prefix (optional; default design_)
# 5: subdir_suffix (optional; default 0.15_beta)
# 6: stride_binary (optional; path to STRIDE binary)
# 7: python_exec (optional; python for ProteinMPNN step)
BASE_DIR=${1:-"temp_peeked_pdb"}
PEEK_BASE_DIR=${2:-"/home/wzarzecki/ds_10000x/results/interventions/discriminant_coefs_2_100x/pdb"}
IDS=${3:-"0 28 49 79 90 1 21 33 45 6"}
ID_PREFIX=${4:-"design_"}
SUBDIR_SUFFIX=${5:-"thr_0.15_beta"}
STRIDE_BINARY=${6:-"stride/stride"}
PYTHON_EXEC_FOR_MPNN=${7:-"/home/wzarzecki/miniforge3/envs/bio_emb/bin/python"}

if [[ -z "${BASE_DIR}" || -z "${PEEK_BASE_DIR}" || -z "${IDS}" ]]; then
    echo "Usage: $0 <base_dir> <peek_base_dir> <ids> [id_prefix] [subdir_suffix] [stride_binary] [python_exec]" >&2
    echo "  base_dir: output root for pdb/ stride/ fasta/" >&2
    echo "  peek_base_dir: source with lambda_* dirs" >&2
    echo "  ids: comma or space separated list of seed IDs" >&2
    exit 1
fi

PDB_DIR="${BASE_DIR%/}/pdb"
STRIDE_DIR="${BASE_DIR%/}/stride"
FASTA_DIR="${BASE_DIR%/}/fasta"

mkdir -p "$PDB_DIR" "$STRIDE_DIR" "$FASTA_DIR"

ROOT_DIR="/data/wzarzecki/SAEtoRuleRFDiffusion"

# 1) Copy selected PDBs using copy_peeked_pdb.py
COPY_SCRIPT="$ROOT_DIR/scripts/structures/peeking_structures/copy_peeked_pdb.py"
if [[ ! -f "$COPY_SCRIPT" ]]; then
    echo "Error: copy_peeked_pdb.py not found at $COPY_SCRIPT" >&2
    exit 1
fi

echo "[1/3] Copying PDBs -> $PDB_DIR"
# Expand comma-separated IDs into space-separated, then pass unquoted so each ID becomes a separate arg
IDS_EXPANDED=${IDS//,/ }
python3 "$COPY_SCRIPT" \
  --base_dir "$PEEK_BASE_DIR" \
  --target_dir "$PDB_DIR" \
  --ids $IDS_EXPANDED \
  --id_prefix "$ID_PREFIX" \
  --subdir_suffix "$SUBDIR_SUFFIX"

# 2) Generate STRIDE files
RUN_STRIDE_PY="$ROOT_DIR/scripts/structures/utils/run_stride.py"
if [[ ! -f "$RUN_STRIDE_PY" ]]; then
    echo "Error: run_stride.py not found at $RUN_STRIDE_PY" >&2
    exit 1
fi

echo "[2/3] Generating STRIDE -> $STRIDE_DIR"
if [[ -n "$STRIDE_BINARY" ]]; then
  python3 "$RUN_STRIDE_PY" --pdb_dir "$PDB_DIR" --stride_dir "$STRIDE_DIR" --stride_binary "$STRIDE_BINARY"
else
  python3 "$RUN_STRIDE_PY" --pdb_dir "$PDB_DIR" --stride_dir "$STRIDE_DIR"
fi

# 3) Generate FASTA with ProteinMPNN inverse folding
RUN_IFOLD_SH="$ROOT_DIR/scripts/protein-struct-pipe/protein_mpnn/run_inverse_folding.sh"
if [[ ! -f "$RUN_IFOLD_SH" ]]; then
    echo "Error: run_inverse_folding.sh not found at $RUN_IFOLD_SH" >&2
    exit 1
fi

echo "[3/3] Generating FASTA -> $FASTA_DIR"

# If there are lambda_* subdirectories under PDB_DIR, run per subdir; otherwise, run once for PDB_DIR
has_lambda_subdirs=false
for d in "$PDB_DIR"/lambda_*; do
  if [[ -d "$d" ]]; then
    has_lambda_subdirs=true
    break
  fi
done

if [[ "$has_lambda_subdirs" == true ]]; then
  for d in "$PDB_DIR"/lambda_*; do
    [[ -d "$d" ]] || continue
    subdir_name=$(basename "$d")
    out_dir="$FASTA_DIR/$subdir_name"
    mkdir -p "$out_dir"
    echo "  - Processing $subdir_name"
    if [[ -n "$PYTHON_EXEC_FOR_MPNN" ]]; then
      bash "$RUN_IFOLD_SH" "$d" "$out_dir" "$PYTHON_EXEC_FOR_MPNN"
    else
      bash "$RUN_IFOLD_SH" "$d" "$out_dir"
    fi
  done
else
  if [[ -n "$PYTHON_EXEC_FOR_MPNN" ]]; then
    bash "$RUN_IFOLD_SH" "$PDB_DIR" "$FASTA_DIR" "$PYTHON_EXEC_FOR_MPNN"
  else
    bash "$RUN_IFOLD_SH" "$PDB_DIR" "$FASTA_DIR"
  fi
fi

echo "Done. Outputs:"
echo "  PDB:    $PDB_DIR"
echo "  STRIDE: $STRIDE_DIR"
echo "  FASTA:  $FASTA_DIR"


