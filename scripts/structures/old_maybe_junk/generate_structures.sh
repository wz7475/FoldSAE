
#!/usr/bin/env bash
# Enable strict mode. pipefail is not supported by all /bin/sh implementations (e.g. dash),
# so only enable it when actually running under bash.
if [ -n "${BASH_VERSION-}" ]; then
	set -euo pipefail
else
	# POSIX sh fallback: keep -e and -u but skip pipefail
	set -eu
fi

usage() {
	cat <<EOF
Usage: $0 <input_dir> [prefix] [num_designs] [lambda_val] [indices_path] [python_executable] [length] [seed]

Positional args:
	input_dir        base output directory (required)
	prefix           file prefix for designs (default: design)
	num_designs      number of designs to request (default: 1)
	lambda_val       lambda value for SAE intervention (default: 0.0)
	indices_path     path to indices file (default: indices.pt)
	python_executable python to run RF scripts (default: /home/wzarzecki/miniforge3/envs/rf/bin/python)
	length           design length (default: 150)
	seed             random seed (default: 1)

This script:
	- generates a SAE intervention config via RFDiffSAE/scripts/generate_config_structures.py
	- runs RFDiffusion via RFDiffSAE/scripts/run_inference.py using that config

Example:
	$0 outputs/designs design 5 0.0 indices.pt
EOF
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
	usage
	exit 0
fi

if [[ -z ${1:-} ]]; then
	echo "ERROR: input_dir is required"
	usage
	exit 2
fi

INPUT_DIR=${1}
PREFIX=${2:-design}
NUM_DESIGNS=${3:-1}
LAMBDA_VAL=${4:-0.0}
INDICES_PATH=${5:-indices.pt}
PYTHON=${6:-/home/wzarzecki/miniforge3/envs/rf/bin/python}
LENGTH=${7:-150}
SEED=${8:-1}

INDICES_FILENAME=$(basename "${INDICES_PATH}")
INDICES_FILENAME="${INDICES_FILENAME%.*}"

LAMBDA_SUBDIR="${INPUT_DIR}/${INDICES_FILENAME}/${LAMBDA_VAL}"
mkdir -p "${LAMBDA_SUBDIR}"

CONFIG_NAME="lambda_${LAMBDA_VAL}_${INDICES_FILENAME}.yaml"

echo "[generate_structures] using python: ${PYTHON}"
echo "[generate_structures] indices: ${INDICES_PATH} -> ${INDICES_FILENAME}"
echo "[generate_structures] config: ${CONFIG_NAME}"
echo "[generate_structures] output prefix dir: ${LAMBDA_SUBDIR}"

echo "Generating SAE config..."
"${PYTHON}" RFDiffSAE/scripts/generate_config_structures.py \
	--lambda_ "${LAMBDA_VAL}" \
	--indices_path "${INDICES_PATH}" \
	--output_config_name "${CONFIG_NAME}"

echo "Running RFdiffusion inference..."
CMD=(
	"${PYTHON}" RFDiffSAE/scripts/run_inference.py
	"inference.output_prefix=${LAMBDA_SUBDIR}/${PREFIX}"
	"contigmap.contigs=[${LENGTH}-${LENGTH}]"
	"inference.num_designs=${NUM_DESIGNS}"
	"inference.final_step=1"
	"inference.use_random_suffix_for_new_design=False"
	"inference.seed=${SEED}"
	"saeinterventions=${CONFIG_NAME}"
)

echo "Executing: ${CMD[*]}"
"${CMD[@]}"

RC=$?
echo "RFdiffusion finished with exit code: ${RC}"
exit ${RC}

