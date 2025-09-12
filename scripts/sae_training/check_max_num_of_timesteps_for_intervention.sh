#!/usr/bin/env bash

PYTHON_EXE="/home/wzarzecki/miniforge3/envs/rf/bin/python"
for i in  $(seq 1 30); do
  # generate config
  $PYTHON_EXE RFDiffSAE/scripts/generate_config.py --limit_timestep $i ;

  # run inference
  $PYTHON_EXE RFDiffSAE/scripts/run_inference.py \
  inference.output_prefix=example_outputs/xxx \
  contigmap.contigs=[100-200] \
  inference.num_designs=1 \
  inference.final_step=1  \
  inference.seed=0 \
  saeinterventions=block4 ;

  # check exit code
  if [ "$?" != "0" ]; then
    echo "error at $i last timestep"
    break
  fi
  echo "success $i last timesteps"
done
