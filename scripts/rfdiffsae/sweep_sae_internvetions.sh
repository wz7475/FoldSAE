#!/usr/bin/env bash

start=${1:--1.1}
stop=${2:-1.15}
step=${3:-0.05}
lowest_timestep=${4:-2}
highest_timestep=${5:-2}
num_designs=${6:-50}
cuda_idx=${7:-1}

current=$start

while (( $(echo "$current < $stop" | bc -l) )); do
  bash scripts/rfdiffsae/sae_intervention_classifiers.sh $num_designs \
  "/home/wzarzecki/sae_interventions/coefs_as_base/${num_designs}_${lowest_timestep}_${highest_timestep}_${current}" \
  1 \
  $current \
  $lowest_timestep \
  $highest_timestep \
  $cuda_idx ;
  current=$(echo "$current + $step" | bc) ;
done
