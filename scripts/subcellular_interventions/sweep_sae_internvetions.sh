#!/usr/bin/env bash

start=${1:--1.1}
stop=${2:-1.15}
step=${3:-0.05}
lowest_timestep=${4:-2}
highest_timestep=${5:-2}
num_designs=${6:-50}
label=${7:-"Cytoplasm"}
cuda_idx=${8:-1}
dir_name=${9:-"sign_as_base"}
num_of_coefs=${10:-10}
coefs_src_dir=${11:-"/home/wzarzecki/ds_sae_latents_1600x/coefs/non_pair"}

current=$start

while (( $(echo "$current < $stop" | bc -l) )); do
  bash scripts/rfdiffsae/sae_intervention_classifiers.sh $num_designs \
  "/home/wzarzecki/sae_interventions/$dir_name/${num_designs}_${lowest_timestep}_${highest_timestep}_${current}" \
  1 \
  $current \
  $lowest_timestep \
  $highest_timestep \
  $label \
  $cuda_idx \
  $num_of_coefs \
  $coefs_src_dir ;
  current=$(echo "$current + $step" | bc) ;
done
