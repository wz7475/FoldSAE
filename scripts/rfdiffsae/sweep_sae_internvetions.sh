#!/usr/bin/env bash



start=-3
stop=2
step=1
current=$start

while (( $(echo "$current < $stop" | bc -l) )); do
  bash scripts/rfdiffsae/sae_intervention_classifiers.sh 40 /home/wzarzecki/sae_interventions/k_40_$current 1 $current
  current=$(echo "$current + $step" | bc)
done
