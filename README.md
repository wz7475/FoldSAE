# SAEtoRuleRFDiffusion
## setup
```shell
git clone --recursive git@github.com:wz7475/SAEtoRuleRFDiffusion.git
```

enviroment
```shell
# we require installed conda and python 3.10 due to depencies of original RFdiffusion
bash ./scripts/envs/rfdiffsae.sh
```

Weights are available at this [link](https://drive.google.com/file/d/1tryqqxtXT6qlLvMOCKSfnfq_hW3y7vt-/view?usp=sharing).

## usage
- each directory contains subproject
- `scripts` contain script to run functionalities of each subproject - [check out here for more info](scripts/readme.md)

### 1) Block choice
First you need to find block which stores the knowledge about concepts of interest

#### Make ablations
Replace the simple call below with the documented CLI usage for scripts/ablations/ablations.sh - checkout script for docs.

Usage:
```shell
bash scripts/ablations/ablations.sh <start_main> <end_main> <start_extra> <end_extra> [num_designs] [final_step] [output_dir] [PYTHON_EXEC] [reference_dir]
```

to speed up process you may also use `scripts/structures/ablations/tmux_ablations.sh`


#### Eval ablations for concept of your interest
This repository includes an evaluation pipeline (scripts/structures/ablations/eval_pipeline.sh) to analyze ablation outputs and summarize structural changes without prescribing an exact invocation here. Checkout scripts for description of each argument.

Usage:
```shell
bash scripts/structures/ablations/eval_pipeline.sh [--pdb_dir <path>] [--stride_dir <path>] [--plot_dir <path>] [--results_file <file>] [--stride_binary <path>] [--python <path>]
```

### 2) SAE trainig
Train SAE in unsupervised manner on collected activations

#### Collect activations for chosen block
```shell
bash ./scripts/sae/collect_activations.sh [num_designs] [input_dir] [protein_length] [config_name] [final_step] [log_file] [PYTHON_RFDIF] [PYTHON_SAE]
```
put config into `RFDiffSAE/config/activations` it may look like
```yaml
map:
  simulator.main_block.4: block4
#  as many pairs as needed
dataset_path: temp_activations
keep_every_n_timestep: 10
keep_every_n_token: 10
save_activations_after_n_designs: 200
```


#### train SAE
run 
```shell
python universaldiffsae/src/scripts/train.py --dataset_path=/home/wzarzecki/ds_10000x_block_2/activations   --effective_batch_size=4096 --expansion_factor=4 --hookpoints=block4_non_pair --k=32 --lr=0.005 --max_trainer_steps=500 --wandb_project=SAE_main_02
```
for details check `RunConfig` in `universaldiffsae/src/sae/config.py`

to automate running SAE trainig with various hyper-params run grid search, you may use wandb setup
```shell
wandb sweep scripts/sae/sweep_train_sae.yaml
wandb agent <sweep_id>
#or
bash scripts/sae/tmux_wandb_agents.sh <sweep_id_with_prefix> <cuda_idx> <num_of_agents>
```

### 3) find feature indicies by training probes
train probing models and map their coefficients to feature indices -> let's learn which feature are responsible for concepts of interest

#### create auxiliary dataset with latents and associated concepts
for secondary structure you may use this script
```shell
bash scripts/structures/create_ds/probes_ds_from_block_act.sh <input_dir> <block_act_dir> <log> <sae_for_pair> <sae_for_non_pair> <stride_bin> <python_bin>
```

#### train probes on it
for secondary structure you may use
```shell
bash scripts/structures/probes/probes_sweep.sh <dataset_dir> <dir_to_store_coefs> <dir_to_store_results> <python_bin>
```

#### choose number of coeficients via visualisation of discriminative features
run notebook `./notebooks/strucutres/coefs_visualization.ipynb` to analyze how many discriminitive features can be found for given treshold


### 4) causal intervention with SAE
run shell script 
#### run interventions
```shell
bash sweep_structure_interventions.sh [lambda_start] [lambda_stop] [lambda_step] [threshold_start] [threshold_stop] [threshold_step] [first_classes] [input_dir] [num_designs] [seed] [indices_path_pair] [sae_non_pair] [sae_pair] [base_dir_for_config] [python] [prefix] [length] [coef_helix] [coef_beta] [coefs_output_dir]
```

This script performs a grid search over $\lambda$ and threshold parameters to run structure interventions, generating a configurable number of protein designs for specific secondary structure classes. It utilizes pre-trained Sparse Autoencoders (SAEs) and coefficients to guide the RFDiffusion process.


You can split across tmux sessions running
```shell
bash run_sweep_interventions.sh [seed] [lambda_start] [lambda_end] [lambda_step] [threshold_a] [threshold_b] [threshold_c] [num_designs] [classes_string]
```
#### eval interventions
```shell
bash scripts/structures/interventions/eval_pipeline.sh <structures_source_dir_from_sweep> <results_dir>
```
#### validate structures
```shell
bash scripts/structures/validation/val_dir_of_dirs.sh <structures_source_dir_from_sweep/pdb> <val_results_dir> <n_ref>
```
