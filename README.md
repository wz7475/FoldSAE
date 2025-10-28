# SAEtoRuleRFDiffusion
## setup
```shell
git clone --recursive git@github.com:wz7475/SAEtoRuleRFDiffusion.git
```

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
bash ./scripts/sae/collect_activations.sh [num_designs] [input_dir] [protein_length] [config_name] [final_step] [log_file] [PYTHON_RFDIF]
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
for details check `RunConfig` in this scrip

to automate running SAE trainig with various hyper-params run grid search, you may use wandb setup
```shell
wandb sweep scripts/sae/sweep_train_sae.yaml
wandb agent <sweep_id>
```

