"""
Train sparse autoencoders on activations from a diffusion model.
"""

import os
import sys
from contextlib import nullcontext, redirect_stdout

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import torch
import torch.distributed as dist
from simple_parsing import parse

from src.sae.config import RunConfig
from src.sae.trainer import SaeTrainer
from src.tools.dataset import load_datasets_from_dir_of_dirs, load_ds_from_one_dir


def run():
    local_rank = os.environ.get("LOCAL_RANK")
    ddp = local_rank is not None
    rank = int(local_rank) if ddp else 0

    if ddp:
        torch.cuda.set_device(int(local_rank))
        dist.init_process_group("nccl")

        if rank == 0:
            print(f"Using DDP across {dist.get_world_size()} GPUs.")

    args = parse(RunConfig)
    # add output_or_diff to the run name
    args.run_name = args.run_name + f"lr{args.lr}_{''.join(args.dataset_path[0].split('/'))}_{args.hookpoints[0]}"

    dtype = torch.float32
    if args.mixed_precision == "fp16":
        dtype = torch.float16
    elif args.mixed_precision == "bf16" and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    args.dtype = dtype
    print(f"Training in {dtype=}")
    # Awkward hack to prevent other ranks from duplicating data preprocessing
    dataset_dict = {}
    if not ddp or rank == 0:
        if args.ds_with_block_and_timestep_dirs:
            dataset = load_datasets_from_dir_of_dirs(os.path.join(args.dataset_path[0], args.hookpoints[0]), dtype, columns=[args.activation_column])
        else:
            dataset = load_ds_from_one_dir(args.dataset_path[0], dtype, columns=[args.activation_column, args.activations_type_column])
            dataset = dataset.filter(lambda row: row[args.activations_type_column] == args.activations_type)
        dataset = dataset.shuffle(args.seed)
        dataset_dict[args.hookpoints[0]] = dataset


    # Prevent ranks other than 0 from printing
    with nullcontext() if rank == 0 else redirect_stdout(None):
        trainer = SaeTrainer(args, dataset_dict)

        trainer.fit(args.max_trainer_steps)


if __name__ == "__main__":
    run()
