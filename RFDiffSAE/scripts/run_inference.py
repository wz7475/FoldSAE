#!/usr/bin/env python
"""
Inference script.

To run with base.yaml as the config,

> python run_inference.py

To specify a different config,

> python run_inference.py --config-name symmetry

where symmetry can be the filename of any other config (without .yaml extension)
See https://hydra.cc/docs/advanced/hydra-command-line-flags/ for more options.

"""

import re
import os
import time
import pickle
from uuid import uuid4
import torch
from omegaconf import OmegaConf
import hydra
import logging

from rfdiffusion.activations import add_meta_data_and_reduce_activations, save_activations_shard
from rfdiffusion.util import writepdb_multi, writepdb
from rfdiffusion.inference import utils as iu
from hydra.core.hydra_config import HydraConfig
import numpy as np
import random
import glob


def make_deterministic(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


@hydra.main(version_base=None, config_path="../config", config_name="base")
def main(conf: HydraConfig) -> None:
    log = logging.getLogger(__name__)
    if conf.inference.make_deterministic:
        make_deterministic(seed=0) # hard seed same for sampler initialization, later each design will it's own seed for sampling
    seed = conf.inference.seed

    # Check for available GPU and print result of check
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(torch.cuda.current_device())
        log.info(
            f"Found GPU with device_name {device_name}. Will run RFdiffusion on {device_name}"
        )
    else:
        log.info("////////////////////////////////////////////////")
        log.info("///// NO GPU DETECTED! Falling back to CPU /////")
        log.info("////////////////////////////////////////////////")

    # Initialize sampler and target/contig.
    sampler = iu.sampler_selector(conf)

    # Loop over number of designs to sample.
    existing = glob.glob(sampler.inf_conf.output_prefix + "*.pdb")
    indices = [-1]
    for e in existing:
        print(e)
        m = re.match(r".*_(\d+)\.pdb$", e)
        print(m)
        if not m:
            continue
        m = m.groups()[0]
        indices.append(int(m))
    biggest_found_idx = max(indices) + 1


    activations_for_all_designs = {}
    save_activs_every = getattr(conf.activations, "save_activations_after_n_designs", 1)
    shard_idx = 0
    # Allow specifying start_design_index to generate specific design numbers
    for i_des in range(seed, seed + sampler.inf_conf.num_designs):
        if conf.inference.make_deterministic:
            make_deterministic(i_des)

        start_time = time.time()
        out_prefix = f"{sampler.inf_conf.output_prefix}_{i_des}"
        if conf.inference.use_random_suffix_for_new_design:
            out_prefix += f"_{uuid4()}"
        structure_id = os.path.split(out_prefix)[1]
        log.info(f"Making design {out_prefix}")
        if sampler.inf_conf.cautious and os.path.exists(out_prefix + ".pdb"):
            log.info(
                f"(cautious mode) Skipping this design because {out_prefix}.pdb already exists."
            )
            continue
        if sampler.inf_conf.cautious and os.path.exists(out_prefix + ".txt"):
            log.info(
                f"(cautious mode) Skipping this design because {out_prefix} failed before - {out_prefix}.txt  already exists."
            )
            continue

        x_init, seq_init = sampler.sample_init()
        denoised_xyz_stack = []
        px0_xyz_stack = []
        seq_stack = []
        plddt_stack = []

        x_t = torch.clone(x_init)
        seq_t = torch.clone(seq_init)
        activations_per_design = {}
        try:
            for t in range(
                int(sampler.t_step_input), sampler.inf_conf.final_step - 1, -1
            ):
                px0, x_t, seq_t, plddt, activations_dict = sampler.sample_step(
                    t=t,
                    x_t=x_t,
                    seq_init=seq_t,
                    final_step=sampler.inf_conf.final_step,
                    structure_id=structure_id,
                )
                px0_xyz_stack.append(px0)
                denoised_xyz_stack.append(x_t)
                seq_stack.append(seq_t)
                plddt_stack.append(plddt[0])
                add_meta_data_and_reduce_activations(
                    activations_per_design,
                    activations_dict,
                    t,
                    conf.activations.keep_every_n_timestep,
                    conf.activations.keep_every_n_token,
                )
        except Exception as e:
            print(f"exception {e}. Skiping to next one. Log into {out_prefix}.txt")
            with open(f"{out_prefix}.txt", "w") as f:
                f.write(
                    f"Structure {structure_id}. Timestep {t}. Caught exception {e}. Skipped design"
                )
            continue
            raise

        if conf.activations.dataset_path:
            activations_for_all_designs[structure_id] = activations_per_design

        # Save activations every N designs as a new shard
        if conf.activations.dataset_path and len(activations_for_all_designs) >= save_activs_every:
            save_activations_shard(activations_for_all_designs, conf.activations.dataset_path, shard_idx)
            activations_for_all_designs.clear()
            shard_idx += 1

        # ...existing code for flipping, saving pdb, trb, trajectory, etc...
        denoised_xyz_stack = torch.stack(denoised_xyz_stack)
        denoised_xyz_stack = torch.flip(denoised_xyz_stack, [0,])
        px0_xyz_stack = torch.stack(px0_xyz_stack)
        px0_xyz_stack = torch.flip(px0_xyz_stack, [0,])
        plddt_stack = torch.stack(plddt_stack)
        os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
        final_seq = seq_stack[-1]
        final_seq = torch.where(torch.argmax(seq_init, dim=-1) == 21, 7, torch.argmax(seq_init, dim=-1))
        bfacts = torch.ones_like(final_seq.squeeze())
        bfacts[torch.where(torch.argmax(seq_init, dim=-1) == 21, True, False)] = 0
        out = f"{out_prefix}.pdb"
        writepdb(
            out,
            denoised_xyz_stack[0, :, :4],
            final_seq,
            sampler.binderlen,
            chain_idx=sampler.chain_idx,
            bfacts=bfacts,
        )
        if conf.inference.write_trb:
            trb = dict(
                config=OmegaConf.to_container(sampler._conf, resolve=True),
                plddt=plddt_stack.cpu().numpy(),
                device=torch.cuda.get_device_name(torch.cuda.current_device())
                if torch.cuda.is_available()
                else "CPU",
                time=time.time() - start_time,
            )
            if hasattr(sampler, "contig_map"):
                for key, value in sampler.contig_map.get_mappings().items():
                    trb[key] = value
            with open(f"{out_prefix}.trb", "wb") as f_out:
                pickle.dump(trb, f_out)
        if sampler.inf_conf.write_trajectory:
            traj_prefix = (
                os.path.dirname(out_prefix) + "/traj/" + os.path.basename(out_prefix)
            )
            os.makedirs(os.path.dirname(traj_prefix), exist_ok=True)
            out = f"{traj_prefix}_Xt-1_traj.pdb"
            writepdb_multi(
                out,
                denoised_xyz_stack,
                bfacts,
                final_seq.squeeze(),
                use_hydrogens=False,
                backbone_only=False,
                chain_ids=sampler.chain_idx,
            )
            out = f"{traj_prefix}_pX0_traj.pdb"
            writepdb_multi(
                out,
                px0_xyz_stack,
                bfacts,
                final_seq.squeeze(),
                use_hydrogens=False,
                backbone_only=False,
                chain_ids=sampler.chain_idx,
            )
        log.info(f"Finished design in {(time.time() - start_time) / 60:.2f} minutes")

    # Save any remaining activations after loop
    if conf.activations.dataset_path and activations_for_all_designs:
        save_activations_shard(activations_for_all_designs, conf.activations.dataset_path, shard_idx)


if __name__ == "__main__":
    main()
