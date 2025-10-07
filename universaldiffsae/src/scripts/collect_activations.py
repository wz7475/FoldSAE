"""
Collect activations from a diffusion model for a given hookpoint and save them to a file.
"""

import os
import sys

from simple_parsing import parse

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from accelerate import Accelerator
from diffusers import DiffusionPipeline

import src.hooked_model.scheduler
from src.hooked_model.hooked_model import HookedDiffusionModel
from src.hooked_model.utils import (
    get_timesteps,
)
from src.sae.cache_activations_runner import CacheActivationsRunner
from src.sae.config import CacheActivationsRunnerConfig


def run():
    args = parse(CacheActivationsRunnerConfig)
    accelerator = Accelerator()
    # define model
    pipe = DiffusionPipeline.from_pretrained(
        args.model_name, torch_dtype=args.dtype, use_safetensors=True, vae=None
    ).to(accelerator.device)
    model = pipe.unet
    scheduler = src.hooked_model.scheduler.DDIMScheduler.from_config(
        pipe.scheduler.config
    )
    hooked_model = HookedDiffusionModel(
        model=model,
        scheduler=scheduler,
        encode_prompt=pipe.encode_prompt,
        get_timesteps=get_timesteps,
        vae=pipe.vae,
    )

    CacheActivationsRunner(args, hooked_model, accelerator).run()


if __name__ == "__main__":
    run()
