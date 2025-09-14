import argparse
import os.path
from typing import Literal

from omegaconf import OmegaConf


def get_sae_conf_dict(
    sae_lambda_: float,
    non_pair_indices_path: str | None,
    pair_indices_path: str | None = None,
) -> dict | None:
    sae_pair_weights_path = "sae-ckpts/picked/patch_topk_expansion_factor16_k32_multi_topkFalse_auxk_alpha0.0lr0.0005_..activations_1200_block4_pair/block4_pair"
    sae_non_pair_weights_path = "sae-ckpts/picked/patch_topk_expansion_factor16_k64_multi_topkFalse_auxk_alpha0.0lr0.0001_..activations_1200_block4_non_pair/block4_non_pair"
    sae_hook_class_path = "rfdiffusion.sae.interventionhook.SAEInterventionHook"
    batch_size = 512
    if not non_pair_indices_path and not pair_indices_path:
        return None
    return {
        "_target_": sae_hook_class_path,
        "sae_pair_path": sae_pair_weights_path if pair_indices_path else None,
        "sae_non_pair_path": (
            sae_non_pair_weights_path if non_pair_indices_path else None
        ),
        "batch_size": batch_size,
        "intervention_indices_for_pair": pair_indices_path,
        "intervention_indices_for_non_pair": non_pair_indices_path,
        "intervention_lambda": sae_lambda_,
        "apply_relu_after_intervention": True
    }


def get_path_for_indices(
    timestep: int,
    label: str,
    base_dir: str,
    filename_suffix: str = "_indices.pt",
) -> str:
    filename = f"{label}_{timestep}{filename_suffix}"
    return os.path.join(base_dir, filename)


def get_dict_timestep_sae_hook_conf(
    indices_tensors_dir_for_non_pair: str,
    indices_tensors_dir_for_pair: str,
    label: str,
    timesteps: list[int],
    intervention_lambda_: float,
    intervention_on_pair_sae: bool,
    intervention_on_non_pair_sae: bool,
) -> dict[int, dict | None]:
    if not intervention_on_pair_sae and not intervention_on_non_pair_sae:
        raise ValueError(
            "Intervention config make sense with at least one sae for intervention"
        )
    conf = {}
    for timestep in timesteps:
        if intervention_on_pair_sae:
            pair_indices_path = get_path_for_indices(
                timestep=timestep,
                label=label,
                base_dir=indices_tensors_dir_for_pair,
            )
        else:
            pair_indices_path = None
        if intervention_on_non_pair_sae:
            non_pair_indices_path = get_path_for_indices(
                timestep=timestep,
                label=label,
                base_dir=indices_tensors_dir_for_non_pair,
            )
        else:
            non_pair_indices_path = None
        conf[timestep] = get_sae_conf_dict(
            intervention_lambda_,
            non_pair_indices_path,
            pair_indices_path,
        )
    return conf


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--highest_timestep", type=int, default=30)
    parser.add_argument("--lowest_timestep", type=int, default=2)
    parser.add_argument("--lambda_", type=float, default=-1.0)
    parser.add_argument("--output_config_name", type=str, default="block4.yaml")
    parser.add_argument("--base_dir_non_pair", default="/home/wzarzecki/ds_sae_latents_1600x/indices_non_pair")
    parser.add_argument("--base_dir_pair", default="/home/wzarzecki/ds_sae_latents_1600x/indices_pair")
    parser.add_argument("--label", default="Cytoplasm")
    args = parser.parse_args()

    saes_conf = get_dict_timestep_sae_hook_conf(
        args.base_dir_non_pair,
        args.base_dir_pair,
        args.label,
        list(range(args.lowest_timestep, args.highest_timestep + 1)),
        args.lambda_,
        intervention_on_pair_sae=False,
        intervention_on_non_pair_sae=True,
    )
    conf_manual = {
        "block": "simulator.main_block.4",
        "saes": saes_conf,
        # "saes": {},
    }
    conf = OmegaConf.create(conf_manual)
    conf_yaml = OmegaConf.to_yaml(conf)

    conf_path =  os.path.join("RFDiffSAE/config/saeinterventions/", args.output_config_name)
    with open(conf_path, "w") as f:
        f.write(conf_yaml)
