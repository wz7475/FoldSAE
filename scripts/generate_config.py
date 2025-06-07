import argparse
import os.path
from typing import Literal

from omegaconf import OmegaConf


def get_sae_conf_dict(
    sae_multiplier: float,
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
        "sae_non_pair_path": sae_non_pair_weights_path if non_pair_indices_path else None,
        "batch_size": batch_size,
        "intervention_indices_for_pair": pair_indices_path,
        "intervention_indices_for_non_pair": non_pair_indices_path,
        "intervention_multiplier": sae_multiplier,
    }


def get_path_for_indices(
    timestep: int,
    label: str,
    base_dir: str,
    sae_type: Literal["pair", "non_pair"],
    filename_suffix: str = "_indices.pt",
) -> str:
    dir_path = os.path.join(base_dir, sae_type)
    filename = f"{label}_{timestep}{filename_suffix}"
    return os.path.join(dir_path, filename)


def get_dict_timestep_sae_hook_conf(
    indices_tensors_dir: str,
    label: str,
    timesteps: list[int],
    intervention_multiplier: float,
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
                base_dir=indices_tensors_dir,
                sae_type="pair",
            )
        else:
            pair_indices_path = None
        if intervention_on_non_pair_sae:
            non_pair_indices_path = get_path_for_indices(
                timestep=timestep,
                label=label,
                base_dir=indices_tensors_dir,
                sae_type="non_pair",
            )
        else:
            non_pair_indices_path = None
        conf[timestep] = get_sae_conf_dict(
            intervention_multiplier, non_pair_indices_path, pair_indices_path
        )
    return conf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit_timestep", type=int, default=30)
    parser.add_argument("--multiplier", type=float, default=-1.0)
    args = parser.parse_args()
    limit_timestep = args.limit_timestep
    multiplier = args.multiplier

    saes_conf = get_dict_timestep_sae_hook_conf(
            "/home/wzarzecki/ds_sae_latents_1600x/indices",
            "Cytoplasm",
            list(range(2, limit_timestep + 1)),
            multiplier,
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

    conf_path = "RFDiffSAE/config/saeinterventions/block4.yaml"
    with open(conf_path, "w") as f:
        f.write(conf_yaml)

    # saes = {
    #     name: instantiate(instance_conf) if instance_conf else None
    #     for name, instance_conf in conf.saes.items()
    # }
    # print(saes)
