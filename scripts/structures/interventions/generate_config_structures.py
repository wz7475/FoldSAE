import argparse
import os.path

from omegaconf import OmegaConf


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda_", type=float, default=0.0)
    parser.add_argument(
        "--output_config_name",
        type=str,
        default="block4_cross_timesteps.yaml",
    )
    parser.add_argument(
        "--indices_path_non_pair",
        type=str,
        default="/home/wzarzecki/ds_secondary_struct/coefs/baseline_top_100.pt",
    )
    parser.add_argument(
        "--indices_path_pair",
        default=None,
    )
    parser.add_argument(
        "--sae_non_pair",
        type=str,
        default="sae-ckpts/picked/patch_topk_expansion_factor16_k64_multi_topkFalse_auxk_alpha0.0lr0.0001_..activations_1200_block4_non_pair/block4_non_pair",
    )
    parser.add_argument(
        "--sae_pair",
        default=None,
    )
    parser.add_argument(
        "--base_dir_for_config", default="RFDiffSAE/config/saeinterventions/"
    )
    return parser.parse_args()


def get_sae_conf_dict(
    sae_lambda_: float,
    sae_non_pair_weights_path: str,
    sae_pair_weights_path: str,
    non_pair_indices_path: str | None,
    pair_indices_path: str | None = None,
) -> dict | None:
    sae_hook_class_path = "rfdiffusion.sae.interventionhook.SAEInterventionHook"
    batch_size = 512
    print(f"pair_indices_path: {pair_indices_path}, {type(pair_indices_path)}")
    return {
        "_target_": sae_hook_class_path,
        "sae_pair_path": sae_pair_weights_path if pair_indices_path else None,
        "sae_non_pair_path": (
            sae_non_pair_weights_path if non_pair_indices_path else None
        ),
        "batch_size": batch_size,
        "intervention_indices_for_pair": pair_indices_path if pair_indices_path else None,
        "intervention_indices_for_non_pair": non_pair_indices_path if non_pair_indices_path else None,
        "intervention_lambda": sae_lambda_,
        "apply_relu_after_intervention": True,
    }


if __name__ == "__main__":
    args = parse_args()

    saes_conf = get_sae_conf_dict(
        args.lambda_,
        args.sae_non_pair,
        args.sae_pair,
        args.indices_path_non_pair,
        args.indices_path_pair,
    )
    conf_manual = {
        "block": "simulator.main_block.4",
        "saes": None,
        "sae_cross_timesteps": saes_conf,
    }
    conf = OmegaConf.create(conf_manual)
    conf_yaml = OmegaConf.to_yaml(conf)

    conf_path = os.path.join(args.base_dir_for_config, args.output_config_name)
    os.makedirs(os.path.dirname(conf_path), exist_ok=True)

    with open(conf_path, "w") as f:
        f.write(conf_yaml)

    print(f"Generated config file: {conf_path}")
