import argparse
import os.path

from omegaconf import OmegaConf


def get_sae_conf_dict(
    sae_lambda_: float,
    non_pair_indices_path: str | None,
    pair_indices_path: str | None = None,
) -> dict | None:
    sae_pair_weights_path = "sae-ckpts/RFDiffSAE/patch_topk_expansion_factor16_k32_multi_topkFalse_auxk_alpha0.0lr1e-05_datawzarzeckiactivations_block4_pair/block4_pair"
    sae_non_pair_weights_path = "sae-ckpts/RFDiffSAE/patch_topk_expansion_factor16_k32_multi_topkFalse_auxk_alpha0.0lr1e-05_datawzarzeckiactivations_block4_non_pair/block4_non_pair"
    sae_hook_class_path = "rfdiffusion.sae.interventionhook.SAEInterventionHook"
    batch_size = 512
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
        "apply_relu_after_intervention": True,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda_", type=float, default=-1.0)
    parser.add_argument(
        "--output_config_name",
        type=str,
        default="block4_cross_timesteps.yaml",
    )
    parser.add_argument(
        "--indices_path",
        type=str,
        default="/home/wzarzecki/ds_secondary_struct/coefs/baseline_top_100.pt",
    )
    parser.add_argument(
        "--indices_non_pair",
        type=str,
        default=None,
    )
    args = parser.parse_args()

    saes_conf = get_sae_conf_dict(
        args.lambda_,
        args.indices_path,
        None
    )
    conf_manual = {
        "block": "simulator.main_block.4",
        "saes": None,
        "sae_cross_timesteps": saes_conf,
    }
    conf = OmegaConf.create(conf_manual)
    conf_yaml = OmegaConf.to_yaml(conf)

    conf_path = os.path.join(
        "RFDiffSAE/config/saeinterventions/", args.output_config_name
    )
    # Ensure directory exists
    os.makedirs(os.path.dirname(conf_path), exist_ok=True)
    
    with open(conf_path, "w") as f:
        f.write(conf_yaml)
    
    print(f"Generated config file: {conf_path}")
