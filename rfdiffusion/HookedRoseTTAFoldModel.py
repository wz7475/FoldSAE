from typing import Callable

import torch

from rfdiffusion.RoseTTAFoldModel import RoseTTAFoldModule
from rfdiffusion.Track_module import IterativeSimulator, IterBlockOutput
from rfdiffusion.sae.blockoutputtransformation import transform_from_iter_block_output
from rfdiffusion.sae.interventionhook import SAEInterventionHook
from rfdiffusion.sae.universalsae import Sae


class HookedRoseTTAFoldModule(RoseTTAFoldModule):
    def __init__(self,
                 n_extra_block,
                 n_main_block,
                 n_ref_block,
                 d_msa,
                 d_msa_full,
                 d_pair,
                 d_templ,
                 n_head_msa,
                 n_head_pair,
                 n_head_templ,
                 d_hidden,
                 d_hidden_templ,
                 p_drop,
                 d_t1d,
                 d_t2d,
                 T,  # total timesteps (used in timestep emb
                 use_motif_timestep,  # Whether to have a distinct emb for motif
                 freeze_track_motif,  # Whether to freeze updates to motif in track
                 SE3_param_full={'l0_in_features': 32, 'l0_out_features': 16, 'num_edge_features': 32},
                 SE3_param_topk={'l0_in_features': 32, 'l0_out_features': 16, 'num_edge_features': 32},
                 input_seq_onehot=False,  # For continuous vs. discrete sequence
                 activations=None,
                 ablations=None,
                 sae_interventions=None,
                 skipped_main_block=-1,
                 skipped_extra_block=-1,
                 ):
        super().__init__(n_extra_block, n_main_block, n_ref_block, d_msa, d_msa_full, d_pair, d_templ, n_head_msa,
                         n_head_pair, n_head_templ, d_hidden, d_hidden_templ, p_drop, d_t1d, d_t2d, T,
                         use_motif_timestep, freeze_track_motif, SE3_param_full, SE3_param_topk, input_seq_onehot)
        self.activations_map = activations["map"] if activations["map"] else {}
        self.blocks_for_ablation = ablations["ablations"]
        self.sae_interventions = sae_interventions
        self.simulator = IterativeSimulator(n_extra_block=n_extra_block,
                                            n_main_block=n_main_block,
                                            n_ref_block=n_ref_block,
                                            d_msa=d_msa, d_msa_full=d_msa_full,
                                            d_pair=d_pair, d_hidden=d_hidden,
                                            n_head_msa=n_head_msa,
                                            n_head_pair=n_head_pair,
                                            SE3_param_full=SE3_param_full,
                                            SE3_param_topk=SE3_param_topk,
                                            p_drop=p_drop,
                                            skipped_main_block=skipped_main_block,
                                            skipped_extra_block=skipped_extra_block,
                                            )
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def _register_hook_by_path(self, block_path: str, hook: Callable):
        module = self
        for block in block_path.split('.'):
            module = getattr(module, block)
        return module.register_forward_hook(hook)

    def _register_cache_hooks(self, cache: dict):
        def getActivation(name):
            def hook(model, input, output):
                if isinstance(output, IterBlockOutput):
                    cache[f"{name}_pair"], cache[f"{name}_non_pair"] = transform_from_iter_block_output(output)
                elif isinstance(output, torch.Tensor):
                    cache[name] = output.detach().cpu()
                else:
                    raise ValueError("Only IterBlockOutput tuple and Tensor supported")

            return hook

        return [self._register_hook_by_path(block_path, getActivation(self.activations_map[block_path])) for block_path
                in self.activations_map]

    def _register_ablation_hooks(self):

        class AblateHook:
            """
            supports only blocks/layers whose output has same shape as input
            """

            @torch.no_grad()
            def __call__(self, module, input, output):
                if isinstance(input, tuple):
                    return (input[0],)
                return input[0]

        return [
            self._register_hook_by_path(block_path, AblateHook()) for block_path in self.blocks_for_ablation
        ]

    def _register_sae_intervention_hooks(self, structure_id: None | str = None, timestep: None | str = None):

        sae_batch_size = self.sae_interventions["batch_size"]
        intervention_indices_for_pair = torch.load(
            self.sae_interventions.intervention_indices_pair_path, weights_only=True) \
            if self.sae_interventions.intervention_indices_pair_path is not None else None
        intervention_indices_for_non_pair = torch.load(self.sae_interventions.intervention_indices_non_pair_path,
                                                       weights_only=True) \
            if self.sae_interventions.intervention_indices_non_pair_path is not None else None
        return [
            self._register_hook_by_path(block_path, SAEInterventionHook(
                Sae.load_from_disk(self.sae_interventions.sae_pair_path, self.device),
                Sae.load_from_disk(self.sae_interventions.sae_non_pair_path, self.device),
                sae_batch_size,
                structure_id,
                self.sae_interventions.sae_latents_base_dir,
                timestep,
                intervention_indices_for_pair,
                intervention_indices_for_non_pair,
                self.sae_interventions.intervention_multiplier
            ))
            for block_path in self.sae_interventions["blocks"]
        ]

    def run_with_hooks(self, msa_latent, msa_full, seq, xyz, idx, t,
                       t1d=None, t2d=None, xyz_t=None, alpha_t=None,
                       msa_prev=None, pair_prev=None, state_prev=None,
                       return_raw=False, return_full=False, return_infer=False,
                       use_checkpoint=False, motif_mask=None, i_cycle=None, n_cycle=None,
                       structure_id: None | str = None):

        activations_dict = {}

        activation_hooks = self._register_cache_hooks(activations_dict)
        ablation_hooks = self._register_ablation_hooks()
        sae_intervention_hooks = self._register_sae_intervention_hooks(structure_id, t.item())

        output = self.forward(msa_latent, msa_full, seq, xyz, idx, t,
                              t1d, t2d, xyz_t, alpha_t,
                              msa_prev, pair_prev, state_prev,
                              return_raw, return_full, return_infer,
                              use_checkpoint, motif_mask, i_cycle, n_cycle)

        for hook in activation_hooks + ablation_hooks + sae_intervention_hooks:
            hook.remove()

        return *output, activations_dict
