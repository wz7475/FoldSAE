import os
import random
from typing import Callable, Tuple

import torch
import torch.nn as nn
from datasets import Dataset
from opt_einsum import contract as einsum
from torch.utils.data import DataLoader, TensorDataset

from rfdiffusion.AuxiliaryPredictor import DistanceNetwork, MaskedTokenNetwork, ExpResolvedNetwork, LDDTNetwork
from rfdiffusion.Embeddings import MSA_emb, Extra_emb, Templ_emb, Recycling
from rfdiffusion.Track_module import IterativeSimulator, IterBlockOutput
from rfdiffusion.sae.universalsae import Sae


class RoseTTAFoldModule(nn.Module):
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
                 T, # total timesteps (used in timestep emb
                 use_motif_timestep, # Whether to have a distinct emb for motif
                 freeze_track_motif, # Whether to freeze updates to motif in track
                 SE3_param_full={'l0_in_features':32, 'l0_out_features':16, 'num_edge_features':32},
                 SE3_param_topk={'l0_in_features':32, 'l0_out_features':16, 'num_edge_features':32},
                 input_seq_onehot=False,     # For continuous vs. discrete sequence
                 ):

        super(RoseTTAFoldModule, self).__init__()

        self.freeze_track_motif = freeze_track_motif

        # Input Embeddings
        d_state = SE3_param_topk['l0_out_features']
        self.latent_emb = MSA_emb(d_msa=d_msa, d_pair=d_pair, d_state=d_state,
                p_drop=p_drop, input_seq_onehot=input_seq_onehot) # Allowed to take onehotseq
        self.full_emb = Extra_emb(d_msa=d_msa_full, d_init=25,
                p_drop=p_drop, input_seq_onehot=input_seq_onehot) # Allowed to take onehotseq
        self.templ_emb = Templ_emb(d_pair=d_pair, d_templ=d_templ, d_state=d_state,
                                   n_head=n_head_templ,
                                   d_hidden=d_hidden_templ, p_drop=0.25, d_t1d=d_t1d, d_t2d=d_t2d)


        # Update inputs with outputs from previous round
        self.recycle = Recycling(d_msa=d_msa, d_pair=d_pair, d_state=d_state)
        #
        self.simulator = IterativeSimulator(n_extra_block=n_extra_block,
                                            n_main_block=n_main_block,
                                            n_ref_block=n_ref_block,
                                            d_msa=d_msa, d_msa_full=d_msa_full,
                                            d_pair=d_pair, d_hidden=d_hidden,
                                            n_head_msa=n_head_msa,
                                            n_head_pair=n_head_pair,
                                            SE3_param_full=SE3_param_full,
                                            SE3_param_topk=SE3_param_topk,
                                            p_drop=p_drop)
        ##
        self.c6d_pred = DistanceNetwork(d_pair, p_drop=p_drop)
        self.aa_pred = MaskedTokenNetwork(d_msa)
        self.lddt_pred = LDDTNetwork(d_state)

        self.exp_pred = ExpResolvedNetwork(d_msa, d_state)

    def forward(self, msa_latent, msa_full, seq, xyz, idx, t,
                t1d=None, t2d=None, xyz_t=None, alpha_t=None,
                msa_prev=None, pair_prev=None, state_prev=None,
                return_raw=False, return_full=False, return_infer=False,
                use_checkpoint=False, motif_mask=None, i_cycle=None, n_cycle=None):

        B, N, L = msa_latent.shape[:3]
        # Get embeddings
        msa_latent, pair, state = self.latent_emb(msa_latent, seq, idx)
        msa_full = self.full_emb(msa_full, seq, idx)

        # Do recycling
        if msa_prev == None:
            msa_prev = torch.zeros_like(msa_latent[:,0])
            pair_prev = torch.zeros_like(pair)
            state_prev = torch.zeros_like(state)
        msa_recycle, pair_recycle, state_recycle = self.recycle(seq, msa_prev, pair_prev, xyz, state_prev)
        msa_latent[:,0] = msa_latent[:,0] + msa_recycle.reshape(B,L,-1)
        pair = pair + pair_recycle
        state = state + state_recycle


        # Get timestep embedding (if using)
        if hasattr(self, 'timestep_embedder'):
            assert t is not None
            time_emb = self.timestep_embedder(L,t,motif_mask)
            n_tmpl = t1d.shape[1]
            t1d = torch.cat([t1d, time_emb[None,None,...].repeat(1,n_tmpl,1,1)], dim=-1)

        # add template embedding
        pair, state = self.templ_emb(t1d, t2d, alpha_t, xyz_t, pair, state, use_checkpoint=use_checkpoint)

        # Predict coordinates from given inputs
        is_frozen_residue = motif_mask if self.freeze_track_motif else torch.zeros_like(motif_mask).bool()
        msa, pair, R, T, alpha_s, state = self.simulator(seq, msa_latent, msa_full, pair, xyz[:,:,:3],
                                                         state, idx, use_checkpoint=use_checkpoint,
                                                         motif_mask=is_frozen_residue)

        if return_raw:
            # get last structure
            xyz = einsum('bnij,bnaj->bnai', R[-1], xyz[:,:,:3]-xyz[:,:,1].unsqueeze(-2)) + T[-1].unsqueeze(-2)
            return msa[:,0], pair, xyz, state, alpha_s[-1]

        # predict masked amino acids
        logits_aa = self.aa_pred(msa)

        # Predict LDDT
        lddt = self.lddt_pred(state)

        if return_infer:
            # get last structure
            xyz = einsum('bnij,bnaj->bnai', R[-1], xyz[:,:,:3]-xyz[:,:,1].unsqueeze(-2)) + T[-1].unsqueeze(-2)

            # get scalar plddt
            nbin = lddt.shape[1]
            bin_step = 1.0 / nbin
            lddt_bins = torch.linspace(bin_step, 1.0, nbin, dtype=lddt.dtype, device=lddt.device)
            pred_lddt = nn.Softmax(dim=1)(lddt)
            pred_lddt = torch.sum(lddt_bins[None,:,None]*pred_lddt, dim=1)

            return msa[:,0], pair, xyz, state, alpha_s[-1], logits_aa.permute(0,2,1), pred_lddt

        #
        # predict distogram & orientograms
        logits = self.c6d_pred(pair)

        # predict experimentally resolved or not
        logits_exp = self.exp_pred(msa[:,0], state)

        # get all intermediate bb structures
        xyz = einsum('rbnij,bnaj->rbnai', R, xyz[:,:,:3]-xyz[:,:,1].unsqueeze(-2)) + T.unsqueeze(-2)

        return logits, logits_aa, logits_exp, xyz, alpha_s, lddt

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
                 T, # total timesteps (used in timestep emb
                 use_motif_timestep, # Whether to have a distinct emb for motif
                 freeze_track_motif, # Whether to freeze updates to motif in track
                 SE3_param_full={'l0_in_features':32, 'l0_out_features':16, 'num_edge_features':32},
                 SE3_param_topk={'l0_in_features':32, 'l0_out_features':16, 'num_edge_features':32},
                 input_seq_onehot=False,     # For continuous vs. discrete sequence
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

    @staticmethod
    def transform_from_iter_block_output(output: IterBlockOutput) -> Tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        output of main block - tuple of 6 tensors
        torch.Size([1, 1, seq_len, 256])
        torch.Size([1, seq_len, seq_len, 128])
        torch.Size([1, seq_len, 3, 3])
        torch.Size([1, seq_len, 3])
        torch.Size([1, seq_len, 8])
        torch.Size([1, seq_len, 10, 2])
        into ->
        - list of tensor of size 128 (flatten on idx 1)
        - list of tensor of size 296 (flatten other: 296 = 256 + 128 + 9 + 3 + 20)
        """
        seq_len = output[0].shape[2]
        buffer = [output[x].detach().cpu().reshape(seq_len, -1).unbind(0) for x in [0, 2, 3, 4, 5]]
        non_pair = []
        for a, b, c, d, e in zip(*buffer):
            non_pair.append(torch.cat((a, b, c, d, e)))
        pair = list(output[1].detach().cpu().reshape(output[1].shape[1] * output[1].shape[2], -1).unbind(0))
        return pair, non_pair

    @staticmethod
    def transform_to_iter_block_output(pair: torch.Tensor, non_pair: torch.Tensor) -> IterBlockOutput:
        """
        Transform tensors back into IterBlockOutput tuple of 6 tensors.

        Args:
            pair: Tensor of shape (seq_len * seq_len, 128)
            non_pair: Tensor of shape (seq_len, 296)

        Returns:
            IterBlockOutput: tuple of 6 tensors with shapes:
            1. torch.Size([1, 1, seq_len, 256])
            2. torch.Size([1, seq_len, seq_len, 128])
            3. torch.Size([1, seq_len, 3, 3])
            4. torch.Size([1, seq_len, 3])
            5. torch.Size([1, seq_len, 8])
            6. torch.Size([1, seq_len, 10, 2])
        """
        seq_len = non_pair.shape[0]

        # Split non_pair into its components
        # 296 = 256 + 9 + 3 + 8 + 20
        splits = [256, 9, 3, 8, 20]
        components = torch.split(non_pair, splits, dim=1)

        # Reshape each component back to its original shape
        tensor1 = components[0].reshape(1, 1, seq_len, 256)
        tensor3 = components[1].reshape(1, seq_len, 3, 3)
        tensor4 = components[2].reshape(1, seq_len, 3)
        tensor5 = components[3].reshape(1, seq_len, 8)
        tensor6 = components[4].reshape(1, seq_len, 10, 2)

        # Reshape pair tensor
        tensor2 = pair.reshape(1, seq_len, seq_len, 128)

        return IterBlockOutput(tensor1, tensor2, tensor3, tensor4, tensor5, tensor6)

    def _register_cache_hooks(self, cache: dict):
        def getActivation(name):
            def hook(model, input, output):
                if isinstance(output, IterBlockOutput):
                    cache[f"{name}_pair"], cache[f"{name}_non_pair"] = self.transform_from_iter_block_output(output)
                elif isinstance(output, torch.Tensor):
                    cache[name] = output.detach().cpu()
                else:
                    raise ValueError("Only IterBlockOutput tuple and Tensor supported")
            return hook

        return [self._register_hook_by_path(block_path, getActivation(self.activations_map[block_path])) for block_path in self.activations_map]

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

    def _register_sae_intervention_hooks(self, structure_id: None | str = None, timestep: None | str = None ):

        class SAEInterventionHook:
            def __init__(self, sae_for_pair: Sae, sae_for_non_pair: Sae, batch_size: int = 512,
                         structure_id: None | str = None, basedir_for_sae_latents: None | str = None, timestep: None | int = None,
                         intervention_indices_for_pair: torch.Tensor | str = None, intervention_indices_for_non_pair: torch.Tensor | str = None,
                         intervention_multiplier: float | None = None
                         ):
                self.basedir_for_sae_latents = basedir_for_sae_latents
                if torch.cuda.is_available():
                    self.device = torch.device('cuda')
                else:
                    self.device = torch.device('cpu')
                self.sae_for_pair = sae_for_pair.to(self.device)
                self.sae_for_non_pair = sae_for_non_pair.to(self.device)
                self.batch_size = batch_size
                self.sae_for_pair.eval()
                self.sae_for_non_pair.eval()
                self.structure_id = structure_id
                self.timestep = timestep
                self.intervention_multiplier = intervention_multiplier
                self.intervention_indices_for_pair = intervention_indices_for_pair.to(self.device)
                self.intervention_indices_for_non_pair = intervention_indices_for_non_pair.to(self.device)
                self.intervention_multiplier = intervention_multiplier

            @staticmethod
            def _update_sae_latents(latents: torch.Tensor, indices: torch.Tensor, multiplier: int) -> torch.Tensor:
                latents[:, indices] *= multiplier
                return latents

            def _reconstruct_batch_with_sae(
                self, 
                sae: Sae, 
                batch: list[torch.Tensor],
                indices_to_modify: torch.Tensor | None = None,
                multiplier: int | None = None,
            ):
                batch = batch[0].to(self.device)
                sae_input, _, _ = sae.preprocess_input(batch.unsqueeze(1))
                pre_acts = sae.pre_acts(sae_input)
                top_acts, top_indices = sae.select_topk(pre_acts)
                buf = top_acts.new_zeros(top_acts.shape[:-1] + (sae.W_dec.mT.shape[-1],))
                latents = buf.scatter_(dim=-1, index=top_indices, src=top_acts)
                if indices_to_modify is not None and multiplier is not None:
                    latents  = self._update_sae_latents(latents, indices_to_modify, multiplier)
                return (latents @ sae.W_dec) + sae.b_dec, latents

            def _save_latents_to_disk_as_hf_dataset(self, latents: torch.Tensor, subdir: str,
                    save_n_random: int | None = None
            ):
                if save_n_random:
                    indices = random.sample(range(latents.shape[0]), save_n_random)
                    latents = latents[indices]
                path = os.path.join(self.basedir_for_sae_latents, subdir, f"{self.timestep}", self.structure_id)
                os.makedirs(path, exist_ok=True)
                Dataset.from_dict({
                    "values": latents
                }).save_to_disk(path)
                print(f"-- saved {latents.shape[0]} activations to {path} --")

            def _reconstruct_with_sae(
                    self,
                    output, path_for_latents: str | None,
                    make_intervention: bool | None = None,
            ):
                pairs, non_pairs = HookedRoseTTAFoldModule.transform_from_iter_block_output(output)
                pairs_dataloader = DataLoader(TensorDataset(torch.stack(pairs)), self.batch_size)
                non_pairs_dataloader = DataLoader(TensorDataset(torch.stack(non_pairs)), self.batch_size)
                reconstructed_pair_batches, reconstructed_non_pair_batches = [], []
                latents_pair_batches, latents_non_pair_batches = [], []
                if make_intervention:
                    probes_pair_indices = self.intervention_indices_for_pair
                    probes_non_pair_indices = self.intervention_indices_for_non_pair
                    intervention_multiplier = self.intervention_multiplier
                else:
                    probes_pair_indices = None
                    probes_non_pair_indices = None
                    intervention_multiplier = None
                with torch.no_grad():
                    for batch in pairs_dataloader:
                        reconstruction, latents = self._reconstruct_batch_with_sae(self.sae_for_pair, batch, probes_pair_indices, intervention_multiplier)
                        reconstructed_pair_batches.append(reconstruction)
                        latents_pair_batches.append(latents)
                    for batch in non_pairs_dataloader:
                        reconstruction, latents = self._reconstruct_batch_with_sae(self.sae_for_non_pair, batch, probes_non_pair_indices, intervention_multiplier)
                        reconstructed_non_pair_batches.append(reconstruction)
                        latents_non_pair_batches.append(latents)
                reconstructed_pair_tensor = torch.cat(reconstructed_pair_batches, dim=0)
                reconstructed_non_pair_tensor = torch.cat(reconstructed_non_pair_batches, dim=0)
                if path_for_latents:  # if path for latents activations given, save path
                    latents_pair_tensor = torch.cat(latents_pair_batches, dim=0)
                    latents_non_pair_tensor = torch.cat(latents_non_pair_batches, dim=0)
                    self._save_latents_to_disk_as_hf_dataset(latents_pair_tensor, "pair",
                                                             save_n_random=int(len(pairs) ** .5))
                    self._save_latents_to_disk_as_hf_dataset(latents_non_pair_tensor, "non_pair")
                return HookedRoseTTAFoldModule.transform_to_iter_block_output(reconstructed_pair_tensor,
                                                                              reconstructed_non_pair_tensor)

            @torch.no_grad()
            def __call__(self, module, input, output):
                x = output
                x2 = self._reconstruct_with_sae(x, path_for_latents=self.basedir_for_sae_latents, make_intervention=False)
                e = IterBlockOutput(*(x_elem - x2_elem for x_elem, x2_elem in zip(x, x2)))
                x3 = self._reconstruct_with_sae(x, path_for_latents=None, make_intervention=True)
                return IterBlockOutput(*(x3_elem + e_elem for x3_elem, e_elem in zip(x3, e)))

        sae_batch_size = self.sae_interventions["batch_size"]
        intervention_indices_for_pair = torch.load(
            self.sae_interventions.probes_indices_pair_path, weights_only=True) \
            if self.sae_interventions.probes_indices_pair_path is not None else None
        intervention_indices_for_non_pair = torch.load(self.sae_interventions.probes_indices_non_pair_path, weights_only=True) \
            if self.sae_interventions.probes_indices_non_pair_path is not None else None
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
                       use_checkpoint=False, motif_mask=None, i_cycle=None, n_cycle=None, structure_id: None | str = None):

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
