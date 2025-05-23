import os
import random

import torch
from datasets import Dataset
from torch.utils.data import DataLoader, TensorDataset

from rfdiffusion.Track_module import IterBlockOutput
from rfdiffusion.sae.blockoutputtransformation import transform_from_iter_block_output, transform_to_iter_block_output
from rfdiffusion.sae.universalsae import Sae


class SAEInterventionHook:
    def __init__(self, sae_for_pair: Sae, sae_for_non_pair: Sae, batch_size: int = 512,
                 structure_id: None | str = None, basedir_for_sae_latents: None | str = None,
                 timestep: None | int = None,
                 intervention_indices_for_pair: torch.Tensor | str = None,
                 intervention_indices_for_non_pair: torch.Tensor | str = None,
                 intervention_multiplier: float | None = None,
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
        self.intervention_indices_for_pair = intervention_indices_for_pair.to(
            self.device) if intervention_indices_for_pair is not None else None
        self.intervention_indices_for_non_pair = intervention_indices_for_non_pair.to(
            self.device) if intervention_indices_for_non_pair is not None else None
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
            latents = self._update_sae_latents(latents, indices_to_modify, multiplier)
        return (latents @ sae.W_dec) + sae.b_dec, latents


    def _reconstruct_with_sae(
            self,
            output, path_for_latents: str | None,
            make_intervention: bool | None = None,
    ):
        pairs, non_pairs = transform_from_iter_block_output(output)
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
                reconstruction, latents = self._reconstruct_batch_with_sae(self.sae_for_pair, batch,
                                                                           probes_pair_indices, intervention_multiplier)
                reconstructed_pair_batches.append(reconstruction)
                latents_pair_batches.append(latents)
            for batch in non_pairs_dataloader:
                reconstruction, latents = self._reconstruct_batch_with_sae(self.sae_for_non_pair, batch,
                                                                           probes_non_pair_indices,
                                                                           intervention_multiplier)
                reconstructed_non_pair_batches.append(reconstruction)
                latents_non_pair_batches.append(latents)
        reconstructed_pair_tensor = torch.cat(reconstructed_pair_batches, dim=0)
        reconstructed_non_pair_tensor = torch.cat(reconstructed_non_pair_batches, dim=0)
        return transform_to_iter_block_output(reconstructed_pair_tensor,
                                              reconstructed_non_pair_tensor)

    @torch.no_grad()
    def __call__(self, module, input, output):
        x = output
        x2 = self._reconstruct_with_sae(x, path_for_latents=self.basedir_for_sae_latents, make_intervention=False)
        e = IterBlockOutput(*(x_elem - x2_elem for x_elem, x2_elem in zip(x, x2)))
        x3 = self._reconstruct_with_sae(x, path_for_latents=None, make_intervention=True)
        return IterBlockOutput(*(x3_elem + e_elem for x3_elem, e_elem in zip(x3, e)))
