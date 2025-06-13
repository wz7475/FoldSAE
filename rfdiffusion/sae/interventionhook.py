from typing import Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset

from rfdiffusion.Track_module import IterBlockOutput
from rfdiffusion.sae.blockoutputtransformation import (
    transform_from_iter_block_output,
    transform_to_iter_block_output,
)
from rfdiffusion.sae.universalsae import Sae


class SAEInterventionHook:
    def __init__(
        self,
        sae_pair_path: str | None,
        sae_non_pair_path: str | None,
        batch_size: int = 512,
        intervention_indices_for_pair: Tuple[torch.Tensor] | str = None,
        intervention_indices_for_non_pair: Tuple[torch.Tensor] | str = None,
        intervention_lambda_: float | None = None,
    ):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.sae_for_pair = Sae.load_from_disk(sae_pair_path, self.device).to(
            self.device
        ) if sae_pair_path else None
        self.sae_for_non_pair = Sae.load_from_disk(sae_non_pair_path, self.device).to(
            self.device
        ) if sae_non_pair_path else None
        self.batch_size = batch_size
        if self.sae_for_pair:
            self.sae_for_pair.eval()
        if self.sae_for_non_pair:
            self.sae_for_non_pair.eval()
        self.intervention_lambda_ = intervention_lambda_
        self.intervention_indices_for_pair = (
            torch.load(
                intervention_indices_for_pair,
                weights_only=True,
            )
            if intervention_indices_for_pair is not None
            else None
        )
        self.intervention_indices_for_non_pair = (
            torch.load(
                intervention_indices_for_non_pair,
                weights_only=True,
            )
            if intervention_indices_for_non_pair is not None
            else None
        )
        self.intervention_lambda_ = intervention_lambda_

    def _update_sae_latents(
        self,
        latents: torch.Tensor, indices: Tuple[torch.Tensor, torch.Tensor], lambda_: int
    ) -> torch.Tensor:
        mask = torch.ones_like(latents)
        coefs_values, indices = indices[0].to(self.device), indices[1].to(self.device)
        for idx, val in zip(indices, coefs_values):
            mask[:, idx] = val * lambda_ + 1
        return latents * mask

    def _reconstruct_batch_with_sae(
        self,
        sae: Sae,
        batch: list[torch.Tensor],
        indices_to_modify: Tuple[torch.Tensor, torch.Tensor] | None = None,
        lambda_: int | None = None,
    ):
        batch = batch[0].to(self.device)
        sae_input, _, _ = sae.preprocess_input(batch.unsqueeze(1))
        pre_acts = sae.pre_acts(sae_input)
        top_acts, top_indices = sae.select_topk(pre_acts)
        buf = top_acts.new_zeros(top_acts.shape[:-1] + (sae.W_dec.mT.shape[-1],))
        latents = buf.scatter_(dim=-1, index=top_indices, src=top_acts)
        if indices_to_modify is not None and lambda_ is not None:
            latents = self._update_sae_latents(latents, indices_to_modify, lambda_)
            print("updated latents")
        else:
            print("not updated latens")
        return (latents @ sae.W_dec) + sae.b_dec, latents

    def _reconstruct_with_sae(
        self,
        output,
        make_intervention: bool = False,
    ):
        pairs, non_pairs = transform_from_iter_block_output(output)
        pairs_dataloader = DataLoader(
            TensorDataset(torch.stack(pairs)), self.batch_size
        )
        non_pairs_dataloader = DataLoader(
            TensorDataset(torch.stack(non_pairs)), self.batch_size
        )
        reconstructed_pair_batches, reconstructed_non_pair_batches = [], []
        latents_pair_batches, latents_non_pair_batches = [], []
        if make_intervention:
            probes_pair_indices = self.intervention_indices_for_pair
            probes_non_pair_indices = self.intervention_indices_for_non_pair
            intervention_lambda_ = self.intervention_lambda_
        else:
            probes_pair_indices = None
            probes_non_pair_indices = None
            intervention_lambda_ = None
        with torch.no_grad():
            if self.sae_for_pair:
                for batch in pairs_dataloader:
                    reconstruction, latents = self._reconstruct_batch_with_sae(
                        self.sae_for_pair,
                        batch,
                        probes_pair_indices,
                        intervention_lambda_,
                    )
                    reconstructed_pair_batches.append(reconstruction)
                    latents_pair_batches.append(latents)
                reconstructed_pair_tensor = torch.cat(reconstructed_pair_batches, dim=0)
            else:
                reconstructed_pair_tensor = torch.stack(pairs).to(self.device)
            if self.sae_for_non_pair:
                for batch in non_pairs_dataloader:
                    reconstruction, latents = self._reconstruct_batch_with_sae(
                        self.sae_for_non_pair,
                        batch,
                        probes_non_pair_indices,
                        intervention_lambda_,
                    )
                    reconstructed_non_pair_batches.append(reconstruction)
                    latents_non_pair_batches.append(latents)
                reconstructed_non_pair_tensor = torch.cat(
                    reconstructed_non_pair_batches, dim=0
                )
            else:
                reconstructed_non_pair_tensor = torch.stack(non_pairs).to(self.device)

        return transform_to_iter_block_output(
            reconstructed_pair_tensor, reconstructed_non_pair_tensor
        )

    @torch.no_grad()
    def __call__(self, module, input, output):
        x = output
        x2 = self._reconstruct_with_sae(x, make_intervention=False)
        e = IterBlockOutput(*(x_elem - x2_elem for x_elem, x2_elem in zip(x, x2)))
        x3 = self._reconstruct_with_sae(x, make_intervention=True)
        return IterBlockOutput(*(x3_elem + e_elem for x3_elem, e_elem in zip(x3, e)))
