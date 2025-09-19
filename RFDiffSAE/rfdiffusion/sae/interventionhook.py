from typing import Tuple, Sequence

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
        intervention_lambda: float | None = None,
        apply_relu_after_intervention: bool = True,
    ):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.sae_for_pair = (
            Sae.load_from_disk(sae_pair_path, self.device).to(self.device)
            if sae_pair_path
            else None
        )
        self.sae_for_non_pair = (
            Sae.load_from_disk(sae_non_pair_path, self.device).to(self.device)
            if sae_non_pair_path
            else None
        )
        self.batch_size = batch_size
        if self.sae_for_pair:
            self.sae_for_pair.eval()
        if self.sae_for_non_pair:
            self.sae_for_non_pair.eval()
        self.intervention_lambda_ = intervention_lambda
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
        self.apply_relu_after_intervention = apply_relu_after_intervention

    @staticmethod
    def _make_masked_multiplication(
        latents: torch.Tensor,
        indices_to_reinforce: torch.Tensor,
        indices_to_block: torch.Tensor,
        low_level_lambda_: float, # must be positive value
        apply_relu: bool,
        residues_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        returns updated latents where residue mask is set to True
        - increases value of latents
        """
        feature_scale = torch.ones_like(latents)
        for idx in indices_to_reinforce:
            feature_scale[:, idx] = 1 + low_level_lambda_
        for idx in indices_to_block:
            feature_scale[:, idx] = 1 - low_level_lambda_
        multiplied = latents * feature_scale
        if residues_mask is not None:
            residues_mask = residues_mask.to(dtype=torch.bool, device=latents.device)
            if apply_relu:
                multiplied = torch.where(
                    residues_mask.unsqueeze(-1), torch.relu(multiplied), latents
                )
            else:
                multiplied = torch.where(
                    residues_mask.unsqueeze(-1), multiplied, latents
                )
            return multiplied
        if apply_relu:
            return torch.relu(multiplied)
        return multiplied

    def _update_sae_latents(
        self,
        latents: torch.Tensor,
        coefs_for_intervention: Sequence[torch.Tensor],
        lambda_: float,
    ) -> torch.Tensor:
        if len(coefs_for_intervention) == 2:
            # older manner of making intervention with coefs and indices of ONE CLASSIFIER
            return self._make_masked_multiplication(
                latents,
                coefs_for_intervention[0].to(self.device),
                coefs_for_intervention[1].to(self.device),
                lambda_,
                self.apply_relu_after_intervention,
            )
        else:
            # recent manner of making intervention with indices and sign of DISCRIMANT COEFS FOR TWO CLASSIFIERS
            positive_coefs_mask, negative_coefs_mask, coefs_a, bias_a = (
                coefs_for_intervention[0].to(self.device),
                coefs_for_intervention[1].to(self.device),
                coefs_for_intervention[2].to(self.device),
                coefs_for_intervention[3].to(self.device),
            )

            is_A_class_mask = torch.sigmoid(torch.matmul(latents, coefs_a) + bias_a) >= 0.5
            if lambda_ > 0:
                # positive lambda means steering towards class A
                # =>  we want to steer no-class-A residues => residues_mask=~is_A_class_mask
                # => reinforce features positively correlated with being class A
                updated_latens = self._make_masked_multiplication(
                    latents,
                    indices_to_reinforce=torch.nonzero(positive_coefs_mask, as_tuple=True)[0], # increase pos features
                    indices_to_block=torch.nonzero(negative_coefs_mask, as_tuple=True)[0], # decrease neg features
                    low_level_lambda_=lambda_,
                    apply_relu=True,
                    residues_mask=~is_A_class_mask,
                )
            elif lambda_ < 0:
                # negative lambda means steering towards "others" class
                # => we want to steer towards class A residues => residues_mask=is_A_class_mask
                # => reinforce features negatively correlated with being class A
                updated_latens = self._make_masked_multiplication(
                    latents,
                    indices_to_reinforce=torch.nonzero(negative_coefs_mask, as_tuple=True)[0],
                    indices_to_block=torch.nonzero(positive_coefs_mask, as_tuple=True)[0],
                    low_level_lambda_=-lambda_,
                    apply_relu=True,
                    residues_mask=is_A_class_mask, # change only class "a" residues towards other classes
                )
            else:
                # do nothing when lambda_ == 0
                updated_latens = latents
            return updated_latens


    def _reconstruct_batch_with_sae(
        self,
        sae: Sae,
        batch: list[torch.Tensor],
        coefs_for_intervention: Sequence[torch.Tensor] | None = None,
        lambda_: int | None = None,
    ):
        batch = batch[0].to(self.device)
        sae_input, _, _ = sae.preprocess_input(batch.unsqueeze(1))
        pre_acts = sae.pre_acts(sae_input)
        top_acts, top_indices = sae.select_topk(pre_acts)
        buf = top_acts.new_zeros(top_acts.shape[:-1] + (sae.W_dec.mT.shape[-1],))
        latents = buf.scatter_(dim=-1, index=top_indices, src=top_acts)
        if coefs_for_intervention is not None and lambda_ is not None:
            latents = self._update_sae_latents(latents, coefs_for_intervention, lambda_)
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
