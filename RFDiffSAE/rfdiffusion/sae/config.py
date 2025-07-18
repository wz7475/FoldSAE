from dataclasses import dataclass

from simple_parsing import Serializable


@dataclass
class SaeConfig(Serializable):
    """
    Configuration for training a sparse autoencoder on a language model.
    """

    expansion_factor: int = 32
    """Multiple of the input dimension to use as the SAE dimension."""

    normalize_decoder: bool = True
    """Normalize the decoder weights to have unit norm."""

    num_latents: int = 0
    """Number of latents to use. If 0, use `expansion_factor`."""

    k: int = 32
    """Number of nonzero features."""

    batch_topk: bool = False
    """Train Batch-TopK SAEs"""

    sample_topk: bool = False
    """Take TopK latents per whole generated sample, not only per patch of the feature map"""

    input_unit_norm: bool = False

    multi_topk: bool = False
    """Use Multi-TopK loss."""
