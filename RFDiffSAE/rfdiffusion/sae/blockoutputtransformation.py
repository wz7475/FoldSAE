import torch

from rfdiffusion.Track_module import IterBlockOutput
from typing import Tuple


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