import torch
from torch import Tensor

EPSILON = 1e-8


def safe_norm(x: Tensor, dim: int | None = None, keepdim: bool = False) -> Tensor:
    """
    Computes the L2 norm of a tensor, adding a small epsilon to avoid NaN
    gradients when the norm is zero.
    """

    norm_sq = x.pow(2).sum(dim, keepdim)
    return torch.sqrt(norm_sq + EPSILON)
