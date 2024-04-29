# adapted from
# https://gist.github.com/bdsaglam/b638b5fe4ddae38495b6b032d5726e33

import torch
from torch import Tensor

from ..fast_bmm import fast_bmm


def expm_simple(x: Tensor, order=10) -> Tensor:
    """
    nth-order Taylor approximation of matrix exponential
    """

    batch_size = x.shape[:-2]  # Get the batch dimensions
    A = x.shape[-1]  # Get the size of the matrix

    I = torch.eye(A, dtype=x.dtype, device=x.device).expand(*batch_size, A, A)
    result = I
    nom = I
    denom = 1.0

    for i in range(1, order):
        nom = fast_bmm(x, nom)
        denom *= i
        result = result + nom / denom

    return result
