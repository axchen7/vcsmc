# adapted from
# https://gist.github.com/bdsaglam/b638b5fe4ddae38495b6b032d5726e33

import torch
from torch import Tensor


def expm_simple(x: Tensor, order=10, n=2) -> Tensor:
    """
    Taylor approximation of matrix exponential.

    Uses a trick for better series convergence:
        expm(X) = expm(X/(2^n))^(2^n)
    """

    batch_size = x.shape[:-2]  # Get the batch dimensions
    A = x.shape[-1]  # Get the size of the matrix

    I = torch.eye(A, dtype=x.dtype, device=x.device).expand(*batch_size, A, A)
    result = I
    nom = I
    denom = 1.0

    x = x / 2**n

    for i in range(1, order):
        nom = torch.matmul(x, nom)
        denom *= i
        result = result + nom / denom

    for _ in range(n):
        result = torch.matmul(result, result)

    return result
