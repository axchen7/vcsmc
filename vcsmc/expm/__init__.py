from typing import Literal

import torch
from torch import Tensor

from . import expm_simple, expm_taylor


def expm(Q: Tensor, fallback: Literal["simple", "taylor"]) -> Tensor:
    """
    On an MPS device (e.g. Apple Silicon), torch.matrix_exp is not supported, so
    use an approximate expm in its place.

    Args:
        Q: Input matrix.
        fallback: Fallback method to use if torch.matrix_exp is not available.
            simple: Use a simple taylor expansion. Good for matrices with small values.
            taylor: Use a more accurate taylor expansion, but may error for
                matrices with large values.
    """

    if Q.device.type != "mps":
        return torch.matrix_exp(Q)

    if fallback == "simple":
        return expm_simple.expm(Q)
    else:
        return expm_taylor.expm(Q)
