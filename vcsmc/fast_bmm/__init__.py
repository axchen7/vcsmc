import torch
from torch import Tensor

from .fast_bmm_2x2 import fast_bmm_2x2
from .fast_bmm_3x3 import fast_bmm_3x3
from .fast_bmm_4x4 import fast_bmm_4x4


def fast_bmm(a: Tensor, b: Tensor) -> Tensor:
    a1 = a.shape[-2]
    a2 = a.shape[-1]
    b1 = b.shape[-2]
    b2 = b.shape[-1]

    if a1 == 2 and a2 == 2 and b1 == 2 and b2 == 2:
        return fast_bmm_2x2(a, b)
    elif a1 == 3 and a2 == 3 and b1 == 3 and b2 == 3:
        return fast_bmm_3x3(a, b)
    elif a1 == 4 and a2 == 4 and b1 == 4 and b2 == 4:
        return fast_bmm_4x4(a, b)
    else:
        return torch.bmm(a, b)
