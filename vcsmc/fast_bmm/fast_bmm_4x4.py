import torch

from .fast_bmm_2x2 import fast_bmm_2x2


@torch.compile
def fast_bmm_4x4(a, b):
    # Split matrices into 2x2 blocks
    a11 = a[..., :2, :2]
    a12 = a[..., :2, 2:]
    a21 = a[..., 2:, :2]
    a22 = a[..., 2:, 2:]

    b11 = b[..., :2, :2]
    b12 = b[..., :2, 2:]
    b21 = b[..., 2:, :2]
    b22 = b[..., 2:, 2:]

    # Perform block matrix multiplication
    c11 = fast_bmm_2x2(a11, b11) + fast_bmm_2x2(a12, b21)
    c12 = fast_bmm_2x2(a11, b12) + fast_bmm_2x2(a12, b22)
    c21 = fast_bmm_2x2(a21, b11) + fast_bmm_2x2(a22, b21)
    c22 = fast_bmm_2x2(a21, b12) + fast_bmm_2x2(a22, b22)

    # Concatenate the blocks to form the result
    result = torch.cat(
        [torch.cat([c11, c12], dim=-1), torch.cat([c21, c22], dim=-1)], dim=-2
    )

    return result
