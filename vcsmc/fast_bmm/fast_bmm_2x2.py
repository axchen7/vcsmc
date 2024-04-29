import torch


@torch.compile
def fast_bmm_2x2(a, b):
    return torch.stack(
        [
            torch.stack(
                [
                    a[..., 0, 0] * b[..., 0, 0] + a[..., 0, 1] * b[..., 1, 0],
                    a[..., 1, 0] * b[..., 0, 0] + a[..., 1, 1] * b[..., 1, 0],
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    a[..., 0, 0] * b[..., 0, 1] + a[..., 0, 1] * b[..., 1, 1],
                    a[..., 1, 0] * b[..., 0, 1] + a[..., 1, 1] * b[..., 1, 1],
                ],
                dim=-1,
            ),
        ],
        dim=-1,
    )
