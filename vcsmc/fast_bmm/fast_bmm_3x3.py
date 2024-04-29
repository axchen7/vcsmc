import torch


@torch.compile
def fast_bmm_3x3(a, b):
    return torch.stack(
        [
            torch.stack(
                [
                    a[..., 0, 0] * b[..., 0, 0]
                    + a[..., 0, 1] * b[..., 1, 0]
                    + a[..., 0, 2] * b[..., 2, 0],
                    a[..., 1, 0] * b[..., 0, 0]
                    + a[..., 1, 1] * b[..., 1, 0]
                    + a[..., 1, 2] * b[..., 2, 0],
                    a[..., 2, 0] * b[..., 0, 0]
                    + a[..., 2, 1] * b[..., 1, 0]
                    + a[..., 2, 2] * b[..., 2, 0],
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    a[..., 0, 0] * b[..., 0, 1]
                    + a[..., 0, 1] * b[..., 1, 1]
                    + a[..., 0, 2] * b[..., 2, 1],
                    a[..., 1, 0] * b[..., 0, 1]
                    + a[..., 1, 1] * b[..., 1, 1]
                    + a[..., 1, 2] * b[..., 2, 1],
                    a[..., 2, 0] * b[..., 0, 1]
                    + a[..., 2, 1] * b[..., 1, 1]
                    + a[..., 2, 2] * b[..., 2, 1],
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    a[..., 0, 0] * b[..., 0, 2]
                    + a[..., 0, 1] * b[..., 1, 2]
                    + a[..., 0, 2] * b[..., 2, 2],
                    a[..., 1, 0] * b[..., 0, 2]
                    + a[..., 1, 1] * b[..., 1, 2]
                    + a[..., 1, 2] * b[..., 2, 2],
                    a[..., 2, 0] * b[..., 0, 2]
                    + a[..., 2, 1] * b[..., 1, 2]
                    + a[..., 2, 2] * b[..., 2, 2],
                ],
                dim=-1,
            ),
        ],
        dim=-1,
    )


fast_bmm_3x3 = torch.compile(fast_bmm_3x3)
