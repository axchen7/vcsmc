import torch
from torch import Tensor, nn

EPSILON = 1e-8


def safe_norm(x: Tensor, dim: int | None = None, keepdim: bool = False) -> Tensor:
    """
    Computes the L2 norm of a tensor, adding a small epsilon to avoid NaN
    gradients when the norm is zero.
    """

    norm_sq = x.pow(2).sum(dim, keepdim)
    return torch.sqrt(norm_sq + EPSILON)


class Distance(nn.Module):
    def normalize(self, vectors_VxD: Tensor) -> Tensor:
        """
        Given a tensor of shape (V, D), normalizes each row so that it is
        suitable for use in distance calculations. Returns a tensor of shape (V,
        D).
        """
        return vectors_VxD

    def feature_expand_shape(self, D: int) -> int:
        """
        Returns the dimensionality after expanding a single D-dimensional vector.
        """
        return D

    def feature_expand(self, vectors_VxD: Tensor) -> Tensor:
        """
        Given a tensor of shape (V, D), expands it to a tensor of shape (V, ?),
        making each vector suitable as inputs to an encoder.
        """
        return vectors_VxD

    def forward(self, vectors1_VxD: Tensor, vectors2_VxD: Tensor) -> Tensor:
        """
        Given two tensors of shape (V, D) containing normalized vectors,returns
        the distance between each pair of rows. Returns a tensor of shape (V,).
        """
        raise NotImplementedError


class Euclidean(Distance):
    def forward(self, vectors1_VxD: Tensor, vectors2_VxD: Tensor) -> Tensor:
        return safe_norm(vectors1_VxD - vectors2_VxD, -1)


class Hyperbolic(Distance):
    def __init__(self, *, max_radius: float = 0.99, scale: float = 0.01):
        super().__init__()

        self.max_radius = max_radius
        self.scale = scale

    def normalize(self, vectors_VxD: Tensor) -> Tensor:
        # return a vector with the same direction but with the norm passed
        # through tanh()

        norms_V = safe_norm(vectors_VxD, -1)
        new_norms_V = norms_V.tanh() * self.max_radius

        # avoid division by zero
        unit_vectors_VxD = vectors_VxD / (norms_V.unsqueeze(-1) + EPSILON)
        return unit_vectors_VxD * new_norms_V.unsqueeze(-1)

    def feature_expand_shape(self, D: int) -> int:
        return D + 1

    def feature_expand(self, vectors_VxD: Tensor) -> Tensor:
        # return a normalized vector with the norm appended as a feature, which
        # captures the vector's distance from the Poincaré disk's center

        # the norm is passed through atanh() to expand it from [0, 1]
        # to [0, inf]

        norms_V = safe_norm(vectors_VxD, -1)
        new_norms_V = norms_V.atanh()

        # avoid division by zero
        unit_vectors_VxD = vectors_VxD / (norms_V.unsqueeze(-1) + EPSILON)
        return torch.cat([unit_vectors_VxD, new_norms_V.unsqueeze(-1)], -1)

    def forward(self, vectors1_VxD: Tensor, vectors2_VxD: Tensor) -> Tensor:
        # see https://en.wikipedia.org/wiki/Poincaré_disk_model#Lines_and_distance

        xy_norm_sq_V = torch.sum((vectors1_VxD - vectors2_VxD) ** 2, -1)
        one_minus_x_norm_sq_V = 1 - torch.sum(vectors1_VxD**2, dim=-1)
        one_minus_y_norm_sq_V = 1 - torch.sum(vectors2_VxD**2, dim=-1)

        delta_V = 2 * xy_norm_sq_V / (one_minus_x_norm_sq_V * one_minus_y_norm_sq_V)

        distance_V = torch.acosh(1 + 2 * delta_V + EPSILON)

        return distance_V * self.scale
