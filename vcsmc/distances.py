import math

import torch
from torch import Tensor, nn

from .utils.distance_utils import EPSILON, safe_norm
from .utils.repr_utils import custom_module_repr

__all__ = ["Distance", "Euclidean", "Hyperbolic"]


class Distance(nn.Module):
    def normalize(self, vectors_VxD: Tensor) -> Tensor:
        """
        Given a tensor of shape (V, D), normalizes each row so that it is
        suitable for being displayed. Returns a tensor of shape (V, D).
        """
        return vectors_VxD

    def unnormalize(self, vectors_VxD: Tensor) -> Tensor:
        """
        Given a tensor of shape (V, D), undoes normalization on each row.
        Returns a tensor of shape (V, D).
        """
        return vectors_VxD

    def feature_expand_shape(self, D: int) -> int:
        """
        Returns the dimensionality after expanding a single D-dimensional vector.
        """
        return D

    def feature_expand(self, vectors_VxD: Tensor) -> Tensor:
        """
        Given a tensor of shape (V, D) containing non-normalized vectors,
        expands it to a tensor of shape (V, ?), making each vector suitable as
        inputs to an encoder.
        """
        return vectors_VxD

    def forward(self, vectors1_VxD: Tensor, vectors2_VxD: Tensor) -> Tensor:
        """
        Given two tensors of shape (V, D) containing non-normalized vectors,
        returns the distance between each pair of rows. Returns a tensor of
        shape (V,).
        """
        raise NotImplementedError


class Euclidean(Distance):
    def forward(self, vectors1_VxD: Tensor, vectors2_VxD: Tensor) -> Tensor:
        return safe_norm(vectors1_VxD - vectors2_VxD, -1)


class Hyperbolic(Distance):
    """
    Embeddings are represented in a form where 0 < |x| < inf. Points can be
    arbitrarily close to the edge of the poincaré disk. The normalization
    function f(|x|)=sqrt(1-exp(-|x|^2)) makes computing distances easier.
    """

    def __init__(self, *, initial_scale: float = 0.1, fixed_scale: bool = False):
        super().__init__()

        self.register_buffer("zero", torch.zeros(1))

        self.initial_scale = initial_scale
        self.fixed_scale = fixed_scale

        if fixed_scale:
            self.log_scale = math.log(initial_scale)
        else:
            self.log_scale = nn.Parameter(torch.tensor(math.log(initial_scale)))

    def extra_repr(self) -> str:
        return custom_module_repr(
            {
                "initial_scale": self.initial_scale,
                "fixed_scale": self.fixed_scale,
            }
        )

    def scale(self):
        if isinstance(self.log_scale, Tensor):
            return self.log_scale.exp()
        else:
            return math.exp(self.log_scale)

    def normalize(self, vectors_VxD: Tensor) -> Tensor:
        # return a vector with the same direction but with the norm passed
        # through f(|x|)=sqrt(1-exp(-x^2))

        norms_V = safe_norm(vectors_VxD, -1)
        norms_sq_V = torch.sum(vectors_VxD**2, -1)
        new_norms_V = torch.sqrt(1 - torch.exp(-norms_sq_V) + EPSILON)

        # avoid division by zero
        unit_vectors_VxD = vectors_VxD / (norms_V.unsqueeze(-1) + EPSILON)
        return unit_vectors_VxD * new_norms_V.unsqueeze(-1)

    def unnormalize(self, vectors_VxD: Tensor) -> Tensor:
        # return a vector with the same direction but with the norm passed
        # through f_inv(|x|)=sqrt(-ln(1-|x|^2))

        norms_V = safe_norm(vectors_VxD, -1)
        norms_sq_V = torch.sum(vectors_VxD**2, -1)
        new_norms_V = torch.sqrt(-torch.log(1 - norms_sq_V + EPSILON) + EPSILON)

        # avoid division by zero
        unit_vectors_VxD = vectors_VxD / (norms_V.unsqueeze(-1) + EPSILON)
        return unit_vectors_VxD * new_norms_V.unsqueeze(-1)

    def feature_expand_shape(self, D: int) -> int:
        return D + 1

    def feature_expand(self, vectors_VxD: Tensor) -> Tensor:
        # encode magnitude and direction separately
        # - use hyperbolic distance from origin for magnitude
        # - use unit vector for direction

        V, D = vectors_VxD.shape

        distances_V = self(vectors_VxD, self.zero.expand(V, D))
        distances_Vx1 = distances_V.unsqueeze(-1)

        # avoid division by zero
        norms_V = safe_norm(vectors_VxD, -1)
        unit_vectors_VxD = vectors_VxD / (norms_V.unsqueeze(-1) + EPSILON)
        return torch.cat([distances_Vx1, unit_vectors_VxD], -1)

    def forward(self, vectors1_VxD: Tensor, vectors2_VxD: Tensor) -> Tensor:
        # see https://en.wikipedia.org/wiki/Poincaré_disk_model#Lines_and_distance

        normalized1_VxD = self.normalize(vectors1_VxD)
        normalized2_VxD = self.normalize(vectors2_VxD)

        xy_norm_sq_V = torch.sum((normalized1_VxD - normalized2_VxD) ** 2, -1)

        norms1_sq_V = torch.sum(vectors1_VxD**2, -1)
        norms2_sq_V = torch.sum(vectors2_VxD**2, -1)

        # accounting for normalization, get simple formula
        # 1/(1-|x_normalized|^2) = 1/(1-f(|x|)^2) = exp(|x|^2)
        one_minus_x_norm_sq_recip_V = torch.exp(norms1_sq_V)
        one_minus_y_norm_sq_recip_V = torch.exp(norms2_sq_V)

        delta_V = (
            2 * xy_norm_sq_V * one_minus_x_norm_sq_recip_V * one_minus_y_norm_sq_recip_V
        )

        # use larger epsilon to make float32 stable
        distance_V = torch.acosh(1 + delta_V + (EPSILON * 10))

        return distance_V * self.scale()
