import torch
from torch import Tensor, nn

from .encoder_utils import MLP


class SitePositionsEncoder(nn.Module):
    """
    Compresses one-hot site positions to a smaller dimension C.
    """

    def forward(self, site_positions_SxSfull: Tensor) -> Tensor:
        """
        Returns:
            site_positions_SxC: Compressed site positions.
        """
        raise NotImplementedError


class DummySitePositionsEncoder(SitePositionsEncoder):
    """
    Returns an empty Sx0 tensor.
    """

    def forward(self, site_positions_SxSfull: Tensor) -> Tensor:
        S = site_positions_SxSfull.shape[0]
        return torch.zeros([S, 0])


class MLPSitePositionsEncoder(SitePositionsEncoder):
    """
    Uses a multi-layer perceptron to encode the site positions to a smaller
    dimension C.
    """

    def __init__(self, *, S: int, C: int, width: int, depth: int):
        """
        Args:
            S: The full number of sites.
            C: The compressed dimension.
            width: Width of each hidden layer.
            depth: Number of hidden layers.
        """

        super().__init__()

        self.mlp = MLP(S, C, width, depth)

    def forward(self, site_positions_SxSfull: Tensor) -> Tensor:
        site_positions_SxC = self.mlp(site_positions_SxSfull)
        return site_positions_SxC
