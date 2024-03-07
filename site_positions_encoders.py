import keras
import tensorflow as tf

from constants import DTYPE_FLOAT
from encoders import mlp_add_hidden_layers
from type_utils import Tensor, tf_function


class SitePositionsEncoder(tf.Module):
    """
    Compresses one-hot site positions to a smaller dimension C.
    """

    def __call__(self, site_positions_SxSfull: Tensor) -> Tensor:
        """
        Returns:
            site_positions_SxC: Compressed site positions.
        """
        raise NotImplementedError


class DummySitePositionsEncoder(SitePositionsEncoder):
    """
    Does nothing to the site positions.
    """

    @tf_function()
    def __call__(self, site_positions_SxSfull):
        return site_positions_SxSfull


class MLPSitePositionsEncoder(SitePositionsEncoder):
    """
    Uses a multi-layer perceptron to encode the site positions to a smaller
    dimension C.
    """

    def __init__(self, *, C: int, width: int, depth: int):
        """
        Args:
            C: The compressed dimension.
            width: Width of each hidden layer.
            depth: Number of hidden layers.
        """

        super().__init__()

        self.C = C
        self.width = width
        self.depth = depth

        self.mlp = None

    def create_mlp(self, Sfull: int):
        mlp = keras.Sequential()
        mlp.add(keras.layers.Input([Sfull], dtype=DTYPE_FLOAT))
        mlp_add_hidden_layers(mlp, width=self.width, depth=self.depth)
        mlp.add(keras.layers.Dense(self.C, dtype=DTYPE_FLOAT))
        return mlp

    @tf_function()
    def __call__(self, site_positions_SxSfull):
        if self.mlp is None:
            Sfull = site_positions_SxSfull.shape[1]
            self.mlp = self.create_mlp(Sfull)

        site_positions_SxC = self.mlp(site_positions_SxSfull)
        return site_positions_SxC
