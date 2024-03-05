import keras
import tensorflow as tf

from constants import DTYPE_FLOAT
from distances import Distance
from type_utils import Tensor, tf_function


def mlp_add_hidden_layers(
    mlp: keras.Sequential, *, width: int, depth: int
) -> keras.Sequential:
    for _ in range(depth):
        mlp.add(keras.layers.Dense(width, activation="relu", dtype=DTYPE_FLOAT))
    return mlp


class SequenceEncoder(tf.Module):
    """Encodes sequences into embeddings."""

    def __call__(self, sequences_VxSxA: Tensor) -> Tensor:
        """
        Args:
            sequences_VxSxA: Sequences to encode.
        Returns:
            embeddings_VxD: Encoded sequences, normalized.
        """
        raise NotImplementedError


class DummySequenceEncoder(SequenceEncoder):
    """A dummy encoder that returns zero embeddings."""

    @tf_function(reduce_retracing=True)
    def __call__(self, sequences_VxSxA):
        V = tf.shape(sequences_VxSxA)[0]  # type: ignore
        return tf.zeros([V, 0], dtype=DTYPE_FLOAT)


class MLPSequenceEncoder(SequenceEncoder):
    """Uses a multi-layer perceptron."""

    def __init__(self, distance: Distance, *, D: int, width: int, depth: int):
        """
        Args:
            distance: Used to normalize the embeddings.
            D: Number of dimensions in output embeddings.
            width: Width of each hidden layer.
            depth: Number of hidden layers.
        """

        super().__init__()

        self.distance = distance
        self.D = D
        self.width = width
        self.depth = depth

        self.mlp = None

    def create_mlp(self, S: int, A: int):
        mlp = keras.Sequential()
        mlp.add(keras.layers.Input([S, A], dtype=DTYPE_FLOAT))
        mlp.add(keras.layers.Flatten())
        mlp_add_hidden_layers(mlp, width=self.width, depth=self.depth)
        mlp.add(keras.layers.Dense(self.D, dtype=DTYPE_FLOAT))
        return mlp

    @tf_function(reduce_retracing=True)
    def __call__(self, sequences_VxSxA):
        if self.mlp is None:
            S = sequences_VxSxA.shape[1]
            A = sequences_VxSxA.shape[2]
            self.mlp = self.create_mlp(S, A)

        return self.distance.normalize(self.mlp(sequences_VxSxA))


class MergeEncoder(tf.Module):
    """Encodes a pair of child embeddings into a parent embedding."""

    def __call__(self, children1_VxD: Tensor, children2_VxD: Tensor) -> Tensor:
        """
        Args:
            children1_VxD: First child embeddings.
            children2_VxD: Second child embeddings.
        Returns:
            embeddings_VxD: Encoded parents, normalized.
        """
        raise NotImplementedError


class MLPMergeEncoder(MergeEncoder):
    def __init__(self, distance: Distance, *, width: int, depth: int):
        """
        Args:
            distance: Used to normalize the embeddings.
            width: Width of each hidden layer.
            depth: Number of hidden layers.
        """

        super().__init__()

        self.distance = distance
        self.width = width
        self.depth = depth

        self.mlp = None

    def create_mlp(self, D: int, D1: int):
        mlp = keras.Sequential()
        mlp.add(keras.layers.Input([2 * D1], dtype=DTYPE_FLOAT))
        mlp_add_hidden_layers(mlp, width=self.width, depth=self.depth)
        mlp.add(keras.layers.Dense(D, dtype=DTYPE_FLOAT))
        return mlp

    @tf_function(reduce_retracing=True)
    def __call__(self, children1_VxD, children2_VxD):
        # D1 is the dimensionality of the expanded children
        expanded1_VxD1 = self.distance.feature_expand(children1_VxD)
        expanded2_VxD1 = self.distance.feature_expand(children2_VxD)

        if self.mlp is None:
            D = children1_VxD.shape[1]
            D1 = expanded1_VxD1.shape[1]
            self.mlp = self.create_mlp(D, D1)

        return self.distance.normalize(
            self.mlp(tf.concat([expanded1_VxD1, expanded2_VxD1], axis=1))
        )


class HyperbolicMLPMergeEncoder(MergeEncoder):
    def __init__(self, distance: Distance, *, width: int, depth: int):
        """
        Args:
            width: Width of each hidden layer.
            depth: Number of hidden layers.
        """

        super().__init__()

        self.distance = distance
        self.width = width
        self.depth = depth

        self.mlp = None

    def create_mlp(self, D1: int):
        mlp = keras.Sequential()
        mlp.add(keras.layers.Input([2 * D1], dtype=DTYPE_FLOAT))
        mlp_add_hidden_layers(mlp, width=self.width, depth=self.depth)
        mlp.add(keras.layers.Dense(2, dtype=DTYPE_FLOAT))
        return mlp

    @tf_function(reduce_retracing=True)
    def __call__(self, children1_VxD, children2_VxD):
        # D1 is the dimensionality of the expanded children
        expanded1_VxD1 = self.distance.feature_expand(children1_VxD)
        expanded2_VxD1 = self.distance.feature_expand(children2_VxD)

        if self.mlp is None:
            D1 = expanded1_VxD1.shape[1]
            self.mlp = self.create_mlp(D1)

        # mlp output is alpha and beta (see definitions below)
        alpha_beta_Vx2 = self.mlp(tf.concat([expanded1_VxD1, expanded2_VxD1], -1))

        # the fractional position between children1 (alpha=0) and children2 (alpha=1)
        alpha_V = alpha_beta_Vx2[:, 0]  # type: ignore
        alpha_V = tf.sigmoid(alpha_V)  # squash to (0, 1)

        # the point a distance alpha on the line between children1 and children2
        mid_V = children1_VxD + alpha_V[:, tf.newaxis] * (children2_VxD - children1_VxD)

        # the fractional distance between the origin (beta=0) and the point mid (beta=1)
        beta_V = alpha_beta_Vx2[:, 1]  # type: ignore
        beta_V = tf.sigmoid(beta_V)  # squash to (0, 1)

        # the point a distance beta on the line between the origin and mid
        parent_VxD = mid_V * beta_V[:, tf.newaxis]  # type: ignore
        return parent_VxD
