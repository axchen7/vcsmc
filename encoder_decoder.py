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

    def create_mlp(self, D: int):
        mlp = keras.Sequential()
        mlp.add(keras.layers.Input([2 * D], dtype=DTYPE_FLOAT))
        mlp_add_hidden_layers(mlp, width=self.width, depth=self.depth)
        mlp.add(keras.layers.Dense(D, dtype=DTYPE_FLOAT))
        return mlp

    @tf_function(reduce_retracing=True)
    def __call__(self, children1_VxD, children2_VxD):
        if self.mlp is None:
            D = children1_VxD.shape[1]
            self.mlp = self.create_mlp(D)

        return self.distance.normalize(
            self.mlp(tf.concat([children1_VxD, children2_VxD], axis=1))
        )


class HyperbolicMLPMergeEncoder(MergeEncoder):
    def __init__(self, *, width: int, depth: int):
        """
        Args:
            width: Width of each hidden layer.
            depth: Number of hidden layers.
        """

        super().__init__()

        self.width = width
        self.depth = depth

        self.mlp = None

    def create_mlp(self, D: int):
        mlp = keras.Sequential()
        mlp.add(keras.layers.Input([2 * D], dtype=DTYPE_FLOAT))
        mlp_add_hidden_layers(mlp, width=self.width, depth=self.depth)
        mlp.add(keras.layers.Dense(2, dtype=DTYPE_FLOAT))
        return mlp

    @tf_function(reduce_retracing=True)
    def __call__(self, children1_VxD, children2_VxD):
        if self.mlp is None:
            D = children1_VxD.shape[1]
            self.mlp = self.create_mlp(D + 1)  # add 1 dimension for the squared norm

        # append squared norm as a feature, which captures the vector's distance
        # from the Poincaré disk's center

        squared_norms1_V = tf.reduce_sum(tf.square(children1_VxD), -1)
        squared_norms2_V = tf.reduce_sum(tf.square(children2_VxD), -1)

        expanded1_VxD1 = tf.concat([children1_VxD, squared_norms1_V[:, tf.newaxis]], -1)
        expanded2_VxD1 = tf.concat([children2_VxD, squared_norms2_V[:, tf.newaxis]], -1)

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


class Decoder(tf.Module):
    """
    Decodes an embedding into a sequence.
    """

    def __call__(self, embeddings_VxD: Tensor) -> Tensor:
        """
        Args:
            embeddings_VxD: Encoded sequences.
        Returns:
            sequences_VxSxA: Decoded sequences.
                The values along axis A are are a probability distribution over
                the alphabet.
        """
        raise NotImplementedError


class StationaryDecoder(Decoder):
    """Returns a learnable stationary distribution."""

    def __init__(self, *, S: int, A: int):
        """
        Args:
            S: Length of output sequences.
            A: Alphabet size.
        """

        super().__init__()

        self.S = S

        self.log_stat_probs = tf.Variable(
            tf.constant(0, DTYPE_FLOAT, [A]), name="log_stat_probs"
        )

    @tf_function(reduce_retracing=True)
    def __call__(self, embeddings_VxD):
        # one stationary probability can be fixed to match degrees of freedom
        stat_probs = tf.tensor_scatter_nd_update(
            self.log_stat_probs, [[0]], [tf.constant(0, DTYPE_FLOAT)]
        )
        # use softmax to ensure all entries are positive
        stat_probs = tf.exp(stat_probs)
        # normalize to ensure sum is 1
        stat_probs /= tf.reduce_sum(stat_probs)

        V = tf.shape(embeddings_VxD)[0]  # type: ignore

        sequences_VxSxA = tf.tile(tf.reshape(stat_probs, [1, 1, -1]), [V, self.S, 1])
        return sequences_VxSxA


class MLPDecoder(Decoder):
    """Uses a multi-layer perceptron."""

    def __init__(
        self,
        *,
        S: int,
        A: int,
        width: int,
        depth: int,
        baseline: float = 0.1,
    ):
        """
        Args:
            S: Length of output sequences.
            A: Alphabet size.
            width: Width of each hidden layer.
            depth: Number of hidden layers.
            baseline: Baseline probability for each of the A letters, from 0 to 1.
                Take this much of the probability mass from the uniform distribution 1/A.
                Helps prevent the model from converging to a degenerate solution.
        """

        super().__init__()

        self.S = S
        self.A = A
        self.width = width
        self.depth = depth
        self.baseline = tf.constant(baseline, dtype=DTYPE_FLOAT)

        self.mlp = None

    def create_mlp(self, D: int):
        mlp = keras.Sequential()
        mlp.add(keras.layers.Input([D], dtype=DTYPE_FLOAT))
        mlp_add_hidden_layers(mlp, width=self.width, depth=self.depth)
        mlp.add(keras.layers.Dense(self.S * self.A, dtype=DTYPE_FLOAT))
        mlp.add(keras.layers.Reshape([self.S, self.A]))
        mlp.add(keras.layers.Softmax(2, dtype=DTYPE_FLOAT))
        return mlp

    @tf_function(reduce_retracing=True)
    def __call__(self, embeddings_VxD):
        if self.mlp is None:
            D = embeddings_VxD.shape[1]
            self.mlp = self.create_mlp(D)

        sequences_VxSxA = self.mlp(embeddings_VxD)
        sequences_VxSxA = sequences_VxSxA * (1 - self.baseline) + self.baseline / self.A  # type: ignore
        return sequences_VxSxA
