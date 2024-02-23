import keras
import tensorflow as tf

from constants import DTYPE_FLOAT
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
            embeddings_VxD: Encoded sequences.
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

    def __init__(self, *, D: int, width: int, depth: int):
        """
        Args:
            D: Number of dimensions in output embeddings.
            width: Width of each hidden layer.
            depth: Number of hidden layers.
        """

        super().__init__()

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

        return self.mlp(sequences_VxSxA)


class MergeEncoder(tf.Module):
    """Encodes a pair of child embeddings into a parent embedding."""

    def __call__(self, children1_VxD: Tensor, children2_VxD: Tensor) -> Tensor:
        """
        Args:
            children1_VxD: First child embeddings.
            children2_VxD: Second child embeddings.
        Returns:
            embeddings_VxD: Encoded parents.
        """
        raise NotImplementedError


class MLPMergeEncoder(MergeEncoder):
    def __init__(self, *, width: int, depth: int):
        """
        Args:
            D: Number of dimensions in input and output embeddings.
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
        mlp.add(keras.layers.Dense(D, dtype=DTYPE_FLOAT))
        return mlp

    @tf_function(reduce_retracing=True)
    def __call__(self, children1_VxD, children2_VxD):
        if self.mlp is None:
            D = children1_VxD.shape[1]
            self.mlp = self.create_mlp(D)

        return self.mlp(tf.concat([children1_VxD, children2_VxD], axis=1))


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
