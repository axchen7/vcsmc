import keras
import tensorflow as tf

from constants import DTYPE_FLOAT
from distances import Distance, safe_norm
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
            distance: Used to feature expand and normalize the embeddings.
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
            distance: Used to feature expand the embeddings.
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


class HyperbolicGeodesicMergeEncoder(MergeEncoder):
    """
    Uses the point on the geodesic between the two children closest to the
    origin as the parent embedding. The parent embedding is thus a deterministic
    function of the children embeddings. Requires embeddings to have dimension
    2.
    """

    @tf_function(reduce_retracing=True)
    def __call__(self, children1_VxD, children2_VxD):
        # require embeddings to have dimension 2
        D = children1_VxD.shape[1]
        tf.assert_equal(D, 2)

        # all values are vectors (shape=[V, 2]) unless otherwise stated

        p = children1_VxD
        q = children2_VxD

        r = (p + q) / 2
        diff = p - q

        n = tf.stack([-diff[:, 1], diff[:, 0]], 1)
        nhat = n / safe_norm(n, axis=1, keepdims=True)

        # ===== scalars (shape=[V] because of vectorization) =====
        p_dot_p = tf.reduce_sum(p * p, 1)
        p_dot_r = tf.reduce_sum(p * r, 1)
        p_dot_nhat = tf.reduce_sum(p * nhat, 1)

        # p_dot_nhat=0 would result in NaNs down the line, so first replace with
        # 1 to ensure that m ultimately has no NaNs. This is needed because
        # although we intend to replace m with r in such cases anyway, m must
        # not contain NaNs as calling tf.where() with NaNs may cause NaN
        # gradients.
        ok = tf.not_equal(p_dot_nhat, 0)
        p_dot_nhat = tf.where(ok, p_dot_nhat, tf.ones_like(p_dot_nhat))

        alpha = (p_dot_p - 2 * p_dot_r + 1) / (2 * p_dot_nhat)
        # ===== end scalars =====

        s = r + alpha[:, tf.newaxis] * nhat

        s_minus_p_norm = safe_norm(s - p, axis=1, keepdims=True)
        s_norm = safe_norm(s, axis=1, keepdims=True)

        m = s * (1 - s_minus_p_norm / s_norm)

        # if any resulting vector is degenerate, replace with midpoint between p
        # and q
        m = tf.where(ok[:, tf.newaxis], m, r)

        return m
