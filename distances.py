import tensorflow as tf

from constants import DTYPE_FLOAT
from type_utils import Tensor, tf_function


class Distance(tf.Module):
    @tf_function()
    def project_single(self, x: Tensor) -> Tensor:
        """
        Given a D-length vector, return the projected vector actually used for
        the distance calculation. The projected vector may have a different shape.
        """
        return x

    @tf_function(reduce_retracing=True)
    def project(self, vectors: Tensor) -> Tensor:
        """
        Given a tensor of shape (V, D), call `project_single()` on each row. Returns a
        tensor of shape (V, ?).
        """
        return tf.vectorized_map(self.project_single, vectors)

    def distance_single(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Given two D-length vectors, return the distance between them.
        """
        raise NotImplementedError

    @tf_function(reduce_retracing=True)
    def __call__(self, vectors1: Tensor, vectors2: Tensor) -> Tensor:
        """
        Given two tensors of shape (V, D), call `distance_single()` on each pair of
        rows. Returns a tensor of shape (V,).
        """
        return tf.vectorized_map(
            lambda xy: self.distance_single(xy[0], xy[1]), (vectors1, vectors2)
        )


class Euclidean(Distance):
    def __init__(self, *, initial_radius: float = 1.0):
        super().__init__()
        # project embeddings onto a sphere of learnable radius
        self.radius = tf.Variable(initial_radius, name="radius", dtype=DTYPE_FLOAT)

    @tf_function()
    def project_single(self, x):
        return x * self.radius / tf.norm(x)

    @tf_function()
    def distance_single(self, x, y):
        x = self.project_single(x)
        y = self.project_single(y)

        return tf.sqrt(tf.reduce_sum(tf.square(x - y)))


class Hyperbolic(Distance):
    def __init__(self):
        super().__init__()
        # cap the radius of vectors to avoid NaNs
        self.logit_max_radius = tf.Variable(
            0, name="logit_max_radius", dtype=DTYPE_FLOAT
        )

    @tf_function()
    def project_single(self, x):
        """
        Treats the tanh of the first coordinate as the radius and projects the
        remaining coordinates onto the hypersphere of that radius. Radius is
        capped to avoid NaNs. Returns a vector of length D-1.
        """
        max_radius = tf.sigmoid(self.logit_max_radius) * 0.99  # cap again at 0.99
        radius = tf.tanh(x[0]) * max_radius
        rest = x[1:]
        return radius * rest / tf.norm(rest)

    @tf_function()
    def distance_single(self, x, y):
        x = self.project_single(x)
        y = self.project_single(y)

        # see https://en.wikipedia.org/wiki/Poincaré_disk_model#Lines_and_distance
        # acosh version causes NaNs, but asinh version works

        xy_norm_sq = tf.reduce_sum(tf.square(x - y))
        one_minus_x_norm_sq = 1 - tf.reduce_sum(tf.square(x))
        one_minus_y_norm_sq = 1 - tf.reduce_sum(tf.square(y))

        return 2 * tf.asinh(
            tf.sqrt(xy_norm_sq / (one_minus_x_norm_sq * one_minus_y_norm_sq))
        )
