import tensorflow as tf

from constants import DTYPE_FLOAT
from type_utils import Tensor, tf_function


class Distance(tf.Module):
    @tf_function()
    def project(self, x: Tensor) -> Tensor:
        """
        Given a D-length tensor, return the projected tensor actually used for
        the distance calculation.
        """
        return x

    @tf_function()
    def project_many(self, x: Tensor) -> Tensor:
        """
        Given a tensor of shape (?, D), call `project()` on each row. Returns a
        tensor of shape (?, D).
        """
        return tf.vectorized_map(self.project, x)

    @tf_function()
    def many(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Given two tensors of shape (?, D), call `__call__()` on each pair of
        rows. Returns a tensor of shape (?,).
        """
        return tf.vectorized_map(lambda xy: self(xy[0], xy[1]), (x, y))

    def __call__(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Given two D-length tensors, return the distance between them.
        """
        raise NotImplementedError


class Euclidean(Distance):
    def __init__(self, *, initial_radius: float = 1.0):
        super().__init__()
        # project leaf and inner embeddings onto a sphere of learnable radius
        self.radius = tf.Variable(initial_radius, name="radius", dtype=DTYPE_FLOAT)

    @tf_function()
    def project(self, x):
        return x * self.radius / tf.norm(x)

    @tf_function()
    def __call__(self, x, y):
        x = self.project(x)
        y = self.project(y)

        return tf.sqrt(tf.reduce_sum(tf.square(x - y)))


class Hyperbolic(Distance):
    def __init__(self, *, initial_radius: float = 0.5):
        super().__init__()
        # project leaf and inner embeddings onto a sphere of learnable radius
        self.radius = tf.Variable(initial_radius, name="radius", dtype=DTYPE_FLOAT)

    @tf_function()
    def project(self, x):
        return x * self.radius / tf.norm(x)

    @tf_function()
    def __call__(self, x, y):
        x = self.project(x)
        y = self.project(y)

        # see https://en.wikipedia.org/wiki/Poincaré_disk_model#Lines_and_distance
        # acosh version causes NaNs, but asinh version works

        xy_norm_sq = tf.reduce_sum(tf.square(x - y))
        one_minus_x_norm_sq = 1 - tf.reduce_sum(tf.square(x))
        one_minus_y_norm_sq = 1 - tf.reduce_sum(tf.square(y))

        return 2 * tf.asinh(
            tf.sqrt(xy_norm_sq / (one_minus_x_norm_sq * one_minus_y_norm_sq))
        )
