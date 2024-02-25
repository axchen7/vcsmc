import tensorflow as tf

from constants import DTYPE_FLOAT
from type_utils import Tensor, tf_function


class Distance(tf.Module):
    @tf_function(reduce_retracing=True)
    def normalize(self, vectors_VxD: Tensor) -> Tensor:
        """
        Given a tensor of shape (V, D), normalizes each row so that it is
        suitable for use in distance calculations. Returns a tensor of shape (V,
        D).
        """
        return vectors_VxD

    def __call__(self, vectors1_VxD: Tensor, vectors2_VxD: Tensor) -> Tensor:
        """
        Given two tensors of shape (V, D) containing normalized vectors,returns
        the distance between each pair of rows. Returns a tensor of shape (V,).
        """
        raise NotImplementedError


class Euclidean(Distance):
    @tf_function()
    def __call__(self, vectors1_VxD, vectors2_VxD):
        return tf.norm(vectors1_VxD - vectors2_VxD, axis=-1)


class Hyperbolic(Distance):
    def __init__(self):
        super().__init__()

        self.logit_max_radius = tf.Variable(
            0, name="logit_max_radius", dtype=DTYPE_FLOAT
        )
        self.log_scale = tf.Variable(0.0, dtype=DTYPE_FLOAT, name="log_scale")

    @tf_function()
    def normalize(self, vectors_VxD):
        # return a vector with the same direction but with the norm passed
        # through tanh()

        max_radius = tf.sigmoid(self.logit_max_radius)

        norms_V = tf.norm(vectors_VxD, axis=-1)
        new_norms_V = tf.tanh(norms_V) * max_radius

        unit_vectors_VxD = vectors_VxD / norms_V[:, tf.newaxis]
        return unit_vectors_VxD * new_norms_V[:, tf.newaxis]

    @tf_function()
    def __call__(self, vectors1_VxD, vectors2_VxD):
        # see https://en.wikipedia.org/wiki/Poincaré_disk_model#Lines_and_distance
        # acosh version causes NaNs, but asinh version works

        xy_norm_sq_V = tf.reduce_sum(tf.square(vectors1_VxD - vectors2_VxD), axis=-1)
        one_minus_x_norm_sq_V = 1 - tf.reduce_sum(tf.square(vectors1_VxD), axis=-1)
        one_minus_y_norm_sq_V = 1 - tf.reduce_sum(tf.square(vectors2_VxD), axis=-1)

        distance_V = 2 * tf.asinh(
            tf.sqrt(xy_norm_sq_V / (one_minus_x_norm_sq_V * one_minus_y_norm_sq_V))
        )

        scale = tf.exp(self.log_scale)
        return distance_V * scale
