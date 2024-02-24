import tensorflow as tf

from constants import DTYPE_FLOAT
from type_utils import Tensor, tf_function


class QMatrix(tf.Module):
    @tf_function()
    def regularization(self) -> Tensor:
        """Add to cost."""
        return tf.constant(0, DTYPE_FLOAT)

    def __call__(self) -> Tensor:
        """Returns the Q matrix."""
        raise NotImplementedError


class DenseQMatrix(QMatrix):
    """
    Use a trainable variable for every entry in the Q matrix (except the
    diagonal).
    """

    def __init__(self, *, A: int):
        super().__init__()

        self.A = A
        self.log_Q = tf.Variable(tf.constant(0, DTYPE_FLOAT, [A, A]), name="log_Q")

    @tf_function()
    def __call__(self):
        # first non-diagonal entry in each row can be fixed to match degrees of
        # freedom
        Q = tf.tensor_scatter_nd_update(
            self.log_Q,
            [[i, (i + 1) % self.A] for i in range(self.A)],
            [tf.constant(0, DTYPE_FLOAT)] * self.A,
        )

        # use exp to ensure all off-diagonal entries are positive
        Q = tf.exp(Q)

        # exclude diagonal entry for now...
        Q = tf.linalg.set_diag(Q, [0] * self.A)

        # normalize off-diagonal entries within each row
        denom = tf.reduce_sum(Q, 1, True)
        denom = tf.repeat(denom, self.A, 1)
        Q /= denom

        # set diagonal to negative sum of off-diagonal entries
        hyphens = tf.reduce_sum(Q, 1)
        Q = tf.linalg.set_diag(Q, -hyphens)
        return Q


class GT16QMatrix(QMatrix):
    """
    Assumes A=16. Uses the CellPhy GT16 model.
    """

    def __init__(self, *, reg_lambda: float = 0.0):
        """
        Args:
            reg_lambda: Stationary probability regularization coefficient.
        """

        super().__init__()

        self.reg_lambda = tf.constant(reg_lambda, DTYPE_FLOAT)

        self.log_nucleotide_exchanges = tf.Variable(
            tf.constant(0, DTYPE_FLOAT, [6]), name="log_nucleotide_exchanges"
        )
        self.log_stat_probs = tf.Variable(
            tf.constant(0, DTYPE_FLOAT, [16]), name="log_stat_probs"
        )

    @tf_function()
    def nucleotide_exchanges(self):
        # one exchangeability can be fixed to match degrees of freedom
        nucleotide_exchanges = tf.tensor_scatter_nd_update(
            self.log_nucleotide_exchanges, [[0]], [tf.constant(0, DTYPE_FLOAT)]
        )
        # use exp to ensure all entries are positive
        nucleotide_exchanges = tf.exp(nucleotide_exchanges)
        # normalize to ensure mean is 1
        nucleotide_exchanges /= tf.reduce_mean(nucleotide_exchanges)
        return nucleotide_exchanges

    @tf_function()
    def stat_probs(self):
        # one stationary probability can be fixed to match degrees of freedom
        stat_probs = tf.tensor_scatter_nd_update(
            self.log_stat_probs, [[0]], [tf.constant(0, DTYPE_FLOAT)]
        )
        # use softmax to ensure all entries are positive
        stat_probs = tf.exp(stat_probs)
        # normalize to ensure sum is 1
        stat_probs /= tf.reduce_sum(stat_probs)
        return stat_probs

    @tf_function()
    def regularization(self):
        stat_probs_norm_squared = tf.reduce_sum(tf.math.square(self.stat_probs()))
        return self.reg_lambda * stat_probs_norm_squared

    @tf_function()
    def __call__(self):
        pi = self.nucleotide_exchanges()  # length 6
        pi8 = tf.repeat(pi, 8)

        # index helpers for Q matrix
        AA, CC, GG, TT, AC, AG, AT, CG, CT, GT, CA, GA, TA, GC, TC, TG = range(16)

        # fmt: off
        updates = [
          # | first base changes                    | second base changes
            [AA, CA], [AC, CC], [AG, CG], [AT, CT], [AA, AC], [CA, CC], [GA, GC], [TA, TC], # A->C
            [AA, GA], [AC, GC], [AG, GG], [AT, GT], [AA, AG], [CA, CG], [GA, GG], [TA, TG], # A->G
            [AA, TA], [AC, TC], [AG, TG], [AT, TT], [AA, AT], [CA, CT], [GA, GT], [TA, TT], # A->T
            [CA, GA], [CC, GC], [CG, GG], [CT, GT], [AC, AG], [CC, CG], [GC, GG], [TC, TG], # C->G
            [CA, TA], [CC, TC], [CG, TG], [CT, TT], [AC, AT], [CC, CT], [GC, GT], [TC, TT], # C->T
            [GA, TA], [GC, TC], [GG, TG], [GT, TT], [AG, AT], [CG, CT], [GG, GT], [TG, TT], # G->T
        ]
        # fmt: on

        R = tf.scatter_nd(updates, pi8, [16, 16])
        R = R + tf.transpose(R)

        ylog_Q = tf.matmul(R, tf.linalg.diag(self.stat_probs()))
        hyphens = tf.reduce_sum(ylog_Q, 1)

        Q = tf.linalg.set_diag(ylog_Q, -hyphens)
        return Q
