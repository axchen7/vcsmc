import tensorflow as tf

from constants import DTYPE_FLOAT
from type_utils import Tensor, tf_function


class QMatrixDecoder(tf.Module):
    def Q_matrix_VxSxAxA(self, embeddings_VxD: Tensor) -> Tensor:
        """
        Returns the Q matrix for each of the S sites. Operates on a batch of
        embeddings.
        """
        raise NotImplementedError

    def stat_probs_VxSxA(self, embeddings_VxD: Tensor) -> Tensor:
        """
        Returns the stationary probabilities for each of the S sites. Operates
        on a batch of Q matrices.
        """
        raise NotImplementedError

    @tf_function()
    def regularization(self) -> Tensor:
        """Add to cost."""
        return tf.constant(0, DTYPE_FLOAT)


class DenseStationaryQMatrixDecoder(QMatrixDecoder):
    """
    Use a trainable variable for every entry in the Q matrix (except the
    diagonal). Q matrix is the same globally, irrespective of the input
    embeddings.
    """

    def __init__(self, *, A: int, t_inf: float = 1e3):
        """
        Args:
            A: The alphabet size.
            t_inf: Large t value used to approximate the stationary distribution.
        """

        super().__init__()

        self.A = A
        self.t_inf = tf.constant(t_inf, DTYPE_FLOAT)

        self.log_Q_matrix_AxA = tf.Variable(
            tf.constant(0, DTYPE_FLOAT, [A, A]), name="log_Q"
        )

    @tf_function(reduce_retracing=True)
    def Q_matrix_VxSxAxA(self, embeddings_VxD):
        # first non-diagonal entry in each row can be fixed to match degrees of
        # freedom
        Q_matrix_AxA = tf.tensor_scatter_nd_update(
            self.log_Q_matrix_AxA,
            [[i, (i + 1) % self.A] for i in range(self.A)],
            [tf.constant(0, DTYPE_FLOAT)] * self.A,
        )

        # use exp to ensure all off-diagonal entries are positive
        Q_matrix_AxA = tf.exp(Q_matrix_AxA)

        # exclude diagonal entry for now...
        Q_matrix_AxA = tf.linalg.set_diag(Q_matrix_AxA, [0] * self.A)

        # normalize off-diagonal entries within each row
        denom_Ax1 = tf.reduce_sum(Q_matrix_AxA, 1, True)
        Q_matrix_AxA /= denom_Ax1

        # set diagonal to -1 (sum of off-diagonal entries)
        hyphens_A = tf.ones(self.A, DTYPE_FLOAT)
        Q_matrix_AxA = tf.linalg.set_diag(Q_matrix_AxA, -hyphens_A)

        # return only shape (1,1,A,A), but assume broadcasting rules apply...
        Q_matrix_1x1xAxA = Q_matrix_AxA[tf.newaxis, tf.newaxis]
        return Q_matrix_1x1xAxA

    @tf_function(reduce_retracing=True)
    def stat_probs_VxSxA(self, embeddings_VxD):
        # find e^(Qt) as t -> inf; then, stationary distribution is in every row
        Q_matrix_VxSxAxA = self.Q_matrix_VxSxAxA(embeddings_VxD)
        expm_limit_VxSxAxA = tf.linalg.expm(Q_matrix_VxSxAxA * self.t_inf)
        stat_probs_VxSxA = expm_limit_VxSxAxA[:, :, 0]  # type: ignore
        return stat_probs_VxSxA


class GT16StationaryQMatrixDecoder(QMatrixDecoder):
    """
    Assumes A=16. Uses the CellPhy GT16 model. Q matrix is the same globally,
    irrespective of the input embeddings.
    """

    def __init__(self, *, reg_lambda: float = 0.0):
        """
        Args:
            reg_lambda: Stationary probability regularization coefficient.
        """

        super().__init__()

        self.reg_lambda = tf.constant(reg_lambda, DTYPE_FLOAT)

        self.log_nucleotide_exchanges_6 = tf.Variable(
            tf.constant(0, DTYPE_FLOAT, [6]), name="log_nucleotide_exchanges"
        )
        self.log_stat_probs_A = tf.Variable(
            tf.constant(0, DTYPE_FLOAT, [16]), name="log_stat_probs"
        )

    @tf_function()
    def nucleotide_exchanges_6(self):
        # one exchangeability can be fixed to match degrees of freedom
        nucleotide_exchanges_6 = tf.tensor_scatter_nd_update(
            self.log_nucleotide_exchanges_6, [[0]], [tf.constant(0, DTYPE_FLOAT)]
        )
        # use exp to ensure all entries are positive
        nucleotide_exchanges_6 = tf.exp(nucleotide_exchanges_6)
        # normalize to ensure mean is 1
        nucleotide_exchanges_6 /= tf.reduce_mean(nucleotide_exchanges_6)
        return nucleotide_exchanges_6

    @tf_function()
    def stats_probs_A(self):
        """Internal, returns global stationary probabilities."""

        # one stationary probability can be fixed to match degrees of freedom
        stat_probs_A = tf.tensor_scatter_nd_update(
            self.log_stat_probs_A, [[0]], [tf.constant(0, DTYPE_FLOAT)]
        )
        # use softmax to ensure all entries are positive
        stat_probs_A = tf.exp(stat_probs_A)
        # normalize to ensure sum is 1
        stat_probs_A /= tf.reduce_sum(stat_probs_A)
        return stat_probs_A

    @tf_function(reduce_retracing=True)
    def Q_matrix_VxSxAxA(self, embeddings_VxD):
        pi = self.nucleotide_exchanges_6()  # length 6
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

        R_AxA = tf.scatter_nd(updates, pi8, [16, 16])
        R_AxA = R_AxA + tf.transpose(R_AxA)

        stat_probs_A = self.stats_probs_A()
        Q_matrix_AxA = tf.matmul(R_AxA, tf.linalg.diag(stat_probs_A))

        hyphens = tf.reduce_sum(Q_matrix_AxA, 1)
        Q_matrix_AxA = tf.linalg.set_diag(Q_matrix_AxA, -hyphens)

        # return only shape (1,1,A,A), but assume broadcasting rules apply...
        Q_matrix_1x1xAxA = Q_matrix_AxA[tf.newaxis, tf.newaxis]
        return Q_matrix_1x1xAxA

    @tf_function(reduce_retracing=True)
    def stat_probs_VxSxA(self, embeddings_VxD):
        stat_probs_A = self.stats_probs_A()

        # return only shape (1,1,A), but assume broadcasting rules apply...
        stat_probs_VxSxA = stat_probs_A[tf.newaxis, tf.newaxis]
        return stat_probs_VxSxA

    @tf_function()
    def regularization(self):
        stat_probs_A = self.stats_probs_A()
        stat_probs_norm_squared = tf.reduce_sum(tf.math.square(stat_probs_A))
        return self.reg_lambda * stat_probs_norm_squared
