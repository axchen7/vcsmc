import keras
import tensorflow as tf

from constants import DTYPE_FLOAT
from distances import Distance
from encoders import mlp_add_hidden_layers
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


class DenseMLPQMatrixDecoder(QMatrixDecoder):
    """
    Use a multi-layer perceptron to learn every entry in the Q matrix (except
    the diagonal). Q-matrix is local to each embedding, but is the same across
    all sites. The MLP's input dimension is the (feature-expanded) embedding
    dimension.
    """

    def __init__(
        self,
        distance: Distance,
        *,
        A: int,
        width: int,
        depth: int,
        t_inf: float = 1e3,
    ):
        """
        Args:
            A: Alphabet size.
            width: Width of each hidden layer.
            depth: Number of hidden layers.
            t_inf: Large t value used to approximate the stationary distribution.
        """

        super().__init__()

        self.distance = distance
        self.A = A
        self.width = width
        self.depth = depth
        self.t_inf = tf.constant(t_inf, DTYPE_FLOAT)

        self.mlp = None

    def create_mlp(self, D1: int):
        mlp = keras.Sequential()
        mlp.add(keras.layers.Input([D1], dtype=DTYPE_FLOAT))
        mlp_add_hidden_layers(mlp, width=self.width, depth=self.depth)
        mlp.add(keras.layers.Dense(self.A * (self.A - 1), dtype=DTYPE_FLOAT))
        mlp.add(keras.layers.Reshape([self.A, self.A - 1]))
        mlp.add(keras.layers.Softmax(-1, dtype=DTYPE_FLOAT))
        return mlp

    @tf_function(reduce_retracing=True)
    def Q_matrix_VxSxAxA(self, embeddings_VxD):
        V = tf.shape(embeddings_VxD)[0]  # type: ignore

        expanded_VxD1 = self.distance.feature_expand(embeddings_VxD)

        if self.mlp is None:
            D1 = expanded_VxD1.shape[1]
            self.mlp = self.create_mlp(D1)

        # get off-diagonal entries
        Q_matrix_VxAxA1 = self.mlp(expanded_VxD1)

        # expand diagonal entries into new column at the end
        diag_VxA1 = tf.linalg.diag_part(Q_matrix_VxAxA1)
        diag_VxA = tf.concat([diag_VxA1, tf.zeros([V, 1], DTYPE_FLOAT)], -1)
        diag_VxAx1 = tf.expand_dims(diag_VxA, -1)
        Q_matrix_VxAxA = tf.concat([Q_matrix_VxAxA1, diag_VxAx1], -1)

        # set diagonal to -1 (sum of off-diagonal entries)
        hyphens_VxA = tf.ones([V, self.A], DTYPE_FLOAT)
        Q_matrix_VxAxA = tf.linalg.set_diag(Q_matrix_VxAxA, -hyphens_VxA)

        # return only shape (V,1,A,A), but assume broadcasting rules apply...
        Q_matrix_Vx1xAxA = Q_matrix_VxAxA[:, tf.newaxis]
        return Q_matrix_Vx1xAxA

    @tf_function(reduce_retracing=True)
    def stat_probs_VxSxA(self, embeddings_VxD):
        # find e^(Qt) as t -> inf; then, stationary distribution is in every row
        Q_matrix_VxSxAxA = self.Q_matrix_VxSxAxA(embeddings_VxD)
        expm_limit_VxSxAxA = tf.linalg.expm(Q_matrix_VxSxAxA * self.t_inf)
        stat_probs_VxSxA = expm_limit_VxSxAxA[:, :, 0]  # type: ignore
        return stat_probs_VxSxA


class DensePerSiteMLPQMatrixDecoder(QMatrixDecoder):
    """
    Use a multi-layer perceptron to learn every entry in the Q matrix (except
    the diagonal). Q-matrix is local to each embedding, and varies across sites.
    """

    def __init__(
        self,
        distance: Distance,
        *,
        S: int,
        A: int,
        width: int,
        depth: int,
        t_inf: float = 1e3,
    ):
        """
        Args:
            S: Number of sites.
            A: Alphabet size.
            width: Width of each hidden layer.
            depth: Number of hidden layers.
            t_inf: Large t value used to approximate the stationary distribution.
        """

        super().__init__()

        self.distance = distance
        self.S = S
        self.A = A
        self.width = width
        self.depth = depth
        self.t_inf = tf.constant(t_inf, DTYPE_FLOAT)

        self.mlp = None

    def create_mlp(self, D1: int):
        mlp = keras.Sequential()
        mlp.add(keras.layers.Input([D1], dtype=DTYPE_FLOAT))
        mlp_add_hidden_layers(mlp, width=self.width, depth=self.depth)
        mlp.add(keras.layers.Dense(self.S * self.A * (self.A - 1), dtype=DTYPE_FLOAT))
        mlp.add(keras.layers.Reshape([self.S, self.A, self.A - 1]))
        mlp.add(keras.layers.Softmax(-1, dtype=DTYPE_FLOAT))
        return mlp

    @tf_function(reduce_retracing=True)
    def Q_matrix_VxSxAxA(self, embeddings_VxD):
        V = tf.shape(embeddings_VxD)[0]  # type: ignore

        expanded_VxD1 = self.distance.feature_expand(embeddings_VxD)

        if self.mlp is None:
            D1 = expanded_VxD1.shape[1]
            self.mlp = self.create_mlp(D1)

        # get off-diagonal entries
        Q_matrix_VxSxAxA1 = self.mlp(expanded_VxD1)

        # expand diagonal entries into new column at the end
        diag_VxSxA1 = tf.linalg.diag_part(Q_matrix_VxSxAxA1)
        diag_VxSxA = tf.concat([diag_VxSxA1, tf.zeros([V, self.S, 1], DTYPE_FLOAT)], -1)
        diag_VxSxAx1 = tf.expand_dims(diag_VxSxA, -1)
        Q_matrix_VxSxAxA = tf.concat([Q_matrix_VxSxAxA1, diag_VxSxAx1], -1)

        # set diagonal to -1 (sum of off-diagonal entries)
        hyphens_VxSxA = tf.ones([V, self.S, self.A], DTYPE_FLOAT)
        Q_matrix_VxSxAxA = tf.linalg.set_diag(Q_matrix_VxSxAxA, -hyphens_VxSxA)

        return Q_matrix_VxSxAxA

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


class DensePerSiteStatProbsMLPQMatrixDecoder(QMatrixDecoder):
    """
    Parameterize the Q matrix by factoring it into A holding times and A
    stationary probabilities. The holding times are represented by global
    trainable variables, while the stationary probabilities are local to each
    embedding, and vary across sites. An MLP is used to learn the stationary
    probabilities.
    """

    def __init__(
        self,
        distance: Distance,
        *,
        S: int,
        A: int,
        width: int,
        depth: int,
        baseline: float = 0.5,
        t_inf: float = 1e3,
    ):
        """
        Args:
            S: Number of sites.
            A: Alphabet size.
            width: Width of each hidden layer.
            depth: Number of hidden layers.
            baseline: Baseline probabilities for stat_probs, for each of the A letters, from 0 to 1.
                Take this much of the probability mass from the uniform distribution 1/A.
                Helps prevent the model from converging to a degenerate solution.
            t_inf: Large t value used to approximate the stationary distribution.
        """

        super().__init__()

        self.distance = distance
        self.S = S
        self.A = A
        self.width = width
        self.depth = depth
        self.baseline = tf.constant(baseline, dtype=DTYPE_FLOAT)
        self.t_inf = tf.constant(t_inf, DTYPE_FLOAT)

        self.log_holding_times_A = tf.Variable(
            tf.constant(0, DTYPE_FLOAT, [A]), name="log_holding_times"
        )

        self.mlp = None

    @tf_function()
    def reciprocal_holding_times_A(self):
        """
        Reciprocal of the expected holding times of each letter in the alphabet.
        """

        # one holding time can be fixed to match degrees of freedom
        log_holding_times_A = tf.tensor_scatter_nd_update(
            self.log_holding_times_A, [[0]], [tf.constant(0, DTYPE_FLOAT)]
        )

        # use exp to ensure all entries are positive
        reciprocal_holding_times_A = tf.exp(-log_holding_times_A)
        # normalize to ensure mean of reciprocal holding times is 1
        reciprocal_holding_times_A /= tf.reduce_mean(reciprocal_holding_times_A)
        return reciprocal_holding_times_A

    def create_mlp(self, D1: int):
        mlp = keras.Sequential()
        mlp.add(keras.layers.Input([D1], dtype=DTYPE_FLOAT))
        mlp_add_hidden_layers(mlp, width=self.width, depth=self.depth)
        mlp.add(keras.layers.Dense(self.S * self.A, dtype=DTYPE_FLOAT))
        mlp.add(keras.layers.Reshape([self.S, self.A]))
        mlp.add(keras.layers.Softmax(-1, dtype=DTYPE_FLOAT))
        return mlp

    @tf_function(reduce_retracing=True)
    def Q_matrix_VxSxAxA(self, embeddings_VxD):
        V = tf.shape(embeddings_VxD)[0]  # type: ignore

        reciprocal_holding_times_A = self.reciprocal_holding_times_A()
        stat_probs_VxSxA = self.stat_probs_VxSxA(embeddings_VxD)

        reciprocal_holding_times_repeated_AxA = tf.repeat(
            reciprocal_holding_times_A[tf.newaxis], self.A, 0
        )
        stat_probs_diag_VxSxAxA = tf.linalg.diag(stat_probs_VxSxA)

        Q_matrix_VxSxAxA = tf.matmul(
            reciprocal_holding_times_repeated_AxA, stat_probs_diag_VxSxAxA
        )

        # set the diagonals to the sum of the off-diagonal entries
        Q_matrix_VxSxAxA = tf.linalg.set_diag(
            Q_matrix_VxSxAxA, tf.zeros([V, self.S, self.A], DTYPE_FLOAT)
        )
        hyphens_VxSxA = tf.reduce_sum(Q_matrix_VxSxAxA, -1)
        Q_matrix_VxSxAxA = tf.linalg.set_diag(Q_matrix_VxSxAxA, -hyphens_VxSxA)

        return Q_matrix_VxSxAxA

    @tf_function(reduce_retracing=True)
    def stat_probs_VxSxA(self, embeddings_VxD):
        expanded_VxD1 = self.distance.feature_expand(embeddings_VxD)

        if self.mlp is None:
            D1 = expanded_VxD1.shape[1]
            self.mlp = self.create_mlp(D1)

        stat_probs_VxSxA = self.mlp(expanded_VxD1)

        stat_probs_VxSxA = (
            self.baseline * (1 / self.A) + (1 - self.baseline) * stat_probs_VxSxA  # type: ignore
        )

        return stat_probs_VxSxA
