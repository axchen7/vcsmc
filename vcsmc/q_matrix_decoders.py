import torch
from torch import Tensor, nn

from .distances import Distance
from .site_positions_encoders import DummySitePositionsEncoder, SitePositionsEncoder
from .utils.encoder_utils import MLP
from .utils.repr_utils import custom_module_repr

__all__ = [
    "QMatrixDecoder",
    "JC69QMatrixDecoder",
    "FactorizedStationaryQMatrixDecoder",
    "FactorizedMLPQMatrixDecoder",
    "DenseStationaryQMatrixDecoder",
    "DenseMLPQMatrixDecoder",
    "GT16StationaryQMatrixDecoder",
    "DensePerSiteStatProbsMLPQMatrixDecoder",
]


class QMatrixDecoder(nn.Module):
    def __init__(
        self, *, A: int, site_positions_encoder: SitePositionsEncoder | None = None
    ):
        super().__init__()

        self.A = A
        self.site_positions_encoder: SitePositionsEncoder = (
            site_positions_encoder or DummySitePositionsEncoder()
        )

    def extra_repr(self) -> str:
        return custom_module_repr({"A": self.A})

    def Q_matrix_VxSxAxA(
        self, embeddings_VxD: Tensor, site_positions_SxC: Tensor
    ) -> Tensor:
        """
        Returns the Q matrix for each of the S sites. Operates on a batch of
        embeddings.
        """
        raise NotImplementedError

    def stat_probs_VxSxA(
        self, embeddings_VxD: Tensor, site_positions_SxC: Tensor
    ) -> Tensor:
        """
        Returns the stationary probabilities for each of the S sites. Operates
        on a batch of Q matrices.
        """
        raise NotImplementedError


class JC69QMatrixDecoder(QMatrixDecoder):
    """
    Uses the Jukes-Cantor 69 model, which assumes equal stationary frequencies
    and equal exchange rates. Thus, there is a global, fully fixed Q matrix (all
    entries are equal except for the diagonal).
    """

    def __init__(self, *, A: int):
        """
        Args:
            A: The alphabet size.
        """

        super().__init__(A=A)

        # make off-diagonal entries sum to 1 within each row, matching the
        # normalization used in DenseStationaryQMatrixDecoder
        fill_value = 1 / (A - 1)
        Q_matrix_AxA = torch.full([A, A], fill_value)
        # set diagonal to -1 (sum of off-diagonal entries)
        Q_matrix_AxA = Q_matrix_AxA.diagonal_scatter(-torch.ones(A))
        Q_matrix_AxA = Q_matrix_AxA[None, None]
        self.register_buffer("Q_matrix_AxA", Q_matrix_AxA)

        # stationary probabilities are uniform
        stat_probs_A = torch.full([A], 1 / A)
        stat_probs_A = stat_probs_A[None, None]
        self.register_buffer("stat_probs_A", stat_probs_A)

    def Q_matrix_VxSxAxA(
        self, embeddings_VxD: Tensor, site_positions_SxC: Tensor
    ) -> Tensor:
        V = embeddings_VxD.shape[0]
        S = site_positions_SxC.shape[0]

        return self.Q_matrix_AxA.expand(V, S, -1, -1)

    def stat_probs_VxSxA(
        self, embeddings_VxD: Tensor, site_positions_SxC: Tensor
    ) -> Tensor:
        V = embeddings_VxD.shape[0]
        S = site_positions_SxC.shape[0]

        return self.stat_probs_A.expand(V, S, -1)


class FactorizedStationaryQMatrixDecoder(QMatrixDecoder):
    """
    Parameterize the Q matrix by factoring it into A holding times and A
    stationary probabilities, each a trainable variable. The Q matrix is the
    same globally, irrespective of the input embeddings.
    """

    def __init__(self, *, A: int):
        """
        Args:
            A: The alphabet size.
        """

        super().__init__(A=A)
        self.register_buffer("zero", torch.zeros(1))

        self.A = A

        self.log_holding_times_A = nn.Parameter(torch.zeros(A))
        self.log_stat_probs_A = nn.Parameter(torch.zeros(A))

    def reciprocal_holding_times_A(self):
        """
        Reciprocal of the expected holding times of each letter in the alphabet.
        """

        # use exp to ensure all entries are positive
        reciprocal_holding_times_A = torch.exp(-self.log_holding_times_A)
        # normalize to ensure mean of reciprocal holding times is 1
        reciprocal_holding_times_A = (
            reciprocal_holding_times_A / reciprocal_holding_times_A.mean()
        )
        return reciprocal_holding_times_A

    def stat_probs_A(self):
        return self.log_stat_probs_A.softmax(0)

    def Q_matrix_VxSxAxA(
        self, embeddings_VxD: Tensor, site_positions_SxC: Tensor
    ) -> Tensor:
        V = embeddings_VxD.shape[0]
        S = site_positions_SxC.shape[0]

        reciprocal_holding_times_A = self.reciprocal_holding_times_A()
        stat_probs_A = self.stat_probs_A()

        # reciprocal holding times, repeated across each row
        recip_times_AxA = reciprocal_holding_times_A.unsqueeze(1).repeat(1, self.A)
        stat_probs_diag_AxA = torch.diag(stat_probs_A)

        Q_matrix_AxA = torch.matmul(recip_times_AxA, stat_probs_diag_AxA)

        # set the diagonals to the sum of the off-diagonal entries
        Q_matrix_AxA = Q_matrix_AxA.diagonal_scatter(self.zero.expand(self.A))
        diag_A = torch.sum(Q_matrix_AxA, -1)
        Q_matrix_AxA = Q_matrix_AxA.diagonal_scatter(-diag_A)

        return Q_matrix_AxA.expand(V, S, -1, -1)

    def stat_probs_VxSxA(
        self, embeddings_VxD: Tensor, site_positions_SxC: Tensor
    ) -> Tensor:
        V = embeddings_VxD.shape[0]
        S = site_positions_SxC.shape[0]

        return self.stat_probs_A().expand(V, S, -1)


class FactorizedMLPQMatrixDecoder(QMatrixDecoder):
    """
    Parameterize the Q matrix by factoring it into A holding times and A
    stationary probabilities. Uses a single multi-layer perceptron to learn
    these 2A parameters. The Q-matrix is local to each embedding.
    """

    def __init__(
        self,
        distance: Distance,
        *,
        A: int,
        D: int,
        width: int = 16,
        depth: int = 2,
    ):
        """
        Args:
            A: Alphabet size.
            D: Number of dimensions sequence embeddings.
            width: Width of each hidden layer.
            depth: Number of hidden layers.
        """

        super().__init__(A=A)
        self.register_buffer("zero", torch.zeros(1))

        self.distance = distance
        self.A = A

        D1 = distance.feature_expand_shape(D)
        self.mlp = MLP(D1, 2 * A, width, depth)

    def holding_times_VxA_and_stat_probs_VxA(
        self, embeddings_VxD: Tensor
    ) -> tuple[Tensor, Tensor]:
        """
        Returns: holding_times_VxA, stat_probs_VxA
        """

        expanded_VxD1 = self.distance.feature_expand(embeddings_VxD)
        output_Vx2A = self.mlp(expanded_VxD1)

        log_holding_times_VxA = output_Vx2A[:, : self.A]
        log_stat_probs_VxA = output_Vx2A[:, self.A :]

        # use exp to ensure all entries are positive
        reciprocal_holding_times_VxA = torch.exp(-log_holding_times_VxA)
        # normalize to ensure mean of reciprocal holding times is 1
        reciprocal_holding_times_VxA = (
            reciprocal_holding_times_VxA
            / reciprocal_holding_times_VxA.mean(dim=-1, keepdim=True)
        )

        stat_probs_VxA = log_stat_probs_VxA.softmax(-1)

        return reciprocal_holding_times_VxA, stat_probs_VxA

    def Q_matrix_VxSxAxA(
        self, embeddings_VxD: Tensor, site_positions_SxC: Tensor
    ) -> Tensor:
        V = embeddings_VxD.shape[0]
        S = site_positions_SxC.shape[0]

        reciprocal_holding_times_VxA, stat_probs_VxA = (
            self.holding_times_VxA_and_stat_probs_VxA(embeddings_VxD)
        )

        # reciprocal holding times, repeated across each row
        recip_times_VxAxA = reciprocal_holding_times_VxA.unsqueeze(-1).repeat(
            1, 1, self.A
        )
        stat_probs_diag_VxAxA = torch.diag_embed(stat_probs_VxA)

        Q_matrix_VxAxA = torch.matmul(recip_times_VxAxA, stat_probs_diag_VxAxA)

        # set the diagonals to the sum of the off-diagonal entries
        Q_matrix_VxAxA = Q_matrix_VxAxA.diagonal_scatter(
            self.zero.expand(V, self.A), dim1=-2, dim2=-1
        )
        diag_VxA = torch.sum(Q_matrix_VxAxA, -1)
        Q_matrix_VxAxA = Q_matrix_VxAxA.diagonal_scatter(-diag_VxA, dim1=-2, dim2=-1)

        return Q_matrix_VxAxA[:, None].expand(-1, S, -1, -1)

    def stat_probs_VxSxA(
        self, embeddings_VxD: Tensor, site_positions_SxC: Tensor
    ) -> Tensor:
        S = site_positions_SxC.shape[0]

        _, stat_probs_VxA = self.holding_times_VxA_and_stat_probs_VxA(embeddings_VxD)
        return stat_probs_VxA[:, None].expand(-1, S, -1)


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

        super().__init__(A=A)
        self.register_buffer("zeros", torch.zeros(1))
        self.register_buffer("ones", torch.ones(1))

        self.A = A
        self.t_inf = t_inf

        self.log_Q_matrix_AxA = nn.Parameter(torch.zeros(A, A))

    def Q_matrix_VxSxAxA(
        self, embeddings_VxD: Tensor, site_positions_SxC: Tensor
    ) -> Tensor:
        V = embeddings_VxD.shape[0]
        S = site_positions_SxC.shape[0]

        # use exp to ensure all off-diagonal entries are positive
        Q_matrix_AxA = self.log_Q_matrix_AxA.exp()

        # exclude diagonal entry for now...
        Q_matrix_AxA = Q_matrix_AxA.diagonal_scatter(self.zeros.expand(self.A))

        # normalize off-diagonal entries within each row
        denom_Ax1 = torch.sum(Q_matrix_AxA, -1, True)
        Q_matrix_AxA = Q_matrix_AxA / denom_Ax1

        # set diagonal to -1 (sum of off-diagonal entries)
        Q_matrix_AxA = Q_matrix_AxA.diagonal_scatter(-self.ones.expand(self.A))

        return Q_matrix_AxA.expand(V, S, -1, -1)

    def stat_probs_VxSxA(
        self, embeddings_VxD: Tensor, site_positions_SxC: Tensor
    ) -> Tensor:
        # find e^(Qt) as t -> inf; then, stationary distribution is in every row
        Q_matrix_VxSxAxA = self.Q_matrix_VxSxAxA(embeddings_VxD, site_positions_SxC)
        expm_limit_VxSxAxA = torch.matrix_exp(Q_matrix_VxSxAxA * self.t_inf)
        stat_probs_VxSxA = expm_limit_VxSxAxA[:, :, 0]
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
        D: int,
        width: int = 16,
        depth: int = 2,
        t_inf: float = 1e3,
    ):
        """
        Args:
            A: Alphabet size.
            D: Number of dimensions sequence embeddings.
            width: Width of each hidden layer.
            depth: Number of hidden layers.
            t_inf: Large t value used to approximate the stationary distribution.
        """

        super().__init__(A=A)
        self.register_buffer("zeros", torch.zeros(1))
        self.register_buffer("ones", torch.ones(1))

        self.distance = distance
        self.A = A
        self.t_inf = t_inf

        D1 = distance.feature_expand_shape(D)
        self.mlp = MLP(D1, A * A, width, depth)

    def Q_matrix_VxSxAxA(
        self, embeddings_VxD: Tensor, site_positions_SxC: Tensor
    ) -> Tensor:
        V = embeddings_VxD.shape[0]
        S = site_positions_SxC.shape[0]

        expanded_VxD1 = self.distance.feature_expand(embeddings_VxD)

        # get Q matrix and reshape
        log_Q_matrix_VxAA: Tensor = self.mlp(expanded_VxD1)
        log_Q_matrix_VxAxA = log_Q_matrix_VxAA.view(V, self.A, self.A)

        # use exp to ensure all off-diagonal entries are positive
        Q_matrix_VxAxA = log_Q_matrix_VxAxA.exp()

        # exclude diagonal entry for now...
        Q_matrix_VxAxA = Q_matrix_VxAxA.diagonal_scatter(
            self.zeros.expand(V, self.A), dim1=-2, dim2=-1
        )

        # normalize off-diagonal entries within each row
        denom_VxAx1 = torch.sum(Q_matrix_VxAxA, -1, True)
        Q_matrix_VxAxA = Q_matrix_VxAxA / denom_VxAx1

        # set diagonal to -1 (sum of off-diagonal entries)
        Q_matrix_VxAxA = Q_matrix_VxAxA.diagonal_scatter(
            -self.ones.expand(V, self.A), dim1=-2, dim2=-1
        )

        Q_matrix_Vx1xAxA = Q_matrix_VxAxA[:, None]
        return Q_matrix_Vx1xAxA.expand(-1, S, -1, -1)

    def stat_probs_VxSxA(
        self, embeddings_VxD: Tensor, site_positions_SxC: Tensor
    ) -> Tensor:
        # find e^(Qt) as t -> inf; then, stationary distribution is in every row
        Q_matrix_VxSxAxA = self.Q_matrix_VxSxAxA(embeddings_VxD, site_positions_SxC)
        expm_limit_VxSxAxA = torch.matrix_exp(Q_matrix_VxSxAxA * self.t_inf)
        stat_probs_VxSxA = expm_limit_VxSxAxA[:, :, 0]
        return stat_probs_VxSxA


class GT16StationaryQMatrixDecoder(QMatrixDecoder):
    """
    Assumes A=16. Uses the CellPhy GT16 model. Q matrix is the same globally,
    irrespective of the input embeddings.
    """

    def __init__(
        self,
        *,
        exchanges_baseline: float = 0.5,
        stat_props_baseline: float = 0.5,
    ):
        """
        Args:
            exchanges_baseline: Baseline probabilities for nucleotide_exchanges_6, from 0 to 1.
                Take this much of the probability mass from the uniform distribution.
                Helps prevent the model from converging to a degenerate solution.
            stat_props_baseline: Baseline probabilities for stat_probs, for each of the A letters, from 0 to 1.
                Take this much of the probability mass from the uniform distribution.
                Helps prevent the model from converging to a degenerate solution.
        """

        super().__init__(A=16)

        self.exchanges_baseline = exchanges_baseline
        self.stat_props_baseline = stat_props_baseline

        self.log_nucleotide_exchanges_6 = nn.Parameter(torch.zeros(6))
        self.log_stat_probs_A = nn.Parameter(torch.zeros(16))

        self.register_buffer("updates", self.get_updates())

    def get_updates(self):
        # index helpers for Q matrix
        AA, CC, GG, TT, AC, AG, AT, CG, CT, GT, CA, GA, TA, GC, TC, TG = range(16)

        # fmt: off
        return torch.tensor([
          # | first base changes                    | second base changes
            [AA, CA], [AC, CC], [AG, CG], [AT, CT], [AA, AC], [CA, CC], [GA, GC], [TA, TC], # A->C
            [AA, GA], [AC, GC], [AG, GG], [AT, GT], [AA, AG], [CA, CG], [GA, GG], [TA, TG], # A->G
            [AA, TA], [AC, TC], [AG, TG], [AT, TT], [AA, AT], [CA, CT], [GA, GT], [TA, TT], # A->T
            [CA, GA], [CC, GC], [CG, GG], [CT, GT], [AC, AG], [CC, CG], [GC, GG], [TC, TG], # C->G
            [CA, TA], [CC, TC], [CG, TG], [CT, TT], [AC, AT], [CC, CT], [GC, GT], [TC, TT], # C->T
            [GA, TA], [GC, TC], [GG, TG], [GT, TT], [AG, AT], [CG, CT], [GG, GT], [TG, TT], # G->T
        ])
        # fmt: on

    def nucleotide_exchanges_6(self):
        # use exp to ensure all entries are positive
        nucleotide_exchanges_6 = self.log_nucleotide_exchanges_6.exp()
        # normalize to ensure mean is 1
        nucleotide_exchanges_6 = nucleotide_exchanges_6 / torch.mean(
            nucleotide_exchanges_6
        )
        nucleotide_exchanges_6 = (
            self.exchanges_baseline * 1
            + (1 - self.exchanges_baseline) * nucleotide_exchanges_6
        )
        return nucleotide_exchanges_6

    def stats_probs_A(self):
        """Internal, returns global stationary probabilities."""

        stat_probs_A = self.log_stat_probs_A.softmax(0)
        stat_probs_A = (
            self.stat_props_baseline * (1 / 16)
            + (1 - self.stat_props_baseline) * stat_probs_A
        )
        return stat_probs_A

    def Q_matrix_VxSxAxA(
        self, embeddings_VxD: Tensor, site_positions_SxC: Tensor
    ) -> Tensor:
        device = embeddings_VxD.device

        V = embeddings_VxD.shape[0]
        S = site_positions_SxC.shape[0]

        pi = self.nucleotide_exchanges_6()  # length 6
        pi8 = pi.repeat(8)

        R_AxA = torch.zeros(16, 16, device=device)
        R_AxA[self.updates[:, 0], self.updates[:, 1]] = pi8
        R_AxA = R_AxA + R_AxA.t()

        stat_probs_A = self.stats_probs_A()
        Q_matrix_AxA = torch.matmul(R_AxA, torch.diag(stat_probs_A))

        diag_A = torch.sum(Q_matrix_AxA, -1)
        Q_matrix_AxA = Q_matrix_AxA.diagonal_scatter(-diag_A)

        return Q_matrix_AxA.expand(V, S, -1, -1)

    def stat_probs_VxSxA(
        self, embeddings_VxD: Tensor, site_positions_SxC: Tensor
    ) -> Tensor:
        V = embeddings_VxD.shape[0]
        S = site_positions_SxC.shape[0]

        stat_probs_A = self.stats_probs_A()

        return stat_probs_A.expand(V, S, -1)


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
        site_positions_encoder: SitePositionsEncoder,
        *,
        A: int,
        D: int,
        C: int,
        width: int,
        depth: int,
        baseline: float = 0.5,
    ):
        """
        Args:
            distance: Used to feature expand the embeddings.
            site_positions_encoder: Used to compress the site positions.
            A: Alphabet size.
            D: Number of dimensions sequence embeddings.
            C: Number of dimensions compressed site positions.
            width: Width of each hidden layer.
            depth: Number of hidden layers.
            baseline: Baseline probabilities for stat_probs, for each of the A letters, from 0 to 1.
                Take this much of the probability mass from the uniform distribution 1/A.
                Helps prevent the model from converging to a degenerate solution.
        """

        super().__init__(A=A, site_positions_encoder=site_positions_encoder)
        self.register_buffer("zero", torch.zeros(1))

        self.distance = distance
        self.A = A
        self.baseline = baseline

        self.log_holding_times_A = nn.Parameter(torch.zeros(A))

        D1 = distance.feature_expand_shape(D)
        self.mlp = MLP(D1 + C, A, width, depth)

    def reciprocal_holding_times_A(self):
        """
        Reciprocal of the expected holding times of each letter in the alphabet.
        """

        # use exp to ensure all entries are positive
        reciprocal_holding_times_A = torch.exp(-self.log_holding_times_A)
        # normalize to ensure mean of reciprocal holding times is 1
        reciprocal_holding_times_A = (
            reciprocal_holding_times_A / reciprocal_holding_times_A.mean()
        )
        return reciprocal_holding_times_A

    def Q_matrix_VxSxAxA(
        self, embeddings_VxD: Tensor, site_positions_SxC: Tensor
    ) -> Tensor:
        V = embeddings_VxD.shape[0]
        S = site_positions_SxC.shape[0]

        reciprocal_holding_times_A = self.reciprocal_holding_times_A()
        stat_probs_VxSxA = self.stat_probs_VxSxA(embeddings_VxD, site_positions_SxC)

        # reciprocal holding times, repeated across each row
        recip_times_AxA = reciprocal_holding_times_A.unsqueeze(1).repeat(1, self.A)
        stat_probs_diag_VxSxAxA = torch.diag_embed(stat_probs_VxSxA)

        Q_matrix_VxSxAxA = torch.matmul(recip_times_AxA, stat_probs_diag_VxSxAxA)

        # set the diagonals to the sum of the off-diagonal entries
        Q_matrix_VxSxAxA = Q_matrix_VxSxAxA.diagonal_scatter(
            self.zero.expand(V, S, self.A), dim1=-2, dim2=-1
        )
        diag_VxSxA = torch.sum(Q_matrix_VxSxAxA, -1)
        Q_matrix_VxSxAxA = Q_matrix_VxSxAxA.diagonal_scatter(
            -diag_VxSxA, dim1=-2, dim2=-1
        )

        return Q_matrix_VxSxAxA

    def stat_probs_VxSxA(
        self, embeddings_VxD: Tensor, site_positions_SxC: Tensor
    ) -> Tensor:
        V = embeddings_VxD.shape[0]
        S = site_positions_SxC.shape[0]

        expanded_VxD1 = self.distance.feature_expand(embeddings_VxD)

        # shape (V*S, D1); repeat like AAABBBCCC
        expanded_repeated_VSxD1 = expanded_VxD1.repeat_interleave(S, 0)

        # shape (V*S, C); repeat like ABCABCABC
        site_positions_repeated_VSxC = site_positions_SxC.repeat(V, 1)

        # shape (V*S, D1+C); flattened list of embeddings and site positions
        expanded_with_site_positions_VSxD1C = torch.cat(
            [expanded_repeated_VSxD1, site_positions_repeated_VSxC], -1
        )

        stat_probs_VSxA = self.mlp(expanded_with_site_positions_VSxD1C).softmax(-1)

        stat_probs_VxSxA = stat_probs_VSxA.view(V, S, self.A)
        stat_probs_VxSxA = (
            self.baseline * (1 / self.A) + (1 - self.baseline) * stat_probs_VxSxA
        )

        return stat_probs_VxSxA
