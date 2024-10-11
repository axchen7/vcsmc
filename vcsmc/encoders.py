import torch
from torch import Tensor, nn

from .distances import Distance, Euclidean, Hyperbolic
from .utils.distance_utils import EPSILON, safe_norm
from .utils.encoder_utils import MLP
from .utils.repr_utils import custom_module_repr

__all__ = [
    "SequenceEncoder",
    "DummySequenceEncoder",
    "EmbeddingTableSequenceEncoder",
    "MLPSequenceEncoder",
    "MergeEncoder",
    "MLPMergeEncoder",
    "EuclideanMidpointMergeEncoder",
    "HyperbolicGeodesicClosestMergeEncoder",
    "HyperbolicGeodesicMidpointMergeEncoder",
]


class SequenceEncoder(nn.Module):
    """Encodes sequences into embeddings."""

    def __init__(self, distance: Distance | None, *, D: int):
        """
        Args:
            distance: Stored for external use.
            D: Number of dimensions in sequence embeddings.
        """

        super().__init__()

        self.distance = distance
        self.D = D

    def extra_repr(self) -> str:
        return custom_module_repr({"D": self.D})

    def forward(self, sequences_VxSxA: Tensor) -> Tensor:
        """
        Args:
            sequences_VxSxA: Sequences to encode.
        Returns:
            embeddings_VxD: Encoded sequences, normalized.
        """
        raise NotImplementedError


class DummySequenceEncoder(SequenceEncoder):
    """A dummy encoder that returns zero embeddings."""

    def __init__(self):
        super().__init__(None, D=0)
        self.register_buffer("zero", torch.zeros(1))

    def forward(self, sequences_VxSxA: Tensor) -> Tensor:
        V = sequences_VxSxA.shape[0]
        return self.zero.expand(V, 0)


class EmbeddingTableSequenceEncoder(SequenceEncoder):
    """
    Uses a fixed-size lookup table that stores each taxon's embedding.
    Ignores the sequences entirely.
    """

    def __init__(
        self,
        distance: Distance,
        data_NxSxA: Tensor,
        *,
        D: int,
        # these default works well empirically:
        # https://wandb.ai/azc2110-Columbia%20University/vcsmc/reports/Effect-of-Hyperbolic-Embedding-Initialization--Vmlldzo4NzU2MDA1?accessToken=zodozvfqe8onlr7tdhflgq36ee85ysp6w1d0vi6m5tm2zql7fsygsnnzz479gbhn
        initial_mean: float = 0.7,
        initial_std: float = 0.1,
    ):
        """
        Args:
            distance: Stored for external use.
            data_NxSxA: fixed sequences to support.
                On the forward pass, an error is thrown if the input sequences
                don't match these sequences exactly.
            D: Number of dimensions sequence embeddings.
            initial_mean: Mean of the normal distribution used to initialize the embeddings.
            initial_std: Std dev of the normal distribution used to initialize the embeddings.
        """

        super().__init__(distance, D=D)

        N = data_NxSxA.shape[0]

        self.register_buffer("data_NxSxA", data_NxSxA)

        self.embedding_table = nn.Parameter(
            torch.normal(initial_mean, initial_std, size=(N, D))
        )

    def forward(self, sequences_VxSxA: Tensor) -> Tensor:
        if torch.equal(sequences_VxSxA, self.data_NxSxA):
            # shortcut for exact match
            return self.embedding_table
        else:
            # find the indices of the sequences in the data
            indices_V = torch.tensor([], dtype=torch.int, device=sequences_VxSxA.device)
            for sequence_SxA in sequences_VxSxA:
                # hack: torch.all has a segfault bug on MPS with multiple dims
                # mask = torch.all(sequence_SxA == self.data_NxSxA, dim=(1, 2))
                mask = torch.all(sequence_SxA == self.data_NxSxA, -1).all(-1)

                index_1 = torch.nonzero(mask).squeeze(1)
                if index_1.numel() == 0:
                    raise ValueError("Sequence not found in data.")
                # if multiple matches, take the first one
                indices_V = torch.cat((indices_V, index_1[0].unsqueeze(0)))
            return self.embedding_table[indices_V]


class MLPSequenceEncoder(SequenceEncoder):
    """Uses a multi-layer perceptron."""

    def __init__(
        self, distance: Distance, *, S: int, A: int, D: int, width: int, depth: int
    ):
        """
        Args:
            distance: Stored for external use.
            S: Number of sites.
            A: Alphabet size.
            D: Number of dimensions sequence embeddings.
            width: Width of each hidden layer.
            depth: Number of hidden layers.
        """

        super().__init__(distance, D=D)
        self.mlp = MLP(S * A, D, width, depth)

    def forward(self, sequences_VxSxA: Tensor) -> Tensor:
        return self.mlp(sequences_VxSxA)


class MergeEncoder(nn.Module):
    """Encodes a pair of child embeddings into a parent embedding."""

    def forward(self, children1_VxD: Tensor, children2_VxD: Tensor) -> Tensor:
        """
        Args:
            children1_VxD: First child embeddings.
            children2_VxD: Second child embeddings.
        Returns:
            embeddings_VxD: Encoded parents, normalized.
        """
        raise NotImplementedError


class MLPMergeEncoder(MergeEncoder):
    def __init__(self, distance: Distance, *, D: int, width: int, depth: int):
        """
        Args:
            distance: Used to feature expand the embeddings.
            D: Number of dimensions in output embeddings.
            width: Width of each hidden layer.
            depth: Number of hidden layers.
        """

        super().__init__()

        self.distance = distance

        D1 = distance.feature_expand_shape(D)
        self.mlp = MLP(2 * D1, 2, width, depth)

    def forward(self, children1_VxD: Tensor, children2_VxD: Tensor) -> Tensor:
        p = children1_VxD
        q = children2_VxD

        p_norm = safe_norm(p, 1, True)
        q_norm = safe_norm(q, 1, True)

        # "pull" the further child to the same distance from the origin as the
        # closer child
        p = torch.where(p_norm > q_norm, p * q_norm / p_norm, p)
        q = torch.where(q_norm > p_norm, q * p_norm / q_norm, q)

        # use original children embeddings for MLP
        expanded1_VxD1 = self.distance.feature_expand(children1_VxD)
        expanded2_VxD1 = self.distance.feature_expand(children2_VxD)

        alpha_beta_Vx2: Tensor = self.mlp(
            torch.cat([expanded1_VxD1, expanded2_VxD1], -1)
        )

        # sigmoid to ensure alpha and beta are in [0, 1]
        alpha_Vx1 = alpha_beta_Vx2[:, 0].sigmoid().unsqueeze(1)
        beta_Vx1 = alpha_beta_Vx2[:, 1].sigmoid().unsqueeze(1)

        midpoints_VxD = p * alpha_Vx1 + q * (1 - alpha_Vx1)
        parents_VxD = midpoints_VxD * beta_Vx1
        return parents_VxD


class EuclideanMidpointMergeEncoder(MergeEncoder):
    """
    Uses the euclidean midpoint between the two children embeddings as the
    parent embedding. The parent embedding is thus a deterministic function of
    the children embeddings. Works for any dimension of embeddings.
    """

    def __init__(self, distance: Distance):
        """
        Args:
            distance: Must be Euclidean.
        """

        super().__init__()
        assert isinstance(distance, Euclidean)

    def forward(self, children1_VxD: Tensor, children2_VxD: Tensor) -> Tensor:
        midpoints_VxD = (children1_VxD + children2_VxD) / 2
        return midpoints_VxD


class HyperbolicGeodesicClosestMergeEncoder(MergeEncoder):
    """
    Uses the point on the geodesic between the two children closest to the
    origin as the parent embedding. The parent embedding is thus a deterministic
    function of the children embeddings. Requires embeddings to have dimension
    2.
    """

    def __init__(self, distance: Distance, *, D: int):
        """
        Args:
            D: Number of dimensions sequence embeddings (must be 2).
        """

        super().__init__()

        assert D == 2
        assert isinstance(distance, Hyperbolic)

        self.distance = distance

    def forward(self, children1_VxD: Tensor, children2_VxD: Tensor) -> Tensor:
        # all values are vectors (shape=[V, 2]) unless otherwise stated

        p = self.distance.normalize(children1_VxD)
        q = self.distance.normalize(children2_VxD)

        p_norm = safe_norm(p, 1, True)
        q_norm = safe_norm(q, 1, True)

        # "pull" the further child to the same distance from the origin as the
        # closer child
        p = torch.where(p_norm > q_norm, p * q_norm / p_norm, p)
        q = torch.where(q_norm > p_norm, q * p_norm / q_norm, q)

        r = (p + q) / 2
        diff = p - q

        n = torch.stack([-diff[:, 1], diff[:, 0]], 1)
        nhat = n / safe_norm(n, 1, True)

        # ===== scalars (shape=[V] because of vectorization) =====
        p_dot_p = torch.sum(p * p, 1)
        p_dot_r = torch.sum(p * r, 1)
        p_dot_nhat = torch.sum(p * nhat, 1)

        # p_dot_nhat=0 would result in NaNs down the line, so first replace with
        # 1 to ensure that m ultimately has no NaNs. This is needed because
        # although we intend to replace m with r in such cases anyway, m must
        # not contain NaNs as calling torch.where() with NaNs may cause NaN
        # gradients.
        ok = p_dot_nhat.abs() > EPSILON
        p_dot_nhat = torch.where(ok, p_dot_nhat, torch.ones_like(p_dot_nhat))

        alpha = (p_dot_p - 2 * p_dot_r + 1) / (2 * p_dot_nhat)
        # ===== end scalars =====

        s = r + alpha.unsqueeze(1) * nhat

        s_minus_p_norm = safe_norm(s - p, 1, True)
        s_norm = safe_norm(s, 1, True)

        m = s * (1 - s_minus_p_norm / s_norm)

        # if any resulting vector is degenerate, replace with midpoint between p
        # and q
        m = torch.where(ok.unsqueeze(1), m, r)

        return self.distance.unnormalize(m)


class HyperbolicGeodesicMidpointMergeEncoder(MergeEncoder):
    """
    Uses the hyperbolic midpoint along the geodesic between the two children.
    The parent embedding is thus a deterministic function of the children
    embeddings. Works for any dimension of embeddings.
    """

    def __init__(self, distance: Distance):
        """
        Args:
            distance: Used to normalize the embeddings.
        """

        super().__init__()

        assert isinstance(distance, Hyperbolic)
        self.distance = distance

    def forward(self, children1_VxD: Tensor, children2_VxD: Tensor) -> Tensor:
        # all values are vectors (shape=[V, D]) unless otherwise stated

        p = self.distance.normalize(children1_VxD)
        q = self.distance.normalize(children2_VxD)

        p_norm = safe_norm(p, 1, True)
        q_norm = safe_norm(q, 1, True)

        closer_point = torch.where(p_norm < q_norm, p, q)

        # "pull" the further child to the same distance from the origin as the
        # closer child
        p = torch.where(p_norm > q_norm, p * q_norm / p_norm, p)
        q = torch.where(q_norm > p_norm, q * p_norm / q_norm, q)

        m = (p + q) / 2  # midpoint

        # ===== scalars (shape=[V] because of vectorization) =====
        m_norm_sq = torch.sum(m**2, 1)
        # use norm squared of the closer point
        p_norm_sq = torch.sum(closer_point**2, 1)
        # dot product between m and the closer point
        k = torch.sum(m * closer_point, 1)

        # okay only if m_norm_sq != 0 and k != 0
        ok = torch.logical_and(m_norm_sq.abs() > EPSILON, k.abs() > EPSILON)

        # replace with 1 to ensure that m ultimately has no NaNs
        m_norm_sq = torch.where(ok, m_norm_sq, torch.ones_like(m_norm_sq))
        k = torch.where(ok, k, torch.ones_like(k))

        radicand = (1 + p_norm_sq) ** 2 - 4 * (k**2) / m_norm_sq

        # okay if radicand >= 0
        ok = torch.logical_and(ok, radicand >= 0)

        # replace with 1 to ensure that m ultimately has no NaNs
        radicand = torch.where(ok, radicand, torch.ones_like(radicand))

        numerator = (1 + p_norm_sq) - torch.sqrt(radicand + EPSILON)
        denominator = 2 * k

        alpha = numerator / denominator
        # ===== end scalars =====

        s = m * alpha.unsqueeze(1)

        # if any resulting vector is degenerate, replace with midpoint between p
        # and q
        s = torch.where(ok.unsqueeze(1), s, m)

        return self.distance.unnormalize(s)
