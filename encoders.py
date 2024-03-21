import torch
from torch import Tensor, nn

from distances import Distance, safe_norm
from encoder_utils import MLP


class SequenceEncoder(nn.Module):
    """Encodes sequences into embeddings."""

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

    def forward(self, sequences_VxSxA: Tensor) -> Tensor:
        V = sequences_VxSxA.shape[0]
        return torch.zeros([V, 0])


class MLPSequenceEncoder(SequenceEncoder):
    """Uses a multi-layer perceptron."""

    def __init__(
        self, distance: Distance, *, S: int, A: int, D: int, width: int, depth: int
    ):
        """
        Args:
            distance: Used to normalize the embeddings.
            S: Number of sites.
            A: Alphabet size.
            D: Number of dimensions sequence embeddings.
            width: Width of each hidden layer.
            depth: Number of hidden layers.
        """

        super().__init__()

        self.distance = distance
        self.mlp = MLP(S * A, D, width, depth)

    def forward(self, sequences_VxSxA: Tensor) -> Tensor:
        return self.distance.normalize(self.mlp(sequences_VxSxA))


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
    def __init__(
        self, distance: Distance, *, S: int, A: int, D: int, width: int, depth: int
    ):
        """
        Args:
            distance: Used to feature expand and normalize the embeddings.
            S: Number of sites.
            A: Alphabet size.
            D: Number of dimensions in output embeddings.
            width: Width of each hidden layer.
            depth: Number of hidden layers.
        """

        super().__init__()

        self.distance = distance

        D1 = distance.feature_expand_shape(D)
        self.mlp = MLP(2 * D1, D, width, depth)

    def forward(self, children1_VxD: Tensor, children2_VxD: Tensor) -> Tensor:
        expanded1_VxD1 = self.distance.feature_expand(children1_VxD)
        expanded2_VxD1 = self.distance.feature_expand(children2_VxD)

        return self.distance.normalize(
            self.mlp(torch.cat([expanded1_VxD1, expanded2_VxD1], -1))
        )


class HyperbolicGeodesicMergeEncoder(MergeEncoder):
    """
    Uses the point on the geodesic between the two children closest to the
    origin as the parent embedding. The parent embedding is thus a deterministic
    function of the children embeddings. Requires embeddings to have dimension
    2.
    """

    def __init__(self, *, D: int):
        """
        Args:
            D: Number of dimensions sequence embeddings (must be 2).
        """

        super().__init__()

        assert D == 2

    def forward(self, children1_VxD: Tensor, children2_VxD: Tensor) -> Tensor:
        # all values are vectors (shape=[V, 2]) unless otherwise stated

        p = children1_VxD
        q = children2_VxD

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
        ok = p_dot_nhat != 0
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

        return m
