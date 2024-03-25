import torch
from torch import Tensor, nn

from distances import Distance, safe_norm
from encoder_utils import MLP
from q_matrix_decoders import QMatrixDecoder
from vcsmc_utils import compute_log_felsenstein_likelihoods_KxSxA


class SequenceEncoder(nn.Module):
    """Encodes sequences into embeddings."""

    def __init__(self, *, D: int):
        super().__init__()
        self.D = D

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
        super().__init__(D=0)

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

        super().__init__(D=D)

        self.distance = distance
        self.mlp = MLP(S * A, D, width, depth)

    def forward(self, sequences_VxSxA: Tensor) -> Tensor:
        return self.distance.normalize(self.mlp(sequences_VxSxA))


class MergeEncoder(nn.Module):
    """Encodes a pair of child embeddings into a parent embedding."""

    def forward(
        self,
        children1_VxD: Tensor,
        children2_VxD: Tensor,
        log_felsensteins1_VxSxA: Tensor,
        log_felsensteins2_VxSxA: Tensor,
        site_positions_SxC: Tensor,
    ) -> Tensor:
        """
        Args:
            children1_VxD: First child embeddings.
            children2_VxD: Second child embeddings.
            log_felsensteins1_VxSxA: Log Felsenstein likelihoods for the first children.
            log_felsensteins2_VxSxA: Log Felsenstein likelihoods for the second children.
            site_positions_SxC: Compressed site positions.
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

    def forward(
        self,
        children1_VxD: Tensor,
        children2_VxD: Tensor,
        log_felsensteins1_VxSxA: Tensor,
        log_felsensteins2_VxSxA: Tensor,
        site_positions_SxC: Tensor,
    ) -> Tensor:
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

    def forward(
        self,
        children1_VxD: Tensor,
        children2_VxD: Tensor,
        log_felsensteins1_VxSxA: Tensor,
        log_felsensteins2_VxSxA: Tensor,
        site_positions_SxC: Tensor,
    ) -> Tensor:
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


class MaxLikelihoodMergeEncoder(MergeEncoder):
    """
    Finds the parent embedding that maximizes the likelihood as computed by the
    Felsenstein algorithm. Performs gradient ascent to find the maximum. Search
    region is bounded by the triangle formed by the children embeddings and the
    origin.
    """

    def __init__(
        self,
        distance: Distance,
        q_matrix_decoder: QMatrixDecoder,
        *,
        iters: int,
        lr: float,
        clip: float,
    ):
        """
        Args:
            distance: Used to compute distances for branch lengths.
            q_matrix_decoder: Used to decode Q matrices of child embeddings.
            iters: Number of iterations of gradient ascent.
            lr: Learning rate for gradient ascent.
            clip: Clipping value for gradient ascent.
        """

        super().__init__()

        self.distance = distance
        self.q_matrix_decoder = q_matrix_decoder
        self.iters = iters
        self.lr = lr
        self.clip = clip

    def forward(
        self,
        children1_VxD: Tensor,
        children2_VxD: Tensor,
        log_felsensteins1_VxSxA: Tensor,
        log_felsensteins2_VxSxA: Tensor,
        site_positions_SxC: Tensor,
    ) -> Tensor:
        # stop gradients from flowing back to inputs
        children1_VxD = children1_VxD.detach()
        children2_VxD = children2_VxD.detach()
        log_felsensteins1_VxSxA = log_felsensteins1_VxSxA.detach()
        log_felsensteins2_VxSxA = log_felsensteins2_VxSxA.detach()
        site_positions_SxC = site_positions_SxC.detach()

        p = children1_VxD
        q = children2_VxD

        p_norm = safe_norm(p, 1, True)
        q_norm = safe_norm(q, 1, True)

        # "pull" the further child to the same distance from the origin as the
        # closer child
        p = torch.where(p_norm > q_norm, p * q_norm / p_norm, p)
        q = torch.where(q_norm > p_norm, q * p_norm / q_norm, q)

        # re-enable gradients, in case we are in a torch.no_grad() context
        with torch.enable_grad():
            V = children1_VxD.shape[0]
            dtype = children1_VxD.dtype
            device = children1_VxD.device

            # relative position between the first and second child; to be optimized
            alpha_V = torch.full(
                [V], 0.5, dtype=dtype, device=device, requires_grad=True
            )
            # relative distance from the origin; to be optimized
            beta_V = torch.full(
                [V], 0.5, dtype=dtype, device=device, requires_grad=True
            )

            alpha_Vx1 = alpha_V.unsqueeze(1)
            beta_Vx1 = beta_V.unsqueeze(1)

            for _ in range(self.iters):
                midpoints_VxD = p * alpha_Vx1 + q * (1 - alpha_Vx1)
                parents_VxD = midpoints_VxD * beta_Vx1

                Q_matrix_VxSxAxA = self.q_matrix_decoder.Q_matrix_VxSxAxA(
                    parents_VxD, site_positions_SxC
                ).detach()

                stat_probs_VxSxA = self.q_matrix_decoder.stat_probs_VxSxA(
                    parents_VxD, site_positions_SxC
                ).detach()
                log_stat_probs_VxSxA = stat_probs_VxSxA.log()

                branch1_V = self.distance(children1_VxD, parents_VxD)
                branch2_V = self.distance(children2_VxD, parents_VxD)

                log_felsensteins_VxSxA = compute_log_felsenstein_likelihoods_KxSxA(
                    Q_matrix_VxSxAxA,
                    log_felsensteins1_VxSxA,
                    log_felsensteins2_VxSxA,
                    branch1_V,
                    branch2_V,
                )

                # dot Felsenstein probabilities with stationary probabilities (along axis A)
                log_likelihoods_VxS = torch.logsumexp(
                    log_felsensteins_VxSxA + log_stat_probs_VxSxA, -1
                )
                log_likelihood_V = log_likelihoods_VxS.sum(-1)

                # maximize likelihood via gradient ascent
                log_likelihood_V.backward(torch.ones_like(log_likelihood_V))
                alpha_V.data += self.lr * torch.clamp(alpha_V.grad, -self.clip, self.clip)  # type: ignore
                beta_V.data += self.lr * torch.clamp(beta_V.grad, -self.clip, self.clip)  # type: ignore
                alpha_V.grad.zero_()  # type: ignore
                beta_V.grad.zero_()  # type: ignore

                # clamp alpha and beta to limit search to triangular region
                alpha_V.data.clamp_(0, 1)
                beta_V.data.clamp_(0, 1)

        return parents_VxD.detach()  # type: ignore
