import torch
from torch import Tensor, nn

from distances import Distance, Hyperbolic, safe_norm
from encoder_utils import MLP
from q_matrix_decoders import QMatrixDecoder
from vcsmc_utils import compute_log_felsenstein_likelihoods_KxSxA


class SequenceEncoder(nn.Module):
    """Encodes sequences into embeddings."""

    def __init__(self, distance: Distance | None, *, D: int):
        """
        Args:
            distance: Used to normalize the embeddings.
            D: Number of dimensions in sequence embeddings.
        """

        super().__init__()

        self.distance = distance
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
        super().__init__(None, D=0)

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

        super().__init__(distance, D=D)
        self.mlp = MLP(S * A, D, width, depth)

    def forward(self, sequences_VxSxA: Tensor) -> Tensor:
        return self.mlp(sequences_VxSxA)


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

    def forward(
        self,
        children1_VxD: Tensor,
        children2_VxD: Tensor,
        log_felsensteins1_VxSxA: Tensor,
        log_felsensteins2_VxSxA: Tensor,
        site_positions_SxC: Tensor,
    ) -> Tensor:
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


class HyperbolicGeodesicMergeEncoder(MergeEncoder):
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

    def forward(
        self,
        children1_VxD: Tensor,
        children2_VxD: Tensor,
        log_felsensteins1_VxSxA: Tensor,
        log_felsensteins2_VxSxA: Tensor,
        site_positions_SxC: Tensor,
    ) -> Tensor:
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

        return self.distance.unnormalize(m)


class HyperbolicGeodesicMLPMergeEncoder(HyperbolicGeodesicMergeEncoder):
    """
    Like the HyperbolicGeodesicMergeEncoder, but after computing the midpoint,
    applies an MLP on the child embeddings to determine parent as a point along
    the line between the midpoint and origin.
    """

    def __init__(self, distance: Distance, *, D: int, width: int, depth: int):
        """
        Args:
            D: Number of dimensions sequence embeddings (must be 2).
        """

        super().__init__(distance, D=D)

        D1 = distance.feature_expand_shape(D)
        self.mlp = MLP(2 * D1, 1, width, depth)

    def forward(
        self,
        children1_VxD: Tensor,
        children2_VxD: Tensor,
        log_felsensteins1_VxSxA: Tensor,
        log_felsensteins2_VxSxA: Tensor,
        site_positions_SxC: Tensor,
    ) -> Tensor:
        midpoints_VxD = super().forward(
            children1_VxD,
            children2_VxD,
            log_felsensteins1_VxSxA,
            log_felsensteins2_VxSxA,
            site_positions_SxC,
        )

        expanded1_VxD1 = self.distance.feature_expand(children1_VxD)
        expanded2_VxD1 = self.distance.feature_expand(children2_VxD)

        # sigmoid to ensure beta is in [0, 1]
        beta_Vx1 = self.mlp(torch.cat([expanded1_VxD1, expanded2_VxD1], -1)).sigmoid()

        parents_VxD = midpoints_VxD * beta_Vx1
        return parents_VxD


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
        alpha_lr: float,
        beta_lr: float,
        sample: bool = False,
    ):
        """
        Args:
            distance: Used to compute distances for branch lengths.
            q_matrix_decoder: Used to decode Q matrices of child embeddings.
            iters: Number of iterations of gradient ascent.
            alpha_lr: Learning rate for alpha, the relative position between the children.
            beta_lr: Learning rate for beta, the relative distance from the origin.
            clip: Clipping value for gradient ascent.
            sample: Whether to sample a point near the maximum likelihood point.
                If False, the maximum likelihood point is returned.
        """

        super().__init__()

        self.distance = distance
        self.q_matrix_decoder = q_matrix_decoder
        self.iters = iters
        self.alpha_lr = alpha_lr
        self.beta_lr = beta_lr
        self.sample = sample

    def forward(
        self,
        children1_VxD: Tensor,
        children2_VxD: Tensor,
        log_felsensteins1_VxSxA: Tensor,
        log_felsensteins2_VxSxA: Tensor,
        site_positions_SxC: Tensor,
    ) -> Tensor:
        # stop gradients from flowing back to inputs
        log_felsensteins1_VxSxA = log_felsensteins1_VxSxA.detach()
        log_felsensteins2_VxSxA = log_felsensteins2_VxSxA.detach()
        site_positions_SxC = site_positions_SxC.detach()
        # ...the children embeddings are detached explicitly in the loop

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

            # relative position between the first and second child; to be optimized
            alpha_V = torch.full([V], 0.5, dtype=dtype, requires_grad=True)
            # relative distance from the origin; to be optimized
            beta_V = torch.full([V], 0.5, dtype=dtype, requires_grad=True)

            alpha_Vx1 = alpha_V.unsqueeze(1)
            beta_Vx1 = beta_V.unsqueeze(1)

            def compute_log_likelihood_V():
                midpoints_VxD = p.detach() * alpha_Vx1 + q.detach() * (1 - alpha_Vx1)
                parents_VxD = midpoints_VxD * beta_Vx1

                Q_matrix_VxSxAxA = self.q_matrix_decoder.Q_matrix_VxSxAxA(
                    parents_VxD, site_positions_SxC
                ).detach()

                stat_probs_VxSxA = self.q_matrix_decoder.stat_probs_VxSxA(
                    parents_VxD, site_positions_SxC
                ).detach()
                log_stat_probs_VxSxA = stat_probs_VxSxA.log()

                branch1_V = self.distance(children1_VxD.detach(), parents_VxD)
                branch2_V = self.distance(children2_VxD.detach(), parents_VxD)

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
                return log_likelihood_V

            for _ in range(self.iters):
                log_likelihood_V = compute_log_likelihood_V()

                # maximize likelihood via gradient ascent
                log_likelihood_V.backward(torch.ones_like(log_likelihood_V))
                alpha_V.data += self.alpha_lr * alpha_V.grad  # type: ignore
                beta_V.data += self.beta_lr * beta_V.grad  # type: ignore
                alpha_V.grad.data.zero_()  # type: ignore
                beta_V.grad.data.zero_()  # type: ignore

                # clamp alpha and beta to limit search to triangular region
                alpha_V.data.clamp_(0, 1)
                beta_V.data.clamp_(0, 1)

            if self.sample:
                # sample alpha' and beta' from independent normal distributions
                # centered at the maximum likelihood alpha and beta; the
                # distribution variances are such that the second derivative of
                # the distributions match d^2(likelihood)/d(alpha^2) and
                # d^2(likelihood)/d(beta^2) at the maximum likelihood point

                log_likelihood_V = compute_log_likelihood_V()

                # normalize likelihoods distribution such that f(x*) = 1, where
                # f is the likelihood distribution over alpha/beta, and x* is
                # the maximum likelihood alpha/beta
                log_likelihood_V = log_likelihood_V - log_likelihood_V.detach()
                likelihood_V = log_likelihood_V.exp()

                # first derivative
                likelihood_V.backward(  # must create graph for second derivative
                    torch.ones_like(likelihood_V), create_graph=True
                )
                partial1_alpha_V: Tensor = alpha_V.grad  # type: ignore
                partial1_beta_V: Tensor = beta_V.grad  # type: ignore
                alpha_V.grad.data.zero_()  # type: ignore
                beta_V.grad.data.zero_()  # type: ignore

                # second derivative of alpha
                partial1_alpha_V.backward(
                    torch.ones_like(partial1_alpha_V),
                    inputs=[alpha_V],
                    retain_graph=True,
                )
                partial2_alpha_V: Tensor = alpha_V.grad  # type: ignore

                # second derivative of beta
                partial1_beta_V.backward(
                    torch.ones_like(partial1_beta_V),
                    inputs=[beta_V],
                    retain_graph=False,
                )
                partial2_beta_V: Tensor = beta_V.grad  # type: ignore

                # for normal distribution, variance = -f(x*)/f''(x*) = -1/f''(x*),
                # as f(x*) = 1 by normalization
                alpha_var_V = -1 / partial2_alpha_V
                beta_var_V = -1 / partial2_beta_V

                # prevent degenerate variances
                alpha_var_V = torch.max(alpha_var_V, torch.tensor(1e-8))
                beta_var_V = torch.max(beta_var_V, torch.tensor(1e-8))

                # sample alpha' and beta' from normal distributions
                alpha_dist = torch.distributions.Normal(alpha_V, alpha_var_V.sqrt())
                beta_dist = torch.distributions.Normal(beta_V, beta_var_V.sqrt())
                alpha_sample_V = alpha_dist.sample()
                beta_sample_V = beta_dist.sample()

                # clamp sampled values to triangular region
                alpha_sample_V.clamp_(0, 1)
                beta_sample_V.clamp_(0, 1)

                # use sampled values in place of maximum likelihood values
                alpha_Vx1 = alpha_sample_V.unsqueeze(1)
                beta_Vx1 = beta_sample_V.unsqueeze(1)

        # detach `alpha_Vx1` and `beta_Vx1` now that we've finished optimizing
        # them; but, use `p` and `q` directly so that gradients of the returned
        # `parents_VxD` can flow back to `children1_VxD` and `children2_VxD`

        alpha_Vx1 = alpha_Vx1.detach()
        beta_Vx1 = beta_Vx1.detach()

        midpoints_VxD = p * alpha_Vx1 + q * (1 - alpha_Vx1)
        parents_VxD = midpoints_VxD * beta_Vx1
        return parents_VxD
