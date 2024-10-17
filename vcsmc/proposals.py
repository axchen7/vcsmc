import math
from typing import Literal

import torch
from torch import Tensor, nn

from .distances import Distance
from .encoders import DummySequenceEncoder, MergeEncoder, SequenceEncoder
from .manifolds import PoincareBall
from .utils.distance_utils import EPSILON
from .utils.repr_utils import custom_module_repr
from .utils.vcsmc_utils import ArangeFn, gather_K, gather_K2, hash_forest_K

__all__ = ["Proposal", "ExpBranchProposal", "EmbeddingProposal"]


class Proposal(nn.Module):
    """
    Proposal distribution for selecting two nodes to merge and sampling branch lengths.
    """

    def __init__(self, seq_encoder: SequenceEncoder, *, max_sub_particles: int):
        super().__init__()
        self.register_buffer("true", torch.ones(1, dtype=torch.bool))

        self.seq_encoder = seq_encoder
        self.max_sub_particles = max_sub_particles

    def get_lookahead_merge_indexes(self, *, K, t: int) -> tuple[int, Tensor, Tensor]:
        # take all possible (t choose 2) merge pairs
        J = t * (t - 1) // 2

        take_J = self.true.expand(t, t).triu(1).flatten().nonzero().flatten()
        idx1_J = take_J // t
        idx2_J = take_J % t

        idx1_KxJ = idx1_J.repeat(K, 1)
        idx2_KxJ = idx2_J.repeat(K, 1)

        return J, idx1_KxJ, idx2_KxJ

    def uses_deterministic_branches(self) -> bool:
        """
        Returns true if the proposal emits the same branch lengths when merging
        the same nodes.
        """
        raise NotImplementedError

    def forward(
        self, N: int, embeddings_KxtxD: Tensor, hashes_Kxt: Tensor, arange_fn: ArangeFn
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Propose J different particles, each defined by the two nodes being
        merged and their branch lengths.

        Args:
            N: The number of leaf nodes.
            embeddings_KtxD: Embeddings of each subtree of each particle.
            arange_fn: torch.arange or equivalent.
        Returns:
            idx1_KxJ: Indices of the first node to merge.
            idx2_KxJ: Indices of the second node to merge.
            branch1_KxJ: Branch lengths of the first node.
            branch2_KxJ: Branch lengths of the second node.
            embedding_KxJxD: Embeddings of the merged subtree.
            log_v_plus_KxJ: Log probabilities of the returned proposal.
        Note:
            At each step r, there are t = N-r >= 2 trees in the forest.
        """

        raise NotImplementedError


class ExpBranchProposal(Proposal):
    """
    Proposal where branch lengths are sampled from exponential distributions,
    with a learnable parameter for each merge step. Merge pairs are sampled
    uniformly. Returns dummy embeddings. Only works for a fixed number of leaf
    nodes because there is exactly one parameter for each merge step.
    """

    def __init__(
        self,
        *,
        N: int,
        initial_branch_len: float = 1.0,
        lookahead_merge: bool = False,
    ):
        """
        Args:
            N: The number of leaf nodes.
            initial_branch_len: The initial expected value of the branch lengths.
                The exponential distribution from which branch lengths are
                sampled will initially have lambda = 1/initial_branch_len.
            lookahead_merge: if True, will return a particle for each of the J=(t choose 2) possible merges.
                An independent pair of branch lengths will be sampled for each particle.
        """

        super().__init__(
            DummySequenceEncoder(),
            # N choose 2
            max_sub_particles=N * (N - 1) // 2 if lookahead_merge else 1,
        )
        self.register_buffer("zero", torch.zeros(1))

        # under exponential distribution, E[branch] = 1/rate
        initial_rate = 1 / initial_branch_len
        # value of variable is passed through exp() later
        initial_log_rates_N1 = torch.full([N - 1], math.log(initial_rate))

        self.initial_branch_len = initial_branch_len
        self.lookahead_merge = lookahead_merge

        # exponential distribution rates for sampling branch lengths; N1 -> N-1
        self.log_rates1_N1 = nn.Parameter(initial_log_rates_N1)
        self.log_rates2_N1 = nn.Parameter(initial_log_rates_N1)

    def extra_repr(self) -> str:
        return custom_module_repr(
            {
                "initial_branch_len": self.initial_branch_len,
                "lookahead_merge": self.lookahead_merge,
            }
        )

    def rates(self, r: int):
        # use exp to ensure rates are positive
        rate1 = self.log_rates1_N1[r].exp()
        rate2 = self.log_rates2_N1[r].exp()
        return rate1, rate2

    def uses_deterministic_branches(self) -> bool:
        return False

    def forward(
        self, N: int, embeddings_KxtxD: Tensor, hashes_Kxt: Tensor, arange_fn: ArangeFn
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        device = embeddings_KxtxD.device

        K = embeddings_KxtxD.shape[0]
        t = embeddings_KxtxD.shape[1]  # number of subtrees
        r = N - t  # merge step

        # ===== determine nodes to merge =====

        if self.lookahead_merge:
            J, idx1_KxJ, idx2_KxJ = self.get_lookahead_merge_indexes(K=K, t=t)
            log_merge_prob = 0
        else:
            # uniformly sample 2 distinct nodes to merge
            J = 1

            idx1_KxJ = torch.randint(0, t, [K], device=device).unsqueeze(1)
            idx2_KxJ = torch.randint(0, t - 1, [K], device=device).unsqueeze(1)

            # shift to guarantee idx2 > idx1
            idx2_KxJ = torch.where(idx2_KxJ >= idx1_KxJ, idx2_KxJ + 1, idx2_KxJ)

            # merge prob = 1 / (t choose 2)
            log_merge_prob = -math.log(t * (t - 1) // 2)

        # ===== sample branch lengths from exponential distributions =====

        rate1, rate2 = self.rates(r)

        # re-parameterization trick: sample from U[0, 1] and transform to
        # exponential distribution (so gradients can flow through the sample)

        uniform1_KxJ = torch.rand([K, J], device=device)
        uniform2_KxJ = torch.rand([K, J], device=device)

        # branch1 ~ Exp(rate1) and branch2 ~ Exp(rate2)
        branch1_KxJ = -(1 / rate1) * uniform1_KxJ.log()
        branch2_KxJ = -(1 / rate2) * uniform2_KxJ.log()

        # log of exponential pdf
        log_branch1_prior_KxJ = rate1.log() - rate1 * branch1_KxJ
        log_branch2_prior_KxJ = rate2.log() - rate2 * branch2_KxJ

        # ===== compute proposal probability =====

        log_v_plus_KxJ = log_merge_prob + log_branch1_prior_KxJ + log_branch2_prior_KxJ

        # ===== return proposal =====

        # dummy embedding
        embedding_KxJxD = self.zero.expand(K, J, 0)

        return (
            idx1_KxJ,
            idx2_KxJ,
            branch1_KxJ,
            branch2_KxJ,
            embedding_KxJxD,
            log_v_plus_KxJ,
        )


EmbeddingProposalBranchesDistribution = Literal["wrapped_normal", "log_normal"]


class EmbeddingProposal(Proposal):
    """
    Proposal where leaf nodes are embedded into D-dimensional space, and pairs
    of child embeddings are re-embedded to produce merged embeddings. Embeddings
    are performed using a multi-layered perceptron. Branch lengths are
    optionally sampled from exponential distributions parameterized by distance
    between embeddings. Merge pairs are sampled using distances.
    """

    def __init__(
        self,
        distance: Distance,
        seq_encoder: SequenceEncoder,
        merge_encoder: MergeEncoder,
        *,
        N: int,
        lookahead_merge: bool = False,
        sample_merge_temp: float | None = None,
        sample_branches: bool = False,
        initial_sample_branches_sigma: float = 0.1,
        branches_distribution: EmbeddingProposalBranchesDistribution = "log_normal",
        static_merge_log_weights: dict[int, Tensor] | None = None,
    ):
        """
        Only one of `lookahead_merge`, `sample_merge_temp`, and `merge_indexes_N1x2`
        should be set.

        Args:
            distance: The distance function to use for embedding.
            seq_encoder: Sequence encoder.
            merge_encoder: Algorithm that gives the parent embedding given the children embeddings.
            N: Maximum number of leaf nodes.
            lookahead_merge: if True, will return a particle for each of the J=(t choose 2) possible merges.
            sample_merge_temp: Temperature to use for sampling a pair of nodes to merge.
                Negative pairwise node distances divided by `sample_merge_temp` are used log weights.
                Set to a large value to effectively sample nodes uniformly. If None, then a
                pair of nodes will be sampled uniformly. Only used if `lookahead_merge`is false.
            sample_branches: Whether to sample branch lengths or directly use the distance between embeddings.
            initial_sample_branches_sigma: initial standard deviation of the branch length distribution.
            branches_distribution: Distribution to use for sampling branch lengths.
                wrapped_normal: Sample the parent embedding from a wrapped normal distribution centered
                    around the point returned by merge_encoder. Applies a Jacobian to compute the probability
                    over branch lengths.
                log_normal: Sample branch lengths from a log-normal distribution whose mean is the distance
                    between the children and the merged embedding.
            static_merge_log_weights: If not None, sets the fixed merge distribution.
                See compute_merge_log_weights_from_vcsmc(). Tensors should be on the CPU.
        """

        super().__init__(
            seq_encoder,
            # N choose 2
            max_sub_particles=N * (N - 1) // 2 if lookahead_merge else 1,
        )
        self.register_buffer("zero", torch.zeros(1))
        self.register_buffer("inf", torch.tensor(torch.inf))

        self.distance = distance
        self.merge_encoder = merge_encoder
        self.lookahead_merge = lookahead_merge
        self.sample_merge_temp = sample_merge_temp
        self.sample_branches = sample_branches
        self.branches_distribution: EmbeddingProposalBranchesDistribution = (
            branches_distribution
        )
        self.initial_sample_branches_sigma = initial_sample_branches_sigma
        self.log_sample_branches_sigma = nn.Parameter(
            torch.tensor(math.log(initial_sample_branches_sigma))
        )
        self.static_merge_log_weights = static_merge_log_weights  # on CPU

    def extra_repr(self) -> str:
        return custom_module_repr(
            {
                "lookahead_merge": self.lookahead_merge,
                "sample_merge_temp": self.sample_merge_temp,
                "sample_branches": self.sample_branches,
                "initial_sample_branches_sigma": self.initial_sample_branches_sigma,
                "static_merge_log_weights": (
                    "provided" if self.static_merge_log_weights else None
                ),
            }
        )

    def sample_branches_sigma(self):
        return self.log_sample_branches_sigma.exp()

    def uses_deterministic_branches(self) -> bool:
        return not self.sample_branches

    def forward(
        self, N: int, embeddings_KxtxD: Tensor, hashes_Kxt: Tensor, arange_fn: ArangeFn
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        device = embeddings_KxtxD.device

        K = embeddings_KxtxD.shape[0]
        t = embeddings_KxtxD.shape[1]  # number of subtrees
        D = embeddings_KxtxD.shape[2]

        # ===== determine nodes to merge =====

        if self.lookahead_merge:
            J, idx1_KxJ, idx2_KxJ = self.get_lookahead_merge_indexes(K=K, t=t)
            log_merge_prob_K = self.zero.expand(K)
        else:
            J = 1

            if self.static_merge_log_weights is not None:
                forest_hashes_K = hash_forest_K(hashes_Kxt)
                forest_hashes_K = forest_hashes_K.cpu()  # copy to CPU all at once
                merge_log_weights_Kxtxt = torch.stack(
                    [self.static_merge_log_weights[int(h)] for h in forest_hashes_K]
                )
                merge_log_weights_Kxtxt = merge_log_weights_Kxtxt.to(device)
            elif self.sample_merge_temp is not None:
                # randomly select two subtrees to merge, using pairwise distances as
                # negative log probabilities, and incorporating the sample
                # temperature
                Ktt = K * t * t  # for brevity
                # repeat like 123123123...
                flat_embeddings1_KttxD = embeddings_KxtxD.repeat(1, t, 1).view(Ktt, -1)
                # repeat like 111222333...
                flat_embeddings2_KttxD = embeddings_KxtxD.repeat(1, 1, t).view(Ktt, -1)

                pairwise_distances_Ktt: Tensor = self.distance(
                    flat_embeddings1_KttxD, flat_embeddings2_KttxD
                )
                pairwise_distances_Kxtxt = pairwise_distances_Ktt.view(K, t, t)
                merge_log_weights_Kxtxt = (
                    -pairwise_distances_Kxtxt / self.sample_merge_temp
                )
            else:
                # uniformly sample 2 distinct nodes to merge
                merge_log_weights_Kxtxt = self.zero.expand(K, t, t)

            # set diagonal entries to -inf to prevent self-merges
            merge_log_weights_Kxtxt = merge_log_weights_Kxtxt.diagonal_scatter(
                -self.inf.expand(K, t), dim1=1, dim2=2
            )

            flattened_log_weights_Kxtt = merge_log_weights_Kxtxt.view(K, t * t)

            merge_distr = torch.distributions.Categorical(
                logits=flattened_log_weights_Kxtt
            )
            flattened_sample_K = merge_distr.sample()

            idx1_K = flattened_sample_K // t
            idx2_K = flattened_sample_K % t

            # merge prob = merge weight * 2 / sum of all weights

            # the factor of 2 is because merging (idx1, idx2) is equivalent to
            # merging (idx2, idx1)

            log_merge_prob_K = gather_K2(
                merge_log_weights_Kxtxt, idx1_K, idx2_K, arange_fn
            )
            log_merge_prob_K = log_merge_prob_K + math.log(2)
            log_merge_prob_K = log_merge_prob_K - torch.logsumexp(
                merge_log_weights_Kxtxt, [1, 2]
            )

            if self.static_merge_log_weights is not None:
                # in this case, idx1_K and idx2_K are w.r.t. the list of trees sorted by hash,
                # so we must map to the original indexes

                # sort_idx_Kxt[k, i] is the index of the i-th sorted tree of
                # particle k as found in the original list of trees
                sort_idx_Kxt = torch.argsort(hashes_Kxt, dim=1)
                idx1_K = gather_K(sort_idx_Kxt, idx1_K, arange_fn)
                idx2_K = gather_K(sort_idx_Kxt, idx2_K, arange_fn)

            idx1_KxJ = idx1_K.unsqueeze(1)
            idx2_KxJ = idx2_K.unsqueeze(1)

        # ===== get merged embedding =====

        idx1_KJ = idx1_KxJ.flatten()
        idx2_KJ = idx2_KxJ.flatten()
        embeddings_KJxtxD = embeddings_KxtxD.repeat_interleave(J, 0)

        child1_KJxD = gather_K(embeddings_KJxtxD, idx1_KJ, arange_fn)
        child2_KJxD = gather_K(embeddings_KJxtxD, idx2_KJ, arange_fn)

        embedding_KJxD = self.merge_encoder(child1_KJxD, child2_KJxD)

        # ===== sample/get branches parameters =====

        if self.sample_branches and self.branches_distribution == "wrapped_normal":
            sigma = self.sample_branches_sigma()
            zero_D = self.zero.expand(D)
            cov_DxD = torch.eye(D, device=device) * sigma**2
            distr = torch.distributions.MultivariateNormal(zero_D, cov_DxD)
            samples_KJxD = distr.sample(torch.Size([K * J]))
            sample_logprobs_KJ = distr.log_prob(samples_KJxD)

            # parallel transport and exponential map the samples to the manifold,
            # with embedding_KJxD as the mean

            def samples_to_branches_and_embedding(
                samples_D: Tensor,
                embedding_D: Tensor,
                child1_D: Tensor,
                child2_D: Tensor,
            ):
                manifold = PoincareBall()

                # use transported samples as the new embeddings
                embedding_normalized_D = self.distance.normalize(
                    embedding_D.unsqueeze(0)
                ).squeeze(0)
                transported_D = manifold.transp0(embedding_normalized_D, samples_D)
                # expmap only works on CPU
                embedding_normalized_D = manifold.expmap(
                    embedding_normalized_D.to("cpu"), transported_D.to("cpu")
                ).to(device)
                embedding_D = self.distance.unnormalize(
                    embedding_normalized_D.unsqueeze(0)
                ).squeeze(0)

                branch1: Tensor = self.distance(child1_D, embedding_D)
                branch2: Tensor = self.distance(child2_D, embedding_D)

                return branch1, branch2, embedding_D

            # for computing jacobian
            def transform_samples_to_branches(
                samples_D: Tensor,
                embedding_D: Tensor,
                child1_D: Tensor,
                child2_D: Tensor,
            ):
                branch1, branch2, _ = samples_to_branches_and_embedding(
                    samples_D, embedding_D, child1_D, child2_D
                )
                branches_2 = torch.stack([branch1, branch2])
                return branches_2

            branch1_KJ, branch2_KJ, embedding_KJxD = torch.vmap(
                samples_to_branches_and_embedding
            )(samples_KJxD, embedding_KJxD, child1_KJxD, child2_KJxD)

            jacobian_KJxDx2 = torch.vmap(
                torch.func.jacfwd(transform_samples_to_branches, argnums=0)  # type: ignore
            )(samples_KJxD, embedding_KJxD, child1_KJxD, child2_KJxD)

            # requires D=2 !
            determinants_KJ = jacobian_KJxDx2.det().abs()
            log_deteterminants_KJ = determinants_KJ.log()

            # must divide by the jacobian determinant
            log_branches_prob_KJ = sample_logprobs_KJ - log_deteterminants_KJ

        elif self.sample_branches and self.branches_distribution == "log_normal":
            # branch lengths are the random variable Y = X^2, where
            # X ~ N(sqrt(distance), sigma), with `distance` being the distance
            # between the children and the merged embedding; Y will be centered
            # around `distance`

            sigma = self.sample_branches_sigma()

            # # re-parameterization trick: sample from N(0, 1) and transform to
            # squared normal distribution (so gradients can flow through the sample)

            normal1_KJ = torch.randn([K * J], device=device)
            normal2_KJ = torch.randn([K * J], device=device)

            dist1_KJ: Tensor = self.distance(child1_KJxD, embedding_KJxD)
            dist2_KJ: Tensor = self.distance(child2_KJxD, embedding_KJxD)

            dist1_sqrt_KJ = torch.sqrt(dist1_KJ + EPSILON)
            dist2_sqrt_KJ = torch.sqrt(dist2_KJ + EPSILON)

            # X ~ N(sqrt(dist), sigma)
            X1_KJ = dist1_sqrt_KJ + sigma * normal1_KJ
            X2_KJ = dist2_sqrt_KJ + sigma * normal2_KJ

            # Y = X^2
            branch1_KJ = X1_KJ**2
            branch2_KJ = X2_KJ**2

            # compute log of the pdf of Y

            X1_distr_KJ = torch.distributions.Normal(dist1_sqrt_KJ, sigma)
            X2_distr_KJ = torch.distributions.Normal(dist2_sqrt_KJ, sigma)

            # can show: pdf_Y(y) = (1/2|X|) * [pdf_X(X) + pdf_X(-X)]
            log_branch1_prob_KJ = (
                -math.log(2)
                - X1_KJ.abs().log()
                + torch.log(
                    X1_distr_KJ.log_prob(X1_KJ).exp()
                    + X1_distr_KJ.log_prob(-X1_KJ).exp()
                )
            )
            log_branch2_prob_KJ = (
                -math.log(2)
                - X2_KJ.abs().log()
                + torch.log(
                    X2_distr_KJ.log_prob(X2_KJ).exp()
                    + X2_distr_KJ.log_prob(-X2_KJ).exp()
                )
            )

            log_branches_prob_KJ = log_branch1_prob_KJ + log_branch2_prob_KJ

        else:
            branch1_KJ: Tensor = self.distance(child1_KJxD, embedding_KJxD)
            branch2_KJ: Tensor = self.distance(child2_KJxD, embedding_KJxD)
            log_branches_prob_KJ = self.zero.expand(K * J)

        branch1_KxJ = branch1_KJ.reshape(K, J)
        branch2_KxJ = branch2_KJ.reshape(K, J)
        embedding_KxJxD = embedding_KJxD.reshape(K, J, -1)

        # ===== compute proposal probability =====

        log_merge_prob_Kx1 = log_merge_prob_K.unsqueeze(1)
        log_branches_prob_KxJ = log_branches_prob_KJ.reshape(K, J)

        log_v_plus_KxJ = log_merge_prob_Kx1 + log_branches_prob_KxJ

        # ===== return proposal =====

        return (
            idx1_KxJ,
            idx2_KxJ,
            branch1_KxJ,
            branch2_KxJ,
            embedding_KxJxD,
            log_v_plus_KxJ,
        )
