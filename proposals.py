import torch
from torch import Tensor, nn

import distances
import encoders
from vcsmc_utils import compute_log_felsenstein_likelihoods_KxSxA, gather_K, gather_K2


class Proposal(nn.Module):
    """
    Proposal distribution for selecting two nodes to merge and sampling branch lengths.
    """

    def __init__(self, seq_encoder: encoders.SequenceEncoder):
        super().__init__()

        self.seq_encoder = seq_encoder

    def forward(
        self,
        N: int,
        leaf_counts_Kxt: Tensor,
        embeddings_KxtxD: Tensor,
        log_felsensteins_KxtxSxA: Tensor,
        site_positions_SxC: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Propose two nodes to merge, as well as their branch lengths.

        Args:
            N: The number of leaf nodes.
            leaf_counts_Kxt: The number of leaf nodes in each subtree of each particle.
            embeddings_KtxD: Embeddings of each subtree of each particle.
            log_felsensteins_KxtxSxA: Log Felsenstein likelihoods for each subtree of each particle.
            site_positions_SxC: Compressed site positions.
        Returns:
            idx1_K: Indices of the first node to merge.
            idx2_K: Indices of the second node to merge.
            branch1_K: Branch lengths of the first node.
            branch2_K: Branch lengths of the second node.
            embedding_KxD: Embeddings of the merged subtree.
            log_v_plus_K: Log probabilities of the returned proposal.
            log_v_minus_K: Log of the over-counting correction factors.
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
    ):
        """
        Args:
            N: The number of leaf nodes.
            initial_branch_len: The initial expected value of the branch lengths.
                The exponential distribution from which branch lengths are
                sampled will initially have lambda = 1/initial_branch_len.
        """

        super().__init__(encoders.DummySequenceEncoder())

        # under exponential distribution, E[branch] = 1/rate
        initial_rate = 1 / initial_branch_len
        # value of variable is passed through exp() later
        initial_log_rates = torch.tensor(initial_rate).log().repeat(N - 1)

        # exponential distribution rates for sampling branch lengths; N1 -> N-1
        self.log_rates1_N1 = nn.Parameter(initial_log_rates)
        self.log_rates2_N1 = nn.Parameter(initial_log_rates)

    def rates(self, r: int):
        # use exp to ensure rates are positive
        rate1 = self.log_rates1_N1[r].exp()
        rate2 = self.log_rates2_N1[r].exp()
        return rate1, rate2

    def forward(
        self,
        N: int,
        leaf_counts_Kxt: Tensor,
        embeddings_KxtxD: Tensor,
        log_felsensteins_KxtxSxA: Tensor,
        site_positions_SxC: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        K = leaf_counts_Kxt.shape[0]
        t = leaf_counts_Kxt.shape[1]  # number of subtrees
        r = N - t  # merge step

        # ===== uniformly sample 2 distinct nodes to merge =====

        idx1_K = torch.randint(0, t, [K])
        idx2_K = torch.randint(0, t - 1, [K])

        # shift to guarantee idx2 > idx1
        idx2_K = torch.where(idx2_K >= idx1_K, idx2_K + 1, idx2_K)

        # ===== sample branch lengths from exponential distributions =====

        rate1, rate2 = self.rates(r)

        # re-parameterization trick: sample from U[0, 1] and transform to
        # exponential distribution (so gradients can flow through the sample)

        uniform1_K = torch.rand([K])
        uniform2_K = torch.rand([K])

        # branch1 ~ Exp(rate1) and branch2 ~ Exp(rate2)
        branch1_K = -(1 / rate1) * uniform1_K.log()
        branch2_K = -(1 / rate2) * uniform2_K.log()

        # log of exponential pdf
        log_branch1_prior_K = rate1.log() - rate1 * branch1_K
        log_branch2_prior_K = rate2.log() - rate2 * branch2_K

        # ===== compute proposal probability =====

        # log(t choose 2)
        log_num_merge_choices = torch.log(torch.tensor(t * (t - 1) / 2))
        log_merge_prob = -log_num_merge_choices

        log_v_plus_K = log_merge_prob + log_branch1_prior_K + log_branch2_prior_K

        # ===== compute over-counting correction factor =====

        num_subtrees_with_one_leaf_K = torch.sum(leaf_counts_Kxt == 1, 1)

        # exclude trees currently being merged from the count
        num_subtrees_with_one_leaf_K -= (gather_K(leaf_counts_Kxt, idx1_K) == 1).int()
        num_subtrees_with_one_leaf_K -= (gather_K(leaf_counts_Kxt, idx2_K) == 1).int()

        v_minus_K = N - num_subtrees_with_one_leaf_K
        log_v_minus_K = v_minus_K.log()

        # ===== return proposal =====

        # dummy embedding
        embedding_KxD = torch.zeros([K, 0])

        return (
            idx1_K,
            idx2_K,
            branch1_K,
            branch2_K,
            embedding_KxD,
            log_v_plus_K,
            log_v_minus_K,
        )


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
        distance: distances.Distance,
        seq_encoder: encoders.SequenceEncoder,
        merge_encoder: encoders.MergeEncoder,
        q_matrix_decoder: encoders.QMatrixDecoder,
        *,
        lookahead_merge: bool = False,
        sample_merge_temp: float = 1.0,
        sample_branches: bool = False,
    ):
        """
        Args:
            distance: The distance function to use for embedding.
            seq_encoder: Sequence encoder.
            merge_encoder: Merge encoder.
            q_matrix_decoder: Q matrix decoder.
            lookahead_merge: Whether to use lookahead likelihoods for sampling merge pairs.
                If false, pairwise distances are used as merge weights.
            sample_merge_temp: Temperature to use for sampling a pair of nodes to merge.
                Negative pairwise node distances divided by `sample_temp` are used log weights.
                Set to a large value to effectively sample nodes uniformly. Only used if
                `lookahead_merge` is false.
            sample_branches: Whether to sample branch lengths from an exponential distribution.
                If false, simply use the distance between embeddings as the branch length.
        """

        super().__init__(seq_encoder)

        self.distance = distance
        self.merge_encoder = merge_encoder
        self.q_matrix_decoder = q_matrix_decoder
        self.lookahead_merge = lookahead_merge
        self.sample_merge_temp = sample_merge_temp
        self.sample_branches = sample_branches

    def forward(
        self,
        N: int,
        leaf_counts_Kxt: Tensor,
        embeddings_KxtxD: Tensor,
        log_felsensteins_KxtxSxA: Tensor,
        site_positions_SxC: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        K = leaf_counts_Kxt.shape[0]
        t = leaf_counts_Kxt.shape[1]  # number of subtrees
        S = site_positions_SxC.shape[0]  # number of sites

        # ===== sample 2 distinct nodes to merge =====

        # randomly select two subtrees to merge, using pairwise distances as
        # negative log probabilities, and incorporating the sample
        # temperature

        Ktt = K * t * t  # for brevity
        # repeat like 123123123...
        flat_embeddings1_KttxD = embeddings_KxtxD.repeat(1, t, 1).view(Ktt, -1)
        # repeat like 111222333...
        flat_embeddings2_KttxD = embeddings_KxtxD.repeat(1, 1, t).view(Ktt, -1)

        if self.lookahead_merge:
            # ===== compute lookahead likelihoods for merge weights =====

            # repeat like 123123123...
            flat_log_felsensteins1_KttxSxA = log_felsensteins_KxtxSxA.repeat(
                1, t, 1, 1
            ).view(Ktt, S, -1)
            # repeat like 111222333...
            flat_log_felsensteins2_KttxSxA = log_felsensteins_KxtxSxA.repeat(
                1, 1, t, 1
            ).view(Ktt, S, -1)

            lookahead_parents_KttxD = self.merge_encoder(
                flat_embeddings1_KttxD,
                flat_embeddings2_KttxD,
                flat_log_felsensteins1_KttxSxA,
                flat_log_felsensteins2_KttxSxA,
                site_positions_SxC,
            )

            lookahead_branches1_Ktt = self.distance(
                flat_embeddings1_KttxD, lookahead_parents_KttxD
            )
            lookahead_branches2_Ktt = self.distance(
                flat_embeddings2_KttxD, lookahead_parents_KttxD
            )

            lookahead_Q_matrices_KttxSxAxA = self.q_matrix_decoder.Q_matrix_VxSxAxA(
                lookahead_parents_KttxD, site_positions_SxC
            )
            lookahead_stat_probs_KttxSxA = self.q_matrix_decoder.stat_probs_VxSxA(
                lookahead_parents_KttxD, site_positions_SxC
            )
            lookahead_log_stat_probs_KttxSxA = lookahead_stat_probs_KttxSxA.log()

            lookahead_log_felsensteins_KttxSxA = (
                compute_log_felsenstein_likelihoods_KxSxA(
                    lookahead_Q_matrices_KttxSxAxA,
                    flat_log_felsensteins1_KttxSxA,
                    flat_log_felsensteins2_KttxSxA,
                    lookahead_branches1_Ktt,
                    lookahead_branches2_Ktt,
                )
            )

            # dot Felsenstein probabilities with stationary probabilities (along axis A)
            lookahead_log_likelihoods_KttxS = torch.logsumexp(
                lookahead_log_felsensteins_KttxSxA + lookahead_log_stat_probs_KttxSxA,
                -1,
            )
            lookahead_log_likelihoods_Ktt = lookahead_log_likelihoods_KttxS.sum(-1)

            merge_log_weights_Kxtxt = lookahead_log_likelihoods_Ktt.view(K, t, t)
        else:
            # ===== compute pairwise distances for merge weights =====

            pairwise_distances_Ktt: Tensor = self.distance(
                flat_embeddings1_KttxD, flat_embeddings2_KttxD
            )
            pairwise_distances_Kxtxt = pairwise_distances_Ktt.view(K, t, t)
            merge_log_weights_Kxtxt = -pairwise_distances_Kxtxt / self.sample_merge_temp

        # set diagonal entries to -inf to prevent self-merges
        merge_log_weights_Kxtxt = merge_log_weights_Kxtxt.diagonal_scatter(
            torch.full([K, t], -torch.inf), dim1=1, dim2=2
        )

        flattened_log_weights_Kxtt = merge_log_weights_Kxtxt.view(K, t * t)

        merge_distr = torch.distributions.Categorical(logits=flattened_log_weights_Kxtt)
        flattened_sample_K = merge_distr.sample()

        idx1_K = flattened_sample_K // t
        idx2_K = flattened_sample_K % t

        # merge prob = merge weight * 2 / sum of all weights

        # the factor of 2 is because merging (idx1, idx2) is equivalent to
        # merging (idx2, idx1)

        log_merge_prob_K = gather_K2(merge_log_weights_Kxtxt, idx1_K, idx2_K)
        log_merge_prob_K = log_merge_prob_K + torch.log(torch.tensor(2))
        log_merge_prob_K = log_merge_prob_K - torch.logsumexp(
            merge_log_weights_Kxtxt, [1, 2]
        )

        # ===== get merged embedding =====

        child1_KxD = gather_K(embeddings_KxtxD, idx1_K)
        child2_KxD = gather_K(embeddings_KxtxD, idx2_K)

        log_felsensteins1_VxSxA = gather_K(log_felsensteins_KxtxSxA, idx1_K)
        log_felsensteins2_VxSxA = gather_K(log_felsensteins_KxtxSxA, idx2_K)

        embedding_KxD = self.merge_encoder(
            child1_KxD,
            child2_KxD,
            log_felsensteins1_VxSxA,
            log_felsensteins2_VxSxA,
            site_positions_SxC,
        )

        # ===== sample/get branches parameters =====

        dist1_K: Tensor = self.distance(child1_KxD, embedding_KxD)
        dist2_K: Tensor = self.distance(child2_KxD, embedding_KxD)

        if self.sample_branches:
            # sample branch lengths from exp distributions whose expectations
            # are the distances between children and merged embeddings

            # EPSILON**0.5 is needed to prevent division by zero under float32
            rate1_K = 1 / (dist1_K + distances.EPSILON**0.5)
            rate2_K = 1 / (dist2_K + distances.EPSILON**0.5)

            # re-parameterization trick: sample from U[0, 1] and transform to
            # exponential distribution (so gradients can flow through the sample)

            uniform1_K = torch.rand([K])
            uniform2_K = torch.rand([K])

            # branch1 ~ Exp(rate1) and branch2 ~ Exp(rate2)
            branch1_K = -(1 / rate1_K) * uniform1_K.log()
            branch2_K = -(1 / rate2_K) * uniform2_K.log()

            # log of exponential pdf
            log_branch1_prior_K = rate1_K.log() - rate1_K * branch1_K
            log_branch2_prior_K = rate2_K.log() - rate2_K * branch2_K
        else:
            branch1_K = dist1_K
            branch2_K = dist2_K

            log_branch1_prior_K = 0
            log_branch2_prior_K = 0

        # ===== compute proposal probability =====

        log_v_plus_K = log_merge_prob_K + log_branch1_prior_K + log_branch2_prior_K

        # ===== compute over-counting correction factor =====

        num_subtrees_with_one_leaf_K = torch.sum(leaf_counts_Kxt == 1, 1)

        # exclude trees currently being merged from the count
        num_subtrees_with_one_leaf_K -= (gather_K(leaf_counts_Kxt, idx1_K) == 1).int()
        num_subtrees_with_one_leaf_K -= (gather_K(leaf_counts_Kxt, idx2_K) == 1).int()

        v_minus_K = N - num_subtrees_with_one_leaf_K
        log_v_minus_K = v_minus_K.log()

        # ===== return proposal =====

        return (
            idx1_K,
            idx2_K,
            branch1_K,
            branch2_K,
            embedding_KxD,
            log_v_plus_K,
            log_v_minus_K,
        )
