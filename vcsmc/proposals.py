import torch
from torch import Tensor, nn

from .distance_utils import EPSILON
from .distances import Distance
from .encoders import DummySequenceEncoder, MergeEncoder, SequenceEncoder
from .vcsmc_utils import gather_K, gather_K2


def get_lookahead_merge_indexes(*, K, t: int) -> tuple[int, Tensor, Tensor]:
    # take all possible (n choose 2) merge pairs
    J = t * (t - 1) // 2

    take_J = torch.ones([t, t], dtype=torch.bool).triu(1).flatten().nonzero().flatten()
    idx1_J = take_J // t
    idx2_J = take_J % t

    idx1_KxJ = idx1_J.repeat(K, 1)
    idx2_KxJ = idx2_J.repeat(K, 1)

    return J, idx1_KxJ, idx2_KxJ


class Proposal(nn.Module):
    """
    Proposal distribution for selecting two nodes to merge and sampling branch lengths.
    """

    def __init__(self, seq_encoder: SequenceEncoder):
        super().__init__()

        self.seq_encoder = seq_encoder

    def forward(
        self, N: int, leaf_counts_Kxt: Tensor, embeddings_KxtxD: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Propose J different particles, each defined by the two nodes being
        merged and their branch lengths.

        Args:
            N: The number of leaf nodes.
            leaf_counts_Kxt: The number of leaf nodes in each subtree of each particle.
            embeddings_KtxD: Embeddings of each subtree of each particle.
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

        super().__init__(DummySequenceEncoder())

        # under exponential distribution, E[branch] = 1/rate
        initial_rate = 1 / initial_branch_len
        # value of variable is passed through exp() later
        initial_log_rates = torch.tensor(initial_rate).log().repeat(N - 1)

        self.lookahead_merge = lookahead_merge

        # exponential distribution rates for sampling branch lengths; N1 -> N-1
        self.log_rates1_N1 = nn.Parameter(initial_log_rates)
        self.log_rates2_N1 = nn.Parameter(initial_log_rates)

    def rates(self, r: int):
        # use exp to ensure rates are positive
        rate1 = self.log_rates1_N1[r].exp()
        rate2 = self.log_rates2_N1[r].exp()
        return rate1, rate2

    def forward(
        self, N: int, leaf_counts_Kxt: Tensor, embeddings_KxtxD: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        K = leaf_counts_Kxt.shape[0]
        t = leaf_counts_Kxt.shape[1]  # number of subtrees
        r = N - t  # merge step

        # ===== determine nodes to merge =====

        if self.lookahead_merge:
            J, idx1_KxJ, idx2_KxJ = get_lookahead_merge_indexes(K=K, t=t)
            log_merge_prob = 0
        else:
            # uniformly sample 2 distinct nodes to merge
            J = 1

            idx1_KxJ = torch.randint(0, t, [K]).unsqueeze(1)
            idx2_KxJ = torch.randint(0, t - 1, [K]).unsqueeze(1)

            # shift to guarantee idx2 > idx1
            idx2_KxJ = torch.where(idx2_KxJ >= idx1_KxJ, idx2_KxJ + 1, idx2_KxJ)

            # merge prob = 1 / (t choose 2)
            log_merge_prob = -torch.log(torch.tensor(t * (t - 1) // 2))

        # ===== sample branch lengths from exponential distributions =====

        rate1, rate2 = self.rates(r)

        # re-parameterization trick: sample from U[0, 1] and transform to
        # exponential distribution (so gradients can flow through the sample)

        uniform1_KxJ = torch.rand([K, J])
        uniform2_KxJ = torch.rand([K, J])

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
        embedding_KxJxD = torch.zeros([K, J, 0])

        return (
            idx1_KxJ,
            idx2_KxJ,
            branch1_KxJ,
            branch2_KxJ,
            embedding_KxJxD,
            log_v_plus_KxJ,
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
        distance: Distance,
        seq_encoder: SequenceEncoder,
        merge_encoder: MergeEncoder,
        *,
        lookahead_merge: bool = False,
        sample_merge_temp: float | None = None,
        sample_branches: bool = False,
        merge_indexes_N1x2: Tensor | None = None,
    ):
        """
        Only one of `lookahead_merge`, `sample_merge_temp`, and `merge_indexes_N1x2`
        should be set.

        Args:
            distance: The distance function to use for embedding.
            seq_encoder: Sequence encoder.
            merge_encoder: Merge encoder.
            lookahead_merge: if True, will return a particle for each of the J=(t choose 2) possible merges.
            sample_merge_temp: Temperature to use for sampling a pair of nodes to merge.
                Negative pairwise node distances divided by `sample_merge_temp` are used log weights.
                Set to a large value to effectively sample nodes uniformly. If None, then a
                pair of nodes will be sampled uniformly. Only used if `lookahead_merge`is false.
            sample_branches: Whether to sample branch lengths from an exponential distribution.
                If false, simply use the distance between embeddings as the branch length.
            merge_indexes_N1x2: If not None, always use these merge indexes instead of sampling.
                This fixes the tree topology.
        """

        super().__init__(seq_encoder)

        self.distance = distance
        self.merge_encoder = merge_encoder
        self.lookahead_merge = lookahead_merge
        self.sample_merge_temp = sample_merge_temp
        self.sample_branches = sample_branches
        self.merge_indexes_N1x2 = merge_indexes_N1x2

    def forward(
        self, N: int, leaf_counts_Kxt: Tensor, embeddings_KxtxD: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        K = leaf_counts_Kxt.shape[0]
        t = leaf_counts_Kxt.shape[1]  # number of subtrees
        r = N - t  # merge step

        # ===== determine nodes to merge =====

        if self.merge_indexes_N1x2 is not None:
            idx1_K = self.merge_indexes_N1x2[r, 0].repeat(K)
            idx2_K = self.merge_indexes_N1x2[r, 1].repeat(K)
            log_merge_prob_K = torch.zeros([K])
        else:
            if self.sample_merge_temp is not None:
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
                merge_log_weights_Kxtxt = torch.zeros([K, t, t])

            # set diagonal entries to -inf to prevent self-merges
            merge_log_weights_Kxtxt = merge_log_weights_Kxtxt.diagonal_scatter(
                torch.full([K, t], -torch.inf), dim1=1, dim2=2
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

            log_merge_prob_K = gather_K2(merge_log_weights_Kxtxt, idx1_K, idx2_K)
            log_merge_prob_K = log_merge_prob_K + torch.log(torch.tensor(2))
            log_merge_prob_K = log_merge_prob_K - torch.logsumexp(
                merge_log_weights_Kxtxt, [1, 2]
            )

        # ===== get merged embedding =====

        child1_KxD = gather_K(embeddings_KxtxD, idx1_K)
        child2_KxD = gather_K(embeddings_KxtxD, idx2_K)

        embedding_KxD = self.merge_encoder(child1_KxD, child2_KxD)

        # ===== sample/get branches parameters =====

        dist1_K: Tensor = self.distance(child1_KxD, embedding_KxD)
        dist2_K: Tensor = self.distance(child2_KxD, embedding_KxD)

        if self.sample_branches:
            # sample branch lengths from exp distributions whose expectations
            # are the distances between children and merged embeddings

            # EPSILON**0.5 is needed to prevent division by zero under float32
            rate1_K = 1 / (dist1_K + EPSILON**0.5)
            rate2_K = 1 / (dist2_K + EPSILON**0.5)

            # re-parameterization trick: sample from U[0, 1] and transform to
            # exponential distribution (so gradients can flow through the sample)

            uniform1_K = torch.rand([K])
            uniform2_K = torch.rand([K])

            # branch1 ~ Exp(rate1) and branch2 ~ Exp(rate2)
            branch1_K = -(1 / rate1_K) * uniform1_K.log()
            branch2_K = -(1 / rate2_K) * uniform2_K.log()

            # log of exponential pdf
            log_branch1_prob_K = rate1_K.log() - rate1_K * branch1_K
            log_branch2_prob_K = rate2_K.log() - rate2_K * branch2_K
        else:
            branch1_K = dist1_K
            branch2_K = dist2_K

            log_branch1_prob_K = torch.zeros([K])
            log_branch2_prob_K = torch.zeros([K])

        # ===== compute proposal probability =====

        log_v_plus_K = log_merge_prob_K + log_branch1_prob_K + log_branch2_prob_K

        # ===== return proposal =====

        # add singleton dimension for J (propose only one particle)
        idx1_KxJ = idx1_K.unsqueeze(1)
        idx2_KxJ = idx2_K.unsqueeze(1)
        branch1_KxJ = branch1_K.unsqueeze(1)
        branch2_KxJ = branch2_K.unsqueeze(1)
        embedding_KxJxD = embedding_KxD.unsqueeze(1)
        log_v_plus_KxJ = log_v_plus_K.unsqueeze(1)

        return (
            idx1_KxJ,
            idx2_KxJ,
            branch1_KxJ,
            branch2_KxJ,
            embedding_KxJxD,
            log_v_plus_KxJ,
        )
