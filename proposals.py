import torch
from torch import Tensor, nn

import distances
import encoders
from vcsmc_utils import gather_K, gather_K2


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
        log: bool,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Propose two nodes to merge, as well as their branch lengths.

        Args:
            N: The number of leaf nodes.
            leaf_counts_Kxt: The number of leaf nodes in each subtree of each particle.
            embeddings_KtxD: Embeddings of each subtree of each particle.
            log: Whether to log to TensorBoard. Must be in a summary writer context.
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
        seq_encoder: encoders.SequenceEncoder,
        *,
        N: int,
        initial_branch_len: float = 1.0,
    ):
        """
        Args:
            seq_encoder: Sequence encoder.
            N: The number of leaf nodes.
            initial_branch_len: The initial expected value of the branch lengths.
                The exponential distribution from which branch lengths are
                sampled will initially have lambda = 1/initial_branch_len.
        """

        super().__init__(seq_encoder)

        initial_param = 1 / initial_branch_len
        # value of variable is passed through exp() later
        initial_log_param = torch.tensor(initial_param).log().repeat(N - 1)

        # N1 -> N-1
        self.log_branch_params1_N1 = nn.Parameter(initial_log_param)
        self.log_branch_params2_N1 = nn.Parameter(initial_log_param)

    def branch_params(self, r: int):
        # use exp to ensure params are positive
        branch_param1 = self.log_branch_params1_N1[r].exp()
        branch_param2 = self.log_branch_params2_N1[r].exp()
        return branch_param1, branch_param2

    def forward(
        self,
        N: int,
        leaf_counts_Kxt: Tensor,
        embeddings_KxtxD: Tensor,
        log: bool,
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

        branch_param1, branch_param2 = self.branch_params(r)

        branch1_distr = torch.distributions.Exponential(rate=branch_param1)
        branch1_distr = torch.distributions.Exponential(rate=branch_param2)

        branch1_K = branch1_distr.sample(torch.Size([K]))
        branch2_K = branch1_distr.sample(torch.Size([K]))

        log_branch1_prior_K = branch1_distr.log_prob(branch1_K)
        log_branch2_prior_K = branch1_distr.log_prob(branch2_K)

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
        *,
        sample_merge_temp: float = 1.0,
        sample_branches: bool = False,
    ):
        """
        Args:
            distance: The distance function to use for embedding.
            seq_encoder: Sequence encoder.
            merge_encoder: Merge encoder.
            sample_merge_temp: Temperature to use for sampling a pair of nodes to merge.
                Negative pairwise node distances divided by `sample_temp` are used log weights.
                Set to a large value to effectively sample nodes uniformly.
            sample_branches: Whether to sample branch lengths from an exponential distribution.
                If false, simply use the distance between embeddings as the branch length.
        """

        super().__init__(seq_encoder)

        self.distance = distance
        self.merge_encoder = merge_encoder
        self.sample_merge_temp = sample_merge_temp
        self.sample_branches = sample_branches

    def forward(
        self,
        N: int,
        leaf_counts_Kxt: Tensor,
        embeddings_KxtxD: Tensor,
        log: bool,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        K = leaf_counts_Kxt.shape[0]
        t = leaf_counts_Kxt.shape[1]  # number of subtrees
        r = N - t  # merge step

        # ===== sample 2 distinct nodes to merge =====

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
        merge_log_weights_Kxtxt = -pairwise_distances_Kxtxt / self.sample_merge_temp

        # set diagonal entries to -inf to prevent self-merges
        merge_log_weights_Kxtxt = merge_log_weights_Kxtxt.diagonal_scatter(
            torch.full([K, t], -torch.inf), dim1=1, dim2=2
        )

        # TODO convert
        # for debugging
        # if log:
        #     if t == N:
        #         log_weights = tf.exp(merge_log_weights_Kxtxt[0, 0])
        #         log_weights /= tf.reduce_sum(log_weights)
        #         tf.summary.histogram("Merge weights", log_weights)

        flattened_log_weights_Kxtt = merge_log_weights_Kxtxt.view(K, t * t)

        merge_distr = torch.distributions.Categorical(logits=flattened_log_weights_Kxtt)
        flattened_sample_K = merge_distr.sample()

        idx1_K = flattened_sample_K // t
        idx2_K = flattened_sample_K % t

        # ===== get merged embedding =====

        child1_KxD = gather_K(embeddings_KxtxD, idx1_K)
        child2_KxD = gather_K(embeddings_KxtxD, idx2_K)

        embedding_KxD = self.merge_encoder(child1_KxD, child2_KxD)

        # ===== sample/get branches parameters =====

        if self.sample_branches:
            dist1_K: Tensor = self.distance(child1_KxD, embedding_KxD)
            dist2_K: Tensor = self.distance(child2_KxD, embedding_KxD)

            # sample branch lengths from exp distributions whose expectations
            # are the distances between children and merged embeddings

            branch_param1_K = 1 / dist1_K
            branch_param2_K = 1 / dist2_K

            branch1_distr = torch.distributions.Exponential(rate=branch_param1_K)
            branch2_distr = torch.distributions.Exponential(rate=branch_param2_K)

            branch1_K = branch1_distr.sample()
            branch2_K = branch2_distr.sample()

            log_branch1_prior_K = branch1_distr.log_prob(branch1_K)
            log_branch2_prior_K = branch2_distr.log_prob(branch2_K)
        else:
            branch1_K: Tensor = self.distance(child1_KxD, embedding_KxD)
            branch2_K: Tensor = self.distance(child2_KxD, embedding_KxD)

            log_branch1_prior_K = 0
            log_branch2_prior_K = 0

        # ===== compute proposal probability =====

        # merge prob = merge weight * 2 / sum of all weights

        # the factor of 2 is because merging (idx1, idx2) is equivalent to
        # merging (idx2, idx1)

        log_merge_prob_K = gather_K2(merge_log_weights_Kxtxt, idx1_K, idx2_K)
        log_merge_prob_K = log_merge_prob_K + torch.log(torch.tensor(2))
        log_merge_prob_K = log_merge_prob_K - torch.logsumexp(
            merge_log_weights_Kxtxt, [1, 2]
        )

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
