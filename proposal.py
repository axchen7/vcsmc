import math

import tensorflow as tf
import tensorflow_probability as tfp

from constants import DTYPE_FLOAT
from type_utils import Tensor, tf_function


class Proposal(tf.Module):
    """
    Proposal distribution for selecting two nodes to merge and sampling branch.
    """

    @tf_function()
    def embed(self, leaf_SxA: Tensor) -> Tensor:
        """
        Embeds a leaf node into the latent space.

        Args:
            leaf_SxA: The leaf node.
        Returns:
            embedding_D: The embedding of the leaf node.
        """

        # default to dummy embedding
        return tf.zeros([1], DTYPE_FLOAT)

    def __call__(
        self, r: Tensor, leaf_counts_t: Tensor, embeddings_txD: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Propose two nodes to merge, as well as their branch lengths.

        Args:
            r: The current merge step (0 <= r <= N-2).
            leaf_counts_t: The number of leaf nodes in each subtree.
            embeddings_txD: Embeddings of each subtree.
        Returns:
            idx1: Indices of the first node to merge.
            idx2: Indices of the second node to merge.
            branch1: Branch lengths of the first node.
            branch2: Branch lengths of the second node.
            embedding: The embedding of the merged subtree.
            log_v_plus: Log probability of the returned proposal.
            log_v_minus: Log of the over-counting correction factor.
            log_branch1_prior: Log prior probability of branch1.
            log_branch2_prior: Log prior probability of branch2.
        Note:
            At each step r, there are t = N-r >= 2 trees in the forest.
        """

        raise NotImplementedError


class ExpBranchProposal(Proposal):
    """
    Proposal where branch lengths are sampled from exponential distributions,
    with a learnable parameter for each merge step.
    """

    def __init__(self, *, N: int, initial_branch_len: float = 1.0):
        """
        Args:
            N: The number of leaf nodes.
            initial_branch_len: The initial expected value of the branch
            lengths. The exponential distribution from which branch lengths are
            sampled will initially have lambda = 1/initial_branch_len.
        """

        super().__init__()

        self.N = N

        initial_param = 1 / initial_branch_len
        # value of variable is passed through exp() later
        initial_log_param = tf.constant(math.log(initial_param), DTYPE_FLOAT, [N - 1])

        # N2 -> N-2
        self._branch_params1_N2 = tf.Variable(initial_log_param)
        self._branch_params2_N2 = tf.Variable(initial_log_param)

    @tf_function()
    def branch_params(self, r):
        # use exp to ensure params are positive
        branch_param1 = tf.exp(self._branch_params1_N2[r])  # type: ignore
        branch_param2 = tf.exp(self._branch_params2_N2[r])  # type: ignore
        return branch_param1, branch_param2

    @tf_function(reduce_retracing=True)
    def __call__(self, r, leaf_counts_t, embeddings_txD):
        # TODO vectorize across K

        num_nodes = self.N - r

        # ===== uniformly sample 2 distinct nodes to merge =====

        idx1 = tf.random.uniform([1], 0, num_nodes, tf.int32)[0]
        idx2 = tf.random.uniform([1], 0, num_nodes - 1, tf.int32)[0]

        if idx2 >= idx1:
            idx2 += 1

        # ===== sample branch lengths from exponential distributions =====

        branch_param1, branch_param2 = self.branch_params(r)

        branch_dist1 = tfp.distributions.Exponential(branch_param1)
        branch_dist2 = tfp.distributions.Exponential(branch_param2)

        branch1 = branch_dist1.sample(1)[0]
        branch2 = branch_dist2.sample(1)[0]

        # ===== compute proposal probability =====

        # log(num_nodes choose 2)
        log_num_merge_choices = tf.math.log(num_nodes * (num_nodes - 1) / 2)
        log_merge_prob = -log_num_merge_choices

        log_branch1_prior = branch_dist1.log_prob(branch1)
        log_branch2_prior = branch_dist2.log_prob(branch2)

        log_v_plus = log_merge_prob + log_branch1_prior + log_branch2_prior

        # ===== compute over-counting correction factor =====

        num_subtrees_with_one_leaf = tf.reduce_sum(
            tf.cast(leaf_counts_t == 1, tf.int32)
        )

        # exclude trees currently being merged from the count
        if leaf_counts_t[idx1] == 1:
            num_subtrees_with_one_leaf -= 1
        if leaf_counts_t[idx2] == 1:
            num_subtrees_with_one_leaf -= 1

        v_minus = self.N - num_subtrees_with_one_leaf
        log_v_minus = tf.math.log(tf.cast(v_minus, DTYPE_FLOAT))

        # ===== return proposal =====

        # dummy embedding
        embedding = tf.zeros([1], DTYPE_FLOAT)

        return (
            idx1,
            idx2,
            branch1,
            branch2,
            embedding,
            log_v_plus,
            log_v_minus,
            log_branch1_prior,
            log_branch2_prior,
        )
