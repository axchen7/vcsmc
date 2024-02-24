import math

import tensorflow as tf
import tensorflow_probability as tfp

import distances
import encoder_decoder
from constants import DTYPE_FLOAT
from type_utils import Tensor, tf_function
from vcsmc_utils import gather_K, gather_K2


class Proposal(tf.Module):
    """
    Proposal distribution for selecting two nodes to merge and sampling branch lengths.
    """

    def __init__(self, seq_encoder: encoder_decoder.SequenceEncoder):
        super().__init__()

        self.seq_encoder = seq_encoder

    def __call__(
        self, N: int, r: Tensor, leaf_counts_Kxt: Tensor, embeddings_KxtxD: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Propose two nodes to merge, as well as their branch lengths.

        Args:
            N: The number of leaf nodes.
            r: The current merge step (0 <= r <= N-2).
            leaf_counts_Kxt: The number of leaf nodes in each subtree of each particle.
            embeddings_KtxD: Embeddings of each subtree of each particle.
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
        seq_encoder: encoder_decoder.SequenceEncoder,
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
        initial_log_param = tf.constant(math.log(initial_param), DTYPE_FLOAT, [N - 1])

        # N2 -> N-2
        self.log_branch_params1_N2 = tf.Variable(
            initial_log_param, name="log_branch_params1_N2"
        )
        self.log_branch_params2_N2 = tf.Variable(
            initial_log_param, name="log_branch_params2_N2"
        )

    @tf_function()
    def branch_params(self, r):
        # use exp to ensure params are positive
        branch_param1 = tf.exp(self.log_branch_params1_N2[r])  # type: ignore
        branch_param2 = tf.exp(self.log_branch_params2_N2[r])  # type: ignore
        return branch_param1, branch_param2

    @tf_function(reduce_retracing=True)
    def __call__(self, N, r, leaf_counts_Kxt, embeddings_KxtxD):
        K = leaf_counts_Kxt.shape[0]
        t = N - r  # number of subtrees

        # ===== uniformly sample 2 distinct nodes to merge =====

        idx1_K = tf.random.uniform([K], 0, t, tf.int32)
        idx2_K = tf.random.uniform([K], 0, t - 1, tf.int32)

        # shift to guarantee idx2 > idx1
        idx2_K = tf.where(idx2_K >= idx1_K, idx2_K + 1, idx2_K)

        # ===== sample branch lengths from exponential distributions =====

        branch_param1, branch_param2 = self.branch_params(r)

        branch_dist1 = tfp.distributions.Exponential(branch_param1)
        branch_dist2 = tfp.distributions.Exponential(branch_param2)

        (
            branch1_K,
            log_branch1_prior_K,
        ) = branch_dist1.experimental_sample_and_log_prob(K)
        (
            branch2_K,
            log_branch2_prior_K,
        ) = branch_dist2.experimental_sample_and_log_prob(K)

        # ===== compute proposal probability =====

        # log(t choose 2)
        log_num_merge_choices = tf.math.log(t * (t - 1) / 2)
        log_merge_prob = -log_num_merge_choices

        log_v_plus_K = log_merge_prob + log_branch1_prior_K + log_branch2_prior_K

        # ===== compute over-counting correction factor =====

        num_subtrees_with_one_leaf_K = tf.reduce_sum(
            tf.cast(leaf_counts_Kxt == 1, tf.int32), 1
        )

        # exclude trees currently being merged from the count
        num_subtrees_with_one_leaf_K -= tf.cast(
            gather_K(leaf_counts_Kxt, idx1_K) == 1, tf.int32
        )
        num_subtrees_with_one_leaf_K -= tf.cast(
            gather_K(leaf_counts_Kxt, idx2_K) == 1, tf.int32
        )

        v_minus_K = N - num_subtrees_with_one_leaf_K
        log_v_minus_K = tf.math.log(tf.cast(v_minus_K, DTYPE_FLOAT))

        # ===== return proposal =====

        # dummy embedding
        embedding_KxD = tf.zeros([K, 0], DTYPE_FLOAT)

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
        seq_encoder: encoder_decoder.SequenceEncoder,
        merge_encoder: encoder_decoder.MergeEncoder,
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
        self.sample_temp = tf.constant(sample_merge_temp, DTYPE_FLOAT)
        self.sample_branches = sample_branches

    @tf_function(reduce_retracing=True)
    def __call__(self, N, r, leaf_counts_Kxt, embeddings_KxtxD):
        K = leaf_counts_Kxt.shape[0]
        t = N - r  # number of subtrees

        # ===== sample 2 distinct nodes to merge =====

        # randomly select two subtrees to merge, using pairwise distances as
        # negative log probabilities, and incorporating the sample
        # temperature

        flat_embeddings1_KttxD = tf.reshape(  # repeat like 123123123...
            tf.tile(embeddings_KxtxD, [1, t, 1]), [K * t * t, -1]
        )
        flat_embeddings2_KttxD = tf.reshape(  # repeat like 111222333...
            tf.tile(embeddings_KxtxD, [1, 1, t]), [K * t * t, -1]
        )

        pairwise_distances_Ktt = self.distance(
            flat_embeddings1_KttxD, flat_embeddings2_KttxD
        )
        pairwise_distances_Kxtxt = tf.reshape(pairwise_distances_Ktt, [K, t, t])
        merge_log_weights_Kxtxt = -pairwise_distances_Kxtxt / self.sample_temp  # type: ignore

        # set diagonal entries to -inf to prevent self-merges
        merge_log_weights_Kxtxt = tf.linalg.set_diag(
            merge_log_weights_Kxtxt,
            tf.fill([K, t], tf.constant(-math.inf, DTYPE_FLOAT)),
        )

        # for debugging
        # if t == N:
        #     log_weights = tf.exp(merge_log_weights_Kxtxt[0, 0])
        #     log_weights /= tf.reduce_sum(log_weights)
        #     tf.summary.histogram("Merge weights", log_weights)

        flattened_log_weights_Kxtt = tf.reshape(merge_log_weights_Kxtxt, [K, t * t])

        # sample a single pair of subtrees for each of the K particles
        flattened_sample_K = tf.random.categorical(
            flattened_log_weights_Kxtt, 1, tf.int32
        )
        flattened_sample_K = tf.squeeze(flattened_sample_K)

        tf.assert_less(flattened_sample_K, t * t, message="sample out of range")
        idx1_K = flattened_sample_K // t
        idx2_K = flattened_sample_K % t
        tf.assert_equal(idx1_K == idx2_K, False, message="subtrees are equal")

        # ===== get merged embedding =====

        child1_KxD = gather_K(embeddings_KxtxD, idx1_K)
        child2_KxD = gather_K(embeddings_KxtxD, idx2_K)

        embedding_KxD = self.merge_encoder(child1_KxD, child2_KxD)

        # ===== sample/get branches parameters =====

        if self.sample_branches:
            dist1_K = self.distance(child1_KxD, embedding_KxD)
            dist2_K = self.distance(child2_KxD, embedding_KxD)

            # sample from exponential distributions whose expectations are the
            # distances between children and merged embeddings
            branch_param1_K = 1 / dist1_K
            branch_param2_K = 1 / dist2_K

            # ===== sample branch lengths from exponential distributions =====

            uniform1_K = tf.random.uniform([K], 0, 1, dtype=DTYPE_FLOAT)
            uniform2_K = tf.random.uniform([K], 0, 1, dtype=DTYPE_FLOAT)

            # use uniform sample + inverse CDF to sample from exponential
            # distribution
            branch1_K = -tf.math.log(uniform1_K) / branch_param1_K
            branch2_K = -tf.math.log(uniform2_K) / branch_param2_K

            # plug samples back into log exponential PDF
            log_branch1_prior_K = (
                tf.math.log(branch_param1_K) - branch_param1_K * branch1_K
            )
            log_branch2_prior_K = (
                tf.math.log(branch_param2_K) - branch_param2_K * branch2_K
            )
        else:
            branch1_K = self.distance(child1_KxD, embedding_KxD)
            branch2_K = self.distance(child2_KxD, embedding_KxD)

            log_branch1_prior_K = 0
            log_branch2_prior_K = 0

        # ===== compute proposal probability =====

        # merge prob = merge weight * 2 / sum of all weights

        # the factor of 2 is because merging (idx1, idx2) is equivalent to
        # merging (idx2, idx1)

        log_merge_prob_K = gather_K2(merge_log_weights_Kxtxt, idx1_K, idx2_K)
        log_merge_prob_K += tf.math.log(tf.constant(2, DTYPE_FLOAT))
        log_merge_prob_K -= tf.math.reduce_logsumexp(merge_log_weights_Kxtxt, [1, 2])

        log_v_plus_K = log_merge_prob_K + log_branch1_prior_K + log_branch2_prior_K

        # ===== compute over-counting correction factor =====

        num_subtrees_with_one_leaf_K = tf.reduce_sum(
            tf.cast(leaf_counts_Kxt == 1, tf.int32), 1
        )

        # exclude trees currently being merged from the count
        num_subtrees_with_one_leaf_K -= tf.cast(
            gather_K(leaf_counts_Kxt, idx1_K) == 1, tf.int32
        )
        num_subtrees_with_one_leaf_K -= tf.cast(
            gather_K(leaf_counts_Kxt, idx2_K) == 1, tf.int32
        )

        v_minus_K = N - num_subtrees_with_one_leaf_K
        log_v_minus_K = tf.math.log(tf.cast(v_minus_K, DTYPE_FLOAT))

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
