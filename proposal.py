import math

import keras
import tensorflow as tf
import tensorflow_probability as tfp

import distances
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
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
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
            embedding_D: The embedding of the merged subtree.
            log_v_plus: Log probability of the returned proposal.
            log_v_minus: Log of the over-counting correction factor.
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
    def __call__(self, r, leaf_counts_t, embeddings_txD):
        # TODO vectorize across K

        t = self.N - r  # number of subtrees

        # ===== uniformly sample 2 distinct nodes to merge =====

        idx1 = tf.random.uniform([1], 0, t, tf.int32)[0]
        idx2 = tf.random.uniform([1], 0, t - 1, tf.int32)[0]

        if idx2 >= idx1:
            idx2 += 1

        # ===== sample branch lengths from exponential distributions =====

        branch_param1, branch_param2 = self.branch_params(r)

        branch_dist1 = tfp.distributions.Exponential(branch_param1)
        branch_dist2 = tfp.distributions.Exponential(branch_param2)

        branch1 = branch_dist1.sample(1)[0]
        branch2 = branch_dist2.sample(1)[0]

        # ===== compute proposal probability =====

        # log(t choose 2)
        log_num_merge_choices = tf.math.log(t * (t - 1) / 2)
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
        embedding_D = tf.zeros([1], DTYPE_FLOAT)

        return (
            idx1,
            idx2,
            branch1,
            branch2,
            embedding_D,
            log_v_plus,
            log_v_minus,
        )


class EmbeddingExpBranchProposal(Proposal):
    """
    Proposal where leaf nodes are embedded into D-dimensional space, and pairs
    of child embeddings are re-embedded to produce merged embeddings. Embeddings
    are performed using a multi-layered perceptron. Branch lengths are sampled
    from exponential distributions parameterized by distance between embeddings.
    """

    def __init__(
        self,
        distance: distances.Distance,
        *,
        N: int,
        S: int,
        A: int,
        D: int,
        leaf_mlp_width: int,
        leaf_mlp_depth: int,
        merge_mlp_width: int,
        merge_mlp_depth: int,
        sample_temp: float = 1.0,
    ):
        """
        Args:
            distance: The distance function to use for embedding.
            N: The number of leaf nodes.
            S: Number of sites in input. All inputs must have this many sites.
            A: Alphabet size.
            D: Number of dimensions in output embedding.
            leaf_mlp_width: Width of each layer of the leaf MLP.
            leaf_mlp_depth: Number of hidden layers in the leaf MLP.
            merge_mlp_width: Width of each layer of the merge MLP.
            merge_mlp_depth: Number of hidden layers in the merge MLP.
            sample_temp: Temperature to use for sampling a pair of nodes to merge.
                Negative pairwise node distances divided by `sample_temp` are used log weights.
                Set to a large value to effectively sample nodes uniformly.
        """

        super().__init__()

        self.distance = distance
        self.N = N
        self.S = S
        self.A = A
        self.D = D
        self.leaf_mlp_width = leaf_mlp_width
        self.leaf_mlp_depth = leaf_mlp_depth
        self.merge_mlp_width = merge_mlp_width
        self.merge_mlp_depth = merge_mlp_depth
        self.sample_temp = tf.constant(sample_temp, DTYPE_FLOAT)

        self.leaf_mlp = self.create_leaf_mlp()
        self.merge_mlp = self.create_merge_mlp()

    def create_leaf_mlp(self):
        mlp = keras.Sequential()
        mlp.add(keras.layers.Input([self.S, self.A], dtype=DTYPE_FLOAT))
        mlp.add(keras.layers.Flatten())

        for _ in range(self.leaf_mlp_depth):
            mlp.add(
                keras.layers.Dense(
                    self.leaf_mlp_width, activation="relu", dtype=DTYPE_FLOAT
                )
            )

        mlp.add(keras.layers.Dense(self.D, dtype=DTYPE_FLOAT))
        return mlp

    def create_merge_mlp(self):
        mlp = keras.Sequential()
        mlp.add(keras.layers.Input([2 * self.D], dtype=DTYPE_FLOAT))

        for _ in range(self.merge_mlp_depth):
            mlp.add(
                keras.layers.Dense(
                    self.merge_mlp_width, activation="relu", dtype=DTYPE_FLOAT
                )
            )

        mlp.add(keras.layers.Dense(self.D, dtype=DTYPE_FLOAT))
        return mlp

    @tf_function()
    def embed(self, leaf_SxA):
        return self.leaf_mlp(tf.expand_dims(leaf_SxA, 0))[0]  # type: ignore

    @tf_function()
    def __call__(self, r, leaf_counts_t, embeddings_txD):
        # TODO vectorize across K

        t = self.N - r  # number of subtrees

        # ===== sample 2 distinct nodes to merge =====

        # randomly select two subtrees to merge, using pairwise distances as
        # negative log probabilities, and incorporating the sample
        # temperature

        pairwise_distances_txt = tf.vectorized_map(
            lambda x: tf.vectorized_map(lambda y: self.distance(x, y), embeddings_txD),
            embeddings_txD,
        )
        merge_log_weights_txt = -pairwise_distances_txt / self.sample_temp  # type: ignore

        # set diagonal entries to -inf to prevent self-merges
        merge_log_weights_txt = tf.linalg.set_diag(
            merge_log_weights_txt,
            tf.fill([t], tf.constant(-math.inf, DTYPE_FLOAT)),
        )

        # sample a single pair of subtrees
        flattened_sample_tt = tf.random.categorical(
            [tf.reshape(merge_log_weights_txt, [-1])], 1, tf.int32
        )[0][0]
        tf.assert_less(flattened_sample_tt, t * t, message="sample out of range")
        idx1 = flattened_sample_tt // t
        idx2 = flattened_sample_tt % t
        tf.assert_equal(idx1 == idx2, False, message="subtrees are equal")

        # ===== get merged embedding =====

        embedding1 = embeddings_txD[idx1]
        embedding2 = embeddings_txD[idx2]

        concat_child_embeddings = tf.concat([embedding1, embedding2], axis=0)
        embedding_D = self.merge_mlp(tf.expand_dims(concat_child_embeddings, 0))[0]  # type: ignore

        # ===== compute branch distribution parameters =====

        dist1 = self.distance(embedding1, embedding_D)
        dist2 = self.distance(embedding2, embedding_D)

        # sample from exponential distributions whose expectations are the
        # distances between children and merged embeddings
        branch_param1 = 1 / dist1
        branch_param2 = 1 / dist2

        # ===== sample branch lengths from exponential distributions =====

        branch_dist1 = tfp.distributions.Exponential(branch_param1)
        branch_dist2 = tfp.distributions.Exponential(branch_param2)

        branch1 = branch_dist1.sample(1)[0]
        branch2 = branch_dist2.sample(1)[0]

        # ===== compute proposal probability =====

        # merge prob = merge weight * 2 / sum of all weights

        # the factor of 2 is because merging (idx1, idx2) is equivalent to
        # merging (idx2, idx1)

        log_merge_prob = merge_log_weights_txt[idx1, idx2]
        log_merge_prob += tf.math.log(tf.constant(2, DTYPE_FLOAT))
        log_merge_prob -= tf.math.reduce_logsumexp(merge_log_weights_txt)

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

        return (
            idx1,
            idx2,
            branch1,
            branch2,
            embedding_D,
            log_v_plus,
            log_v_minus,
        )
