import math

import tensorflow as tf

import markov
import proposal
from constants import DTYPE_FLOAT
from type_utils import Tensor, tf_function


@tf_function()
def compute_log_double_factorials_2N(N):
    """
    Pre-compute log double factorials that get baked into the TensorFlow
    graph.

    Returns: Tensor of log double factorials, where the nth entry is n!!,
    for n in [0, 2N).
    """

    # log double factorials
    even_values = [0.0]
    odd_values = [0.0]

    for i in range(1, N):
        even_factor = 2 * i
        odd_factor = 2 * i + 1

        even_values.append(even_values[-1] + math.log(even_factor))
        odd_values.append(odd_values[-1] + math.log(odd_factor))

    all_values = []

    for i in range(N):
        all_values.append(even_values[i])
        all_values.append(odd_values[i])

    return tf.constant(all_values, DTYPE_FLOAT)


@tf_function()
def compute_felsenstein_likelihoods_SxA(
    Q,
    likelihoods1_SxA,
    likelihoods2_SxA,
    branch1,
    branch2,
):
    P1_AxA = tf.linalg.expm(Q * branch1)
    P2_AxA = tf.linalg.expm(Q * branch2)

    prob1_SxA = tf.matmul(likelihoods1_SxA, P1_AxA)
    prob2_SxA = tf.matmul(likelihoods2_SxA, P2_AxA)

    return prob1_SxA * prob2_SxA


@tf_function(reduce_retracing=True)
def compute_log_likelihood_and_pi(
    stat_probs,
    branch_lengths_rx2,
    leaf_counts_t,
    felsensteins_txSxA,
    prior_branch_len,
    log_double_factorials_2N,
):
    """
    Dots Felsenstein probabilities with stationary probabilities and multiplies
    across sites and subtrees, yielding likelihood P(Y|forest,theta). Then
    multiplies by the prior over topologies and branch lengths to add the
    P(forest|theta) factor, yielding the measure pi(forest) = P(Y,forest|theta).

    Returns:
        log_likelihood: log P(Y|forest,theta)
        log_pi: log P(Y,forest|theta)
    """
    likelihoods_txS = tf.tensordot(felsensteins_txSxA, stat_probs, 1)
    log_likelihoods_txS = tf.math.log(likelihoods_txS)
    log_likelihood = tf.reduce_sum(log_likelihoods_txS)  # reduce along both axes

    leaf_counts_2timesminus3 = 2 * leaf_counts_t - 3
    leaf_counts_2timesminus3 = tf.maximum(leaf_counts_2timesminus3, 0)

    leaf_log_double_factorials_t = tf.gather(
        log_double_factorials_2N, leaf_counts_2timesminus3
    )
    log_topology_prior = tf.reduce_sum(-leaf_log_double_factorials_t)

    # compute probability of branches under exponential distribution with
    # lambda=1 / (expected branch length)
    branch_prior_param = 1 / prior_branch_len
    log_branch_prior = tf.reduce_sum(
        tf.math.log(branch_prior_param) - branch_prior_param * branch_lengths_rx2
    )

    log_pi = log_likelihood + log_topology_prior + log_branch_prior
    return log_likelihood, log_pi


@tf_function(reduce_retracing=True)
def replace_with_merged(arr, idx1, idx2, new_val) -> Tensor:
    """
    Removes elements at idx1 and idx2, and appends new_val to the end.
    Acts on axis=0.
    """

    # ensure idx1 < idx2
    if idx1 > idx2:
        idx1, idx2 = idx2, idx1

    return tf.concat(
        [arr[:idx1], arr[idx1 + 1 : idx2], arr[idx2 + 1 :], [new_val]],
        0,
    )


@tf_function()
def build_newick_tree(
    taxa_N: Tensor, merge_indexes_N1x2: Tensor, branch_lengths_N1x2: Tensor
):
    """
    Converts the merge indexes and branch lengths output by VCSMC into a Newick
    tree.

    Args:
        taxa_N: Tensor of taxa names of shape (N,).
        merge_indexes_N1x2: Tensor of merge indexes, where the nth entry is the
            merge indexes at the nth merge step. Shape is (N-1, 2)
        branch_lengths_N1x2: Tensor of branch lengths, where the nth entry is
            the branch lengths at the nth merge step. Shape is (N-1, 2)

    Returns: Tensor of the single Newick tree.
    """

    N = taxa_N.shape[0]

    # At each step r, there are t = N-r trees in the forest.
    # After N-1 steps, there is one tree left.
    trees_t = taxa_N

    for r in tf.range(N - 1):
        tf.autograph.experimental.set_loop_options(
            shape_invariants=[(trees_t, tf.TensorShape([None]))]
        )

        idx1 = merge_indexes_N1x2[r, 0]
        idx2 = merge_indexes_N1x2[r, 1]
        branch1 = tf.strings.as_string(branch_lengths_N1x2[r, 0])
        branch2 = tf.strings.as_string(branch_lengths_N1x2[r, 1])

        tree1 = trees_t[idx1]
        tree2 = trees_t[idx2]

        new_tree = "(" + tree1 + ":" + branch1 + "," + tree2 + ":" + branch2 + ")"

        trees_t = replace_with_merged(trees_t, idx1, idx2, new_tree)

    root_tree = trees_t[0] + ";"
    return root_tree


class VCSMC(tf.Module):
    def __init__(
        self,
        markov: markov.Markov,
        proposal: proposal.Proposal,
        taxa_N: Tensor,
        *,
        K: int,
        prior_branch_len: float = 1.0,
    ):
        """
        Args:
            markov: Markov object
            proposal: Proposal object
            taxa_N: Tensor of taxa names
            K: Number of particles
            prior_branch_len: Expected branch length under the prior
        """

        super().__init__()

        self.markov = markov
        self.proposal = proposal
        self.taxa_N = taxa_N
        self.K = K
        self.prior_branch_len = tf.constant(prior_branch_len, DTYPE_FLOAT)

    @tf_function()
    def get_init_embeddings_KxtxD(self, data_NxSxA):
        """Sets the embedding for all K particles to the same initial value."""
        embeddings_txD = tf.vectorized_map(self.proposal.embed, data_NxSxA)
        return tf.repeat(embeddings_txD[tf.newaxis], self.K, axis=0)  # type: ignore

    @tf_function()
    def __call__(self, data_NxSxA: Tensor) -> Tensor:
        """
        Returns:
            log_Z_SMC: lower bound to the likelihood; should set cost = -log_Z_SMC
            log_likelihoods_K: log likelihoods for each particle at the last merge step
            best_newick_tree: Newick tree with the highest likelihood
        """

        N, S, A = data_NxSxA.shape
        K = self.K

        log_double_factorials_2N = compute_log_double_factorials_2N(N)
        stat_probs = self.markov.stat_probs()
        Q = self.markov.Q()

        # at each step r, there are t = N-r >= 2 trees in the forest.
        # initially, r = 0 and t = N

        # for tracking tree topologies
        merge_indexes_Kxrx2 = tf.zeros([K, 0, 2], dtype=tf.int32)
        branch_lengths_Kxrx2 = tf.zeros([K, 0, 2], dtype=DTYPE_FLOAT)

        leaf_counts_Kxt = tf.ones([K, N], dtype=tf.int32)
        embeddings_KxtxD = self.get_init_embeddings_KxtxD(data_NxSxA)
        # Felsenstein probabilities for computing pi(s)
        felsensteins_KxtxSxA = tf.repeat(data_NxSxA[tf.newaxis], K, axis=0)
        log_pis_K = tf.zeros(K, DTYPE_FLOAT)

        # for computing empirical measure pi_rk(s)
        log_weights_K = tf.zeros(K, DTYPE_FLOAT)

        # must record all weights to compute Z_SMC
        log_weights_rxK = tf.TensorArray(DTYPE_FLOAT, N - 1)

        # for displaying at the end
        log_likelihoods_K = tf.zeros(K, DTYPE_FLOAT)  # initial value isn't used

        # for setting shape_invariants
        D = embeddings_KxtxD.shape[2]

        # iterate over merge steps
        for r in tf.range(N - 1):
            tf.autograph.experimental.set_loop_options(
                shape_invariants=[
                    (merge_indexes_Kxrx2, tf.TensorShape([K, None, 2])),
                    (branch_lengths_Kxrx2, tf.TensorShape([K, None, 2])),
                    (leaf_counts_Kxt, tf.TensorShape([K, None])),
                    (embeddings_KxtxD, tf.TensorShape([K, None, D])),
                    (felsensteins_KxtxSxA, tf.TensorShape([K, None, S, A])),
                ]
            )

            # ===== resample =====

            indexes = tf.random.categorical([log_weights_K], K)
            indexes = tf.squeeze(indexes)

            merge_indexes_Kxrx2 = tf.gather(merge_indexes_Kxrx2, indexes)
            branch_lengths_Kxrx2 = tf.gather(branch_lengths_Kxrx2, indexes)
            leaf_counts_Kxt = tf.gather(leaf_counts_Kxt, indexes)
            embeddings_KxtxD = tf.gather(embeddings_KxtxD, indexes)
            felsensteins_KxtxSxA = tf.gather(felsensteins_KxtxSxA, indexes)
            log_pis_K = tf.gather(log_pis_K, indexes)
            log_weights_K = tf.gather(log_weights_K, indexes)

            # ===== extend partial states using proposal =====

            # TODO vectorize across K

            new_merge_indexes_Kxrx2 = tf.TensorArray(tf.int32, K)
            new_branch_lengths_Kxrx2 = tf.TensorArray(DTYPE_FLOAT, K)
            new_leaf_counts_Kxt = tf.TensorArray(tf.int32, K)
            new_embeddings_KxtxD = tf.TensorArray(DTYPE_FLOAT, K)
            new_felsensteins_KxtxSxA = tf.TensorArray(DTYPE_FLOAT, K)
            new_log_likelihoods_K = tf.TensorArray(DTYPE_FLOAT, K)
            new_log_pis_K = tf.TensorArray(DTYPE_FLOAT, K)
            new_log_weights_K = tf.TensorArray(DTYPE_FLOAT, K)

            # iterate over particles
            for k in tf.range(K):
                # sample from proposal distribution
                (
                    idx1,
                    idx2,
                    branch1,
                    branch2,
                    embedding_D,
                    log_v_plus,
                    log_v_minus,
                ) = self.proposal(r, leaf_counts_Kxt[k], embeddings_KxtxD[k])

                # helper function
                def merge(arr, new_val):
                    return replace_with_merged(arr, idx1, idx2, new_val)

                # ===== post-proposal bookkeeping =====

                new_branch_lengths_rx2 = tf.concat(
                    [branch_lengths_Kxrx2[k], [[branch1, branch2]]], 0
                )
                new_leaf_counts_t = merge(
                    leaf_counts_Kxt[k],
                    leaf_counts_Kxt[k, idx1] + leaf_counts_Kxt[k, idx2],
                )
                new_felsensteins_txSxA = merge(
                    felsensteins_KxtxSxA[k],
                    compute_felsenstein_likelihoods_SxA(
                        Q,
                        felsensteins_KxtxSxA[k, idx1],
                        felsensteins_KxtxSxA[k, idx2],
                        branch1,
                        branch2,
                    ),
                )

                new_merge_indexes_Kxrx2 = new_merge_indexes_Kxrx2.write(
                    k, tf.concat([merge_indexes_Kxrx2[k], [[idx1, idx2]]], 0)
                )
                new_branch_lengths_Kxrx2 = new_branch_lengths_Kxrx2.write(
                    k, new_branch_lengths_rx2
                )
                new_leaf_counts_Kxt = new_leaf_counts_Kxt.write(k, new_leaf_counts_t)
                new_embeddings_KxtxD = new_embeddings_KxtxD.write(
                    k,
                    merge(embeddings_KxtxD[k], embedding_D),
                )
                new_felsensteins_KxtxSxA = new_felsensteins_KxtxSxA.write(
                    k, new_felsensteins_txSxA
                )

                # ===== compute new likelihoods, pis, and weights =====

                new_log_likelihood, new_log_pi = compute_log_likelihood_and_pi(
                    stat_probs,
                    new_branch_lengths_rx2,
                    new_leaf_counts_t,
                    new_felsensteins_txSxA,
                    self.prior_branch_len,
                    log_double_factorials_2N,
                )

                # equation (7) in the VCSMC paper
                new_log_weight = new_log_pi - log_pis_K[k] + log_v_minus - log_v_plus

                new_log_likelihoods_K = new_log_likelihoods_K.write(
                    k, new_log_likelihood
                )
                new_log_pis_K = new_log_pis_K.write(k, new_log_pi)
                new_log_weights_K = new_log_weights_K.write(k, new_log_weight)

            # ===== update states =====

            merge_indexes_Kxrx2 = new_merge_indexes_Kxrx2.stack()
            branch_lengths_Kxrx2 = new_branch_lengths_Kxrx2.stack()
            leaf_counts_Kxt = new_leaf_counts_Kxt.stack()
            embeddings_KxtxD = new_embeddings_KxtxD.stack()
            felsensteins_KxtxSxA = new_felsensteins_KxtxSxA.stack()
            log_likelihoods_K = new_log_likelihoods_K.stack()
            log_pis_K = new_log_pis_K.stack()
            log_weights_K = new_log_weights_K.stack()

            log_weights_rxK = log_weights_rxK.write(r, log_weights_K)

        # ===== compute Z_SMC =====

        # Forms the estimator log_Z_SMC, a multi sample lower bound to the
        # likelihood. Z_SMC is formed by averaging over weights (across k) and
        # multiplying over coalescent events (across r).
        # See equation (8) in the VCSMC paper.

        log_weights_rxK = log_weights_rxK.stack()
        log_scaled_weights_rxK = log_weights_rxK - tf.math.log(tf.cast(K, DTYPE_FLOAT))
        log_sum_weights_r = tf.reduce_logsumexp(log_scaled_weights_rxK, axis=1)
        log_Z_SMC = tf.reduce_sum(log_sum_weights_r)

        # ==== build best Newick tree ====

        best_tree_idx = tf.math.argmax(log_likelihoods_K)
        best_newick_tree = build_newick_tree(
            self.taxa_N,
            merge_indexes_Kxrx2[best_tree_idx],  # type: ignore
            branch_lengths_Kxrx2[best_tree_idx],  # type: ignore
        )

        # ===== return final results =====

        return log_Z_SMC, log_likelihoods_K, best_newick_tree
