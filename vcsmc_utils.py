import math

import tensorflow as tf

from constants import DTYPE_FLOAT
from type_utils import Tensor, tf_function


@tf_function()
def compute_log_double_factorials_2N(N) -> Tensor:
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
def compute_felsenstein_likelihoods_KxSxA(
    Q,
    likelihoods1_KxSxA,
    likelihoods2_KxSxA,
    branch1_K,
    branch2_K,
) -> Tensor:
    Qbranch1_KxAxA = tf.tensordot(branch1_K, Q, 0)
    Qbranch2_KxAxA = tf.tensordot(branch2_K, Q, 0)

    P1_KxAxA = tf.linalg.expm(Qbranch1_KxAxA)
    P2_KxAxA = tf.linalg.expm(Qbranch2_KxAxA)

    prob1_KxSxA = tf.matmul(likelihoods1_KxSxA, P1_KxAxA)
    prob2_KxSxA = tf.matmul(likelihoods2_KxSxA, P2_KxAxA)

    return prob1_KxSxA * prob2_KxSxA


@tf_function(reduce_retracing=True)
def compute_log_likelihood_and_pi_K(
    stat_probs,
    branch1_lengths_Kxr,
    branch2_lengths_Kxr,
    leaf_counts_Kxt,
    felsensteins_KxtxSxA,
    prior_branch_len,
    log_double_factorials_2N,
) -> Tensor:
    """
    Dots Felsenstein probabilities with stationary probabilities and multiplies
    across sites and subtrees, yielding likelihood P(Y|forest,theta). Then
    multiplies by the prior over topologies and branch lengths to add the
    P(forest|theta) factor, yielding the measure pi(forest) = P(Y,forest|theta).
    Performs this computation across all particles.

    Returns:
        log_likelihood_K: log P(Y|forest,theta)
        log_pi_K: log P(Y,forest|theta)
    """

    likelihoods_KxtxS = tf.tensordot(felsensteins_KxtxSxA, stat_probs, 1)
    log_likelihoods_KxtxS = tf.math.log(likelihoods_KxtxS)
    log_likelihood_K = tf.reduce_sum(log_likelihoods_KxtxS, [1, 2])

    leaf_counts_2timesminus3_Kxt = 2 * leaf_counts_Kxt - 3
    leaf_counts_2timesminus3_Kxt = tf.maximum(leaf_counts_2timesminus3_Kxt, 0)

    leaf_log_double_factorials_Kxt = tf.vectorized_map(
        lambda entry_t: tf.gather(log_double_factorials_2N, entry_t),
        leaf_counts_2timesminus3_Kxt,
    )
    log_topology_prior_K = tf.reduce_sum(-leaf_log_double_factorials_Kxt, 1)  # type: ignore

    # compute probability of branches under exponential distribution with
    # lambda = 1 / (expected branch length)
    branch_prior_param = 1 / prior_branch_len
    log_branch1_prior_K = tf.reduce_sum(
        tf.math.log(branch_prior_param) - branch_prior_param * branch1_lengths_Kxr, 1
    )
    log_branch2_prior_K = tf.reduce_sum(
        tf.math.log(branch_prior_param) - branch_prior_param * branch2_lengths_Kxr, 1
    )

    log_pi_K = (
        log_likelihood_K
        + log_topology_prior_K
        + log_branch1_prior_K
        + log_branch2_prior_K
    )
    return log_likelihood_K, log_pi_K


@tf_function(reduce_retracing=True)
def gather_K(arr_K, index_K) -> Tensor:
    """
    Given an array of shape (K, None, ...), gathers the K elements at
    [k, index[k]] for each k in [0, K). Returns a tensor of shape (K, ...).
    """
    K = index_K.shape[0]
    indexes_Kx2 = tf.stack([tf.range(K), index_K], 1)
    return tf.gather_nd(arr_K, indexes_Kx2)


@tf_function(reduce_retracing=True)
def gather_K2(arr_K, index1_K, index2_K) -> Tensor:
    """
    Given an array of shape (K, None, None, ...), gathers the K elements at
    [k, index1[k], index2[k]] for each k in [0, K).
    Returns a tensor of shape (K, ...).
    """
    K = index1_K.shape[0]
    indexes_Kx3 = tf.stack([tf.range(K), index1_K, index2_K], 1)
    return tf.gather_nd(arr_K, indexes_Kx3)


@tf_function(reduce_retracing=True)
def concat_K(arr_Kxr, val_K) -> Tensor:
    """
    Concatenates val_K to the end of arr_Kxr. Acts on axis=1.
    """
    return tf.concat([arr_Kxr, val_K[:, tf.newaxis]], 1)


@tf_function(reduce_retracing=True)
def replace_with_merged(arr, idx1, idx2, new_val) -> Tensor:
    """
    Removes elements at idx1 and idx2, and appends new_val to the end. Acts on
    axis=0.
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
    taxa_N: Tensor,
    merge1_indexes_N1: Tensor,
    merge2_indexes_N1: Tensor,
    branch1_lengths_N1: Tensor,
    branch2_lengths_N1: Tensor,
) -> Tensor:
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

        idx1 = merge1_indexes_N1[r]
        idx2 = merge2_indexes_N1[r]
        branch1 = tf.strings.as_string(branch1_lengths_N1[r])
        branch2 = tf.strings.as_string(branch2_lengths_N1[r])

        tree1 = trees_t[idx1]
        tree2 = trees_t[idx2]

        new_tree = "(" + tree1 + ":" + branch1 + "," + tree2 + ":" + branch2 + ")"

        trees_t = replace_with_merged(trees_t, idx1, idx2, new_tree)

    root_tree = trees_t[0] + ";"
    return root_tree
