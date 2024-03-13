import math
from typing import Literal

import torch
from torch import Tensor


def compute_log_double_factorials_2N(N: int) -> Tensor:
    """
    Pre-compute log double factorials.

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

    return torch.tensor(all_values)


def compute_felsenstein_likelihoods_KxSxA(
    Q_matrix_KxSxAxA: Tensor,
    likelihoods1_KxSxA: Tensor,
    likelihoods2_KxSxA: Tensor,
    branch1_K: Tensor,
    branch2_K: Tensor,
) -> Tensor:
    Qbranch1_KxSxAxA = Q_matrix_KxSxAxA * branch1_K[:, None, None, None]
    Qbranch2_KxSxAxA = Q_matrix_KxSxAxA * branch2_K[:, None, None, None]

    # matrix_exp is not implemented for mps
    P1_KxSxAxA = torch.matrix_exp(Qbranch1_KxSxAxA.cpu()).to(Q_matrix_KxSxAxA)
    P2_KxSxAxA = torch.matrix_exp(Qbranch2_KxSxAxA.cpu()).to(Q_matrix_KxSxAxA)

    likelihoods1_KxSxAx1 = likelihoods1_KxSxA.unsqueeze(-1)
    likelihoods2_KxSxAx1 = likelihoods2_KxSxA.unsqueeze(-1)

    prob1_KxSxAx1 = torch.matmul(P1_KxSxAxA, likelihoods1_KxSxAx1)
    prob2_KxSxAx1 = torch.matmul(P2_KxSxAxA, likelihoods2_KxSxAx1)

    prob1_KxSxA = prob1_KxSxAx1.squeeze(-1)
    prob2_KxSxA = prob2_KxSxAx1.squeeze(-1)

    return prob1_KxSxA * prob2_KxSxA


def compute_log_likelihood_and_pi_K(
    branch1_lengths_Kxr: Tensor,
    branch2_lengths_Kxr: Tensor,
    leaf_counts_Kxt: Tensor,
    felsensteins_KxtxSxA: Tensor,
    stat_probs_KxtxSxA: Tensor,
    prior_dist: Literal["gamma", "exp"],
    prior_branch_len: float,
    log_double_factorials_2N: Tensor,
) -> tuple[Tensor, Tensor]:
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

    # dot Felsenstein probabilities with stationary probabilities (along axis A)
    likelihoods_KxtxS = torch.sum(felsensteins_KxtxSxA * stat_probs_KxtxSxA, -1)
    log_likelihoods_KxtxS = torch.log(likelihoods_KxtxS)
    log_likelihood_K = torch.sum(log_likelihoods_KxtxS, [1, 2])

    leaf_counts_2timesminus3_Kxt = 2 * leaf_counts_Kxt - 3
    leaf_counts_2timesminus3_Kxt = torch.clamp(leaf_counts_2timesminus3_Kxt, min=0)

    # trick: by adding a dim to log_double_factorials_2N, we can index it with
    # leaf_counts_2timesminus3_Kxt
    leaf_log_double_factorials_Kxt = log_double_factorials_2N.unsqueeze(0)[
        0, leaf_counts_2timesminus3_Kxt
    ]

    log_topology_prior_K = torch.sum(-leaf_log_double_factorials_Kxt, 1)

    if prior_dist == "exp":
        # distribution has a mean of prior_branch_len
        branch_prior_distr = torch.distributions.Exponential(
            rate=1.0 / prior_branch_len
        )
    elif prior_dist == "gamma":
        # distribution has a mean of prior_branch_len
        branch_prior_distr = torch.distributions.Gamma(
            concentration=2.0, rate=2.0 / prior_branch_len
        )
    else:
        raise ValueError

    log_branch1_prior_K = torch.sum(branch_prior_distr.log_prob(branch1_lengths_Kxr), 1)
    log_branch2_prior_K = torch.sum(branch_prior_distr.log_prob(branch2_lengths_Kxr), 1)

    log_pi_K = (
        log_likelihood_K
        + log_topology_prior_K
        + log_branch1_prior_K
        + log_branch2_prior_K
    )

    return log_likelihood_K, log_pi_K


def gather_K(arr_K: Tensor, index_K: Tensor) -> Tensor:
    """
    Given an array of shape (K, None, ...), gathers the K elements at
    [k, index[k]] for each k in [0, K). Returns a tensor of shape (K, ...).
    """
    K = arr_K.shape[0]
    return arr_K[torch.arange(K), index_K]


def gather_K2(arr_K: Tensor, index1_K: Tensor, index2_K: Tensor) -> Tensor:
    """
    Given an array of shape (K, None, None, ...), gathers the K elements at
    [k, index1[k], index2[k]] for each k in [0, K).
    Returns a tensor of shape (K, ...).
    """
    K = arr_K.shape[0]
    return arr_K[torch.arange(K), index1_K, index2_K]


def concat_K(arr_Kxr, val_K) -> Tensor:
    """
    Concatenates val_K to the end of arr_Kxr. Acts on dim=1.
    """
    return torch.cat([arr_Kxr, val_K.unsqueeze(1)], 1)


def replace_with_merged_K(
    arr_K: Tensor, idx1_K: Tensor, idx2_K: Tensor, new_val_K: Tensor
) -> Tensor:
    """
    For each k in [0, K), modify row arr[k] by removing elements at idx1[k] and
    idx2[k], and appending new_val[k] to the end.
    """

    K = arr_K.shape[0]
    arange_K = torch.arange(K)

    # ensure idx1 < idx2 (element-wise)
    idx1_K, idx2_K = torch.min(idx1_K, idx2_K), torch.max(idx1_K, idx2_K)

    new_arr_K = arr_K.clone()

    # move new_val into arr at idx1
    new_arr_K[arange_K, idx1_K] = new_val_K
    # move last element of arr to idx2
    new_arr_K[arange_K, idx2_K] = arr_K[:, -1]
    # remove last element of arr
    new_arr_K = new_arr_K[:, :-1]

    return new_arr_K


def replace_with_merged_list(arr: list, idx1: int, idx2: int, new_val) -> list:
    """
    Mirrors merge ordering in `replace_with_merged_K`. Does not modify the input
    """

    # ensure idx1 < idx2
    if idx1 > idx2:
        idx1, idx2 = idx2, idx1

    new_arr = arr.copy()

    # move new_val into arr at idx1
    new_arr[idx1] = new_val
    # move last element of arr to idx2
    new_arr[idx2] = arr[-1]
    # remove last element of arr
    new_arr.pop()

    return new_arr


def build_newick_tree(
    taxa_N: list[str],
    merge1_indexes_N1: Tensor,
    merge2_indexes_N1: Tensor,
    branch1_lengths_N1: Tensor,
    branch2_lengths_N1: Tensor,
) -> str:
    """
    Converts the merge indexes and branch lengths output by VCSMC into a Newick
    tree.

    Args:
        taxa_N: List of taxa names of length N.
        merge_indexes_N1x2: Tensor of merge indexes, where the nth entry is the
            merge indexes at the nth merge step. Shape is (N-1, 2)
        branch_lengths_N1x2: Tensor of branch lengths, where the nth entry is
            the branch lengths at the nth merge step. Shape is (N-1, 2)

    Returns: Tensor of the single Newick tree.
    """

    N = len(taxa_N)

    # At each step r, there are t = N-r trees in the forest.
    # After N-1 steps, there is one tree left.
    trees_t = taxa_N

    for r in range(N - 1):
        idx1 = int(merge1_indexes_N1[r])
        idx2 = int(merge2_indexes_N1[r])

        branch1 = str(float(branch1_lengths_N1[r]))
        branch2 = str(float(branch2_lengths_N1[r]))

        tree1 = trees_t[idx1]
        tree2 = trees_t[idx2]

        new_tree = "(" + tree1 + ":" + branch1 + "," + tree2 + ":" + branch2 + ")"

        trees_t = replace_with_merged_list(trees_t, idx1, idx2, new_tree)

    root_tree = trees_t[0] + ";"
    return root_tree
