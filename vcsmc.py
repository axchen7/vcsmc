from typing import Literal, TypedDict

import torch
from torch import Tensor, nn

from proposals import Proposal
from q_matrix_decoders import QMatrixDecoder
from vcsmc_utils import (
    build_newick_tree,
    compute_log_double_factorials_2N,
    compute_log_felsenstein_likelihoods_KxSxA,
    compute_log_likelihood_and_pi_K,
    concat_K,
    gather_K,
    replace_with_merged_K,
)


class VCSMC_Result(TypedDict):
    log_Z_SMC: Tensor
    log_likelihood_K: Tensor
    best_newick_tree: str
    best_merge1_indexes_N1: Tensor
    best_merge2_indexes_N1: Tensor
    best_branch1_lengths_N1: Tensor
    best_branch2_lengths_N1: Tensor


class VCSMC(nn.Module):
    def __init__(
        self,
        q_matrix_decoder: QMatrixDecoder,
        proposal: Proposal,
        taxa_N: list[str],
        *,
        K: int,
        prior_dist: Literal["gamma", "exp"] = "exp",
        prior_branch_len: float = 1.0,
    ):
        """
        Args:
            q_matrix_decoder: QMatrix object
            proposal: Proposal object
            taxa_N: List of taxa names of length N
            K: Number of particles
            prior_dist: Prior distribution for branch lengths
            prior_branch_len: Expected branch length under the prior
        """

        super().__init__()

        self.q_matrix_decoder = q_matrix_decoder
        self.proposal = proposal
        self.taxa_N = taxa_N
        self.K = K
        self.prior_dist: Literal["gamma", "exp"] = prior_dist
        self.prior_branch_len = prior_branch_len

        N = len(taxa_N)
        self.log_double_factorials_2N = compute_log_double_factorials_2N(N)

    def get_init_embeddings_KxNxD(self, data_NxSxA: Tensor):
        """Sets the embedding for all K particles to the same initial value."""
        embeddings_NxD: Tensor = self.proposal.seq_encoder(data_NxSxA)
        return embeddings_NxD.repeat(self.K, 1, 1)

    def forward(
        self,
        data_NxSxA: Tensor,
        data_batched_NxSxA: Tensor,
        site_positions_batched_SxSfull: Tensor,
        *,
        log=False,
    ) -> VCSMC_Result:
        """
        Args:
            data_NxSxA: Tensor of N full sequences (not batched).
                Used to compute initial embeddings.
                S = total number of sites.
            data_batched_NxSxA: Tensor of N sequences, batched along sequences and/or sites.
                S = number of sites in the batch.
            site_positions_SxSfull: One-hot encodings of the true site positions.
                S = number of sites in the batch.
                Sfull = total number of sites.
            log: Whether to log to TensorBoard. Must be in a summary writer context.
        Returns a dict containing:
            log_Z_SMC: lower bound to the likelihood; should set cost = -log_Z_SMC
            log_likelihood_K: log likelihoods for each particle at the last merge step
            best_newick_tree: Newick tree with the highest likelihood
            best_merge1_indexes_r: left node merge indexes for the best tree
            best_merge2_indexes_r: right node merge indexes for the best tree
            best_branch1_lengths_r: left branch lengths for the best tree
            best_branch2_lengths_r: right branch lengths for the best tree
        """

        N = data_batched_NxSxA.shape[0]
        A = data_batched_NxSxA.shape[2]

        K = self.K

        # compress site positions
        site_positions_SxC = self.q_matrix_decoder.site_positions_encoder(
            site_positions_batched_SxSfull
        )

        # at each step r, there are t = N-r >= 2 trees in the forest.
        # initially, r = 0 and t = N

        # for tracking tree topologies
        merge1_indexes_Kxr = torch.zeros(K, 0, dtype=torch.int)
        merge2_indexes_Kxr = torch.zeros(K, 0, dtype=torch.int)
        branch1_lengths_Kxr = torch.zeros(K, 0)
        branch2_lengths_Kxr = torch.zeros(K, 0)

        leaf_counts_Kxt = torch.ones(K, N, dtype=torch.int)
        embeddings_KxtxD = self.get_init_embeddings_KxNxD(data_NxSxA)

        # Felsenstein probabilities for computing pi(s)
        log_felsensteins_KxtxSxA = data_batched_NxSxA.log().repeat(K, 1, 1, 1)

        # difference of current and last iteration's values are used to compute weights
        log_pi_K = torch.zeros(K)
        # for computing empirical measure pi_rk(s)
        log_weight_K = torch.zeros(K)

        # must record all weights to compute Z_SMC
        log_weights_list_rxK: list[Tensor] = []

        # for displaying at the end
        log_likelihood_K = torch.zeros(K)  # initial value isn't used

        # for flattening embeddings
        D = embeddings_KxtxD.shape[-1]

        # iterate over merge steps
        for _ in range(N - 1):
            # ===== resample =====

            resample_distr = torch.distributions.Categorical(logits=log_weight_K)
            indexes_K = resample_distr.sample(torch.Size([K]))

            merge1_indexes_Kxr = merge1_indexes_Kxr[indexes_K]
            merge2_indexes_Kxr = merge2_indexes_Kxr[indexes_K]
            branch1_lengths_Kxr = branch1_lengths_Kxr[indexes_K]
            branch2_lengths_Kxr = branch2_lengths_Kxr[indexes_K]
            leaf_counts_Kxt = leaf_counts_Kxt[indexes_K]
            embeddings_KxtxD = embeddings_KxtxD[indexes_K]
            log_felsensteins_KxtxSxA = log_felsensteins_KxtxSxA[indexes_K]
            log_pi_K = log_pi_K[indexes_K]
            log_weight_K = log_weight_K[indexes_K]

            # ===== extend partial states using proposal =====

            # sample from proposal distribution
            (
                idx1_K,
                idx2_K,
                branch1_K,
                branch2_K,
                embedding_KxD,
                log_v_plus_K,
                log_v_minus_K,
            ) = self.proposal(
                N,
                leaf_counts_Kxt,
                embeddings_KxtxD,
                log_felsensteins_KxtxSxA,
                site_positions_SxC,
                log,
            )

            # helper function
            def merge_K(arr_K: Tensor, new_val_K: Tensor):
                return replace_with_merged_K(arr_K, idx1_K, idx2_K, new_val_K)

            # ===== post-proposal bookkeeping =====

            merge1_indexes_Kxr = concat_K(merge1_indexes_Kxr, idx1_K)
            merge2_indexes_Kxr = concat_K(merge2_indexes_Kxr, idx2_K)
            branch1_lengths_Kxr = concat_K(branch1_lengths_Kxr, branch1_K)
            branch2_lengths_Kxr = concat_K(branch2_lengths_Kxr, branch2_K)

            leaf_counts_Kxt = merge_K(
                leaf_counts_Kxt,
                gather_K(leaf_counts_Kxt, idx1_K) + gather_K(leaf_counts_Kxt, idx2_K),
            )
            embeddings_KxtxD = merge_K(embeddings_KxtxD, embedding_KxD)

            # ===== compute Felsenstein likelihoods =====

            Q_matrix_KxSxAxA = self.q_matrix_decoder.Q_matrix_VxSxAxA(
                embedding_KxD, site_positions_SxC
            )

            log_felsensteins_KxSxA = compute_log_felsenstein_likelihoods_KxSxA(
                Q_matrix_KxSxAxA,
                gather_K(log_felsensteins_KxtxSxA, idx1_K),
                gather_K(log_felsensteins_KxtxSxA, idx2_K),
                branch1_K,
                branch2_K,
            )
            log_felsensteins_KxtxSxA = merge_K(
                log_felsensteins_KxtxSxA, log_felsensteins_KxSxA
            )

            # ===== compute new likelihood, pi, and weight =====

            def compute_log_stat_probs_KxtxSxA():
                t = embeddings_KxtxD.shape[1]

                # flatten embeddings to compute stat_probs, then reshape back
                embeddings_KtxD = embeddings_KxtxD.reshape(K * t, D)
                stat_probs_KtxSxA = self.q_matrix_decoder.stat_probs_VxSxA(
                    embeddings_KtxD, site_positions_SxC
                )
                log_stat_probs_KtxSxA = stat_probs_KtxSxA.log()

                Kt = stat_probs_KtxSxA.shape[0]
                S = stat_probs_KtxSxA.shape[1]

                # if stat_probs_VxSxA() returned a tensor with S=1 and/or Kt=1,
                # continue to use broadcasting
                if Kt == 1:
                    return log_stat_probs_KtxSxA.view(1, 1, S, A)
                else:
                    return log_stat_probs_KtxSxA.view(K, t, S, A)

            prev_log_pi_K = log_pi_K
            log_stat_probs_KxtxSxA = compute_log_stat_probs_KxtxSxA()

            log_likelihood_K, log_pi_K = compute_log_likelihood_and_pi_K(
                branch1_lengths_Kxr,
                branch2_lengths_Kxr,
                leaf_counts_Kxt,
                log_felsensteins_KxtxSxA,
                log_stat_probs_KxtxSxA,
                self.prior_dist,
                self.prior_branch_len,
                self.log_double_factorials_2N,
            )

            # equation (7) in the VCSMC paper
            log_weight_K = log_pi_K - prev_log_pi_K + log_v_minus_K - log_v_plus_K

            log_weights_list_rxK.append(log_weight_K)

        # ===== compute Z_SMC =====

        # Forms the estimator log_Z_SMC, a multi sample lower bound to the
        # likelihood. Z_SMC is formed by averaging over weights (across k) and
        # multiplying over coalescent events (across r).
        # See equation (8) in the VCSMC paper.

        log_weights_rxK = torch.stack(log_weights_list_rxK)
        log_scaled_weights_rxK = log_weights_rxK - torch.log(torch.tensor(K))
        log_sum_weights_r = torch.logsumexp(log_scaled_weights_rxK, 1)
        log_Z_SMC = torch.sum(log_sum_weights_r)

        # ==== build best Newick tree ====

        best_tree_idx = torch.argmax(log_likelihood_K)
        best_newick_tree = build_newick_tree(
            self.taxa_N,
            merge1_indexes_Kxr[best_tree_idx],
            merge2_indexes_Kxr[best_tree_idx],
            branch1_lengths_Kxr[best_tree_idx],
            branch2_lengths_Kxr[best_tree_idx],
        )

        # ===== return final results =====

        return {
            "log_Z_SMC": log_Z_SMC,
            "log_likelihood_K": log_likelihood_K,
            "best_newick_tree": best_newick_tree,
            "best_merge1_indexes_N1": merge1_indexes_Kxr[best_tree_idx],
            "best_merge2_indexes_N1": merge2_indexes_Kxr[best_tree_idx],
            "best_branch1_lengths_N1": branch1_lengths_Kxr[best_tree_idx],
            "best_branch2_lengths_N1": branch2_lengths_Kxr[best_tree_idx],
        }
