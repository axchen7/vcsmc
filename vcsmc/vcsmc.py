import math
from typing import Literal, TypedDict

import torch
from torch import Tensor, nn

from .proposals import Proposal
from .q_matrix_decoders import QMatrixDecoder
from .vcsmc_utils import (
    build_newick_tree,
    compute_log_double_factorials_2N,
    compute_log_felsenstein_likelihoods_KxSxA,
    compute_log_likelihood_and_pi_K,
    compute_log_v_minus_K,
    concat_K,
    gather_K,
    replace_with_merged_K,
)


class VcsmcResult(TypedDict):
    log_Z_SMC: Tensor
    log_likelihood_K: Tensor
    best_newick_tree: str
    best_merge1_indexes_N1: Tensor  # left node index at each step
    best_merge2_indexes_N1: Tensor  # right node index at each step
    best_branch1_lengths_N1: Tensor  # left branch length at each step
    best_branch2_lengths_N1: Tensor  # right branch length at each step
    best_embeddings_N1xD: Tensor  # merged embedding at each step


class VCSMC(nn.Module):
    def __init__(
        self,
        q_matrix_decoder: QMatrixDecoder,
        proposal: Proposal,
        taxa_N: list[str],
        *,
        K: int,
        hash_trick: bool = False,
        # assume Exp(10) branch length prior
        prior_dist: Literal["gamma", "exp", "unif"] = "exp",
        prior_branch_len: float = 0.1,
    ):
        """
        Args:
            q_matrix_decoder: QMatrix object
            proposal: Proposal object
            taxa_N: List of taxa names of length N
            K: Number of particles
            hash_trick: Whether to use the hash trick to speed up computation
            prior_dist: Prior distribution for branch lengths
            prior_branch_len: Expected branch length under the prior
        """

        super().__init__()

        self.q_matrix_decoder = q_matrix_decoder
        self.proposal = proposal
        self.taxa_N = taxa_N
        self.K = K
        self.hash_trick = hash_trick
        self.prior_dist: Literal["gamma", "exp", "unif"] = prior_dist
        self.prior_branch_len = prior_branch_len

        N = len(taxa_N)
        log_double_factorials_2N = compute_log_double_factorials_2N(N)
        self.register_buffer("log_double_factorials_2N", log_double_factorials_2N)

    def get_init_embeddings_KxNxD(self, data_NxSxA: Tensor):
        """Sets the embedding for all K particles to the same initial value."""
        embeddings_NxD: Tensor = self.proposal.seq_encoder(data_NxSxA)
        return embeddings_NxD.repeat(self.K, 1, 1)

    def forward(
        self,
        data_NxSxA: Tensor,
        data_batched_NxSxA: Tensor,
        site_positions_batched_SxSfull: Tensor,
    ) -> VcsmcResult:
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
        Returns a dict containing:
            log_Z_SMC: lower bound to the likelihood; should set cost = -log_Z_SMC
            log_likelihood_K: log likelihoods for each particle at the last merge step
            best_newick_tree: Newick tree with the highest likelihood
            best_merge1_indexes_r: left node merge indexes for the best tree
            best_merge2_indexes_r: right node merge indexes for the best tree
            best_branch1_lengths_r: left branch lengths for the best tree
            best_branch2_lengths_r: right branch lengths for the best tree
        """

        device = data_batched_NxSxA.device

        N = data_batched_NxSxA.shape[0]
        A = data_batched_NxSxA.shape[2]

        K = self.K
        D = self.proposal.seq_encoder.D

        # compress site positions
        site_positions_SxC = self.q_matrix_decoder.site_positions_encoder(
            site_positions_batched_SxSfull
        )

        # at each step r, there are t = N-r >= 2 trees in the forest.
        # initially, r = 0 and t = N

        # for tracking tree topologies
        merge1_indexes_Kxr = torch.zeros(K, 0, dtype=torch.int, device=device)
        merge2_indexes_Kxr = torch.zeros(K, 0, dtype=torch.int, device=device)
        branch1_lengths_Kxr = torch.zeros(K, 0, device=device)
        branch2_lengths_Kxr = torch.zeros(K, 0, device=device)
        # merged embedding at each step
        embeddings_KxrxD = torch.zeros(K, 0, D, device=device)

        leaf_counts_Kxt = torch.ones(K, N, dtype=torch.int, device=device)
        # embeddings of each tree currently in the forest
        embeddings_KxtxD = self.get_init_embeddings_KxNxD(data_NxSxA)

        # Felsenstein probabilities for computing pi(s)
        log_felsensteins_KxtxSxA = data_batched_NxSxA.log().repeat(K, 1, 1, 1)

        # difference of current and last iteration's values are used to compute weights
        log_pi_K = torch.zeros(K, device=device)
        # for computing empirical measure pi_rk(s); initialize to log(1/K)
        log_weight_K = torch.full([K], -math.log(K), device=device)

        # must record all weights to compute Z_SMC
        log_weights_list_rxK: list[Tensor] = []

        # for returning at the end
        log_likelihood_K = torch.zeros(K, device=device)  # initial value isn't used

        # iterate over merge steps
        for _ in range(N - 1):
            # ===== resample =====

            resample_distr = torch.distributions.Categorical(logits=log_weight_K)
            indexes_K = resample_distr.sample(torch.Size([K]))

            merge1_indexes_Kxr = merge1_indexes_Kxr[indexes_K]
            merge2_indexes_Kxr = merge2_indexes_Kxr[indexes_K]
            branch1_lengths_Kxr = branch1_lengths_Kxr[indexes_K]
            branch2_lengths_Kxr = branch2_lengths_Kxr[indexes_K]
            embeddings_KxrxD = embeddings_KxrxD[indexes_K]
            leaf_counts_Kxt = leaf_counts_Kxt[indexes_K]
            embeddings_KxtxD = embeddings_KxtxD[indexes_K]
            log_felsensteins_KxtxSxA = log_felsensteins_KxtxSxA[indexes_K]
            log_pi_K = log_pi_K[indexes_K]

            # ===== extend partial states using proposal =====

            # sample from proposal distribution
            (
                idx1_KxJ,
                idx2_KxJ,
                branch1_KxJ,
                branch2_KxJ,
                embedding_KxJxD,
                log_v_plus_KxJ,
            ) = self.proposal(N, leaf_counts_Kxt, embeddings_KxtxD)

            # ===== deal with sub-particle (J) dimension =====

            J = idx1_KxJ.shape[1]

            idx1_KJ = idx1_KxJ.flatten()
            idx2_KJ = idx2_KxJ.flatten()
            branch1_KJ = branch1_KxJ.flatten()
            branch2_KJ = branch2_KxJ.flatten()
            embedding_KJxD = embedding_KxJxD.reshape(K * J, D)
            log_v_plus_KJ = log_v_plus_KxJ.flatten()

            # ===== handle particles with matching hashes together =====

            if self.hash_trick:
                with torch.no_grad():
                    # construct new embeddings to hash
                    embeddings_KJxtxD = embeddings_KxtxD.repeat_interleave(J, 0)
                    embeddings_KJxtxD = replace_with_merged_K(
                        embeddings_KJxtxD, idx1_KJ, idx2_KJ, embedding_KJxD
                    )

                    embedding_norm_sqs_KJxt = torch.sum(embeddings_KJxtxD**2, 2)
                    # sort first for predictable sum
                    embedding_norm_sqs_KJxt = embedding_norm_sqs_KJxt.sort(1).values
                    # use sum of norm square embeddings as hashes
                    hashes_KJ = torch.sum(embedding_norm_sqs_KJxt, 1)

                    sorted_hashes_KJ, hash_sort_idx_KJ = torch.sort(hashes_KJ)
                    hash_unsort_idx_KJ = torch.argsort(hash_sort_idx_KJ)
                    # Z = number of unique hashes
                    _, hash_counts_Z = sorted_hashes_KJ.unique_consecutive(
                        return_counts=True
                    )
                    # first occurrence index of each unique hash within the sorted hashes
                    sorted_unique_hash_idx_Z = (
                        torch.cumsum(hash_counts_Z, 0) - hash_counts_Z[0]
                    )

                    Z = hash_counts_Z.shape[0]

                    # first occurrence index of each unique hash within the original hashes
                    unique_hash_idx_Z = hash_sort_idx_KJ[sorted_unique_hash_idx_Z]
                    # undo the above mapping
                    undo_unique_hash_idx_KJ = torch.arange(
                        Z, device=device
                    ).repeat_interleave(hash_counts_Z, 0)[hash_unsort_idx_KJ]

                def squeeze_Z(arr_KJ: Tensor):
                    """
                    Returns unique values with respect to the hashes (and sorting).
                    """
                    return arr_KJ[unique_hash_idx_Z]

                def unsqueeze_KJ(arr_Z: Tensor):
                    """
                    Expands unique values to original size (and undoing the sorting).
                    """
                    return arr_Z[undo_unique_hash_idx_KJ]

            else:
                Z = K * J

                def squeeze_Z(arr_KJ: Tensor):
                    return arr_KJ

                def unsqueeze_KJ(arr_Z: Tensor):
                    return arr_Z

            # ===== squeeze particles with matching hashes =====

            # helper function
            def K_to_Z(arr_K: Tensor):
                arr_KJ = arr_K.repeat_interleave(J, 0)
                return squeeze_Z(arr_KJ)

            idx1_Z = squeeze_Z(idx1_KJ)
            idx2_Z = squeeze_Z(idx2_KJ)
            branch1_Z = squeeze_Z(branch1_KJ)
            branch2_Z = squeeze_Z(branch2_KJ)
            embedding_ZxD = squeeze_Z(embedding_KJxD)
            log_v_plus_Z = squeeze_Z(log_v_plus_KJ)

            merge1_indexes_Zxr = K_to_Z(merge1_indexes_Kxr)
            merge2_indexes_Zxr = K_to_Z(merge2_indexes_Kxr)
            branch1_lengths_Zxr = K_to_Z(branch1_lengths_Kxr)
            branch2_lengths_Zxr = K_to_Z(branch2_lengths_Kxr)
            embeddings_ZxrxD = K_to_Z(embeddings_KxrxD)
            leaf_counts_Zxt = K_to_Z(leaf_counts_Kxt)
            embeddings_ZxtxD = K_to_Z(embeddings_KxtxD)
            log_felsensteins_ZxtxSxA = K_to_Z(log_felsensteins_KxtxSxA)
            log_pi_Z = K_to_Z(log_pi_K)

            # ===== post-proposal bookkeeping =====

            # helper function
            def merge_Z(arr_Z: Tensor, new_val_Z: Tensor):
                return replace_with_merged_K(arr_Z, idx1_Z, idx2_Z, new_val_Z)

            branch1_lengths_Zxr = concat_K(branch1_lengths_Zxr, branch1_Z)
            branch2_lengths_Zxr = concat_K(branch2_lengths_Zxr, branch2_Z)
            leaf_counts_Zxt = merge_Z(
                leaf_counts_Zxt,
                gather_K(leaf_counts_Zxt, idx1_Z) + gather_K(leaf_counts_Zxt, idx2_Z),
            )
            embeddings_ZxtxD = merge_Z(embeddings_ZxtxD, embedding_ZxD)

            # ===== compute Felsenstein likelihoods =====

            Q_matrix_ZxSxAxA = self.q_matrix_decoder.Q_matrix_VxSxAxA(
                embedding_ZxD, site_positions_SxC
            )

            log_felsensteins_ZxSxA = compute_log_felsenstein_likelihoods_KxSxA(
                Q_matrix_ZxSxAxA,
                gather_K(log_felsensteins_ZxtxSxA, idx1_Z),
                gather_K(log_felsensteins_ZxtxSxA, idx2_Z),
                branch1_Z,
                branch2_Z,
            )
            log_felsensteins_ZxtxSxA = merge_Z(
                log_felsensteins_ZxtxSxA, log_felsensteins_ZxSxA
            )

            # ===== compute new likelihood and pi values =====

            def compute_log_stat_probs_ZxtxSxA():
                t = embeddings_ZxtxD.shape[1]

                # flatten embeddings to compute stat_probs, then reshape back
                embeddings_ZtxD = embeddings_ZxtxD.reshape(Z * t, D)
                stat_probs_ZtxSxA = self.q_matrix_decoder.stat_probs_VxSxA(
                    embeddings_ZtxD, site_positions_SxC
                )
                log_stat_probs_ZtxSxA = stat_probs_ZtxSxA.log()

                Zt = stat_probs_ZtxSxA.shape[0]  # broadcasting is possible (Zt=1)
                S = stat_probs_ZtxSxA.shape[1]  # broadcasting is possible (S=1)

                # handle special case of stat_probs_VxSxA() broadcasting along
                # the batch dimension
                if Zt == 1:
                    return log_stat_probs_ZtxSxA.reshape(1, 1, S, A)
                else:
                    return log_stat_probs_ZtxSxA.reshape(Z, -1, S, A)

            prev_log_pi_Z = log_pi_Z
            log_stat_probs_ZxtxSxA = compute_log_stat_probs_ZxtxSxA()

            log_likelihood_Z, log_pi_Z = compute_log_likelihood_and_pi_K(
                branch1_lengths_Zxr,
                branch2_lengths_Zxr,
                leaf_counts_Zxt,
                log_felsensteins_ZxtxSxA,
                log_stat_probs_ZxtxSxA,
                self.prior_dist,
                self.prior_branch_len,
                self.log_double_factorials_2N,
            )

            # ===== compute sub-particle weights =====

            # compute over-counting correction
            log_v_minus_Z = compute_log_v_minus_K(N, leaf_counts_Zxt)

            # equation (7) in the VCSMC paper
            log_weight_Z = log_pi_Z - prev_log_pi_Z + log_v_minus_Z - log_v_plus_Z

            # ===== compute particle weights =====

            log_weight_KxJ = unsqueeze_KJ(log_weight_Z).reshape(K, J)

            # for each initial particle, average over sub-particle weights
            log_weight_K = torch.logsumexp(log_weight_KxJ, 1)
            # divide by J
            log_weight_K = log_weight_K - math.log(J)

            log_weights_list_rxK.append(log_weight_K)

            #  ===== sample particles from sub-particles =====

            if J > 1:
                # distr has K batches, with J sub-particle weights per batch
                sub_resample_distr_K = torch.distributions.Categorical(
                    logits=log_weight_KxJ
                )
                sub_indexes_K = sub_resample_distr_K.sample()
            else:
                sub_indexes_K = torch.zeros(K, dtype=torch.int, device=device)

            # ===== post sub-particle resampling bookkeeping =====

            # helper function
            def gather_sub_K(arr_Z: Tensor):
                arr_KJ = unsqueeze_KJ(arr_Z)
                arr_KxJ = arr_KJ.reshape((K, J) + arr_KJ.shape[1:])
                return gather_K(arr_KxJ, sub_indexes_K)

            # helper function
            def concat_sub_K(arr_Zxr, val_K):
                arr_Kxr = gather_sub_K(arr_Zxr)
                return concat_K(arr_Kxr, val_K)

            idx1_K = gather_sub_K(idx1_Z)
            idx2_K = gather_sub_K(idx2_Z)
            branch1_K = gather_sub_K(branch1_Z)
            branch2_K = gather_sub_K(branch2_Z)
            embedding_KxD = gather_sub_K(embedding_ZxD)
            merge1_indexes_Kxr = concat_sub_K(merge1_indexes_Zxr, idx1_K)
            merge2_indexes_Kxr = concat_sub_K(merge2_indexes_Zxr, idx2_K)
            branch1_lengths_Kxr = concat_sub_K(branch1_lengths_Zxr, branch1_K)
            branch2_lengths_Kxr = concat_sub_K(branch2_lengths_Zxr, branch2_K)
            embeddings_KxrxD = concat_sub_K(embeddings_ZxrxD, embedding_KxD)
            leaf_counts_Kxt = gather_sub_K(leaf_counts_Zxt)
            embeddings_KxtxD = gather_sub_K(embeddings_ZxtxD)
            log_felsensteins_KxtxSxA = gather_sub_K(log_felsensteins_ZxtxSxA)
            log_likelihood_K = gather_sub_K(log_likelihood_Z)
            log_pi_K = gather_sub_K(log_pi_Z)

        # ===== compute Z_SMC =====

        # Forms the estimator log_Z_SMC, a multi sample lower bound to the
        # likelihood. Z_SMC is formed by averaging over weights (across k) and
        # multiplying over coalescent events (across r).
        # See equation (8) in the VCSMC paper.

        log_weights_rxK = torch.stack(log_weights_list_rxK)
        log_scaled_weights_rxK = log_weights_rxK - math.log(K)
        log_sum_weights_r = torch.logsumexp(log_scaled_weights_rxK, 1)
        log_Z_SMC = torch.sum(log_sum_weights_r)

        # ===== build best Newick tree =====

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
            "best_embeddings_N1xD": embeddings_KxrxD[best_tree_idx],
        }
