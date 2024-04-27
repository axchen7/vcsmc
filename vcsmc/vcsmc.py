from typing import Literal, TypedDict

import torch
from torch import Tensor, nn

from .proposals import Proposal
from .q_matrix_decoders import QMatrixDecoder
from .vcsmc_regularization import VcsmcRegularization
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
    loss: Tensor
    regularization: Tensor  # proportional to S, the number of sites in the batch
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
        # assume Exp(10) branch length prior
        prior_dist: Literal["gamma", "exp", "unif"] = "exp",
        prior_branch_len: float = 0.1,
        regularization: VcsmcRegularization | None = None,
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
        self.prior_dist: Literal["gamma", "exp", "unif"] = prior_dist
        self.prior_branch_len = prior_branch_len
        self.regularization = regularization

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

        # S = number of sites in the batch
        N, S, A = data_batched_NxSxA.shape

        K = self.K
        D = self.proposal.seq_encoder.D

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
        embeddings_KxrxD = torch.zeros(K, 0, D)  # merged embedding at each step
        branch1_child_KxrxD = torch.zeros(K, 0, D)
        branch1_parent_KxrxD = torch.zeros(K, 0, D)
        branch2_child_KxrxD = torch.zeros(K, 0, D)
        branch2_parent_KxrxD = torch.zeros(K, 0, D)

        leaf_counts_Kxt = torch.ones(K, N, dtype=torch.int)
        # embeddings of each tree currently in the forest
        embeddings_KxtxD = self.get_init_embeddings_KxNxD(data_NxSxA)

        # Felsenstein probabilities for computing pi(s)
        log_felsensteins_KxtxSxA = data_batched_NxSxA.log().repeat(K, 1, 1, 1)

        # difference of current and last iteration's values are used to compute weights
        log_pi_K = torch.zeros(K)
        # for computing empirical measure pi_rk(s); initialize to log(1/K)
        log_weight_K = -torch.log(torch.tensor(K)).repeat(K)

        # must record all weights to compute Z_SMC
        log_weights_list_rxK: list[Tensor] = []

        # for returning at the end
        log_likelihood_K = torch.zeros(K)  # initial value isn't used

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
            branch1_child_KxrxD = branch1_child_KxrxD[indexes_K]
            branch1_parent_KxrxD = branch1_parent_KxrxD[indexes_K]
            branch2_child_KxrxD = branch2_child_KxrxD[indexes_K]
            branch2_parent_KxrxD = branch2_parent_KxrxD[indexes_K]
            leaf_counts_Kxt = leaf_counts_Kxt[indexes_K]
            embeddings_KxtxD = embeddings_KxtxD[indexes_K]
            log_felsensteins_KxtxSxA = log_felsensteins_KxtxSxA[indexes_K]
            log_pi_K = log_pi_K[indexes_K]
            log_weight_K = log_weight_K[indexes_K]

            # ===== extend partial states using proposal =====

            # sample from proposal distribution
            (
                idx1_KxJ,
                idx2_KxJ,
                branch1_KxJ,
                branch2_KxJ,
                embedding_KxJxD,
                log_v_plus_KxJ,
            ) = self.proposal(
                N,
                leaf_counts_Kxt,
                embeddings_KxtxD,
                log_felsensteins_KxtxSxA,
                site_positions_SxC,
            )

            # ===== deal with sub-particle (J) dimension =====

            J = idx1_KxJ.shape[1]

            idx1_KJ = idx1_KxJ.flatten()
            idx2_KJ = idx2_KxJ.flatten()
            branch1_KJ = branch1_KxJ.flatten()
            branch2_KJ = branch2_KxJ.flatten()
            embedding_KJxD = embedding_KxJxD.reshape(K * J, D)
            log_v_plus_KJ = log_v_plus_KxJ.flatten()

            branch1_lengths_KJxr = branch1_lengths_Kxr.repeat_interleave(J, 0)
            branch2_lengths_KJxr = branch2_lengths_Kxr.repeat_interleave(J, 0)
            embeddings_KJxrxD = embeddings_KxrxD.repeat_interleave(J, 0)
            leaf_counts_KJxt = leaf_counts_Kxt.repeat_interleave(J, 0)
            embeddings_KJxtxD = embeddings_KxtxD.repeat_interleave(J, 0)
            log_felsensteins_KJxtxSxA = log_felsensteins_KxtxSxA.repeat_interleave(J, 0)
            log_pi_KJ = log_pi_K.repeat_interleave(J, 0)

            # ===== post-proposal bookkeeping =====

            # helper function
            def merge_KJ(arr_KJ: Tensor, new_val_KJ: Tensor):
                return replace_with_merged_K(arr_KJ, idx1_KJ, idx2_KJ, new_val_KJ)

            branch1_lengths_KJxr = concat_K(branch1_lengths_KJxr, branch1_KJ)
            branch2_lengths_KJxr = concat_K(branch2_lengths_KJxr, branch2_KJ)
            embeddings_KJxrxD = concat_K(embeddings_KJxrxD, embedding_KJxD)

            leaf_counts_KJxt = merge_KJ(
                leaf_counts_KJxt,
                gather_K(leaf_counts_KJxt, idx1_KJ)
                + gather_K(leaf_counts_KJxt, idx2_KJ),
            )
            embeddings_KJxtxD = merge_KJ(embeddings_KJxtxD, embedding_KJxD)

            # ===== compute Felsenstein likelihoods =====

            Q_matrix_KJxSxAxA = self.q_matrix_decoder.Q_matrix_VxSxAxA(
                embedding_KJxD, site_positions_SxC
            )

            log_felsensteins_KJxSxA = compute_log_felsenstein_likelihoods_KxSxA(
                Q_matrix_KJxSxAxA,
                gather_K(log_felsensteins_KJxtxSxA, idx1_KJ),
                gather_K(log_felsensteins_KJxtxSxA, idx2_KJ),
                branch1_KJ,
                branch2_KJ,
            )
            log_felsensteins_KJxtxSxA = merge_KJ(
                log_felsensteins_KJxtxSxA, log_felsensteins_KJxSxA
            )

            # ===== compute new likelihood and pi values =====

            def compute_log_stat_probs_KJxtxSxA():
                t = embeddings_KxtxD.shape[1]

                # flatten embeddings to compute stat_probs, then reshape back
                embeddings_KJtxD = embeddings_KJxtxD.reshape(K * J * t, D)
                stat_probs_KJtxSxA = self.q_matrix_decoder.stat_probs_VxSxA(
                    embeddings_KJtxD, site_positions_SxC
                )
                log_stat_probs_KJtxSxA = stat_probs_KJtxSxA.log()

                KJt = stat_probs_KJtxSxA.shape[0]  # broadcasting is possible (KJt=1)
                S = stat_probs_KJtxSxA.shape[1]  # broadcasting is possible (S=1)

                # handle special case of stat_probs_VxSxA() broadcasting along
                # the batch dimension
                if KJt == 1:
                    return log_stat_probs_KJtxSxA.reshape(1, 1, S, A)
                else:
                    return log_stat_probs_KJtxSxA.reshape(K * J, -1, S, A)

            prev_log_pi_KJ = log_pi_KJ
            log_stat_probs_KJxtxSxA = compute_log_stat_probs_KJxtxSxA()

            log_likelihood_KJ, log_pi_KJ = compute_log_likelihood_and_pi_K(
                branch1_lengths_KJxr,
                branch2_lengths_KJxr,
                leaf_counts_KJxt,
                log_felsensteins_KJxtxSxA,
                log_stat_probs_KJxtxSxA,
                self.prior_dist,
                self.prior_branch_len,
                self.log_double_factorials_2N,
            )

            # ===== compute weights =====

            # compute over-counting correction
            log_v_minus_KJ = compute_log_v_minus_K(N, leaf_counts_KJxt)

            # equation (7) in the VCSMC paper
            log_weight_KJ = log_pi_KJ - prev_log_pi_KJ + log_v_minus_KJ - log_v_plus_KJ

            # for each initial particle, average over sub-particle weights
            log_weight_KxJ = log_weight_KJ.reshape(K, J)
            log_weight_K = torch.logsumexp(log_weight_KxJ, 1)
            log_weight_K = log_weight_K - torch.log(torch.tensor(J))  # divide by J

            log_weights_list_rxK.append(log_weight_K)

            #  ===== sample particles from sub-particles =====

            if J > 1:
                # distr has K batches, with J sub-particle weights per batch
                sub_resample_distr_K = torch.distributions.Categorical(
                    logits=log_weight_KxJ
                )
                sub_indexes_K = sub_resample_distr_K.sample()
            else:
                sub_indexes_K = torch.zeros(K, dtype=torch.int)

            idx1_K = gather_K(idx1_KxJ, sub_indexes_K)
            idx2_K = gather_K(idx2_KxJ, sub_indexes_K)
            branch1_K = gather_K(branch1_KxJ, sub_indexes_K)
            branch2_K = gather_K(branch2_KxJ, sub_indexes_K)
            embedding_KxD = gather_K(embedding_KxJxD, sub_indexes_K)

            log_felsensteins_KxJxtxSxA = log_felsensteins_KJxtxSxA.reshape(
                K, J, -1, S, A
            )
            log_likelihood_KxJ = log_likelihood_KJ.reshape(K, J)
            log_pi_KxJ = log_pi_KJ.reshape(K, J)

            log_felsensteins_KxtxSxA = gather_K(
                log_felsensteins_KxJxtxSxA, sub_indexes_K
            )
            log_likelihood_K = gather_K(log_likelihood_KxJ, sub_indexes_K)
            log_pi_K = gather_K(log_pi_KxJ, sub_indexes_K)

            # ===== post sub-particle resampling bookkeeping =====

            # helper function
            def merge_K(arr_K: Tensor, new_val_K: Tensor):
                return replace_with_merged_K(arr_K, idx1_K, idx2_K, new_val_K)

            merge1_indexes_Kxr = concat_K(merge1_indexes_Kxr, idx1_K)
            merge2_indexes_Kxr = concat_K(merge2_indexes_Kxr, idx2_K)
            branch1_lengths_Kxr = concat_K(branch1_lengths_Kxr, branch1_K)
            branch2_lengths_Kxr = concat_K(branch2_lengths_Kxr, branch2_K)
            embeddings_KxrxD = concat_K(embeddings_KxrxD, embedding_KxD)

            branch1_child_KxrxD = concat_K(
                branch1_child_KxrxD, gather_K(embeddings_KxtxD, idx1_K)
            )
            branch1_parent_KxrxD = concat_K(branch1_parent_KxrxD, embedding_KxD)
            branch2_child_KxrxD = concat_K(
                branch2_child_KxrxD, gather_K(embeddings_KxtxD, idx2_K)
            )
            branch2_parent_KxrxD = concat_K(branch2_parent_KxrxD, embedding_KxD)

            leaf_counts_Kxt = merge_K(
                leaf_counts_Kxt,
                gather_K(leaf_counts_Kxt, idx1_K) + gather_K(leaf_counts_Kxt, idx2_K),
            )
            embeddings_KxtxD = merge_K(embeddings_KxtxD, embedding_KxD)

        # ===== compute Z_SMC =====

        # Forms the estimator log_Z_SMC, a multi sample lower bound to the
        # likelihood. Z_SMC is formed by averaging over weights (across k) and
        # multiplying over coalescent events (across r).
        # See equation (8) in the VCSMC paper.

        log_weights_rxK = torch.stack(log_weights_list_rxK)
        log_scaled_weights_rxK = log_weights_rxK - torch.log(torch.tensor(K))
        log_sum_weights_r = torch.logsumexp(log_scaled_weights_rxK, 1)
        log_Z_SMC = torch.sum(log_sum_weights_r)

        # ===== compute regularization and loss =====

        regularization = torch.tensor(0.0)

        if self.regularization:
            regularization = self.regularization(
                S,
                branch1_child_KxrxD,
                branch1_parent_KxrxD,
                branch2_child_KxrxD,
                branch2_parent_KxrxD,
            )
            regularization = regularization * self.regularization.coeff

        loss = -log_Z_SMC + regularization

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
            "loss": loss,
            "regularization": regularization,
            "log_Z_SMC": log_Z_SMC,
            "log_likelihood_K": log_likelihood_K,
            "best_newick_tree": best_newick_tree,
            "best_merge1_indexes_N1": merge1_indexes_Kxr[best_tree_idx],
            "best_merge2_indexes_N1": merge2_indexes_Kxr[best_tree_idx],
            "best_branch1_lengths_N1": branch1_lengths_Kxr[best_tree_idx],
            "best_branch2_lengths_N1": branch2_lengths_Kxr[best_tree_idx],
            "best_embeddings_N1xD": embeddings_KxrxD[best_tree_idx],
        }
