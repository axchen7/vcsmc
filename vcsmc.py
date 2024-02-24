import tensorflow as tf

from constants import DTYPE_FLOAT
from encoder_decoder import Decoder
from proposal import Proposal
from q_matrix import QMatrix
from type_utils import Tensor, tf_function
from vcsmc_utils import (
    build_newick_tree,
    compute_felsenstein_likelihoods_KxSxA,
    compute_log_double_factorials_2N,
    compute_log_likelihood_and_pi_K,
    concat_K,
    gather_K,
    replace_with_merged,
)


class VCSMC(tf.Module):
    def __init__(
        self,
        q_matrix: QMatrix,
        proposal: Proposal,
        decoder: Decoder,
        taxa_N: Tensor,
        *,
        K: int,
        prior_branch_len: float = 1.0,
    ):
        """
        Args:
            q_matrix: QMatrix object
            proposal: Proposal object
            taxa_N: Tensor of taxa names
            K: Number of particles
            prior_branch_len: Expected branch length under the prior
        """

        super().__init__()

        self.q_matrix = q_matrix
        self.proposal = proposal
        self.decoder = decoder
        self.taxa_N = taxa_N
        self.K = K
        self.prior_branch_len = tf.constant(prior_branch_len, DTYPE_FLOAT)

    @tf_function()
    def get_init_embeddings_KxNxD(self, data_NxSxA):
        """Sets the embedding for all K particles to the same initial value."""
        embeddings_NxD = self.proposal.seq_encoder(data_NxSxA)
        return tf.repeat(embeddings_NxD[tf.newaxis], self.K, axis=0)  # type: ignore

    @tf_function()
    def __call__(self, data_NxSxA: Tensor) -> Tensor:
        """
        Returns a dict containing:
            log_Z_SMC: lower bound to the likelihood; should set cost = -log_Z_SMC
            log_likelihood_K: log likelihoods for each particle at the last merge step
            best_newick_tree: Newick tree with the highest likelihood
            best_merge1_indexes_r: left node merge indexes for the best tree
            best_merge2_indexes_r: right node merge indexes for the best tree
            best_branch1_lengths_r: left branch lengths for the best tree
            best_branch2_lengths_r: right branch lengths for the best tree
        """

        N, S, A = data_NxSxA.shape
        K = self.K

        log_double_factorials_2N = compute_log_double_factorials_2N(N)
        Q = self.q_matrix()

        # at each step r, there are t = N-r >= 2 trees in the forest.
        # initially, r = 0 and t = N

        # for tracking tree topologies
        merge1_indexes_Kxr = tf.zeros([K, 0], dtype=tf.int32)
        merge2_indexes_Kxr = tf.zeros([K, 0], dtype=tf.int32)
        branch1_lengths_Kxr = tf.zeros([K, 0], dtype=DTYPE_FLOAT)
        branch2_lengths_Kxr = tf.zeros([K, 0], dtype=DTYPE_FLOAT)

        leaf_counts_Kxt = tf.ones([K, N], dtype=tf.int32)
        embeddings_KxtxD = self.get_init_embeddings_KxNxD(data_NxSxA)
        # Felsenstein probabilities for computing pi(s)
        felsensteins_KxtxSxA = tf.repeat(data_NxSxA[tf.newaxis], K, axis=0)

        # difference of current and last iteration's values are used to compute weights
        log_pi_K = tf.zeros(K, DTYPE_FLOAT)
        # for computing empirical measure pi_rk(s)
        log_weight_K = tf.zeros(K, DTYPE_FLOAT)

        # must record all weights to compute Z_SMC
        log_weights_rxK = tf.TensorArray(DTYPE_FLOAT, N - 1)

        # for displaying at the end
        log_likelihood_K = tf.zeros(K, DTYPE_FLOAT)  # initial value isn't used

        # for setting shape_invariants
        D = embeddings_KxtxD.shape[2]

        # iterate over merge steps
        for r in tf.range(N - 1):
            tf.autograph.experimental.set_loop_options(
                shape_invariants=[
                    (merge1_indexes_Kxr, tf.TensorShape([K, None])),
                    (merge2_indexes_Kxr, tf.TensorShape([K, None])),
                    (branch1_lengths_Kxr, tf.TensorShape([K, None])),
                    (branch2_lengths_Kxr, tf.TensorShape([K, None])),
                    (leaf_counts_Kxt, tf.TensorShape([K, None])),
                    (embeddings_KxtxD, tf.TensorShape([K, None, D])),
                    (felsensteins_KxtxSxA, tf.TensorShape([K, None, S, A])),
                ]
            )

            # ===== resample =====

            indexes = tf.random.categorical([log_weight_K], K)
            indexes = tf.squeeze(indexes)

            merge1_indexes_Kxr = tf.gather(merge1_indexes_Kxr, indexes)
            merge2_indexes_Kxr = tf.gather(merge2_indexes_Kxr, indexes)
            branch1_lengths_Kxr = tf.gather(branch1_lengths_Kxr, indexes)
            branch2_lengths_Kxr = tf.gather(branch2_lengths_Kxr, indexes)
            leaf_counts_Kxt = tf.gather(leaf_counts_Kxt, indexes)
            embeddings_KxtxD = tf.gather(embeddings_KxtxD, indexes)
            felsensteins_KxtxSxA = tf.gather(felsensteins_KxtxSxA, indexes)
            log_pi_K = tf.gather(log_pi_K, indexes)
            log_weight_K = tf.gather(log_weight_K, indexes)

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
            ) = self.proposal(N, r, leaf_counts_Kxt, embeddings_KxtxD)

            # helper function
            def merge_K(arr_K, new_val_K):
                # TODO optimize
                return tf.map_fn(
                    lambda args: replace_with_merged(
                        args[0], idx1_K[args[2]], idx2_K[args[2]], args[1]
                    ),
                    (arr_K, new_val_K, tf.range(K)),
                    fn_output_signature=arr_K.dtype,
                )

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
            felsensteins_KxtxSxA = merge_K(
                felsensteins_KxtxSxA,
                compute_felsenstein_likelihoods_KxSxA(
                    Q,
                    gather_K(felsensteins_KxtxSxA, idx1_K),
                    gather_K(felsensteins_KxtxSxA, idx2_K),
                    branch1_K,
                    branch2_K,
                ),
            )

            # ===== compute new likelihood, pi, and weight =====

            prev_log_pi_K = log_pi_K

            decoded_embeddings_KxtxSxA = tf.vectorized_map(
                self.decoder, embeddings_KxtxD
            )

            log_likelihood_K, log_pi_K = compute_log_likelihood_and_pi_K(
                branch1_lengths_Kxr,
                branch2_lengths_Kxr,
                leaf_counts_Kxt,
                felsensteins_KxtxSxA,
                decoded_embeddings_KxtxSxA,
                self.prior_branch_len,
                log_double_factorials_2N,
            )

            # equation (7) in the VCSMC paper
            log_weight_K = log_pi_K - prev_log_pi_K + log_v_minus_K - log_v_plus_K

            log_weights_rxK = log_weights_rxK.write(r, log_weight_K)

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

        best_tree_idx = tf.math.argmax(log_likelihood_K)
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
