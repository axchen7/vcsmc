import tensorflow as tf

from constants import DTYPE_FLOAT
from type_utils import Tensor, tf_function


class Proposal(tf.Module):
    """
    Proposal distribution for selecting two nodes to merge and sampling branch.
    """

    @tf.function
    def embed(self, leaf_SxA: Tensor) -> Tensor:
        """
        Embeds a leaf node into the latent space.

        Args:
            leaf_SxA: The leaf node.
        Returns:
            embedding_D: The embedding of the leaf node.
        """

        return tf.zeros([1], DTYPE_FLOAT)

    def __call__(
        self, r: Tensor, leaf_counts_R: Tensor, embeddings_RxD: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Propose two nodes to merge, as well as their branch lengths.

        Args:
            r: The current merge step (0 <= r <= N-2).
            leaf_counts_R: The number of leaf nodes in each subtree.
            embeddings_RxD: Embeddings of each subtree.
        Returns:
            idx1: Indices of the first node to merge.
            idx2: Indices of the second node to merge.
            branch1: Branch lengths of the first node.
            branch2: Branch lengths of the second node.
            log_v_plus: Log probability of the returned proposal.
            log_v_minus: Log of the over-counting correction factor.
        Note:
            Along dimension R, there are N elements, but only the first N-r >= 2
            elements are used.
        """

        raise NotImplementedError
