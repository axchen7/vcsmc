from typing import TypedDict

from torch import Tensor

__all__ = ["VcsmcResult"]


class VcsmcResult(TypedDict):
    log_ZCSMC: Tensor
    log_likelihood_K: Tensor
    merge_indexes_KxN1x2: Tensor
    """left/right node indexes at each step, for all particles"""
    branch_lengths_KxN1x2: Tensor
    """left/right branch lengths at each step, for all particles"""
    best_newick_tree: str
    best_merge_indexes_N1x2: Tensor
    """left/right node indexes at each step"""
    best_embeddings_N1xD: Tensor
    """merged embedding at each step"""
