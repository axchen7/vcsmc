# import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import cast

import torch
from torch import Tensor
from torch.nn import functional as F

from ..proposals import EmbeddingProposal
from .vcsmc_types import VcsmcResult
from .vcsmc_utils import replace_with_merged_list


@dataclass
class _Node:
    leaf_index: int | None = None  # only for leaf nodes
    edges: list["_OutgoingEdge"] = field(default_factory=list)


@dataclass
class _OutgoingEdge:
    destination: "_Node"
    length: Tensor  # scalar


def _join_nodes(node1: _Node, node2: _Node, distance: Tensor):
    node1.edges.append(_OutgoingEdge(destination=node2, length=distance))
    node2.edges.append(_OutgoingEdge(destination=node1, length=distance))


def _merge_nodes(
    child1: _Node, branch1_length: Tensor, child2: _Node, branch2_length: Tensor
) -> _Node:
    parent = _Node()
    _join_nodes(parent, child1, branch1_length)
    _join_nodes(parent, child2, branch2_length)
    return parent


def _build_tree(
    merge1_indexes_N1: Tensor,
    merge2_indexes_N1: Tensor,
    branch1_lengths_N1: Tensor,
    branch2_lengths_N1: Tensor,
) -> list[_Node]:
    N = merge1_indexes_N1.shape[0] + 1
    nodes = [_Node(leaf_index=i) for i in range(N)]
    leaf_nodes = nodes

    for r in range(N - 1):
        idx1 = int(merge1_indexes_N1[r])
        idx2 = int(merge2_indexes_N1[r])

        parent = _merge_nodes(
            nodes[idx1], branch1_lengths_N1[r], nodes[idx2], branch2_lengths_N1[r]
        )

        nodes = replace_with_merged_list(nodes, idx1, idx2, parent)

    return leaf_nodes


def _compute_tree_distance_matrix_NxN(
    merge1_indexes_N1: Tensor,
    merge2_indexes_N1: Tensor,
    branch1_lengths_N1: Tensor,
    branch2_lengths_N1: Tensor,
) -> Tensor:
    device = branch1_lengths_N1.device
    N = merge1_indexes_N1.shape[0] + 1

    leaf_nodes = _build_tree(
        merge1_indexes_N1,
        merge2_indexes_N1,
        branch1_lengths_N1,
        branch2_lengths_N1,
    )

    rows_NxN: list[Tensor] = []  # list of shape (N,) row tensors

    for i in range(N):
        row_N: list[Tensor | None] = [None] * N

        def traverse(node: _Node, distance_from_start: Tensor, from_node: _Node | None):
            if node.leaf_index is not None:
                row_N[node.leaf_index] = distance_from_start

            for edge in node.edges:
                if edge.destination != from_node:
                    traverse(edge.destination, distance_from_start + edge.length, node)

        traverse(leaf_nodes[i], torch.tensor(0.0, device=device), None)

        assert all(x is not None for x in row_N)
        rows_NxN.append(torch.stack(cast(list[Tensor], row_N)))

    return torch.stack(rows_NxN)


def _compute_avg_tree_distance_matrix_NxN(
    merge1_indexes_KxN1: Tensor,
    merge2_indexes_KxN1: Tensor,
    branch1_lengths_KxN1: Tensor,
    branch2_lengths_KxN1: Tensor,
) -> Tensor:
    device = branch1_lengths_KxN1.device
    K = merge1_indexes_KxN1.shape[0]
    N = merge1_indexes_KxN1.shape[1] + 1

    sum_matrix_NxN = torch.zeros(N, N, device=device)

    for k in range(K):
        sum_matrix_NxN = sum_matrix_NxN + _compute_tree_distance_matrix_NxN(
            merge1_indexes_KxN1[k],
            merge2_indexes_KxN1[k],
            branch1_lengths_KxN1[k],
            branch2_lengths_KxN1[k],
        )

    return sum_matrix_NxN / K


def _compute_direct_distance_matrix_NxN(
    proposal: EmbeddingProposal, data_NxSxA: Tensor
) -> Tensor:
    N = data_NxSxA.shape[0]

    embeddings_NxD = proposal.seq_encoder(data_NxSxA)
    rows_NxN: list[Tensor] = []  # list of shape (N,) row tensors

    for i in range(N):
        row_N = proposal.distance(embeddings_NxD[i].unsqueeze(0), embeddings_NxD)
        rows_NxN.append(row_N)

    return torch.stack(rows_NxN)


def compute_embedding_distance_mse(
    proposal: EmbeddingProposal, data_NxSxA: Tensor, result: VcsmcResult
) -> Tensor:
    direct_distance_matrix_NxN = _compute_direct_distance_matrix_NxN(
        proposal, data_NxSxA
    )

    tree_distance_matrix_NxN = _compute_avg_tree_distance_matrix_NxN(
        merge1_indexes_KxN1=result["merge_indexes_KxN1x2"][:, :, 0],
        merge2_indexes_KxN1=result["merge_indexes_KxN1x2"][:, :, 1],
        branch1_lengths_KxN1=result["branch_lengths_KxN1x2"][:, :, 0],
        branch2_lengths_KxN1=result["branch_lengths_KxN1x2"][:, :, 1],
    )

    return F.mse_loss(direct_distance_matrix_NxN, tree_distance_matrix_NxN)
