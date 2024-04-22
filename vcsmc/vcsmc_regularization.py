import torch
from torch import Tensor, nn

from .distance_utils import EPSILON


class VcsmcRegularization(nn.Module):
    def __init__(self, *, coeff: float):
        """
        Args:
            coeff: Coefficient for the regularization term.
        """

        super().__init__()
        self.coeff = coeff

    def forward(
        self,
        S: int,
        branch1_child_KxN1xD: Tensor,
        branch1_parent_KxN1xD: Tensor,
        branch2_child_KxN1xD: Tensor,
        branch2_parent_KxN1xD: Tensor,
    ) -> Tensor:
        """
        Returns the regularization term, NOT multiplied by the coefficient.
        Regularization should be proportional to S, the number of sites.
        """
        raise NotImplementedError


class CrossingPenaltyVcsmcRegularization(VcsmcRegularization):
    """
    Penalizes branches for crossing over each other. The closer the crossing
    point is to the parent end of the branches, the higher the penalty.
    """

    def __init__(self, *, coeff: float, D: int):
        """
        Args:
            coeff: Coefficient for the regularization term.
            D: Number of dimensions sequence embeddings (must be 2).
        """

        super().__init__(coeff=coeff)
        assert D == 2

    def crossing_penalty(
        self,
        branch1_child_Vx2: Tensor,
        branch1_parent_Vx2: Tensor,
        branch2_child_Vx2: Tensor,
        branch2_parent_Vx2: Tensor,
    ) -> Tensor:
        # all tensors are of shape (V,)

        x1 = branch1_child_Vx2[:, 0]
        y1 = branch1_child_Vx2[:, 1]
        x2 = branch1_parent_Vx2[:, 0]
        y2 = branch1_parent_Vx2[:, 1]
        x3 = branch2_child_Vx2[:, 0]
        y3 = branch2_child_Vx2[:, 1]
        x4 = branch2_parent_Vx2[:, 0]
        y4 = branch2_parent_Vx2[:, 1]

        A = x2 - x1
        B = x4 - x3
        C = y2 - y1
        D = y4 - y3
        E = x3 - x1
        F = y3 - y1

        denom = A * D - B * C
        parallel = denom.abs() < EPSILON

        # avoid nan
        denom = torch.where(parallel, torch.ones_like(denom), denom)

        alpha = (D * E - B * F) / denom
        beta = (C * E - A * F) / denom

        non_crossing = parallel
        non_crossing = torch.logical_or(non_crossing, alpha < 0)
        non_crossing = torch.logical_or(non_crossing, alpha > 1)
        non_crossing = torch.logical_or(non_crossing, beta < 0)
        non_crossing = torch.logical_or(non_crossing, beta > 1)

        penalty = torch.where(
            non_crossing,
            torch.zeros_like(alpha),
            torch.min(alpha, beta),
        )

        return penalty

    def compute_avg_tree_crossing_penalty(
        self,
        branch1_child_KxN1xD: Tensor,
        branch1_parent_KxN1xD: Tensor,
        branch2_child_KxN1xD: Tensor,
        branch2_parent_KxN1xD: Tensor,
    ) -> Tensor:
        # N1 = N-1
        K, N1, D = branch1_child_KxN1xD.shape
        V = 2 * N1

        assert D == 2  # crossing penalty only works in 2D

        # all branches
        branch_child_KxVxD = torch.cat([branch1_child_KxN1xD, branch2_child_KxN1xD], 1)
        branch_parent_KxVxD = torch.cat(
            [branch1_parent_KxN1xD, branch2_parent_KxN1xD], 1
        )

        # compute pairwise crossing penalties

        KVV = K * V * V  # for brevity
        # repeat like 123123123...
        branch_child_A_KVVxD = branch_child_KxVxD.repeat(1, V, 1).view(KVV, D)
        branch_parent_A_KVVxD = branch_parent_KxVxD.repeat(1, V, 1).view(KVV, D)
        # repeat like 111222333...
        branch_child_B_KVVxD = branch_child_KxVxD.repeat(1, 1, V).view(KVV, D)
        branch_parent_B_KVVxD = branch_parent_KxVxD.repeat(1, 1, V).view(KVV, D)

        penalties_KVV = self.crossing_penalty(
            branch_child_A_KVVxD,
            branch_parent_A_KVVxD,
            branch_child_B_KVVxD,
            branch_parent_B_KVVxD,
        )

        penalties_K_V_V = penalties_KVV.view(K, V, V)

        # zero out penalties for self-comparisons (e.g. diagonal)
        penalties_K_V_V[:, torch.arange(V), torch.arange(V)] = 0

        # add penalties across pairwise comparisons
        penalties_K = penalties_K_V_V.sum([1, 2])

        # average across particles
        avg_penalty = penalties_K.mean()
        return avg_penalty

    def forward(
        self,
        S: int,
        branch1_child_KxN1xD: Tensor,
        branch1_parent_KxN1xD: Tensor,
        branch2_child_KxN1xD: Tensor,
        branch2_parent_KxN1xD: Tensor,
    ):
        avg_crossing_penalty = self.compute_avg_tree_crossing_penalty(
            branch1_child_KxN1xD,
            branch1_parent_KxN1xD,
            branch2_child_KxN1xD,
            branch2_parent_KxN1xD,
        )
        return avg_crossing_penalty * S
