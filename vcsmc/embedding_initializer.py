import torch
from torch import Tensor, nn
from torch.optim.adam import Adam
from tqdm import tqdm

import wandb

from .distances import Hyperbolic
from .encoders import EmbeddingTableSequenceEncoder
from .utils.wandb_utils import WANDB_PROJECT, WandbRunType

__all__ = ["SequenceDistanceEmbeddingInitializer"]


class SequenceDistanceEmbeddingInitializer(nn.Module):
    """
    Given the raw sequences, forms a distance matrix of hamming distances. Then,
    optimizes the embeddings by minimizing the difference between the sequence
    distances and distances in the embedding space.
    """

    def __init__(
        self,
        seq_encoder: EmbeddingTableSequenceEncoder,
        distance: Hyperbolic,
    ):
        super().__init__()

        self.seq_encoder = seq_encoder
        self.distance = distance

    def compute_hamming_distance_matrix_NxN(self, data_NxSxA: Tensor) -> Tensor:
        """Returns a tensor on the CPU"""

        N = data_NxSxA.shape[0]
        S = data_NxSxA.shape[1]

        distance_matrix_NxN = torch.zeros(N, N, device=data_NxSxA.device)

        for i in range(N):
            for j in range(N):
                seq1_SxA = data_NxSxA[i]
                seq2_SxA = data_NxSxA[j]

                # sequences match at position s if seq1[s] == seq2[s] (across the entire A dimension)
                matches_SxA = seq1_SxA == seq2_SxA
                matches_S = matches_SxA.all(dim=-1)

                hamming_dist = S - matches_S.sum()

                # normalize distance to be in [0, 1]
                distance_matrix_NxN[i, j] = hamming_dist / S

        return distance_matrix_NxN

    def compute_embedding_distance_matrix_NxN(self, data_NxSxA: Tensor) -> Tensor:
        N = data_NxSxA.shape[0]

        embeddings_NxD: Tensor = self.seq_encoder(data_NxSxA)
        # repeat like 123123123...
        embeddings1_NNxD = embeddings_NxD.repeat(N, 1).view(N * N, -1)
        # repeat like 111222333...
        embeddings2_NNxD = embeddings_NxD.repeat(1, N).view(N * N, -1)

        distances_NN: Tensor = self.distance(embeddings1_NNxD, embeddings2_NNxD)
        distance_matrix_NxN = distances_NN.view(N, N)

        return distance_matrix_NxN

    def forward(self, data_NxSxA: Tensor) -> Tensor:
        """Returns the loss"""
        hamming_dist_NxN = self.compute_hamming_distance_matrix_NxN(data_NxSxA)
        embedding_dist_NxN = self.compute_embedding_distance_matrix_NxN(data_NxSxA)
        # TODO use more sophisticated loss? (see sync notes)
        return nn.functional.mse_loss(hamming_dist_NxN, embedding_dist_NxN)

    def fit(self, data_NxSxA: Tensor, *, epochs: int, lr: float):
        optimizer = Adam(self.parameters(), lr=lr)

        run = wandb.init(
            project=WANDB_PROJECT, job_type=WandbRunType.FIT_INITIAL_EMBEDDINGS
        )

        for epoch in tqdm(range(epochs), desc="Fitting initial embeddings"):
            loss = self(data_NxSxA)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            run.log(
                {
                    "Loss": loss,
                    "Hyperbolic scale": self.distance.scale(),
                },
                step=epoch,
            )

        run.finish()
