import typer
from torch.optim.adam import Adam

from vcsmc import *

INITIAL_MEANS = [0.0, 0.3, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
INITIAL_STDS = [0.05, 0.1, 0.2]


def embedding_initialization(
    file: str = "data/hohna/DS1.phy",
    lr: float = 0.01,
    epochs: int = 200,
    K: int = 512,
    D: int = 2,
):
    device = detect_device()

    N, _S, A, data_NxSxA, taxa_N = load_phy(file, A4_ALPHABET)
    data_NxSxA = data_NxSxA.to(device)

    for initial_mean in INITIAL_MEANS:
        for initial_std in INITIAL_STDS:

            q_matrix_decoder = JC69QMatrixDecoder(A=A)

            distance = Hyperbolic()
            seq_encoder = EmbeddingTableSequenceEncoder(
                distance,
                data_NxSxA,
                D=D,
                initial_mean=initial_mean,
                initial_std=initial_std,
            )
            merge_encoder = HyperbolicGeodesicMidpointMergeEncoder(distance)
            proposal = EmbeddingProposal(
                distance, seq_encoder, merge_encoder, N=N, sample_branches=True
            )

            vcsmc = VCSMC(q_matrix_decoder, proposal, N=N, K=K).to(device)

            optimizer = Adam(vcsmc.parameters(), lr=lr)

            train(
                vcsmc,
                optimizer,
                taxa_N,
                data_NxSxA,
                file,
                epochs=epochs,
                run_name=f"HYP-SMC-INIT-MEAN-{initial_mean}-STD-{initial_std}",
            )


if __name__ == "__main__":
    typer.run(embedding_initialization)
