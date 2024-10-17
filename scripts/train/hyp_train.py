from enum import Enum
from typing import Annotated, Optional

import typer
from torch.optim.adam import Adam

from vcsmc import *


class QMatrixType(Enum):
    JC69 = "jc69"
    STATIONARY = "stationary"
    MLP_FACTORIZED = "mlp_factorized"
    MLP_DENSE = "mlp_dense"


def hyp_train(
    file: str,
    lr: Annotated[float, typer.Option()],
    epochs: Annotated[int, typer.Option()],
    K: Annotated[int, typer.Option()],
    D: int = 2,
    grad_accumulation_steps: int = 1,
    sites_batch_size: Optional[int] = None,
    hyperbolic: bool = True,
    q_matrix: QMatrixType = QMatrixType.JC69,
    sample_branches: bool = False,
    lookahead_merge: bool = False,
    hash_trick: bool = False,
    checkpoint_grads: bool = False,
    run_name: Optional[str] = None,
):
    """Train a VCSMC model on a phylogenetic dataset."""

    device = detect_device()

    N, _S, A, data_NxSxA, taxa_N = load_phy(file, A4_ALPHABET)
    data_NxSxA = data_NxSxA.to(device)

    if hyperbolic:
        distance = Hyperbolic()
        seq_encoder = EmbeddingTableSequenceEncoder(distance, data_NxSxA, D=D)
        merge_encoder = HyperbolicGeodesicMidpointMergeEncoder(distance)
        proposal = EmbeddingProposal(
            distance,
            seq_encoder,
            merge_encoder,
            N=N,
            lookahead_merge=lookahead_merge,
            sample_branches=sample_branches,
        )
    else:
        distance = None
        proposal = ExpBranchProposal(N=N, lookahead_merge=lookahead_merge)

    match q_matrix:
        case QMatrixType.JC69:
            q_matrix_decoder = JC69QMatrixDecoder(A=A)
        case QMatrixType.STATIONARY:
            q_matrix_decoder = FactorizedStationaryQMatrixDecoder(A=A)
        case QMatrixType.MLP_FACTORIZED:
            assert distance is not None, "MLP Q-matrix requires hyperbolic distance"
            q_matrix_decoder = FactorizedMLPQMatrixDecoder(distance, A=A, D=D)
        case QMatrixType.MLP_DENSE:
            assert distance is not None, "MLP Q-matrix requires hyperbolic distance"
            q_matrix_decoder = DenseMLPQMatrixDecoder(distance, A=A, D=D)

    vcsmc = VCSMC(
        q_matrix_decoder,
        proposal,
        N=N,
        K=K,
        hash_trick=hash_trick,
        checkpoint_grads=checkpoint_grads,
    ).to(device)

    optimizer = Adam(vcsmc.parameters(), lr=lr)

    train(
        vcsmc,
        optimizer,
        taxa_N,
        data_NxSxA,
        file,
        epochs=epochs,
        grad_accumulation_steps=grad_accumulation_steps,
        sites_batch_size=sites_batch_size,
        run_name=run_name,
    )


if __name__ == "__main__":
    typer.run(hyp_train)
