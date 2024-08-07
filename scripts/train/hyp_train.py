from typing import Annotated, Optional

import typer
from torch.optim.adam import Adam

from vcsmc import *


def hyp_train(
    file: str,
    lr: Annotated[float, typer.Option()],
    epochs: Annotated[int, typer.Option()],
    K: Annotated[int, typer.Option()],
    D: int = 2,
    grad_accumulation_steps: int = 1,
    sites_batch_size: Optional[int] = None,
    jc69: bool = False,
    hyperbolic: bool = False,
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

    if jc69:
        q_matrix_decoder = JC69QMatrixDecoder(A=A)
    else:
        q_matrix_decoder = DenseStationaryQMatrixDecoder(A=A)

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
        proposal = ExpBranchProposal(N=N, lookahead_merge=lookahead_merge)

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
