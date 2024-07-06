from typing import Annotated, Optional

import typer

from vcsmc import *


def main(
    file: str,
    K: Annotated[int, typer.Option()],
    D: Annotated[int, typer.Option()],
    lr: Annotated[float, typer.Option()],
    epochs: Annotated[int, typer.Option()],
    sites_batch_size: Optional[int] = None,
    jc69: bool = False,
    hyperbolic: bool = False,
    sample_branches: bool = False,
    lookahead_merge: bool = False,
    hash_trick: bool = False,
    checkpoint_grads: bool = False,
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
    optimizer = torch.optim.Adam(vcsmc.parameters(), lr=lr)

    train(
        vcsmc,
        optimizer,
        taxa_N,
        data_NxSxA,
        file,
        epochs=epochs,
        sites_batch_size=sites_batch_size,
    )


if __name__ == "__main__":
    typer.run(main)
