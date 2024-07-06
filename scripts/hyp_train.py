from typing import Annotated, Optional

import typer

from vcsmc import *


def main(
    file: str,
    K: Annotated[int, typer.Option()],
    D: Annotated[int, typer.Option()],
    lr: Annotated[float, typer.Option()],
    epochs: Annotated[int, typer.Option()],
    sites_batch_size: Annotated[Optional[int], typer.Option()] = None,
    jc69: Annotated[bool, typer.Option()] = False,
    hyperbolic: Annotated[bool, typer.Option()] = False,
    sample_branches: Annotated[bool, typer.Option()] = False,
    lookahead_merge: Annotated[bool, typer.Option()] = False,
    hash_trick: Annotated[bool, typer.Option()] = False,
    checkpoint_grads: Annotated[bool, typer.Option()] = False,
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
        hash_trick = False  # can't use hash trick with ExpBranchProposal

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
