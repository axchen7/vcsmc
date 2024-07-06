from typing import Annotated, Optional

import typer

from vcsmc import *


def main(
    file: str,
    lr1: Annotated[float, typer.Option()],
    lr2: Annotated[float, typer.Option()],
    epochs1: Annotated[int, typer.Option()],
    epochs2: Annotated[int, typer.Option()],
    K1: Annotated[int, typer.Option()],
    K2: Annotated[int, typer.Option()],
    D: int = 2,
    sites_batch_size: Optional[int] = None,
    jc69: bool = False,
    hash_trick: bool = False,
    checkpoint_grads: bool = False,
):
    """
    Train Hyperbolic SMC model on a phylogenetic dataset. Performs two phases:
    - First, trains using lookahead merge and deterministic branch lengths.
    - Second, fixes the tree topology to the best one found and continues training
        using sampled branch lengths and no lookahead merge.
    """

    device = detect_device()

    N, _S, A, data_NxSxA, taxa_N = load_phy(file, A4_ALPHABET)
    data_NxSxA = data_NxSxA.to(device)

    if jc69:
        q_matrix_decoder = JC69QMatrixDecoder(A=A)
    else:
        q_matrix_decoder = DenseStationaryQMatrixDecoder(A=A)

    distance = Hyperbolic()
    seq_encoder = EmbeddingTableSequenceEncoder(distance, data_NxSxA, D=D)
    merge_encoder = HyperbolicGeodesicMidpointMergeEncoder(distance)

    # ===== phase 1 =====

    print("Running phase 1...")

    proposal = EmbeddingProposal(
        distance,
        seq_encoder,
        merge_encoder,
        N=N,
        lookahead_merge=True,
        sample_branches=False,
    )

    vcsmc = VCSMC(
        q_matrix_decoder,
        proposal,
        N=N,
        K=K1,
        hash_trick=hash_trick,
        checkpoint_grads=checkpoint_grads,
    ).to(device)

    optimizer = torch.optim.Adam(vcsmc.parameters(), lr=lr1)

    train(
        vcsmc,
        optimizer,
        taxa_N,
        data_NxSxA,
        file,
        epochs=epochs1,
        sites_batch_size=sites_batch_size,
    )

    phase1_result = evaluate(vcsmc, taxa_N, data_NxSxA)

    # ===== phase 2 =====

    print("Running phase 2...")

    proposal = EmbeddingProposal(
        distance,
        seq_encoder,
        merge_encoder,
        N=N,
        lookahead_merge=False,
        sample_branches=True,
        merge_indexes_KxN1x2=phase1_result["merge_indexes_KxN1x2"],
    )

    vcsmc = VCSMC(
        q_matrix_decoder,
        proposal,
        N=N,
        K=K2,
        hash_trick=False,  # no hash trick when sampling branches
        checkpoint_grads=checkpoint_grads,
    ).to(device)

    optimizer = torch.optim.Adam(vcsmc.parameters(), lr=lr2)

    train(
        vcsmc,
        optimizer,
        taxa_N,
        data_NxSxA,
        file,
        start_epoch=epochs1,
        epochs=epochs1 + epochs2,
        sites_batch_size=sites_batch_size,
    )


if __name__ == "__main__":
    typer.run(main)
