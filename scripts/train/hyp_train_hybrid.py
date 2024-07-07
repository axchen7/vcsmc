from typing import Annotated, Optional

import torch
import typer

from vcsmc import *


def hyp_train_hybrid(
    file: str,
    lr1: Annotated[float, typer.Option()],
    lr2: Annotated[float, typer.Option()],
    epochs1: Annotated[int, typer.Option()],
    epochs2: Annotated[int, typer.Option()],
    merge_samples: Annotated[int, typer.Option()],
    K1: Annotated[int, typer.Option()],
    K2: Annotated[int, typer.Option()],
    D: int = 2,
    sites_batch_size: Optional[int] = None,
    jc69: bool = False,
    lookahead_merge1: bool = False,
    hash_trick1: bool = False,
    checkpoint_grads: bool = False,
    run_name: Optional[str] = None,
):
    """
    Train Hyperbolic SMC model on a phylogenetic dataset. Performs two phases:
    - First, trains using deterministic branch lengths and maybe lookahead merge.
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
        lookahead_merge=lookahead_merge1,
        sample_branches=False,
    )

    vcsmc = VCSMC(
        q_matrix_decoder,
        proposal,
        N=N,
        K=K1,
        hash_trick=hash_trick1,
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
        run_name=f"{run_name}-phase1" if run_name else None,
    )

    # ===== phase 2 =====

    print("Running phase 2...")

    # find best checkpoint from phase 1
    _args, checkpoint = load_checkpoint(start_epoch="best")

    best_vcsmc = checkpoint["vcsmc"]
    best_proposal = vcsmc.proposal

    assert isinstance(best_proposal, EmbeddingProposal)

    # initialize new models

    static_merge_log_weights = compute_merge_log_weights_from_vcsmc(
        best_vcsmc, taxa_N, data_NxSxA, samples=merge_samples
    )

    proposal = EmbeddingProposal(
        best_proposal.distance,
        best_proposal.seq_encoder,
        best_proposal.merge_encoder,
        N=N,
        lookahead_merge=False,
        sample_branches=True,
        static_merge_log_weights=static_merge_log_weights,
    )

    vcsmc = VCSMC(
        best_vcsmc.q_matrix_decoder,
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
        start_epoch=checkpoint["start_epoch"],
        epochs=checkpoint["start_epoch"] + epochs2,
        sites_batch_size=sites_batch_size,
        run_name=f"{run_name}-phase2" if run_name else None,
    )


if __name__ == "__main__":
    typer.run(hyp_train_hybrid)
