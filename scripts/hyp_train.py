from argparse import ArgumentParser

from vcsmc import *


def main():
    parser = ArgumentParser(description="Train a VCSMC model on a phylogenetic dataset")
    parser.add_argument("file", type=str)
    parser.add_argument("--K", type=int, required=True)
    parser.add_argument("--D", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--sites-batch-size", type=int, required=False)
    parser.add_argument("--jc69", action="store_true")
    parser.add_argument("--lookahead-merge", action="store_true")
    parser.add_argument("--hash-trick", action="store_true")
    parser.add_argument("--checkpoint-grads", action="store_true")
    args = parser.parse_args()

    device = detect_device()

    N, _S, A, data_NxSxA, taxa_N = load_phy(args.file, A4_ALPHABET)
    data_NxSxA = data_NxSxA.to(device)

    distance = Hyperbolic()
    seq_encoder = EmbeddingTableSequenceEncoder(distance, data_NxSxA, D=args.D)
    merge_encoder = HyperbolicGeodesicMidpointMergeEncoder(distance)

    proposal = EmbeddingProposal(
        distance, seq_encoder, merge_encoder, N=N, lookahead_merge=args.lookahead_merge
    )

    if args.jc69:
        q_matrix_decoder = JC69QMatrixDecoder(A=A)
    else:
        q_matrix_decoder = DenseStationaryQMatrixDecoder(A=A)

    vcsmc = VCSMC(
        q_matrix_decoder,
        proposal,
        N=N,
        K=args.K,
        hash_trick=args.hash_trick,
        checkpoint_grads=args.checkpoint_grads,
    ).to(device)
    optimizer = torch.optim.Adam(vcsmc.parameters(), lr=args.lr)

    train(
        vcsmc,
        optimizer,
        taxa_N,
        data_NxSxA,
        args.file,
        epochs=args.epochs,
        sites_batch_size=args.sites_batch_size,
    )


if __name__ == "__main__":
    main()
