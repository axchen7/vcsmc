import sys

sys.path.append("..")

from vcsmc import *

# if torch.cuda.is_available():
#     torch.set_default_device("cuda")
# elif torch.backends.mps.is_available():
#     torch.set_default_device("mps")


D = 2
lr = 0.01
epochs = 100

file = "../data/primates.phy"


def train_with_proposal(proposal_type: type[Proposal], *, K: int):
    N, S, A, data_NxSxA, taxa_N = load_phy(file, A4_ALPHABET)

    if proposal_type is ExpBranchProposal:
        proposal = ExpBranchProposal(N=N)
        run_name = f"VCSMC_K{K}"
    elif proposal_type is EmbeddingProposal:
        distance = Hyperbolic()
        seq_encoder = EmbeddingTableSequenceEncoder(distance, data_NxSxA, D=D)
        merge_encoder = HyperbolicGeodesicMidpointMergeEncoder(distance)
        proposal = EmbeddingProposal(distance, seq_encoder, merge_encoder)
        run_name = f"Hyp_SMC_K{K}"
    else:
        raise ValueError()

    q_matrix_decoder = JC69QMatrixDecoder(A=A)
    vcsmc = VCSMC(q_matrix_decoder, proposal, taxa_N, K=K)
    optimizer = torch.optim.Adam(vcsmc.parameters(), lr=lr)

    print(f"Starting {run_name}")

    train(vcsmc, optimizer, taxa_N, data_NxSxA, file, epochs=epochs, run_name=run_name)


K_vals = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

for K in K_vals:
    train_with_proposal(ExpBranchProposal, K=K)
    train_with_proposal(EmbeddingProposal, K=K)
