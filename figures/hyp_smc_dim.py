import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.chdir("..")
sys.path.append(".")

from vcsmc import *

if torch.cuda.is_available():
    torch.set_default_device("cuda")
elif torch.backends.mps.is_available():
    torch.set_default_device("mps")


K = 128
lr = 0.01
epochs = 100

file = "data/primates.phy"


def train_with_proposal(D: int):
    N, S, A, data_NxSxA, taxa_N = load_phy(file, A4_ALPHABET)

    distance = Hyperbolic()
    seq_encoder = EmbeddingTableSequenceEncoder(distance, data_NxSxA, D=D)
    merge_encoder = HyperbolicGeodesicMidpointMergeEncoder(distance)
    proposal = EmbeddingProposal(distance, seq_encoder, merge_encoder)
    q_matrix_decoder = JC69QMatrixDecoder(A=A)
    vcsmc = VCSMC(q_matrix_decoder, proposal, taxa_N, K=K)
    optimizer = torch.optim.Adam(vcsmc.parameters(), lr=lr)

    run_name = f"Hyp_SMC_D{D}"

    print(f"Starting {run_name}")

    train(vcsmc, optimizer, taxa_N, data_NxSxA, file, epochs=epochs, run_name=run_name)


D_vals = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]

for D in D_vals:
    train_with_proposal(D)
