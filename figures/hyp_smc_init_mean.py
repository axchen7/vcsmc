import os

os.chdir("..")

from vcsmc import *

if torch.cuda.is_available():
    torch.set_default_device("cuda")
elif torch.backends.mps.is_available():
    torch.set_default_device("mps")


D = 2
K = 256
lr = 0.01
epochs = 100

file = "data/hohna/DS1.phy"


def train_with_proposal(initial_mean: float):
    N, S, A, data_NxSxA, taxa_N = load_phy(file, A4_ALPHABET)

    distance = Hyperbolic()
    seq_encoder = EmbeddingTableSequenceEncoder(
        distance, data_NxSxA, D=D, initial_mean=initial_mean
    )
    merge_encoder = HyperbolicGeodesicMidpointMergeEncoder(distance)
    proposal = EmbeddingProposal(distance, seq_encoder, merge_encoder)
    q_matrix_decoder = JC69QMatrixDecoder(A=A)
    vcsmc = VCSMC(q_matrix_decoder, proposal, taxa_N, K=K)
    optimizer = torch.optim.Adam(vcsmc.parameters(), lr=lr)

    run_name = f"Hyp_SMC_init_mean{initial_mean}"

    print(f"Starting {run_name}")

    train(vcsmc, optimizer, taxa_N, data_NxSxA, file, epochs=epochs, run_name=run_name)


initial_mean_vals = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95]

for initial_mean in initial_mean_vals:
    train_with_proposal(initial_mean)
