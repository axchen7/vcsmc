# %%
from figures_util import set_path

set_path()

from vcsmc import *

device = detect_device()

K = 128
lr = 0.01
epochs = 100

file = "data/primates.phy"


def train_with_proposal(D: int):
    N, S, A, data_NxSxA, taxa_N = load_phy(file, A4_ALPHABET)
    data_NxSxA = data_NxSxA.to(device)

    distance = Hyperbolic()
    seq_encoder = EmbeddingTableSequenceEncoder(distance, data_NxSxA, D=D)
    merge_encoder = HyperbolicGeodesicMidpointMergeEncoder(distance)
    proposal = EmbeddingProposal(distance, seq_encoder, merge_encoder)
    q_matrix_decoder = JC69QMatrixDecoder(A=A)
    vcsmc = VCSMC(q_matrix_decoder, proposal, taxa_N, K=K).to(device)
    optimizer = torch.optim.Adam(vcsmc.parameters(), lr=lr)

    run_name = f"Hyp_SMC_D{D}"

    print(f"Starting {run_name}")

    train(vcsmc, optimizer, taxa_N, data_NxSxA, file, epochs=epochs, run_name=run_name)


D_vals = [1, 2, 3, 4, 6, 8, 12, 16]

for D in D_vals:
    train_with_proposal(D)

# %%
from figures_util import make_output_dir, set_path

set_path()

import os

import matplotlib.pyplot as plt

from vcsmc import *


def load_log_likelihoods(D):
    run_name = f"Hyp_SMC_D{D}"

    results: TrainResults = torch.load(
        find_most_recent_path(f"runs/*{run_name}", "results.pt")
    )
    return results["log_likelihood_avgs"]


D_vals = [1, 2, 3, 4, 6, 8, 12, 16]

plt.title("Effect of Dimensionality")
plt.xlabel("Epochs")
plt.ylabel("Log Likelihood")
plt.ylim(-8000, -6300)

for i, D in enumerate(D_vals):
    ll = load_log_likelihoods(D)
    linestyle = "solid" if D == 2 else "dotted"
    plt.plot(ll[5:], label=f"D = {D}", linestyle=linestyle)

plt.legend()
plt.tight_layout()

file = f"{make_output_dir()}/hyp_smc_dim.png"
if os.path.exists(file):
    os.remove(file)

plt.savefig(file)
plt.show()

# %%
