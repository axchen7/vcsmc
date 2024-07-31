# %%
from vcsmc import *

device = detect_device()

D = 2
K = 256
lr = 0.01
epochs = 100

file = "data/hohna/DS1.phy"


def train_with_proposal(initial_mean: float):
    N, S, A, data_NxSxA, taxa_N = load_phy(file, A4_ALPHABET)
    data_NxSxA = data_NxSxA.to(device)

    distance = Hyperbolic()
    seq_encoder = EmbeddingTableSequenceEncoder(
        distance, data_NxSxA, D=D, initial_mean=initial_mean
    )
    merge_encoder = HyperbolicGeodesicMidpointMergeEncoder(distance)
    proposal = EmbeddingProposal(distance, seq_encoder, merge_encoder, N=N)
    q_matrix_decoder = JC69QMatrixDecoder(A=A)
    vcsmc = VCSMC(q_matrix_decoder, proposal, N=N, K=K).to(device)
    optimizer = torch.optim.Adam(vcsmc.parameters(), lr=lr)

    run_name = f"Hyp_SMC_init_mean{initial_mean}"

    print(f"Starting {run_name}")

    train(vcsmc, optimizer, taxa_N, data_NxSxA, file, epochs=epochs, run_name=run_name)


initial_mean_vals = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95]

for initial_mean in initial_mean_vals:
    train_with_proposal(initial_mean)

# %%
import os

import matplotlib.pyplot as plt
import torch

from vcsmc import *


def load_log_likelihoods(initial_mean: float):
    run_name = f"Hyp_SMC_init_mean{initial_mean}"

    results: TrainResults = torch.load(
        find_most_recent_path(f"runs/*{run_name}", "results.pt"),
        weights_only=False,
    )
    return results["log_likelihood_avgs"]


initial_mean_vals = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95]

plt.title("Effect of Embedding Initialization")
plt.xlabel("Epochs")
plt.ylabel("Log Likelihood")
# plt.ylim(-8000, -6300)

for i, initial_mean in enumerate(initial_mean_vals):
    ll = load_log_likelihoods(initial_mean)
    plt.plot(ll[5:], label=f"Initial mean = {initial_mean}")

plt.legend()
plt.tight_layout()

os.makedirs("outputs/figures", exist_ok=True)

file = "outputs/figures/hyp_smc_init_mean.png"
if os.path.exists(file):
    os.remove(file)

plt.savefig(file)
plt.show()

# %%
