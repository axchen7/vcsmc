# %%
from figures_util import set_path

set_path()

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

# %%
from figures_util import make_output_dir, set_path

set_path()

import os

import matplotlib.pyplot as plt

from vcsmc import *


def load_log_likelihoods(initial_mean: float):
    run_name = f"Hyp_SMC_init_mean{initial_mean}"

    results: TrainResults = torch.load(
        find_most_recent_path(f"runs/*{run_name}", "results.pt")
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

file = f"{make_output_dir()}/hyp_smc_init_mean.png"
if os.path.exists(file):
    os.remove(file)

plt.savefig(file)
plt.show()

# %%
