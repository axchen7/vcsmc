# %%
from figures_util import set_path

set_path()

from vcsmc import *

# device = detect_device()
device = "cpu"

D = 2
lr_exp_branch_proposal = 0.1
lr_embedding_proposal = 0.01
epochs = 200

file = "data/primates.phy"


def train_with_proposal(proposal_type: type[Proposal], K: int):
    N, S, A, data_NxSxA, taxa_N = load_phy(file, A4_ALPHABET)
    data_NxSxA = data_NxSxA.to(device)

    if proposal_type is ExpBranchProposal:
        proposal = ExpBranchProposal(N=N)
        lr = lr_exp_branch_proposal
        run_name = f"VCSMC_K{K}"
    elif proposal_type is EmbeddingProposal:
        distance = Hyperbolic()
        seq_encoder = EmbeddingTableSequenceEncoder(distance, data_NxSxA, D=D)
        merge_encoder = HyperbolicGeodesicMidpointMergeEncoder(distance)
        proposal = EmbeddingProposal(distance, seq_encoder, merge_encoder)
        lr = lr_embedding_proposal
        run_name = f"Hyp_SMC_K{K}"
    else:
        raise ValueError()

    q_matrix_decoder = DenseStationaryQMatrixDecoder(A=A)
    vcsmc = VCSMC(q_matrix_decoder, proposal, K=K).to(device)
    optimizer = torch.optim.Adam(vcsmc.parameters(), lr=lr)

    print(f"Starting {run_name}")

    train(vcsmc, optimizer, taxa_N, data_NxSxA, file, epochs=epochs, run_name=run_name)


K_vals = [4, 8, 16, 32, 64, 128, 256, 512]

for K in K_vals:
    train_with_proposal(ExpBranchProposal, K)
    train_with_proposal(EmbeddingProposal, K)

# %%
from figures_util import make_output_dir, set_path

set_path()

import os

import matplotlib.pyplot as plt

from vcsmc import *


def load_log_likelihoods(proposal_type: type[Proposal], K: int):
    if proposal_type is ExpBranchProposal:
        run_name = f"VCSMC_K{K}"
    elif proposal_type is EmbeddingProposal:
        run_name = f"Hyp_SMC_K{K}"
    else:
        raise ValueError()

    results: TrainResults = torch.load(
        find_most_recent_path(f"runs/*{run_name}", "results.pt")
    )
    return results["log_likelihood_avgs"]


K_vals = [4, 8, 16, 32, 64, 128, 256, 512]

fig, axs = plt.subplots(4, 2, figsize=(10, 10))
fig.suptitle("VCSMC vs Hyp SMC with Different K")

for i, K in enumerate(K_vals):
    vcsmc_ll = load_log_likelihoods(ExpBranchProposal, K)
    hyp_ll = load_log_likelihoods(EmbeddingProposal, K)

    ax = axs[i // 2, i % 2]
    ax.set_title(f"K = {K}")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Log Likelihood")
    ax.plot(vcsmc_ll[5:], label="VCSMC")
    ax.plot(hyp_ll[5:], label="Hyp SMC")
    ax.legend()

plt.tight_layout()

file = f"{make_output_dir()}/proposal_K.png"
try:
    os.remove(file)
except FileNotFoundError:
    pass

plt.savefig(file)
plt.show()

# %%
