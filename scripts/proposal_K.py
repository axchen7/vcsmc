# %%
from vcsmc import *

device = detect_device()

D = 2
lr_exp_branch_proposal = 0.1
lr_embedding_proposal = 0.01
epochs = 200

file = "data/primates.phy"


def train_with_proposal(
    proposal_type: Literal["vcsmc", "vncsmc", "hyp_smc", "hyp_nsmc"], K: int
):
    N, S, A, data_NxSxA, taxa_N = load_phy(file, A4_ALPHABET)
    data_NxSxA = data_NxSxA.to(device)

    if proposal_type == "vcsmc" or proposal_type == "vncsmc":
        is_vncsmc = proposal_type == "vncsmc"

        proposal = ExpBranchProposal(N=N, lookahead_merge=is_vncsmc)
        hash_trick = False
        lr = lr_exp_branch_proposal
        run_name = f"VNCSMC_K{K}" if is_vncsmc else f"VCSMC_K{K}"
    elif proposal_type == "hyp_smc" or proposal_type == "hyp_nsmc":
        is_nsmc = proposal_type == "hyp_nsmc"

        distance = Hyperbolic()
        seq_encoder = EmbeddingTableSequenceEncoder(distance, data_NxSxA, D=D)
        merge_encoder = HyperbolicGeodesicMidpointMergeEncoder(distance)
        proposal = EmbeddingProposal(
            distance, seq_encoder, merge_encoder, N=N, lookahead_merge=is_nsmc
        )
        hash_trick = is_nsmc
        lr = lr_embedding_proposal
        run_name = f"Hyp_NSMC_K{K}" if is_nsmc else f"Hyp_SMC_K{K}"

    q_matrix_decoder = DenseStationaryQMatrixDecoder(A=A)
    vcsmc = VCSMC(
        q_matrix_decoder,
        proposal,
        N=N,
        K=K,
        hash_trick=hash_trick,
    ).to(device)
    optimizer = torch.optim.Adam(vcsmc.parameters(), lr=lr)

    print(f"Starting {run_name}")

    train(vcsmc, optimizer, taxa_N, data_NxSxA, file, epochs=epochs, run_name=run_name)


K_vals = [4, 8, 16, 32, 64, 128, 256, 512]
proposal_types = ("vcsmc", "vncsmc", "hyp_smc", "hyp_nsmc")

for K in K_vals:
    for proposal_type in proposal_types:
        train_with_proposal(proposal_type, K)

# %%
import os

import matplotlib.pyplot as plt

from vcsmc import *


def load_log_likelihoods(
    proposal_type: Literal["vcsmc", "vncsmc", "hyp_smc", "hyp_nsmc"], K: int
):
    if proposal_type == "vcsmc":
        run_name = f"VCSMC_K{K}"
    elif proposal_type == "vncsmc":
        run_name = f"VNCSMC_K{K}"
    elif proposal_type == "hyp_smc":
        run_name = f"Hyp_SMC_K{K}"
    elif proposal_type == "hyp_nsmc":
        run_name = f"Hyp_NSMC_K{K}"

    results: TrainResults = torch.load(
        find_most_recent_path(f"runs/*{run_name}", "results.pt")
    )
    return results["log_likelihood_avgs"]


K_vals = [4, 8, 16, 32, 64, 128, 256, 512]

fig, axs = plt.subplots(4, 2, figsize=(10, 10))
fig.suptitle("VCSMC vs Hyp SMC with Different K")

for i, K in enumerate(K_vals):
    vcsmc_ll = load_log_likelihoods("vcsmc", K)
    vncsmc_ll = load_log_likelihoods("vncsmc", K)
    hyp_smc_ll = load_log_likelihoods("hyp_smc", K)
    hyp_nsmc_ll = load_log_likelihoods("hyp_nsmc", K)

    ax = axs[i // 2, i % 2]
    ax.set_title(f"K = {K}")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Log Likelihood")
    ax.plot(vcsmc_ll[5:], label="VCSMC")
    ax.plot(vncsmc_ll[5:], label="VNCSMC")
    ax.plot(hyp_smc_ll[5:], label="Hyp SMC")
    ax.plot(hyp_nsmc_ll[5:], label="Hyp NSMC")
    ax.legend()

plt.tight_layout()

os.makedirs("outputs/figures", exist_ok=True)

file = "outputs/figures/proposal_K.png"
try:
    os.remove(file)
except FileNotFoundError:
    pass

plt.savefig(file)
plt.show()

# %%
