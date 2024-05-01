# %%
from figures_util import set_path

set_path()

from vcsmc import *

device = detect_device()

D = 2
lr = 0.01
# epochs = 200
epochs = 1000
lookahead_merge = True
hash_trick = True
checkpoint_grads = True


def train_with_proposal(file: str):
    # hack to avoid OOM
    if "DS8.phy" in file:
        K = 4
        sites_batch_size = 256
        sample_taxa_count = None
    elif "DS7.phy" in file:
        K = 4
        sites_batch_size = 128
        sample_taxa_count = None
    elif "DS6.phy" in file:
        K = 8
        sites_batch_size = 256
        sample_taxa_count = None
    else:
        K = 16
        sites_batch_size = None
        sample_taxa_count = None

    N, S, A, data_NxSxA, taxa_N = load_phy(file, A4_ALPHABET)
    data_NxSxA = data_NxSxA.to(device)

    distance = Hyperbolic()
    seq_encoder = EmbeddingTableSequenceEncoder(distance, data_NxSxA, D=D)
    merge_encoder = HyperbolicGeodesicMidpointMergeEncoder(distance)
    proposal = EmbeddingProposal(
        distance, seq_encoder, merge_encoder, lookahead_merge=lookahead_merge
    )
    q_matrix_decoder = JC69QMatrixDecoder(A=A)
    vcsmc = VCSMC(
        q_matrix_decoder,
        proposal,
        K=K,
        hash_trick=hash_trick,
        checkpoint_grads=checkpoint_grads,
    ).to(device)
    optimizer = torch.optim.Adam(vcsmc.parameters(), lr=lr)

    fname = file.split("/")[-1]
    run_name = f"Hyp_SMC_{fname}"

    print(f"Starting {run_name}")

    train(
        vcsmc,
        optimizer,
        taxa_N,
        data_NxSxA,
        file,
        epochs=epochs,
        sites_batch_size=sites_batch_size,
        sample_taxa_count=sample_taxa_count,
        run_name=run_name,
    )


files = [f"data/hohna/DS{i}.phy" for i in range(1, 9)]

for file in files:
    train_with_proposal(file)

# %%
from figures_util import make_output_dir, set_path

set_path()

import os

import numpy as np
import tqdm

from vcsmc import *

ESTIMATE_LL_ITERS = 10


def estimate_log_likelihood(file: str) -> tuple[float, float]:
    """
    Estimates the LL by sampling a few log ZCSMCs from the model state at the
    best epoch.

    Returns:
        ll_mean, ll_std_dev
    """
    fname = file.split("/")[-1]
    run_name = f"Hyp_SMC_{fname}"

    data_NxSxA, taxa_N, vcsmc = train_from_checkpoint(
        load_only=True, start_epoch="best", search_dir=f"runs/*{run_name}"
    )

    # hack to avoid OOM
    if fname == "DS7.phy":
        vcsmc.K = 1

    print(f"Evaluating {run_name}")

    ll_list: list[float] = []

    for _ in tqdm(range(ESTIMATE_LL_ITERS)):
        result = evaluate(vcsmc, taxa_N, data_NxSxA)
        ll_list.append(result["log_ZCSMC"].item())

    ll_mean = float(np.mean(ll_list))
    ll_std_dev = float(np.std(ll_list))

    return ll_mean, ll_std_dev


output_file = f"{make_output_dir()}/hyp_smc_benchmark.csv"
if os.path.exists(output_file):
    os.remove(output_file)

files = [f"data/hohna/DS{i}.phy" for i in range(1, 9)]

with open(output_file, "w") as f:
    f.write("file,ll_mean,ll_std_dev\n")
    f.flush()

    for file in files:
        ll_mean, ll_std_dev = estimate_log_likelihood(file)
        f.write(f"{file},{ll_mean:.2f},{ll_std_dev:.2f}\n")
        f.flush()

# %%
