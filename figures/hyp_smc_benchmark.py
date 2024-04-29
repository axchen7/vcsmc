# %%
from figures_util import set_path

set_path()

from vcsmc import *

device = detect_device()

D = 2
lr = 0.01
epochs = 200
lookahead_merge = True
hash_trick = True


def train_with_proposal(file: str):
    # hack to avoid OOM
    if "DS7.phy" in file:
        K = 4
        sites_batch_size = 128
    elif "DS6.phy" in file:
        K = 8
        sites_batch_size = 256
    elif "DS4.phy" in file:
        K = 8
        sites_batch_size = None
    else:
        K = 16
        sites_batch_size = None

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
        taxa_N,
        K=K,
        hash_trick=hash_trick,
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
        run_name=run_name,
    )


files = [f"data/hohna/DS{i}.phy" for i in range(1, 8)]

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
    Estimates the LL by sampling a few log ZSMCs from the model state at the
    best epoch.

    Returns:
        ll_mean, ll_std_dev
    """
    fname = file.split("/")[-1]
    run_name = f"Hyp_SMC_{fname}"

    # find epoch with the largest log likelihood
    def find_best_epoch():
        results: TrainResults = torch.load(
            find_most_recent_path(f"runs/*{run_name}", "results.pt")
        )
        # there is no off-by-one error here: say epoch 1 has the highest LL;
        # then, results[0] is max, and loading epoch 0 will give the model state
        # before the optimizer step at epoch 1
        return int(np.argmax(results["elbos"]))

    data_NxSxA, taxa_N, vcsmc = train_from_checkpoint(
        load_only=True,
        start_epoch=find_best_epoch(),
        search_dir=f"runs/*{run_name}",
    )

    # hack to avoid OOM
    if "DS7.phy" in file:
        vcsmc.K = 2

    print(f"Evaluating {run_name}")

    ll_list: list[float] = []

    for _ in tqdm(range(ESTIMATE_LL_ITERS)):
        result = evaluate(vcsmc, data_NxSxA)
        ll_list.append(result["log_Z_SMC"].item())

    ll_mean = float(np.mean(ll_list))
    ll_std_dev = float(np.std(ll_list))

    return ll_mean, ll_std_dev


output_file = f"{make_output_dir()}/hyp_smc_benchmark.csv"
if os.path.exists(output_file):
    os.remove(output_file)

files = [f"data/hohna/DS{i}.phy" for i in range(1, 8)]

with open(output_file, "w") as f:
    f.write("file,ll_mean,ll_std_dev\n")
    f.flush()

    for file in files:
        ll_mean, ll_std_dev = estimate_log_likelihood(file)
        f.write(f"{file},{ll_mean:.2f},{ll_std_dev:.2f}\n")
        f.flush()

# %%
