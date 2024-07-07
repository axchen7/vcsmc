import numpy as np
import torch
from tqdm import tqdm

from vcsmc import evaluate, load_checkpoint


@torch.no_grad()
def estimate_latest_run_ll(num_samples: int) -> tuple[float, float]:
    """
    Sampling a few log ZCSMCs from the best epoch of the latest run.

    Returns:
        ll_mean, ll_std_dev
    """

    args, checkpoint = load_checkpoint(start_epoch="best")

    data_NxSxA = args["data_NxSxA"]
    taxa_N = args["taxa_N"]
    vcsmc = checkpoint["vcsmc"]

    ll_list: list[float] = []

    for _ in tqdm(range(num_samples), f"Estimating LL for {args['run_name']}"):
        result = evaluate(vcsmc, taxa_N, data_NxSxA)
        ll_list.append(result["log_ZCSMC"].item())

    ll_mean = float(np.mean(ll_list))
    ll_std_dev = float(np.std(ll_list))

    return ll_mean, ll_std_dev
