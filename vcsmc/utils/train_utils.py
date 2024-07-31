import glob
import io
import os
import shutil
from typing import Callable

import torch
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from PIL import Image
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import wandb

from ..vcsmc import VCSMC
from .train_types import TrainArgs, TrainCheckpoint
from .vcsmc_types import VcsmcResult
from .vcsmc_utils import (
    gather_K,
    hash_forest_K,
    hash_K,
    hash_tree_K,
    replace_with_merged_K,
)

__all__ = [
    "SlowStartLRScheduler",
    "find_most_recent_path",
    "filter_runs",
    "detect_device",
    "evaluate",
    "compute_merge_log_weights_from_vcsmc",
]


class SlowStartLRScheduler(LambdaLR):
    """
    Use a smaller learning rate for the first few epochs.
    """

    def __init__(self, optimizer: Optimizer, *, scale: float, cutoff: int):
        """
        Args:
            optimizer: The optimizer.
            scale: The scale factor for the learning rate.
            cutoff: The epoch at which to stop scaling the learning rate.
        """

        self.scale = scale
        self.cutoff = cutoff

        super().__init__(optimizer, self.lr_lambda)

    def lr_lambda(self, epoch: int):
        return self.scale if epoch < self.cutoff else 1.0


def find_most_recent_path(search_dir: str, name: str) -> str:
    """
    Finds the most recently created file or directory with the given name.

    Args:
        search_dir: The path to the directory to recursively search. Can be a glob pattern.
        name: The name of the file or pattern to search for. Can be a glob pattern.

    Returns:
        The path of the most recently created file with the given name.
    """
    file_list = glob.glob(f"{search_dir}/**/{name}", recursive=True)

    if len(file_list) == 0:
        raise FileNotFoundError(
            f'No files or directories matching "{name}" found in "{search_dir}".'
        )

    return max(file_list, key=os.path.getctime)


def filter_runs(filter_fn: Callable[[TrainCheckpoint, TrainArgs], bool]):
    """
    Symlinks entries in `./runs` to `./filtered_runs` based on the `filter_fn`.
    """

    shutil.rmtree("filtered_runs", ignore_errors=True)
    os.mkdir("filtered_runs")

    runs = os.listdir("runs")
    match_count = 0

    for run in runs:
        checkpoints_dir = f"runs/{run}/checkpoints"

        try:
            args: TrainArgs = torch.load(
                find_most_recent_path(checkpoints_dir, "args.pt"),
                weights_only=False,
            )
            checkpoint: TrainCheckpoint = torch.load(
                find_most_recent_path(checkpoints_dir, "checkpoint_*.pt"),
                weights_only=False,
            )

            matches = filter_fn(checkpoint, args)
        except Exception as e:
            print(f"Error filtering run {run}:")
            print(f"{e.__class__.__name__}: {e}")
            continue

        if matches:
            os.symlink(f"../runs/{run}", f"filtered_runs/{run}")
            match_count += 1

    print(f"\nFiltered {match_count} runs out of {len(runs)} total runs.")


def detect_device() -> torch.device:
    """
    Use the GPU if available, otherwise use the CPU.
    """

    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_site_positions_SxSfull(data_NxSxA: Tensor) -> Tensor:
    S = data_NxSxA.shape[1]
    site_positions_SxSfull = torch.eye(
        S, dtype=data_NxSxA.dtype, device=data_NxSxA.device
    )
    return site_positions_SxSfull


def batch_by_sites(data_NxSxA: Tensor, batch_size: int | None) -> DataLoader:
    """
    Returns a (mapped) DataLoader where each element is a tuple
    (data_batched_SxNxA, site_positions_batched_SxSfull).

    Args:
        data_NxSxA: The data.
        batch_size: The batch size. Set to None to use the full dataset.
    """

    if batch_size is None:
        S = data_NxSxA.shape[1]
        batch_size = S

    data_SxNxA = data_NxSxA.permute(1, 0, 2)
    site_positions_SxSfull = get_site_positions_SxSfull(data_NxSxA)

    # shape: S x ([N, A], [Sfull])
    dataset = TensorDataset(data_SxNxA, site_positions_SxSfull)

    # shape: V x ([S, N, A], [S, Sfull]), where now S = batch_size
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


@torch.no_grad()
def evaluate(
    vcsmc: VCSMC,
    taxa_N: list[str],
    data_NxSxA: Tensor,
) -> VcsmcResult:
    dataset = batch_by_sites(data_NxSxA, None)

    # batch is actually the full dataset
    data_batched_SxNxA, site_positions_batched_SxSfull = next(iter(dataset))
    data_batched_NxSxA = data_batched_SxNxA.permute(1, 0, 2)

    return vcsmc(taxa_N, data_NxSxA, data_batched_NxSxA, site_positions_batched_SxSfull)


@torch.no_grad()
def compute_merge_log_weights_from_merge_indexes(
    merge_indexes_KxN1x2: Tensor,
) -> dict[int, Tensor]:
    """
    Given the merge indexes of all K particles from the VCSMC output, computes
    the empirical distribution conditioned on each unique forest.

    Returns a mapping of each unique forest by its hash to a shape (t, t)
    symmetric matrix of log pairwise merge probabilities.

    Notes:
    - t is the number of trees in the forest.
    - The merge probabilities matrix is not normalized.
    - Returned tensors are on the CPU.
    """

    K = merge_indexes_KxN1x2.shape[0]
    N = merge_indexes_KxN1x2.shape[1] + 1

    merge_indexes_KxN1x2 = merge_indexes_KxN1x2.cpu()

    # forest hash -> list of seen merges
    # (includes all merges, not just unique ones)
    seen_merges: dict[int, list[tuple[int, int]]] = {}

    # forest hash -> proposal distribution, shape (t, t)
    merge_proposals: dict[int, Tensor] = {}

    # simultaneously build trees for the K particles

    # tree hashes
    hashes_Kxt = hash_K(torch.arange(N)).repeat(K, 1)

    for r in range(N - 1):
        t = N - r  # number of trees in the forest

        forest_hashes_K = hash_forest_K(hashes_Kxt)
        idx1_K = merge_indexes_KxN1x2[:, r, 0]
        idx2_K = merge_indexes_KxN1x2[:, r, 1]

        # we must output the indexes w.r.t. the list of trees sorted by hash,
        # as the order of trees is otherwise arbitrary

        # unsort_idx_Kxt[k, i] is the index of the i-th original tree of
        # particle k as found in the sorted list of trees
        unsort_idx_Kxt = hashes_Kxt.argsort(dim=1).argsort(dim=1)

        for k in range(K):
            forest_hash = int(forest_hashes_K[k])

            idx1 = idx1_K[k]
            idx2 = idx2_K[k]

            sorted_idx1 = int(unsort_idx_Kxt[k, idx1])
            sorted_idx2 = int(unsort_idx_Kxt[k, idx2])

            if forest_hash not in seen_merges:
                seen_merges[forest_hash] = []

            seen_merges[forest_hash].append((sorted_idx1, sorted_idx2))

            if forest_hash in merge_proposals:
                # sanity check (and also for hash collisions)
                assert merge_proposals[forest_hash].shape[0] == t
            else:
                merge_proposals[forest_hash] = torch.zeros(t, t)

        hashes_Kxt = replace_with_merged_K(
            hashes_Kxt,
            idx1_K,
            idx2_K,
            hash_tree_K(
                gather_K(hashes_Kxt, idx1_K, torch.arange),
                gather_K(hashes_Kxt, idx2_K, torch.arange),
            ),
            torch.arange,
        )

    # compute the empirical distribution

    merge_log_weights: dict[int, Tensor] = {}

    for forest_hash, merges in seen_merges.items():
        for idx1, idx2 in merges:
            merge_proposals[forest_hash][idx1, idx2] += 1
            merge_proposals[forest_hash][idx2, idx1] += 1

        merge_log_weights[forest_hash] = merge_proposals[forest_hash].log()

    return merge_log_weights


@torch.no_grad()
def compute_merge_log_weights_from_vcsmc(
    vcsmc: VCSMC,
    taxa_N: list[str],
    data_NxSxA: Tensor,
    *,
    samples: int,
):
    """
    Performs multiple samples from the VCSMC to compute the empirical
    distribution of merge proposals.
    """

    merge_indexes_KxN1x2_list: list[Tensor] = []

    for _ in tqdm(range(samples), "Sampling merge indexes from VCSMC"):
        results = evaluate(vcsmc, taxa_N, data_NxSxA)
        merge_indexes_KxN1x2_list.append(results["merge_indexes_KxN1x2"])

    # flatten particles
    merge_indexes_KxN1x2 = torch.cat(merge_indexes_KxN1x2_list, dim=0)
    return compute_merge_log_weights_from_merge_indexes(merge_indexes_KxN1x2)


def fig_to_wandb_image(fig: Figure) -> wandb.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    image = wandb.Image(Image.open(buf))
    buf.close()
    return image
