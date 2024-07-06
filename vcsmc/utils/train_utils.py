import glob
import os
import shutil
from typing import Callable

import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, TensorDataset

from ..vcsmc import VCSMC
from .train_types import TrainArgs, TrainCheckpoint
from .vcsmc_types import VcsmcResult


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
                find_most_recent_path(checkpoints_dir, "args.pt")
            )
            checkpoint: TrainCheckpoint = torch.load(
                find_most_recent_path(checkpoints_dir, "checkpoint_*.pt")
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
