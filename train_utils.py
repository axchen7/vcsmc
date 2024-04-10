import glob
import os
import shutil
from typing import Callable, TypedDict

import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler

from vcsmc import VCSMC


class TrainArgs(TypedDict):
    taxa_N: list[str]
    data_NxSxA: Tensor
    file: str
    root: str
    epochs: int
    sites_batch_size: int | None


class TrainCheckpoint(TypedDict):
    vcsmc: VCSMC
    optimizer: Optimizer
    lr_scheduler: LRScheduler | None
    start_epoch: int


class slow_start_lr_scheduler(LambdaLR):
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
        checkpoints = f"runs/{run}/checkpoints"
        checkpoint_files = glob.glob(f"{checkpoints}/checkpoint_*.pt")

        if len(checkpoint_files) == 0:
            print(f"No checkpoints found for run {run}")
            continue
        try:
            args: TrainArgs = torch.load(f"{checkpoints}/args.pt")
        except FileNotFoundError:
            print(f"No args found for run {run}")
            continue
        try:
            checkpoint: TrainCheckpoint = torch.load(checkpoint_files[0])
        except FileNotFoundError:
            print(f"No checkpoint found for run {run}")
            continue

        vcsmc = checkpoint["vcsmc"]
        optimizer = checkpoint["optimizer"]

        try:
            matches = filter_fn(checkpoint, args)
        except Exception as e:
            print(f"Error filtering run {run}:")
            print(f"{e.__class__.__name__}: {e}")
            continue

        if matches:
            os.symlink(f"../runs/{run}", f"filtered_runs/{run}")
            match_count += 1

    print(f"\nFiltered {match_count} runs out of {len(runs)} total runs.")
