import glob
import math
import os
import shutil
from typing import Callable, TypedDict

import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler

from .vcsmc import VCSMC


class TemperatureScheduler:
    def __call__(self, epoch: int) -> float:
        raise NotImplementedError


class TrainArgs(TypedDict):
    taxa_N: list[str]
    data_NxSxA: Tensor
    file: str
    temperature_scheduler: TemperatureScheduler | None
    root: str | None
    epochs: int
    sites_batch_size: int | None
    sample_taxa_count: int | None
    run_name: str | None


class TrainCheckpoint(TypedDict):
    vcsmc: VCSMC
    optimizer: Optimizer
    lr_scheduler: LRScheduler | None
    start_epoch: int


class TrainResults(TypedDict):
    ZCSMCs: list[float]
    log_likelihood_avgs: list[float]


class ExpDecayTemperatureScheduler(TemperatureScheduler):
    def __init__(self, initial_temp: float, decay_rate: float):
        """
        The temperature decays exponentially from initial_temp to 1.
        Example args: initial_temp=10, decay_rate=1/100
        """
        self.initial_temp = initial_temp
        self.decay_rate = decay_rate

    def __call__(self, epoch: int) -> float:
        return 1 + (self.initial_temp - 1) * math.exp(-self.decay_rate * epoch)


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
