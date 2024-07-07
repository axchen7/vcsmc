from typing import TypedDict

from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from ..vcsmc import VCSMC


class TrainArgs(TypedDict):
    taxa_N: list[str]
    data_NxSxA: Tensor
    file: str
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
    """Checkpoint contains the state at the start of this epoch (before parameter update)."""


class TrainResults(TypedDict):
    start_epoch: int
    """First result is from this epoch (before parameter update)."""
    ZCSMCs: list[float]
    log_likelihood_avgs: list[float]
