import glob
import os

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


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
