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


def find_most_recent_file(dir: str, filename: str) -> str:
    """
    Args:
        dir: The path to the directory to recursively search.
        filename: The name of the file to search for.

    Returns:
        The path of the most recently created file with the given name.
    """
    file_list = glob.glob(f"{dir}/**/{filename}", recursive=True)

    if len(file_list) == 0:
        raise FileNotFoundError(
            f'No files found with the name "{filename}" in directory "{dir}".'
        )

    return max(file_list, key=os.path.getctime)
