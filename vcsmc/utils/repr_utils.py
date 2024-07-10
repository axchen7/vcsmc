import re

from torch import nn


def custom_module_repr(extra_args: dict) -> str:
    return "\n".join([f"({k}): {v}" for k, v in extra_args.items()])


def module_to_config(module: nn.Module) -> dict:
    """
    Example input:
    (seq_encoder): EmbeddingTableSequenceEncoder(
        (D): 2
        (distance): Hyperbolic(
            (initial_scale): 0.1
            (fixed_scale): False
        )
    )
    Example output:
    {
        "D": 2
        "distance": "Hyperbolic",
        "initial_scale": 0.1,
        "fixed_scale": False
    }
    """

    def str_to_val(s: str):
        if s == "None":
            return None
        if s == "True":
            return True
        if s == "False":
            return False
        try:
            return int(s)
        except ValueError:
            pass
        try:
            return float(s)
        except ValueError:
            pass
        return s

    config = {}

    for line in repr(module).split("\n"):
        line = line.strip()
        match = re.match(r"\((\w+)\):\s([^(]+)", line)
        if match:
            key, val = match.groups()
            config[key] = str_to_val(val)

    return config
