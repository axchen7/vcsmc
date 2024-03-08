import numpy as np
import tensorflow as tf

from constants import DTYPE_FLOAT
from type_utils import Tensor

# fmt: off

A4_ALPHABET = {
    "A": [1, 0, 0, 0],
    "C": [0, 1, 0, 0],
    "G": [0, 0, 1, 0],
    "T": [0, 0, 0, 1],
    "N": [1 / 4] * 4,  # blank
    "-": [1 / 4] * 4,  # blank
}

GT10_ALPHABET = {
    #     AA CC GG TT AC AG AT CG CT GT CA GA TA GC TC TG
    "A": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # A/A
    "C": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # C/C
    "G": [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # G/G
    "T": [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # T/T
    "M": [0, 0, 0, 0,.5, 0, 0, 0, 0, 0,.5, 0, 0, 0, 0, 0],  # A/C
    "R": [0, 0, 0, 0, 0,.5, 0, 0, 0, 0, 0,.5, 0, 0, 0, 0],  # A/G
    "W": [0, 0, 0, 0, 0, 0,.5, 0, 0, 0, 0, 0,.5, 0, 0, 0],  # A/T
    "S": [0, 0, 0, 0, 0, 0, 0,.5, 0, 0, 0, 0, 0,.5, 0, 0],  # C/G
    "Y": [0, 0, 0, 0, 0, 0, 0, 0,.5, 0, 0, 0, 0, 0,.5, 0],  # C/T
    "K": [0, 0, 0, 0, 0, 0, 0, 0, 0,.5, 0, 0, 0, 0, 0,.5],  # G/T
    "N": [1 / 16] * 16,  # blank
    "-": [1 / 16] * 16,  # blank
}

# fmt: on


def load_phy(
    file: str, alphabet: dict[str, list[float]]
) -> tuple[int, int, int, Tensor, Tensor]:
    """
    Returns:
        N: Number of taxa.
        S: Number of sites.
        A: Alphabet size.
        data_NxSxA: Tensor of shape (N, S, A).
        taxa_N: Tensor of taxa names of shape (N,).
    """

    with open(file) as f:
        lines = f.readlines()

    genome_lines = lines[1:]

    taxa_N = [line[: line.find(" ")].strip() for line in genome_lines]
    taxa_N = tf.convert_to_tensor(taxa_N, dtype=tf.string)

    genome_strings = [line[line.find(" ") :].strip() for line in genome_lines]

    sample_char = genome_strings[0][0]

    N = len(genome_strings)
    S = len(genome_strings[0])
    A = len(alphabet[sample_char])

    data_NxSxA = np.zeros((N, S, A))

    for n in range(N):
        for s in range(S):
            data_NxSxA[n, s, :] = alphabet[genome_strings[n][s]]

    data_NxSxA = tf.convert_to_tensor(data_NxSxA, DTYPE_FLOAT)

    return N, S, A, data_NxSxA, taxa_N
