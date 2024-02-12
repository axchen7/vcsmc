import numpy as np
import tensorflow as tf

from type_utils import Tensor

# fmt: off

A4_ALPHABET = {
    "A": [1, 0, 0, 0],
    "C": [0, 1, 0, 0],
    "G": [0, 0, 1, 0],
    "T": [0, 0, 0, 1],
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
    file: str, alphabet: dict[str, list[float]] = GT10_ALPHABET
) -> tuple[int, int, int, list[str], Tensor]:
    """
    Returns:
        N: Number of taxa.
        S: Number of sites.
        A: Alphabet size.
        taxa: List of taxa names.
        data_SxNxA: A tensor of shape (S, N, A).
    """

    with open(file) as f:
        lines = f.readlines()

    genome_lines = lines[1:]
    taxa = [line[: line.find(" ")].strip() for line in genome_lines]
    genome_strings = [line[line.find(" ") :].strip() for line in genome_lines]

    sample_char = genome_strings[0][0]

    N = len(genome_strings)
    S = len(genome_strings[0])
    A = len(alphabet[sample_char])

    data_SxNxA = np.zeros((S, N, A))

    for n in range(N):
        for s in range(S):
            data_SxNxA[s, n, :] = alphabet[genome_strings[n][s]]

    return N, S, A, taxa, tf.convert_to_tensor(data_SxNxA)
