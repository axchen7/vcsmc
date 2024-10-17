# Setup

First, install the required packages:

```bash
pip install -e .
```

Next, create a [Wandb](https://wandb.ai) account to save run metrics. Link your
account to the CLI by running:

```bash
wandb login
```

# Example Runs

Notes:

- `--q-matrix`: specifies the Q matrix to use.
  - `jc69`: fixed JC69 Q matrix.
  - `stationary`: one global Q matrix with each entry free.
  - `mlp_factorized`: An MLP maps embeddings to holding times and stationary
    probabilities, forming the Q matrix. More memory efficient than `mlp_dense`.
  - `mlp_dense`: An MLP maps embeddings to all entries of the Q matrix.
- `--lookahead-merge`: performs H-VNCSMC if `--hyperbolic` is set, or VNCSMC
  otherwise.
- `--hash-trick`: memoizes compute over tree topologies to speed up computation.
  Only applies when `--hyperbolic` is set, and essentially required if
  `--lookahead-merge` is set.
- `--checkpoint-grads`: use gradient checkpointing to reduce memory usage.

Run H-VCSMC on primates using K=512 and a learned Q matrix:

```bash
python -m scripts.train.hyp_train --lr 0.01 --epochs 200 --k 512 --q-matrix mlp_dense --hash-trick data/primates.phy
```

Run H-VNCSMC (nested proposal) on primates using K=16 and a learned Q matrix:

```bash
python -m scripts.train.hyp_train --lr 0.01 --epochs 200 --k 16 --q-matrix mlp_dense --lookahead-merge --hash-trick data/primates.phy
```

Run H-VNCSMC on a larger benchmark dataset (DS1) using K=16 and a factorized Q
matrix:

```bash
python -m scripts.train.hyp_train --lr 0.05 --epochs 200 --k 16 --q-matrix mlp_factorized --lookahead-merge --hash-trick data/hohna/DS1.phy
```

Run H-VNCSMC on benchmark datasets DS1-DS7, with deferred branch sampling to
better learn the embeddings:

```bash
python -m scripts.benchmarks.hyp_smc_benchmark --q-matrix mlp_factorized
```
