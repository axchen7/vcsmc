# Setup

First, install the required packages:

```bash
pip install -e .
```

Next, create a [Wandb](https://wandb.ai) account to save run metrics. Link
your account to the CLI by running:

```bash
wandb login
```

# Example Runs

Notes:

- `--q-matrix`: specifies the Q matrix to use.
  - "jc69": fixed JC69 Q matrix.
  - "stationary": one global Q matrix with each entry free.
  - "mlp": Q matrix is the output of a MLP that takes embeddings as input.
- `--lookahead-merge`: performs H-VNCSMC if `--hyperbolic` is set, or VNCSMC
  otherwise.
- `--hash-trick`: memoizes compute over tree topologies to speed up computation.
  Only applies when `--hyperbolic` is set, and essentially required if
  `--lookahead-merge` is set.
- `--checkpoint-grads`: use gradient checkpointing to reduce memory usage.

Run H-VCSMC on primates using K=512:

```bash
python -m scripts.train.hyp_train --lr 0.01 --epochs 200 --k 512 --q-matrix mlp --hash-trick data/primates.phy
```

Run H-VNCSMC (nested proposal) on primates using K=16:

```bash
python -m scripts.train.hyp_train --lr 0.01 --epochs 200 --k 16 --q-matrix mlp --lookahead-merge --hash-trick data/primates.phy
```

Run H-VNCSMC on a larger benchmark dataset (DS1) using K=16:

```bash
python -m scripts.train.hyp_train --lr 0.01 --epochs 200 --k 16 --q-matrix mlp --lookahead-merge --hash-trick --checkpoint-grads data/hohna/DS1.phy
```

Run H-VNCSMC on benchmark datasets DS1-DS7, with deferred branch sampling to better learn the embeddings:

```bash
python -m scripts.benchmarks.hyp_smc_benchmark --q-matrix mlp
```
