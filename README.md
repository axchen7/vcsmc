# Setup

First, create a conda environment:

```bash
conda create --yes --prefix .conda python=3.11
```

Next, activate the environment:

```bash
conda activate ./.conda
```

Then, install the project in editable mode:

```bash
pip install -e .
```

On Mac OS, you will also need `cairo` to rasterize SVGs of poincare plots:

```bash
brew install cairo
```

Finally, create a [Wandb](https://wandb.ai) account to save run metrics. Link
your account to the CLI by running:

```bash
wandb login
```

# Example Runs

Notes:

- `--q-matrix`:
  - "jc69": fixed JC69 Q matrix.
  - "stationary": one global Q matrix with each entry free.
  - "mlp": Q matrix is the output of a MLP that takes embeddings as input.
- `--no-hyperbolic`: regular VCSMC rather than Hyperbolic SMC.Z
- `--lookahead-merge`: performs nested hyperbolic SMC if `--hyperbolic` is set,
  or VNCSMC otherwise.
- `--hash-trick`: uses the hash trick to speed up computation. Only applies when
  `--hyperbolic` is set, and essentially required if `--lookahead-merge` is set.
- `--checkpoint-grads`: use gradient checkpointing to reduce memory usage.

Train primates, JC69:

```bash
python -m scripts.train.hyp_train --lr 0.01 --epochs 200 --k 16 --q-matrix jc69 --lookahead-merge --hash-trick data/primates.phy
```

Train DS1, JC69:

```bash
python -m scripts.train.hyp_train --lr 0.01 --epochs 200 --k 16 --q-matrix jc69 --lookahead-merge --hash-trick --checkpoint-grads data/hohna/DS1.phy
```

Benchmark (output to a .csv):

```bash
python -m scripts.benchmarks.hyp_smc_benchmark
```
