# Setup

First, create a conda environment:

```bash
conda create --yes --prefix .conda python=3.11
```

Next, activate the environment:

```bash
conda activate ./.conda
```

Finally, install the project in editable mode:

```bash
pip install -e .
```

# Example Runs

Notes:

- `--jc69`: uses the JC69 model of evolution. Without this flag, every entry in
  the Q matrix is learned.
- `--hyperbolic`: uses Hyperbolic SMC rather than regular VCSMC.
- `--lookahead-merge`: performs nested hyperbolic SMC if `--hyperbolic` is set,
  or VNCSMC otherwise.
- `--hash-trick`: uses the hash trick to speed up computation. Only applies when
  `--hyperbolic` is set, and essentially required if `--lookahead-merge` is set.
- `--checkpoint-grads`: use gradient checkpointing to reduce memory usage.

Train primates, JC69:

```bash
python scripts/train/hyp_train.py --lr 0.01 --epochs 200 --k 16 --jc69 --hyperbolic --lookahead-merge --hash-trick data/primates.phy
```

Train DS1, JC69:

```bash
python scripts/train/hyp_train.py --lr 0.01 --epochs 200 --k 16 --jc69 --hyperbolic --lookahead-merge --hash-trick --checkpoint-grads data/hohna/DS1.phy
```

# Generate Paper Figures

Title: VCSMC vs Hyp SMC with Different K

```bash
python scripts/figures/proposal_K.py
```

Title: Effect of Embedding Initialization

```bash
python scripts/figures/hyp_smc_init_mean.py
```

Title: Effect of Dimensionality

```bash
python scripts/figures/hyp_smc_dim.py
```

Benchmark (output to a .csv):

```bash
python scripts/benchmark/hyp_smc_benchmark.py
```
