# Setup

First, create a conda environment:

```bash
conda create --prefix .conda python=3.11
```

Next, activate the environment:

```bash
conda activate .conda
```

Finally, install the project in editable mode:

```bash
pip install -e .
```

# Example Runs

Train primates, JC69:

```bash
python scripts/hyp_train.py --K 16 --D 2 --lr 0.01 --epochs 200 --jc69 data/primates.phy
```

Train DS1, JC69:

```bash
python scripts/hyp_train.py --K 16 --D 2 --lr 0.01 --epochs 200 --jc69 data/hohna/DS1.phy
```

# Generate Paper Figures

Title: VCSMC vs Hyp SMC with Different K

```bash
python scripts/proposal_K.py
```

Title: Effect of Embedding Initialization

```bash
python scripts/hyp_smc_init_mean.py
```

Title: Effect of Dimensionality

```bash
python scripts/hyp_smc_dim.py
```

Benchmark (output to a .csv):

```bash
python scripts/hyp_smc_benchmark.py
```
