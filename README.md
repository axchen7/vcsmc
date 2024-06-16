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
python scripts/train_driver.py --K 16 --D 2 --lr 0.01 --epochs 200 --jc69 data/primates.phy
```

Train DS1, JC69:

```bash
python scripts/train_driver.py --K 16 --D 2 --lr 0.01 --epochs 200 --jc69 data/hohna/DS1.phy
```
