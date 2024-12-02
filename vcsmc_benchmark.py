from scripts.train.hyp_train import QMatrixType, hyp_train

for dataset in ["DS1", "DS2", "DS3", "DS4", "DS5", "DS6", "DS8"]:
    print(f"Running {dataset}...")
    hyp_train(
        lr=0.1,
        epochs=100,
        K=512,
        q_matrix=QMatrixType.STATIONARY,
        hyperbolic=False,
        file=f"data/hohna/{dataset}.phy",
        run_name=f"VCSMC-{dataset}",
    )
