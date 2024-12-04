from scripts.train.hyp_train import QMatrixType, hyp_train

for dataset in ["DS3", "DS4", "DS5", "DS6", "DS8"]:
    for K in [1024]:
        print(f"Running {dataset}, K={K}")
        hyp_train(
            lr=0.1,
            epochs=200,
            K=K,
            q_matrix=QMatrixType.STATIONARY,
            hyperbolic=False,
            checkpoint_grads=False,
            file=f"data/hohna/{dataset}.phy",
            run_name=f"VCSMC-{dataset}-K{K}",
        )
