from scripts.train.hyp_train import QMatrixType, hyp_train

for K in [128, 256, 512]:
    for dataset in ["DS3", "DS4", "DS5", "DS6", "DS8"]:
        print(f"Running {dataset}, K={K}")
        hyp_train(
            lr=0.1,
            epochs=200,
            K=K,
            q_matrix=QMatrixType.STATIONARY,
            hyperbolic=False,
            checkpoint_grads=False,
            lookahead_merge=True,
            file=f"data/hohna/{dataset}.phy",
            run_name=f"VCSMC-{dataset}-K{K}",
        )
