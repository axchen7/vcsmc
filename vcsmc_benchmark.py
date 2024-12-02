from scripts.train.hyp_train import QMatrixType, hyp_train

for dataset in ["DS1", "DS2", "DS3", "DS4", "DS5", "DS6", "DS8"]:
    for K in [512, 1024, 2048]:
        print(f"Running {dataset}, K={K}")
        try:
            hyp_train(
                lr=0.1,
                epochs=200,
                K=K,
                q_matrix=QMatrixType.STATIONARY,
                hyperbolic=False,
                checkpoint_grads=True,
                file=f"data/hohna/{dataset}.phy",
                run_name=f"VCSMC-{dataset}-K{K}",
            )
        except Exception as e:
            print(f"Error running {dataset}, K={K}: {e}")
