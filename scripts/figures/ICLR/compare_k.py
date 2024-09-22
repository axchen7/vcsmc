from scripts.train.hyp_train import QMatrixType, hyp_train

for i in range(3):
    for K in [16, 32, 64, 128, 256, 512]:
        hyp_train(
            run_name=f"compare_k-{K}",
            file="data/primates.phy",
            lr=0.01,
            epochs=200,
            K=K,
            q_matrix=QMatrixType.MLP,
            sample_branches=False,
            lookahead_merge=False,
            hash_trick=True,
        )
