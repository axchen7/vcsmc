from scripts.train.hyp_train import QMatrixType, hyp_train

for i in range(10):
    for K in [1, 2, 4, 8, 16, 32]:
        hyp_train(
            run_name=f"compare_k-{K}",
            file="data/primates.phy",
            lr=0.05,
            epochs=200,
            K=K,
            q_matrix=QMatrixType.MLP,
            sample_branches=False,
            lookahead_merge=True,
            hash_trick=True,
        )
