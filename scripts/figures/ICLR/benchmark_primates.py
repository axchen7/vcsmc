from scripts.train.hyp_train_hybrid import QMatrixType, hyp_train_hybrid

for i in range(5):
    hyp_train_hybrid(
        run_name=f"benchmark_primates",
        file="data/primates.phy",
        lr1=0.05,
        lr2=0.01,
        epochs1=100,
        epochs2=200,
        merge_samples=100,
        K1=512,
        K2=256,
        q_matrix=QMatrixType.MLP,
        lookahead_merge1=True,
        hash_trick1=True,
    )
