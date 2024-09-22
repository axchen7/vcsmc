from scripts.train.hyp_train import QMatrixType, hyp_train

for i in range(5):
    for q_matrix in [QMatrixType.JC69, QMatrixType.STATIONARY, QMatrixType.MLP]:
        hyp_train(
            run_name=f"compare_q_matrix-{q_matrix.value}",
            file="data/primates.phy",
            lr=0.01,
            epochs=200,
            K=512,
            q_matrix=q_matrix,
            sample_branches=False,
            lookahead_merge=True,
            hash_trick=True,
        )
