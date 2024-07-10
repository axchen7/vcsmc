import typer

from scripts.train.hyp_train import hyp_train

NO_LOOKAHEAD_K_VALS = [4, 8, 16, 32, 64, 128, 256, 512]
LOOKAHEAD_K_VALS = [4, 8, 16, 32, 64, 128]


def different_proposals_and_K(file: str = "data/primates.phy", epochs: int = 200):
    for K in NO_LOOKAHEAD_K_VALS:
        # fmt: off
        args = dict(epochs=epochs, K=K, jc69=True, sample_branches=True, lookahead_merge=False, checkpoint_grads=False)
        # fmt: on
        hyp_train(file=file, lr=0.1, hyperbolic=False, run_name=f"VCSMC-K{K}", **args)  # type: ignore
        hyp_train(file=file, lr=0.01, hyperbolic=True, run_name=f"HYP-SMC-K{K}", **args)  # type: ignore

    for K in LOOKAHEAD_K_VALS:
        # fmt: off
        args = dict(epochs=epochs, K=K, jc69=True, sample_branches=True, lookahead_merge=True, checkpoint_grads=True)
        # fmt: on
        hyp_train(file=file, lr=0.1, hyperbolic=False, run_name=f"VCSMC-K{K}-LOOKAHEAD", **args)  # type: ignore
        hyp_train(file=file, lr=0.01, hyperbolic=True, run_name=f"HYP-SMC-K{K}-LOOKAHEAD", **args)  # type: ignore


if __name__ == "__main__":
    typer.run(different_proposals_and_K)
