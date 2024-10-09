import csv
import os
from datetime import datetime

import typer

import wandb
from scripts.train.hyp_train_hybrid import QMatrixType, hyp_train_hybrid
from scripts.utils.estimate_latest_run_ll import estimate_latest_run_ll
from vcsmc.utils.wandb_utils import WANDB_PROJECT, WandbRunType

# skip DS7; some literature refers to DS8 as DS7
DATASETS = ["DS1", "DS2", "DS3", "DS4", "DS5", "DS6", "DS8"]

ESTIMATE_LL_SAMPLES = 100


def train(dataset: str, q_matrix: QMatrixType):
    hyp_train_hybrid(
        file=f"data/hohna/{dataset}.phy",
        # below are good parameters...
        lr1=0.05,
        lr2=0.01,
        epochs1=200,
        epochs2=200,
        merge_samples=100,
        K1=16,
        K2=512,
        D=2,
        q_matrix=q_matrix,
        lookahead_merge1=True,
        hash_trick1=True,
        checkpoint_grads1=True,
        run_name=dataset,
    )


def hyp_smc_benchmark(q_matrix: QMatrixType = QMatrixType.JC69):
    date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = f"outputs/benchmarks/hyp_smc_benchmark_{date}.csv"

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)

        # write csv headers
        writer.writerow(["dataset", "LL mean", "LL std dev"])
        f.flush()

        for dataset in DATASETS:
            try:
                train(dataset, q_matrix)
                ll_mean, ll_std_dev = estimate_latest_run_ll(ESTIMATE_LL_SAMPLES)
            except Exception as e:
                print(f"Error while processing {dataset}: {e}")
                wandb.finish(exit_code=1)
                continue

            ll_mean = round(ll_mean, 2)
            ll_std_dev = round(ll_std_dev, 2)

            print(f"{dataset} LL estimate: {ll_mean} ± {ll_std_dev}")

            # write row
            writer.writerow([dataset, ll_mean, ll_std_dev])
            f.flush()

    run = wandb.init(
        project=WANDB_PROJECT,
        job_type=WandbRunType.HYP_SMC_BENCHMARK,
        name="Benchmark Results",
    )
    run.log_artifact(output_file, name="hyp_smc_benchmark", type="result")
    run.finish()


if __name__ == "__main__":
    typer.run(hyp_smc_benchmark)
