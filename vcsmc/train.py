import os
from datetime import datetime
from io import StringIO
from os import path
from typing import Callable, Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
from Bio import Phylo
from torch import Tensor
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb

from .encoders import Hyperbolic
from .proposals import EmbeddingProposal
from .q_matrix_decoders import JC69QMatrixDecoder
from .site_positions_encoders import DummySitePositionsEncoder
from .utils.poincare_utils import PoincarePlot
from .utils.repr_utils import module_to_config
from .utils.train_types import TrainArgs, TrainCheckpoint, TrainResults
from .utils.train_utils import (
    batch_by_sites,
    fig_to_wandb_image,
    find_most_recent_path,
    get_site_positions_SxSfull,
)
from .utils.vcsmc_types import VcsmcResult
from .utils.wandb_utils import WANDB_PROJECT, WandbRunType
from .vcsmc import VCSMC

__all__ = ["train", "load_checkpoint", "train_from_checkpoint"]


def get_checkpoints_dir(run_name: str | None = None) -> str:
    date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    parent_dir = "checkpoints"
    child_dir = date
    if run_name is not None:
        child_dir += f"_{run_name}"
    return path.join(parent_dir, child_dir)


def train(
    vcsmc: VCSMC,
    optimizer: Optimizer,
    taxa_N: list[str],
    data_NxSxA: Tensor,
    file: str,
    *,
    lr_scheduler: LRScheduler | None = None,
    root: str | None = None,
    epochs: int,
    start_epoch: int = 0,
    grad_accumulation_steps: int = 1,
    sites_batch_size: int | None = None,
    sample_taxa_count: int | None = None,
    run_name: str | None = None,
):
    # ===== setup =====

    checkpoints_dir = get_checkpoints_dir(run_name)
    os.makedirs(checkpoints_dir, exist_ok=True)

    if root is None:
        outgroup_root = None
        root_idx = 0
    else:
        outgroup_root = root
        root_idx = taxa_N.index(root)

    site_positions_SxSfull = get_site_positions_SxSfull(data_NxSxA)

    # track data across epochs
    ZCSMCs: list[float] = []
    log_likelihood_avgs: list[float] = []

    # ===== helper functions =====

    def get_args() -> TrainArgs:
        return {
            "taxa_N": taxa_N,
            "data_NxSxA": data_NxSxA,
            "file": file,
            "root": root,
            "epochs": epochs,
            "grad_accumulation_steps": grad_accumulation_steps,
            "sites_batch_size": sites_batch_size,
            "sample_taxa_count": sample_taxa_count,
            "run_name": run_name,
        }

    def get_checkpoint() -> TrainCheckpoint:
        return {
            "vcsmc": vcsmc,
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "start_epoch": start_epoch,
        }

    def init_wandb():
        config = {
            **get_args(),
            **get_checkpoint(),
            **module_to_config(vcsmc),
            "lr": optimizer.param_groups[0].get("lr"),
        }
        config = {
            k: v for k, v in config.items() if isinstance(v, (bool, int, float, str))
        }

        return wandb.init(
            project=WANDB_PROJECT,
            job_type=WandbRunType.TRAIN,
            name=run_name,
            config=config,
        )

    def save_args():
        filename = "args.pt"
        torch.save(get_args(), path.join(checkpoints_dir, filename))

    def save_checkpoint(start_epoch: int):
        filename = f"checkpoint_{start_epoch}.pt"
        torch.save(get_checkpoint(), path.join(checkpoints_dir, filename))

    def save_results():
        results: TrainResults = {
            "start_epoch": start_epoch,  # passed in via train() args
            "ZCSMCs": ZCSMCs,
            "log_likelihood_avgs": log_likelihood_avgs,
        }
        filename = "results.pt"
        torch.save(results, path.join(checkpoints_dir, filename))

    def train_step(
        dataloader: DataLoader,
    ) -> tuple[float, list[float] | None, float, str, VcsmcResult | None]:
        """
        Trains one epoch, iterating through batches.

        Returns:
            log_ZCSMC_sum: Sum across batches of ZCSMCs.
            log_likelihoods: log likelihoods (across all grad accumulation steps), or None if there are multiple batches.
            log_likelihood_sum: Sum across batches of log likelihoods averaged across particles.
            best_newick_tree: best of the K newick trees from the first epoch (last grad accumulation step).
            result: Vcsmc result, or None if there are multiple batches (last grad accumulation step).
        """

        log_ZCSMC_sum = 0.0
        log_likelihood_sum = 0.0

        best_newick_tree = ""

        log_likelihoods: list[float] | None = None
        result: VcsmcResult | None = None

        for data_batched_SxNxA, site_positions_batched_SxSfull in dataloader:
            data_batched_NxSxA = data_batched_SxNxA.permute(1, 0, 2)

            if sample_taxa_count is not None:
                # sample sample_taxa_count indices out of N total, without replacement
                N = len(taxa_N)
                indices = torch.randperm(N)[:sample_taxa_count]

                samp_taxa_N = [taxa_N[i] for i in indices]
                samp_data_NxSxA = data_NxSxA[indices]
                samp_data_batched_NxSxA = data_batched_NxSxA[indices]
            else:
                samp_taxa_N = taxa_N
                samp_data_NxSxA = data_NxSxA
                samp_data_batched_NxSxA = data_batched_NxSxA

            log_likelihoods = []

            for _ in range(grad_accumulation_steps):
                cur_result: VcsmcResult = vcsmc(
                    samp_taxa_N,
                    samp_data_NxSxA,
                    samp_data_batched_NxSxA,
                    site_positions_batched_SxSfull,
                )
                result = cur_result

                log_ZCSMC = cur_result["log_ZCSMC"]
                log_likelihood_K = cur_result["log_likelihood_K"]

                log_likelihood_avg = log_likelihood_K.mean()

                if best_newick_tree == "":
                    best_newick_tree = cur_result["best_newick_tree"]

                loss = -log_ZCSMC
                loss = loss / grad_accumulation_steps
                loss.backward()

                log_ZCSMC_sum += log_ZCSMC.item() / grad_accumulation_steps
                log_likelihood_sum += (
                    log_likelihood_avg.item() / grad_accumulation_steps
                )

                log_likelihoods.extend(log_likelihood_K.tolist())

            optimizer.step()
            optimizer.zero_grad()

        if lr_scheduler is not None:
            lr_scheduler.step()

        if len(dataloader) > 1:
            log_likelihoods = None
            result = None

        return (
            log_ZCSMC_sum,
            log_likelihoods,
            log_likelihood_sum,
            best_newick_tree,
            result,
        )

    @torch.no_grad()
    def get_avg_root_Q_matrix_AxA():
        root_embedding_1xD = vcsmc.proposal.seq_encoder(
            data_NxSxA[root_idx].unsqueeze(0)
        )
        site_positions_SxC = vcsmc.q_matrix_decoder.site_positions_encoder(
            site_positions_SxSfull
        )
        root_Q_SxAxA = vcsmc.q_matrix_decoder.Q_matrix_VxSxAxA(
            root_embedding_1xD, site_positions_SxC
        ).squeeze(0)
        avg_root_Q_AxA = root_Q_SxAxA.mean(0)
        return avg_root_Q_AxA

    @torch.no_grad()
    def get_data_reconstruction_cosine_similarity():
        embeddings_NxD = vcsmc.proposal.seq_encoder(data_NxSxA)
        site_positions_SxC = vcsmc.q_matrix_decoder.site_positions_encoder(
            site_positions_SxSfull
        )
        reconstructed_NxSxA = vcsmc.q_matrix_decoder.stat_probs_VxSxA(
            embeddings_NxD, site_positions_SxC
        )
        sim = torch.sum(data_NxSxA * reconstructed_NxSxA)
        sim = sim / (data_NxSxA.norm() * reconstructed_NxSxA.norm())
        return sim

    # ===== batch data =====

    dataloader = batch_by_sites(data_NxSxA, sites_batch_size)

    # ===== train =====

    run = init_wandb()
    save_args()
    save_checkpoint(start_epoch)

    for epoch in tqdm(range(epochs - start_epoch), desc="Training"):
        epoch += start_epoch

        (
            log_ZCSMC_sum,
            log_likelihoods,
            log_likelihood_avg,
            best_newick_tree,
            result,
        ) = train_step(dataloader)

        save_checkpoint(epoch + 1)
        save_results()  # overwrite each time

        ZCSMCs.append(log_ZCSMC_sum)
        log_likelihood_avgs.append(log_likelihood_avg)

        log: dict = {
            "Log ZCSMC": log_ZCSMC_sum,
            "Log likelihood avg": log_likelihood_avg,
        }

        if log_likelihoods is not None:
            log["Log likelihoods"] = wandb.Histogram(log_likelihoods)

        if not isinstance(
            vcsmc.q_matrix_decoder.site_positions_encoder, DummySitePositionsEncoder
        ):
            log["Data reconstruction cosine similarity"] = (
                get_data_reconstruction_cosine_similarity()
            )

        if isinstance(vcsmc.proposal.seq_encoder.distance, Hyperbolic):
            log["Hyperbolic scale"] = vcsmc.proposal.seq_encoder.distance.scale()

        if (
            isinstance(vcsmc.proposal, EmbeddingProposal)
            and vcsmc.proposal.sample_branches
        ):
            log["Sample branches sigma"] = vcsmc.proposal.sample_branches_sigma()

        if (epoch + 1) % 4 == 0:
            # ===== best tree =====

            N = sample_taxa_count if sample_taxa_count is not None else len(taxa_N)
            fig, ax = plt.subplots(figsize=(10, N * 0.2))

            phylo_tree = Phylo.read(  # type: ignore
                StringIO(best_newick_tree), "newick"
            )
            if outgroup_root is not None:
                phylo_tree.root_with_outgroup(outgroup_root)
            Phylo.draw(phylo_tree, axes=ax, do_show=False)  # type: ignore

            log["Best tree"] = fig_to_wandb_image(fig)

            # ===== Q matrix =====

            if not isinstance(vcsmc.q_matrix_decoder, JC69QMatrixDecoder):
                fig, ax = plt.subplots()
                ax.imshow(get_avg_root_Q_matrix_AxA().cpu())
                log["Root Q matrix (average across sites)"] = fig_to_wandb_image(fig)

            # ===== poincare plot =====

            if (
                result
                and isinstance(vcsmc.proposal, EmbeddingProposal)
                and isinstance(vcsmc.proposal.seq_encoder.distance, Hyperbolic)
                and vcsmc.proposal.seq_encoder.D == 2
            ):
                interactive_poincare = PoincarePlot(vcsmc, taxa_N, data_NxSxA, result)
                poincare_image = interactive_poincare.to_wandb_image()
                if poincare_image:
                    log["Poincare plot"] = poincare_image

        run.log(log, step=epoch, commit=True)

    # ===== done training! =====

    print("Training complete!")

    run.log_artifact(checkpoints_dir, name="checkpoints", type="checkpoint")
    run.finish()


def load_checkpoint(
    *,
    start_epoch: int | Literal["best"] | None = None,
    search_dir: str = "checkpoints",
) -> tuple[TrainArgs, TrainCheckpoint]:
    """
    Args:
        start_epoch: Load the state at the start of this epoch (before parameter update).
            If None, loads the latest checkpoint.
            If "best", loads the epoch with the highest ZCSMC.
        search_dir: The directory to search for the checkpoint. E.g. "checkpoints/*label".

    Returns:
        args, checkpoint
    """

    def find_best_epoch():
        """Returns the epoch with the highest ZCSMC"""
        results: TrainResults = torch.load(
            find_most_recent_path(search_dir, "results.pt"),
            weights_only=False,
        )
        # there is no off-by-one error here: say epoch i has the highest
        # pre-update LL, so we want to start at epoch i; in this case,
        # results[i] is max, and loading epoch i will give the model state
        # before the parameter update at step i
        return int(np.argmax(results["ZCSMCs"])) + results["start_epoch"]

    if start_epoch == "best":
        start_epoch = find_best_epoch()

    checkpoint_glob = (
        "checkpoint_*.pt" if start_epoch is None else f"checkpoint_{start_epoch}.pt"
    )

    args: TrainArgs = torch.load(
        find_most_recent_path(search_dir, "args.pt"),
        weights_only=False,
    )
    checkpoint: TrainCheckpoint = torch.load(
        find_most_recent_path(search_dir, checkpoint_glob),
        weights_only=False,
    )

    start_epoch = checkpoint["start_epoch"]
    print(
        f"Loaded checkpoint at the start of epoch {start_epoch} (after epoch {start_epoch - 1})."
    )

    return args, checkpoint


def train_from_checkpoint(
    *,
    additional_epochs: int,
    start_epoch: int | Literal["best"] | None = None,
    search_dir: str = "checkpoints",
    modify_args: Callable[[TrainArgs, TrainCheckpoint], None] | None = None,
) -> tuple[Tensor, list[str], VCSMC]:
    """
    Args:
        additional_epochs: Train for this many additional epochs.
            If set, overrides the number of epochs in the checkpoint.
        start_epoch: Start training from the state at the start of this epoch (before parameter update).
            If None, starts from the latest checkpoint.
            If "best", starts from the epoch with the highest ZCSMC.
        search_dir: The directory to search for the checkpoint. E.g. "checkpoints/*label".
        modify_args: Function to mutate the args and checkpoint before training.

    Returns:
        data_NxSxA, taxa_N, vcsmc
    """

    args, checkpoint = load_checkpoint(start_epoch=start_epoch, search_dir=search_dir)

    if modify_args:
        modify_args(args, checkpoint)

    data_NxSxA = args["data_NxSxA"]
    taxa_N = args["taxa_N"]
    vcsmc = checkpoint["vcsmc"]

    args["epochs"] = checkpoint["start_epoch"] + additional_epochs

    train(
        vcsmc,
        checkpoint["optimizer"],
        lr_scheduler=checkpoint["lr_scheduler"],
        start_epoch=checkpoint["start_epoch"],
        **args,
    )

    return data_NxSxA, taxa_N, vcsmc
