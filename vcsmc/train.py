import os
from io import StringIO
from typing import Callable, Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
from Bio import Phylo
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter  # type: ignore
from tqdm import tqdm

from .encoders import Hyperbolic
from .train_utils import (
    TemperatureScheduler,
    TrainArgs,
    TrainCheckpoint,
    TrainResults,
    find_most_recent_path,
)
from .vcsmc import VCSMC, VcsmcResult


def get_site_positions_SxSfull(data_NxSxA: Tensor) -> Tensor:
    S = data_NxSxA.shape[1]
    site_positions_SxSfull = torch.eye(
        S, dtype=data_NxSxA.dtype, device=data_NxSxA.device
    )
    return site_positions_SxSfull


def batch_by_sites(data_NxSxA: Tensor, batch_size: int | None) -> DataLoader:
    """
    Returns a (mapped) DataLoader where each element is a tuple
    (data_batched_SxNxA, site_positions_batched_SxSfull).

    Args:
        data_NxSxA: The data.
        batch_size: The batch size. Set to None to use the full dataset.
    """

    if batch_size is None:
        S = data_NxSxA.shape[1]
        batch_size = S

    data_SxNxA = data_NxSxA.permute(1, 0, 2)
    site_positions_SxSfull = get_site_positions_SxSfull(data_NxSxA)

    # shape: S x ([N, A], [Sfull])
    dataset = TensorDataset(data_SxNxA, site_positions_SxSfull)

    # shape: V x ([S, N, A], [S, Sfull]), where now S = batch_size
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def train(
    vcsmc: VCSMC,
    optimizer: torch.optim.Optimizer,
    taxa_N: list[str],
    data_NxSxA: Tensor,
    file: str,
    *,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    temperature_scheduler: TemperatureScheduler | None = None,
    root: str | None = None,
    epochs: int,
    start_epoch: int = 0,
    sites_batch_size: int | None = None,
    sample_taxa_count: int | None = None,
    run_name: str | None = None,
    profile: bool = False,
):
    # ===== setup =====

    if root is None:
        outgroup_root = None
        root_idx = 0
    else:
        outgroup_root = root
        root_idx = taxa_N.index(root)

    site_positions_SxSfull = get_site_positions_SxSfull(data_NxSxA)

    writer = SummaryWriter(
        comment=f"-{run_name}" if run_name is not None else "",
    )

    # track data across epochs
    ZCSMCs: list[float] = []
    log_likelihood_avgs: list[float] = []

    # ===== helper functions =====

    def get_checkpoints_dir():
        dirname = os.path.join(writer.get_logdir(), "checkpoints")
        os.makedirs(dirname, exist_ok=True)
        return dirname

    def save_args():
        args: TrainArgs = {
            "taxa_N": taxa_N,
            "data_NxSxA": data_NxSxA,
            "file": file,
            "temperature_scheduler": temperature_scheduler,
            "root": root,
            "epochs": epochs,
            "sites_batch_size": sites_batch_size,
            "sample_taxa_count": sample_taxa_count,
            "run_name": run_name,
        }
        filename = "args.pt"
        torch.save(args, os.path.join(get_checkpoints_dir(), filename))

    def save_checkpoint(start_epoch: int):
        checkpoint: TrainCheckpoint = {
            "vcsmc": vcsmc,
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "start_epoch": start_epoch,
        }
        filename = f"checkpoint_{start_epoch}.pt"
        torch.save(checkpoint, os.path.join(get_checkpoints_dir(), filename))

    def save_results():
        results: TrainResults = {
            "ZCSMCs": ZCSMCs,
            "log_likelihood_avgs": log_likelihood_avgs,
        }
        filename = "results.pt"
        torch.save(results, os.path.join(get_checkpoints_dir(), filename))

    def train_step(
        dataloader: DataLoader,
        epoch: int,
    ) -> tuple[float, Tensor | None, float, str]:
        """
        Trains one epoch, iterating through batches.

        Returns:
            log_ZCSMC_sum: Sum across batches of ZCSMCs.
            log_likelihood_K: log likelihoods, or None if there are multiple batches.
            log_likelihood_sum: Sum across batches of log likelihoods averaged across particles.
            best_newick_tree: best of the K newick trees from the first epoch.
        """

        log_ZCSMC_sum = 0.0
        log_likelihood_sum = 0.0

        best_newick_tree = ""

        log_likelihood_K = None

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

            temperature = None
            if temperature_scheduler is not None:
                temperature = temperature_scheduler(epoch)

            result: VcsmcResult = vcsmc(
                samp_taxa_N,
                samp_data_NxSxA,
                samp_data_batched_NxSxA,
                site_positions_batched_SxSfull,
                temperature=temperature,
            )

            log_ZCSMC = result["log_ZCSMC"]
            log_likelihood_K = result["log_likelihood_K"]

            log_likelihood_avg = log_likelihood_K.mean()

            if best_newick_tree == "":
                best_newick_tree = result["best_newick_tree"]

            loss = -log_ZCSMC

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            log_ZCSMC_sum += log_ZCSMC.item()
            log_likelihood_sum += log_likelihood_avg.item()

        if lr_scheduler is not None:
            lr_scheduler.step()

        if len(dataloader) > 1:
            log_likelihood_K = None

        return log_ZCSMC_sum, log_likelihood_K, log_likelihood_sum, best_newick_tree

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

    save_args()
    save_checkpoint(start_epoch)

    def train_epochs(prof: torch.profiler.profile | None):
        for epoch in tqdm(range(epochs - start_epoch)):
            if prof:
                prof.step()

            epoch += start_epoch

            log_ZCSMC_sum, log_likelihood_K, log_likelihood_avg, best_newick_tree = (
                train_step(dataloader, epoch)
            )

            save_checkpoint(epoch + 1)

            cosine_similarity = get_data_reconstruction_cosine_similarity()

            ZCSMCs.append(log_ZCSMC_sum)
            log_likelihood_avgs.append(log_likelihood_avg)

            writer.add_scalar("Log ZCSMC", log_ZCSMC_sum, epoch)
            writer.add_scalar("Log likelihood avg", log_likelihood_avg, epoch)
            writer.add_scalar(
                "Data reconstruction cosine similarity",
                cosine_similarity,
                epoch,
            )

            if isinstance(vcsmc.proposal.seq_encoder.distance, Hyperbolic):
                writer.add_scalar(
                    "Hyperbolic scale",
                    vcsmc.proposal.seq_encoder.distance.scale(),
                    epoch,
                )

            if log_likelihood_K is not None:
                writer.add_histogram("Log likelihoods", log_likelihood_K, epoch)

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

                writer.add_figure("Best tree", fig, epoch)

                # ===== Q matrix =====

                fig, ax = plt.subplots()
                ax.imshow(get_avg_root_Q_matrix_AxA().cpu())
                writer.add_figure("Root Q matrix (average across sites)", fig, epoch)

            writer.flush()

    if profile:
        with torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                writer.get_logdir()
            ),
            record_shapes=True,
            profile_memory=True,
            # with_stack=True, (deprecated/broken)
        ) as prof:
            train_epochs(prof)
    else:
        train_epochs(None)

    # ===== done training! =====

    print("Training complete!")
    save_results()


def train_from_checkpoint(
    *,
    epochs: int | None = None,
    load_only: bool = False,
    start_epoch: int | Literal["best"] | None = None,
    modify_args: Callable[[TrainArgs, TrainCheckpoint], None] | None = None,
    search_dir: str = "runs",
) -> tuple[Tensor, list[str], VCSMC]:
    """
    Args:
        epochs: The epoch to stop training at.
            If set, overrides the number of epochs in the checkpoint.
        load_only: If True, loads the checkpoint and returns the data and model without training.
        start_epoch: The epoch to start training from. If None, starts from the latest checkpoint.
            If "best", starts from the epoch with the highest ZCSMC.
        modify_args: A function that modifies the args and checkpoint before training.
        search_dir: The directory to search for the checkpoint. E.g. "runs/*label".

    Returns:
        data_NxSxA, taxa_N, vcsmc
    """

    checkpoints_dir = find_most_recent_path(search_dir, "checkpoints")

    def find_best_epoch():
        """Returns the epoch with the highest ZCSMC"""
        results: TrainResults = torch.load(
            find_most_recent_path(checkpoints_dir, "results.pt")
        )
        # there is no off-by-one error here: say epoch 1 has the highest LL;
        # then, results[0] is max, and loading epoch 0 will give the model state
        # before the optimizer step at epoch 1
        return int(np.argmax(results["ZCSMCs"]))

    if start_epoch == "best":
        start_epoch = find_best_epoch()

    checkpoint_glob = (
        "checkpoint_*.pt" if start_epoch is None else f"checkpoint_{start_epoch}.pt"
    )

    args: TrainArgs = torch.load(find_most_recent_path(checkpoints_dir, "args.pt"))
    checkpoint: TrainCheckpoint = torch.load(
        find_most_recent_path(checkpoints_dir, checkpoint_glob)
    )

    if modify_args is not None:
        modify_args(args, checkpoint)

    print(f"Loaded checkpoint at epoch {checkpoint['start_epoch']}.")

    data_NxSxA = args["data_NxSxA"]
    taxa_N = args["taxa_N"]

    if epochs is not None:
        args["epochs"] = epochs

    vcsmc = checkpoint["vcsmc"]

    if not load_only:
        train(
            vcsmc,
            checkpoint["optimizer"],
            lr_scheduler=checkpoint["lr_scheduler"],
            start_epoch=checkpoint["start_epoch"],
            **args,
        )

    return data_NxSxA, taxa_N, vcsmc


@torch.no_grad()
def evaluate(
    vcsmc: VCSMC,
    taxa_N: list[str],
    data_NxSxA: Tensor,
    *,
    temperature: float | None = None,
) -> VcsmcResult:
    dataset = batch_by_sites(data_NxSxA, None)

    # batch is actually the full dataset
    data_batched_SxNxA, site_positions_batched_SxSfull = next(iter(dataset))
    data_batched_NxSxA = data_batched_SxNxA.permute(1, 0, 2)

    return vcsmc(
        taxa_N,
        data_NxSxA,
        data_batched_NxSxA,
        site_positions_batched_SxSfull,
        temperature=temperature,
    )
