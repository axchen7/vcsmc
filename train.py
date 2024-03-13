from io import StringIO

import matplotlib.pyplot as plt
import torch
from Bio import Phylo
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter  # type: ignore
from tqdm import tqdm

from vcsmc import VCSMC, VCSMC_Result


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
    *,
    root: str = "Healthy",
    epochs: int,
    start_epoch: int = 0,
    sites_batch_size: int | None = None,
    graph_and_profile: bool = False,
):
    # ===== setup =====

    N = data_NxSxA.shape[0]
    root_idx = taxa_N.index(root)

    site_positions_SxSfull = get_site_positions_SxSfull(data_NxSxA)

    writer = SummaryWriter()

    # ===== helper functions =====

    def train_step(
        dataloader: DataLoader,
    ) -> tuple[Tensor, Tensor | None, Tensor, str]:
        """
        Trains one epoch, iterating through batches.

        Returns:
            log_Z_SMC_sum: Sum across batches of log_Z_SMC.
            log_likelihood_K: log likelihoods, or None if there are multiple batches.
            log_likelihood_sum: Sum across batches of log likelihoods averaged across particles.
            best_newick_tree: best of the K newick trees from the first epoch.
        """

        log_Z_SMC_sum = torch.tensor(0.0)
        log_likelihood_sum = torch.tensor(0.0)

        best_newick_tree = ""

        log_likelihood_K = None

        for data_batched_SxNxA, site_positions_batched_SxSfull in dataloader:
            data_batched_NxSxA = data_batched_SxNxA.permute(1, 0, 2)

            result: VCSMC_Result = vcsmc(
                data_NxSxA,
                data_batched_NxSxA,
                site_positions_batched_SxSfull,
                log=True,
            )

            log_Z_SMC = result["log_Z_SMC"]
            log_likelihood_K = result["log_likelihood_K"]

            log_likelihood_avg = log_likelihood_K.mean()

            if best_newick_tree == "":
                best_newick_tree = result["best_newick_tree"]

            loss = -log_Z_SMC

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            log_Z_SMC_sum = log_Z_SMC_sum + log_Z_SMC
            log_likelihood_sum = log_likelihood_sum + log_likelihood_avg

        if len(dataloader) > 1:
            log_likelihood_K = None

        return (log_Z_SMC_sum, log_likelihood_K, log_likelihood_sum, best_newick_tree)

    def get_avg_root_Q_matrix_AxA():
        with torch.no_grad():
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

    def get_data_reconstruction_cosine_similarity():
        with torch.no_grad():
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

    # ===== generate graph and profile =====

    # TODO migrate to torch

    # if graph_and_profile:
    #     data_batched_NxSxA, site_positions_batched_SxSfull = next(iter(dataset))

    #     tf.summary.trace_on(graph=True, profiler=True)  # type: ignore
    #     vcsmc(data_NxSxA, data_batched_NxSxA, site_positions_batched_SxSfull)
    #     with summary_writer.as_default():
    #         tf.summary.trace_export(name="vcsmc", step=0, profiler_outdir=results_dir)  # type: ignore

    # ===== train =====

    for epoch in tqdm(range(epochs)):
        epoch += start_epoch

        log_Z_SMC_sum, log_likelihood_K, log_likelihood_avg, best_newick_tree = (
            train_step(dataloader)
        )

        cosine_similarity = get_data_reconstruction_cosine_similarity()

        writer.add_scalar("Elbo", log_Z_SMC_sum, epoch)
        writer.add_scalar("Log likelihood avg", log_likelihood_avg, epoch)
        writer.add_scalar(
            "Data reconstruction cosine similarity",
            cosine_similarity,
            epoch,
        )

        if log_likelihood_K is not None:
            writer.add_histogram("Log likelihoods", log_likelihood_K, epoch)

        if (epoch + 1) % 4 == 0:
            # ===== best tree =====

            fig, ax = plt.subplots(figsize=(10, N * 0.2))

            phylo_tree = Phylo.read(  # type: ignore
                StringIO(best_newick_tree), "newick"
            )
            phylo_tree.root_with_outgroup(root)
            Phylo.draw(phylo_tree, axes=ax, do_show=False)  # type: ignore

            writer.add_figure("Best tree", fig, epoch)

            # ===== Q matrix =====

            fig, ax = plt.subplots()
            ax.imshow(get_avg_root_Q_matrix_AxA())
            writer.add_figure("Root Q matrix (average across sites)", fig, epoch)

        writer.flush()

    # ===== done training! =====

    print("Training complete!")
