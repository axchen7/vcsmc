from io import StringIO

import keras
import matplotlib.pyplot as plt
import tensorflow as tf
from Bio import Phylo
from tqdm import tqdm

import utils
from constants import DTYPE_FLOAT
from type_utils import Dataset, Tensor, tf_function
from vcsmc import VCSMC


@tf_function()
def get_site_positions_SxSfull(data_NxSxA: Tensor) -> Tensor:
    S = data_NxSxA.shape[1]
    site_positions_SxSfull = tf.eye(S, dtype=DTYPE_FLOAT)
    return site_positions_SxSfull


@tf_function()
def batch_by_sites(data_NxSxA: Tensor, batch_size: int | None) -> Dataset:
    """
    Returns a dataset where each element is a tuple
    (data_batched_SxNxA, site_positions_batched_SxSfull).

    Args:
        data_NxSxA: The data.
        batch_size: The batch size. Set to None to use the full dataset.
    """

    if batch_size is None:
        S = data_NxSxA.shape[1]
        batch_size = S

    data_SxNxA = tf.transpose(data_NxSxA, [1, 0, 2])
    site_positions_SxSfull = get_site_positions_SxSfull(data_NxSxA)

    # V = variable batch dimension
    data_ds_VxNxA = tf.data.Dataset.from_tensor_slices(data_SxNxA)
    site_positions_ds_VxSfull = tf.data.Dataset.from_tensor_slices(
        site_positions_SxSfull
    )

    # shape: V x ([N, A], [Sfull])
    dataset = tf.data.Dataset.zip((data_ds_VxNxA, site_positions_ds_VxSfull))

    dataset = dataset.shuffle(dataset.cardinality(), reshuffle_each_iteration=True)

    # shape: V x ([S, N, A], [S, Sfull]), where now S = batch_size
    dataset = dataset.batch(batch_size)

    # convert to shape: V x ([N, S, A], [S, Sfull])
    dataset = dataset.map(
        lambda data_batched_SxNxA, site_positions_batched_SxSfull: (
            tf.transpose(data_batched_SxNxA, [1, 0, 2]),
            site_positions_batched_SxSfull,
        )
    )

    return dataset


def train(
    vcsmc: VCSMC,
    optimizer: keras.optimizers.Optimizer,
    taxa_N: Tensor,
    data_NxSxA: Tensor,
    *,
    root: str = "Healthy",
    epochs: int,
    start_epoch: int = 0,
    sites_batch_size: int | None = None,
    graph_and_profile: bool = False,
):
    site_positions_SxSfull = get_site_positions_SxSfull(data_NxSxA)

    @tf_function()
    def train_step(dataset: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Trains one epoch, iterating through batches.

        Returns:
            log_Z_SMC_sum: Sum across batches of log_Z_SMC.
            log_likelihood_sum: Sum across batches of log likelihoods averaged across particles.
            best_newick_tree: best of the K newick trees from the first epoch.
        """

        log_Z_SMC_sum = tf.constant(0.0, dtype=DTYPE_FLOAT)
        log_likelihood_sum = tf.constant(0.0, dtype=DTYPE_FLOAT)

        best_newick_tree = tf.constant("", dtype=tf.string)

        for data_batched_SxNxA, site_positions_batched_SxSfull in dataset:
            with tf.GradientTape() as tape:
                result = vcsmc(
                    data_NxSxA,
                    data_batched_SxNxA,
                    site_positions_batched_SxSfull,
                    log=True,
                )

                log_Z_SMC = result["log_Z_SMC"]
                log_likelihood_K = result["log_likelihood_K"]

                if best_newick_tree == "":
                    best_newick_tree = result["best_newick_tree"]

                cost = -log_Z_SMC
                # TODO remove regularization
                cost += vcsmc.q_matrix_decoder.regularization()  # scale by N and S?

            variables = tape.watched_variables()
            grads = tape.gradient(cost, variables)
            optimizer.apply_gradients(zip(grads, variables))  # type: ignore

            log_Z_SMC_sum += log_Z_SMC
            log_likelihood_sum += tf.reduce_mean(log_likelihood_K)

        return log_Z_SMC_sum, log_likelihood_sum, best_newick_tree

    @tf_function()
    def get_avg_root_Q_matrix_AxA():
        root_idx = tf.argmax(taxa_N == root)

        root_embedding_1xD = vcsmc.proposal.seq_encoder(
            tf.expand_dims(data_NxSxA[root_idx], 0)
        )
        site_positions_SxC = vcsmc.q_matrix_decoder.site_positions_encoder(
            site_positions_SxSfull
        )
        root_Q_SxAxA = tf.squeeze(
            vcsmc.q_matrix_decoder.Q_matrix_VxSxAxA(
                root_embedding_1xD, site_positions_SxC
            ),
            0,
        )
        avg_root_Q_AxA = tf.reduce_mean(root_Q_SxAxA, 0)
        return avg_root_Q_AxA

    @tf_function()
    def get_data_reconstruction_cosine_similarity():
        embeddings_NxD = vcsmc.proposal.seq_encoder(data_NxSxA)
        site_positions_SxC = vcsmc.q_matrix_decoder.site_positions_encoder(
            site_positions_SxSfull
        )
        reconstructed_NxSxA = vcsmc.q_matrix_decoder.stat_probs_VxSxA(
            embeddings_NxD, site_positions_SxC
        )
        sim = tf.reduce_sum(data_NxSxA * reconstructed_NxSxA)
        sim /= tf.norm(data_NxSxA) * tf.norm(reconstructed_NxSxA)
        return sim

    N = data_NxSxA.shape[0]

    results_dir = utils.create_results_dir()
    summary_writer = tf.summary.create_file_writer(results_dir)  # type: ignore

    # ===== batch data =====

    dataset = batch_by_sites(data_NxSxA, sites_batch_size)

    # ===== generate graph and profile =====

    if graph_and_profile:
        data_batched_NxSxA, site_positions_batched_SxSfull = next(iter(dataset))

        tf.summary.trace_on(graph=True, profiler=True)  # type: ignore
        vcsmc(data_NxSxA, data_batched_NxSxA, site_positions_batched_SxSfull)
        with summary_writer.as_default():
            tf.summary.trace_export(name="vcsmc", step=0, profiler_outdir=results_dir)  # type: ignore

    # ===== train =====

    for epoch in tqdm(range(epochs)):
        with summary_writer.as_default(step=epoch + start_epoch):
            log_Z_SMC_sum, log_likelihood_avg, best_newick_tree = train_step(dataset)

            cosine_similarity = get_data_reconstruction_cosine_similarity()

            tf.summary.scalar("Elbo", log_Z_SMC_sum)
            tf.summary.scalar("Log likelihood avg", log_likelihood_avg)
            tf.summary.scalar(
                "Data reconstruction cosine similarity",
                cosine_similarity,
            )

            if (epoch + 1) % 4 == 0:
                # # ===== leaf embeddings =====

                # leaf_embeddings = evsmc.leaf_embeddings(data_SxNxA)
                # leaf_embeddings = evsmc.distance.project_many(leaf_embeddings)

                # plt.scatter(leaf_embeddings[:, 0], leaf_embeddings[:, 1])

                # for i, taxon in enumerate(taxa):
                #     plt.annotate(
                #         taxon,
                #         (leaf_embeddings[i, 0], leaf_embeddings[i, 1]),
                #         fontsize="xx-small",
                #     )

                # tf.summary.image("Leaf embeddings", utils.cur_plt_to_tf_image())

                # ===== best tree =====

                _, axes = plt.subplots(figsize=(10, N * 0.2))

                phylo_tree = Phylo.read(  # type: ignore
                    StringIO(best_newick_tree.numpy().decode()), "newick"
                )
                phylo_tree.root_with_outgroup(root)
                Phylo.draw(phylo_tree, axes=axes, do_show=False)  # type: ignore

                tf.summary.image("Best tree", utils.cur_plt_to_tf_image())

                # ===== Q matrix =====

                plt.imshow(get_avg_root_Q_matrix_AxA())
                tf.summary.image(
                    "Root Q matrix (average across sites)", utils.cur_plt_to_tf_image()
                )

    # ===== done training! =====

    print("Training complete!")
