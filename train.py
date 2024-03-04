from io import StringIO

import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from Bio import Phylo
from tqdm import tqdm

import utils
from type_utils import Tensor, tf_function
from vcsmc import VCSMC


def train(
    vcsmc: VCSMC,
    optimizer: keras.optimizers.Optimizer,
    taxa_N: Tensor,
    data_NxSxA: Tensor,
    *,
    root: str = "Healthy",
    epochs: int,
    start_epoch: int = 0,
    graph_and_profile: bool = False,
):
    @tf_function()
    def train_step(batch_NxSxA: Tensor):
        with tf.GradientTape() as tape:
            result = vcsmc(batch_NxSxA, log=True)
            log_Z_SMC = result["log_Z_SMC"]
            log_likelihood_K = result["log_likelihood_K"]
            best_newick_tree = result["best_newick_tree"]

            cost = -log_Z_SMC
            cost += vcsmc.q_matrix_decoder.regularization()  # scale by N and S?

        variables = tape.watched_variables()
        grads = tape.gradient(cost, variables)
        optimizer.apply_gradients(zip(grads, variables))  # type: ignore

        return log_Z_SMC, log_likelihood_K, best_newick_tree

    @tf_function()
    def get_avg_root_Q_matrix_AxA():
        root_idx = tf.argmax(taxa_N == root)

        root_embedding_1xD = vcsmc.proposal.seq_encoder(
            tf.expand_dims(data_NxSxA[root_idx], 0)
        )
        root_Q_SxAxA = tf.squeeze(
            vcsmc.q_matrix_decoder.Q_matrix_VxSxAxA(root_embedding_1xD), 0
        )
        avg_root_Q_AxA = tf.reduce_mean(root_Q_SxAxA, 0)
        return avg_root_Q_AxA

    @tf_function()
    def get_data_reconstruction_cosine_similarity():
        embeddings_NxD = vcsmc.proposal.seq_encoder(data_NxSxA)
        reconstructed_NxSxA = vcsmc.q_matrix_decoder.stat_probs_VxSxA(embeddings_NxD)
        sim = tf.reduce_sum(data_NxSxA * reconstructed_NxSxA)
        sim /= tf.norm(data_NxSxA) * tf.norm(reconstructed_NxSxA)
        return sim

    N = data_NxSxA.shape[0]

    results_dir = utils.create_results_dir()
    summary_writer = tf.summary.create_file_writer(results_dir)  # type: ignore

    # ===== generate graph and profile =====

    if graph_and_profile:
        tf.summary.trace_on(graph=True, profiler=True)  # type: ignore
        vcsmc(data_NxSxA)
        with summary_writer.as_default():
            tf.summary.trace_export(name="vcsmc", step=0, profiler_outdir=results_dir)  # type: ignore

    # ===== train =====

    log_likelihoods_across_epochs = []
    avg_log_likelihoods_across_epochs = []

    for epoch in tqdm(range(epochs)):
        with summary_writer.as_default(step=epoch + start_epoch):
            log_Z_SMC, log_likelihood_K, best_newick_tree = train_step(data_NxSxA)

            log_likelihoods_avg = tf.math.reduce_mean(log_likelihood_K)
            log_likelihoods_std_dev = tf.math.reduce_std(log_likelihood_K)

            avg_log_likelihoods_across_epochs.append(log_likelihoods_avg)
            log_likelihoods_across_epochs.append(log_likelihood_K)

            cosine_similarity = get_data_reconstruction_cosine_similarity()

            tf.summary.scalar("Elbo", log_Z_SMC)
            tf.summary.scalar("Log likelihood avg", log_likelihoods_avg)
            tf.summary.scalar("Log likelihoods std dev", log_likelihoods_std_dev)
            tf.summary.scalar(
                "Data reconstruction cosine similarity",
                cosine_similarity,
            )
            tf.summary.histogram("Log likelihoods", log_likelihood_K)

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

    # ===== final log likelihoods across epochs =====

    K = log_likelihoods_across_epochs[0].shape[0]

    plt.scatter(
        np.arange(epochs).repeat(K),
        np.array(log_likelihoods_across_epochs).flatten(),
        c="black",
        alpha=0.2,
    )
    plt.plot(
        np.arange(epochs),
        avg_log_likelihoods_across_epochs,
        c="orange",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Log Likelihood")
    plt.title("Log Likelihoods Across Epochs")
    plt.show()
