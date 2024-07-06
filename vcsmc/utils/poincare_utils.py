import math

import matplotlib.pyplot as plt
import torch
from drawsvg import Drawing, Text
from hyperbolic import euclid, poincare
from IPython.display import display
from ipywidgets import FloatSlider, interactive
from torch import Tensor

from ..distances import Hyperbolic
from ..proposals import EmbeddingProposal
from ..train import evaluate, get_site_positions_SxSfull
from ..vcsmc import VCSMC
from .vcsmc_utils import replace_with_merged_list


@torch.no_grad()
def interactive_poincare(vcsmc: VCSMC, data_NxSxA: Tensor, taxa_N: list[str]):
    N = len(taxa_N)

    proposal = vcsmc.proposal
    assert isinstance(proposal, EmbeddingProposal)

    distance = proposal.seq_encoder.distance
    assert isinstance(distance, Hyperbolic)

    def normalize(embeddings_VxD: Tensor):
        """Normalize and ensure |x| < 1"""
        max_norm = 1 - 1e-6
        embeddings_VxD = distance.normalize(embeddings_VxD)
        norms_V = torch.norm(embeddings_VxD, dim=-1)
        unit_vectors_VxD = embeddings_VxD / norms_V.unsqueeze(-1) * max_norm
        return torch.where(
            norms_V.unsqueeze(-1) < max_norm, embeddings_VxD, unit_vectors_VxD
        )

    result = evaluate(vcsmc, taxa_N, data_NxSxA)

    merge1_indexes_N1 = result["best_merge_indexes_N1x2"][:, 0]
    merge2_indexes_N1 = result["best_merge_indexes_N1x2"][:, 1]

    embeddings_N1xD = normalize(result["best_embeddings_N1xD"])

    points = []
    lines = []
    texts = []

    min_x = math.inf
    max_x = -math.inf
    min_y = math.inf
    max_y = -math.inf

    embeddings_txD: list[Tensor] = list(normalize(proposal.seq_encoder(data_NxSxA)))
    labels_t = taxa_N

    for r in range(N - 1):
        idx1 = int(merge1_indexes_N1[r])
        idx2 = int(merge2_indexes_N1[r])

        emb1_D = embeddings_txD[idx1]
        emb2_D = embeddings_txD[idx2]
        parent_emb_D = embeddings_N1xD[r]

        # flip y coordinate to match matplotlib display orientation
        unpack = lambda x: (float(x[0]), -float(x[1]))

        emb1 = unpack(emb1_D)
        emb2 = unpack(emb2_D)
        parent_embed = unpack(parent_emb_D)

        label1 = labels_t[idx1]
        label2 = labels_t[idx2]

        p1 = poincare.Point(emb1[0], emb1[1])
        p2 = poincare.Point(emb2[0], emb2[1])
        p3 = poincare.Point(parent_embed[0], parent_embed[1])
        points.extend([p1, p2, p3])

        if p1 != p3:
            l1 = poincare.Line.from_points(*p1, *p3, segment=True)
            lines.append(l1)
        if p2 != p3:
            l2 = poincare.Line.from_points(*p2, *p3, segment=True)
            lines.append(l2)

        if label1 != "":
            t1 = (label1, emb1[0], emb1[1])
            texts.append(t1)
        if label2 != "":
            t2 = (label2, emb2[0], emb2[1])
            texts.append(t2)

        min_x = min(min_x, emb1[0], emb2[0], parent_embed[0])
        max_x = max(max_x, emb1[0], emb2[0], parent_embed[0])
        min_y = min(min_y, emb1[1], emb2[1], parent_embed[1])
        max_y = max(max_y, emb1[1], emb2[1], parent_embed[1])

        embeddings_txD = replace_with_merged_list(
            embeddings_txD, idx1, idx2, parent_emb_D
        )
        labels_t = replace_with_merged_list(labels_t, idx1, idx2, "")

    dx = max_x - min_x
    dy = max_y - min_y

    initial_size = max(dx, dy) * 1.1

    initial_origin_x = -initial_size / 2 + (min_x + max_x) / 2
    initial_origin_y = -initial_size / 2 + (min_y + max_y) / 2

    def plot_poincare(size, origin_x, origin_y):
        stroke_width = size / 500
        radius = 2 * stroke_width
        text_size = size / 100

        d = Drawing(size, size, origin=(origin_x, origin_y))
        d.draw(euclid.Circle(0, 0, 1), fill="silver")

        for l in lines:
            d.draw(l, stroke_width=stroke_width, stroke="green", fill="none")

        for p in points:
            d.draw(p, radius=radius, fill="orange")

        for t in texts:
            d.draw(Text(t[0], text_size, t[1], t[2], fill="black"))

        d.set_render_size(w=750)
        display(d)

    return interactive(
        plot_poincare,
        size=FloatSlider(value=initial_size, min=0, max=2, step=0.001),
        origin_x=FloatSlider(value=initial_origin_x, min=-1, max=1, step=0.001),
        origin_y=FloatSlider(value=initial_origin_y, min=-1, max=1, step=0.001),
    )


@torch.no_grad()
def plot_embeddings(vcsmc: VCSMC, data_NxSxA: Tensor, taxa_N: list[str]):
    distance = vcsmc.proposal.seq_encoder.distance
    assert distance is not None

    embeddings = distance.normalize(vcsmc.proposal.seq_encoder(data_NxSxA))
    embeddings = embeddings.cpu()

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    ax.scatter(embeddings[:, 0], embeddings[:, 1])

    for i, txt in enumerate(taxa_N):
        ax.text(float(embeddings[i, 0]), float(embeddings[i, 1]), txt, fontsize=6)

    plt.show()


@torch.no_grad()
def interactive_q_matrix(vcsmc: VCSMC, data_NxSxA: Tensor):
    S = data_NxSxA.shape[1]

    distance = vcsmc.proposal.seq_encoder.distance
    assert isinstance(distance, Hyperbolic)

    site_positions_SxC = vcsmc.q_matrix_decoder.site_positions_encoder(
        get_site_positions_SxSfull(data_NxSxA)
    )

    @torch.no_grad()
    def plot_q_matrix(r, theta, s):
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        embedding_1xD = distance.unnormalize(
            torch.tensor([x, y], device=data_NxSxA.device).unsqueeze(0)
        )
        q_matrix_SxAxA = vcsmc.q_matrix_decoder.Q_matrix_VxSxAxA(
            embedding_1xD, site_positions_SxC
        )[0]
        q_matrix_AxA = q_matrix_SxAxA[s]
        plt.imshow(q_matrix_AxA.cpu())
        plt.show()

    return interactive(
        plot_q_matrix,
        r=(0.01, 0.99, 0.01),
        theta=(0, 2 * math.pi, 0.01),
        s=(0, S - 1, 1),
    )


@torch.no_grad()
def interactive_stat_probs(vcsmc: VCSMC, data_NxSxA: Tensor):
    S_RANGE = 10

    S = data_NxSxA.shape[1]
    A = data_NxSxA.shape[2]

    distance = vcsmc.proposal.seq_encoder.distance
    assert isinstance(distance, Hyperbolic)

    site_positions_SxC = vcsmc.q_matrix_decoder.site_positions_encoder(
        get_site_positions_SxSfull(data_NxSxA)
    )

    @torch.no_grad()
    def plot_stat_probs(r, theta, s_start):
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        embedding_1xD = distance.unnormalize(
            torch.tensor([x, y], device=data_NxSxA.device).unsqueeze(0)
        )
        stat_probs_SxA = vcsmc.q_matrix_decoder.stat_probs_VxSxA(
            embedding_1xD, site_positions_SxC
        )[0]

        fig = plt.figure(figsize=(20, 4))
        ax = fig.add_subplot(111)

        ax.set_xticks(range(S_RANGE))
        ax.set_xticklabels(range(s_start, s_start + S_RANGE))  # type: ignore
        ax.set_yticks(range(A))
        ax.set_yticklabels(["A", "C", "G", "T"])

        im = ax.imshow(
            stat_probs_SxA.T[:, s_start : s_start + S_RANGE].cpu(), cmap="Blues"
        )
        plt.colorbar(im)
        plt.show()

    return interactive(
        plot_stat_probs,
        r=(0.01, 0.99, 0.01),
        theta=(0, 2 * math.pi, 0.01),
        s_start=(0, S - S_RANGE, 1),
    )
