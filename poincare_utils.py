import math

import tensorflow as tf
from drawsvg import Drawing, Text
from hyperbolic import euclid, poincare

from proposal import EmbeddingProposal
from type_utils import Tensor
from vcsmc import VCSMC
from vcsmc_utils import replace_with_merged


def render_poincare(
    vcsmc: VCSMC,
    proposal: EmbeddingProposal,
    data_NxSxA: Tensor,
    taxa_N: Tensor,
) -> Drawing:
    N = taxa_N.shape[0]

    result = vcsmc(data_NxSxA)

    merge1_indexes_N1 = result["best_merge1_indexes_N1"]
    merge2_indexes_N1 = result["best_merge2_indexes_N1"]

    points = []
    lines = []
    texts = []

    min_x = math.inf
    max_x = -math.inf
    min_y = math.inf
    max_y = -math.inf

    embeddings_txD = proposal.seq_encoder(data_NxSxA)
    labels_t = taxa_N

    for r in range(N - 1):
        idx1 = merge1_indexes_N1[r]
        idx2 = merge2_indexes_N1[r]

        emb1_D = embeddings_txD[idx1]
        emb2_D = embeddings_txD[idx2]
        parent_emb_D = proposal.merge_encoder(emb1_D[tf.newaxis], emb2_D[tf.newaxis])[0]

        unpack = lambda x: (float(x[0]), float(x[1]))

        emb1 = unpack(emb1_D)
        emb2 = unpack(emb2_D)
        parent_embed = unpack(parent_emb_D)

        label1 = str(labels_t[idx1].numpy())[2:-1]
        label2 = str(labels_t[idx2].numpy())[2:-1]

        p1 = poincare.Point(emb1[0], emb1[1])
        p2 = poincare.Point(emb2[0], emb2[1])
        p3 = poincare.Point(parent_embed[0], parent_embed[1])
        points.extend([p1, p2, p3])

        l1 = poincare.Line.from_points(*p1, *p3, segment=True)
        l2 = poincare.Line.from_points(*p2, *p3, segment=True)
        lines.extend([l1, l2])

        if label1 != "":
            t1 = Text(label1, 0.001, emb1[0], emb1[1], fill="black")
            texts.append(t1)
        if label2 != "":
            t2 = Text(label2, 0.001, emb2[0], emb2[1], fill="black")
            texts.append(t2)

        min_x = min(min_x, emb1[0], emb2[0], parent_embed[0])
        max_x = max(max_x, emb1[0], emb2[0], parent_embed[0])
        min_y = min(min_y, emb1[1], emb2[1], parent_embed[1])
        max_y = max(max_y, emb1[1], emb2[1], parent_embed[1])

        embeddings_txD = replace_with_merged(embeddings_txD, idx1, idx2, parent_emb_D)
        labels_t = replace_with_merged(labels_t, idx1, idx2, tf.constant(""))

    dx = max_x - min_x
    dy = max_y - min_y

    size = max(dx, dy) * 1.1

    origin_x = -size / 2 + (min_x + max_x) / 2
    origin_y = -size / 2 + (min_y + max_y) / 2

    stroke_width = size / 500
    radius = 2 * stroke_width

    d = Drawing(size, size, origin=(origin_x, origin_y))
    d.draw(euclid.Circle(0, 0, 1), fill="silver")

    for l in lines:
        d.draw(l, stroke_width=stroke_width, stroke="green", fill="none")

    for p in points:
        d.draw(p, radius=radius, fill="orange")

    for t in texts:
        d.draw(t)

    d.set_render_size(w=800)
    return d
