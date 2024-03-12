import math

from drawsvg import Drawing, Text
from hyperbolic import euclid, poincare
from torch import Tensor

from proposals import EmbeddingProposal
from train import batch_by_sites
from vcsmc import VCSMC
from vcsmc_utils import replace_with_merged_list


def render_poincare(
    vcsmc: VCSMC,
    proposal: EmbeddingProposal,
    data_NxSxA: Tensor,
    taxa_N: list[str],
) -> Drawing:
    N = len(taxa_N)

    dataset = batch_by_sites(data_NxSxA, None)

    # batch is actually the full dataset
    data_batched_SxNxA, site_positions_batched_SxSfull = next(iter(dataset))
    data_batched_NxSxA = data_batched_SxNxA.permute(1, 0, 2)

    result = vcsmc(data_NxSxA, data_batched_NxSxA, site_positions_batched_SxSfull)

    merge1_indexes_N1 = result["best_merge1_indexes_N1"]
    merge2_indexes_N1 = result["best_merge2_indexes_N1"]

    points = []
    lines = []
    texts = []

    min_x = math.inf
    max_x = -math.inf
    min_y = math.inf
    max_y = -math.inf

    embeddings_txD: list[Tensor] = list(proposal.seq_encoder(data_NxSxA))
    labels_t = taxa_N

    for r in range(N - 1):
        idx1 = merge1_indexes_N1[r]
        idx2 = merge2_indexes_N1[r]

        emb1_D = embeddings_txD[idx1]
        emb2_D = embeddings_txD[idx2]

        parent_emb_1xD = proposal.merge_encoder(
            emb1_D.unsqueeze(0), emb2_D.unsqueeze(0)
        )
        parent_emb_D = parent_emb_1xD[0]

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

    size = max(dx, dy) * 1.1

    origin_x = -size / 2 + (min_x + max_x) / 2
    origin_y = -size / 2 + (min_y + max_y) / 2

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

    d.set_render_size(w=800)
    return d
