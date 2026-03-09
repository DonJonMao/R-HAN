from __future__ import annotations

import random
from typing import List, Sequence, Tuple

from .math_utils import l2_sq, mean

Vector = List[float]


def _closest_center_idx(vec: Vector, centers: Sequence[Vector]) -> int:
    best_idx = 0
    best_dist = float("inf")
    for i, center in enumerate(centers):
        d = l2_sq(vec, center)
        if d < best_dist:
            best_dist = d
            best_idx = i
    return best_idx


def kmeans(
    vectors: Sequence[Vector],
    k: int,
    max_iters: int,
    seed: int = 7,
) -> Tuple[List[Vector], List[List[int]]]:
    if not vectors:
        return [], []
    n = len(vectors)
    k = max(1, min(k, n))
    rng = random.Random(seed)

    initial_indices = list(range(n))
    rng.shuffle(initial_indices)
    centers = [vectors[i][:] for i in initial_indices[:k]]

    assignments = [0] * n
    for _ in range(max_iters):
        changed = False
        clusters = [[] for _ in range(k)]
        for i, vec in enumerate(vectors):
            cid = _closest_center_idx(vec, centers)
            clusters[cid].append(i)
            if assignments[i] != cid:
                changed = True
            assignments[i] = cid

        new_centers: List[Vector] = []
        for cid, idxs in enumerate(clusters):
            if idxs:
                new_centers.append(mean(vectors[i] for i in idxs))
            else:
                # Re-seed empty cluster with a random point.
                new_centers.append(vectors[rng.randrange(0, n)][:])

        centers = new_centers
        if not changed:
            break

    final_clusters = [[] for _ in range(k)]
    for i, cid in enumerate(assignments):
        final_clusters[cid].append(i)
    return centers, final_clusters
