from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Sequence, Tuple

from .math_utils import add, dot, normalize, scale
from .types import DAGState
from .vectorizer import HashingVectorizer

Vector = List[float]


class GraphRepresentationModel:
    """Encodes DAG with GNN-style message passing + self-attention.

    Following Yang et al. (2025), the encoder returns:
    - global graph embedding z
    - node embeddings as source view u_k
    - node embeddings as target view v_k
    """

    def __init__(
        self,
        vectorizer: HashingVectorizer,
        smooth_alpha: float = 0.70,
        message_steps: int = 2,
        attention_edge_bias: float = 0.20,
        attention_self_bias: float = 0.10,
        seed: int = 7,
    ):
        self.vectorizer = vectorizer
        self.smooth_alpha = smooth_alpha
        self.message_steps = max(1, message_steps)
        self.attn_edge_bias = attention_edge_bias
        self.attn_self_bias = attention_self_bias

        rng = random.Random(seed)
        dim = self.vectorizer.dim

        # Message passing weights.
        self.w_self = [rng.uniform(0.6, 1.0) for _ in range(dim)]
        self.w_in = [rng.uniform(0.2, 0.7) for _ in range(dim)]
        self.w_out = [rng.uniform(0.2, 0.7) for _ in range(dim)]
        self.w_res = [rng.uniform(0.4, 0.8) for _ in range(dim)]

        # Self-attention projections.
        self.w_q = [rng.uniform(-1.0, 1.0) for _ in range(dim)]
        self.w_k = [rng.uniform(-1.0, 1.0) for _ in range(dim)]
        self.w_v = [rng.uniform(-1.0, 1.0) for _ in range(dim)]
        self.w_pool = [rng.uniform(-1.0, 1.0) for _ in range(dim)]

        # Source/target view projections conditioned on graph-level context.
        self.w_src = [rng.uniform(0.6, 1.2) for _ in range(dim)]
        self.w_dst = [rng.uniform(0.6, 1.2) for _ in range(dim)]
        self.w_src_ctx = [rng.uniform(-0.5, 0.5) for _ in range(dim)]
        self.w_dst_ctx = [rng.uniform(-0.5, 0.5) for _ in range(dim)]

    @staticmethod
    def _zeros(dim: int) -> Vector:
        return [0.0] * dim

    @staticmethod
    def _elem_mul(a: Vector, b: Vector) -> Vector:
        return [x * y for x, y in zip(a, b)]

    @staticmethod
    def _tanh(v: Vector) -> Vector:
        return [math.tanh(x) for x in v]

    def _neighbor_mean(self, neigh: Sequence[int], h: Sequence[Vector], dim: int) -> Vector:
        if not neigh:
            return self._zeros(dim)
        out = self._zeros(dim)
        inv = 1.0 / float(len(neigh))
        for j in neigh:
            out = [a + b for a, b in zip(out, h[j])]
        return [x * inv for x in out]

    def _build_neighbors(self, n: int, edges: Sequence[Tuple[int, int]]) -> Tuple[List[List[int]], List[List[int]]]:
        in_neigh = [[] for _ in range(n)]
        out_neigh = [[] for _ in range(n)]
        for src, dst in edges:
            if 0 <= src < n and 0 <= dst < n:
                in_neigh[dst].append(src)
                out_neigh[src].append(dst)
        return in_neigh, out_neigh

    def encode_nodes(
        self, dag: DAGState, agent_vectors: Dict[str, Vector]
    ) -> Tuple[Dict[int, Vector], Dict[int, Vector]]:
        n = len(dag.nodes)
        if n == 0:
            return {}, {}
        dim = len(next(iter(agent_vectors.values())))

        # Initialize node states from external/agent embeddings.
        h: List[Vector] = [normalize(agent_vectors[name]) for name in dag.nodes]
        in_neigh, out_neigh = self._build_neighbors(n, dag.edges)

        # Lightweight directed message passing.
        for _ in range(self.message_steps):
            h_new: List[Vector] = []
            for i in range(n):
                h_in = self._neighbor_mean(in_neigh[i], h, dim)
                h_out = self._neighbor_mean(out_neigh[i], h, dim)
                pre = [
                    self.w_self[d] * h[i][d]
                    + self.w_in[d] * h_in[d]
                    + self.w_out[d] * h_out[d]
                    for d in range(dim)
                ]
                upd = self._tanh(pre)
                mixed = [
                    self.w_res[d] * h[i][d] + (1.0 - self.w_res[d]) * upd[d]
                    for d in range(dim)
                ]
                h_new.append(normalize(mixed))
            h = h_new

        # Self-attention contextualization.
        q = [normalize(self._elem_mul(x, self.w_q)) for x in h]
        k = [normalize(self._elem_mul(x, self.w_k)) for x in h]
        v = [normalize(self._elem_mul(x, self.w_v)) for x in h]
        edge_set = set((int(s), int(t)) for s, t in dag.edges)
        scale_attn = 1.0 / math.sqrt(max(1.0, float(dim)))

        h_ctx: List[Vector] = []
        for i in range(n):
            logits: List[float] = []
            for j in range(n):
                b = 0.0
                if i == j:
                    b += self.attn_self_bias
                if (i, j) in edge_set or (j, i) in edge_set:
                    b += self.attn_edge_bias
                logits.append(scale_attn * dot(q[i], k[j]) + b)

            m = max(logits)
            exps = [math.exp(x - m) for x in logits]
            zsum = sum(exps)
            attn = [x / max(1e-12, zsum) for x in exps]

            ctx = self._zeros(dim)
            for j in range(n):
                ctx = [a + attn[j] * b for a, b in zip(ctx, v[j])]
            h_ctx.append(normalize(add(h[i], ctx)))

        # Graph-level pooling.
        pool_logits = [dot(self.w_pool, x) for x in h_ctx]
        m_pool = max(pool_logits)
        pool_exps = [math.exp(x - m_pool) for x in pool_logits]
        pool_sum = sum(pool_exps)
        pool = [x / max(1e-12, pool_sum) for x in pool_exps]

        z_local = self._zeros(dim)
        for i in range(n):
            z_local = [a + pool[i] * b for a, b in zip(z_local, h_ctx[i])]
        z_local = normalize(z_local)

        src_vectors: Dict[int, Vector] = {}
        dst_vectors: Dict[int, Vector] = {}
        for i in range(n):
            src_pre = [
                self.w_src[d] * h_ctx[i][d] + self.w_src_ctx[d] * z_local[d]
                for d in range(dim)
            ]
            dst_pre = [
                self.w_dst[d] * h_ctx[i][d] + self.w_dst_ctx[d] * z_local[d]
                for d in range(dim)
            ]
            src_vectors[i] = normalize(self._tanh(src_pre))
            dst_vectors[i] = normalize(self._tanh(dst_pre))
        return src_vectors, dst_vectors

    def encode_graph(
        self,
        dag: DAGState,
        agent_vectors: Dict[str, Vector],
        prev_z: Optional[Vector] = None,
    ) -> Tuple[Vector, Dict[int, Vector], Dict[int, Vector]]:
        src_vecs, dst_vecs = self.encode_nodes(dag, agent_vectors)
        if not dag.nodes:
            return [0.0] * self.vectorizer.dim, src_vecs, dst_vecs

        merged = normalize(self.vectorizer.merge(list(src_vecs.values()) + list(dst_vecs.values())))

        if prev_z is None:
            z = merged
        else:
            # Smooth z in latent space: z_t = a * z_{t-1} + (1-a) * z_new.
            z = normalize(add(scale(prev_z, self.smooth_alpha), scale(merged, 1.0 - self.smooth_alpha)))

        return z, src_vecs, dst_vecs
