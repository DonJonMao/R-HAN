from __future__ import annotations

import random
from typing import Iterable, List, Tuple

from .math_utils import add, scale
from .types import MASConfig, ProxyPair

Vector = List[float]


def _relu(x: float) -> float:
    return x if x > 0.0 else 0.0


def _relu_grad(x: float) -> float:
    return 1.0 if x > 0.0 else 0.0


class ProxyModel:
    """MLP proxy S(z) with MSE + pairwise ranking loss."""

    def __init__(self, dim: int, hidden_dim: int, seed: int = 7):
        rng = random.Random(seed)
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w1 = [[rng.uniform(-0.08, 0.08) for _ in range(dim)] for _ in range(hidden_dim)]
        self.b1 = [0.0 for _ in range(hidden_dim)]
        self.w2 = [rng.uniform(-0.08, 0.08) for _ in range(hidden_dim)]
        self.b2 = 0.0

    def _forward(self, z: Vector) -> Tuple[Vector, Vector, float]:
        pre_h = [sum(wij * zj for wij, zj in zip(row, z)) + b for row, b in zip(self.w1, self.b1)]
        h = [_relu(v) for v in pre_h]
        pred = sum(w * hv for w, hv in zip(self.w2, h)) + self.b2
        return pre_h, h, pred

    def predict(self, z: Vector) -> float:
        _, _, pred = self._forward(z)
        return pred

    def gradient_wrt_z(self, z: Vector) -> Vector:
        pre_h, _, _ = self._forward(z)
        # d pred / dz = sum_j w2_j * relu'(pre_h_j) * w1_j
        grad = [0.0 for _ in range(self.dim)]
        for j in range(self.hidden_dim):
            gate = self.w2[j] * _relu_grad(pre_h[j])
            if gate == 0.0:
                continue
            row = self.w1[j]
            grad = [g + gate * r for g, r in zip(grad, row)]
        return grad

    def fit(self, pairs: Iterable[ProxyPair], config: MASConfig) -> None:
        pairs = list(pairs)
        if not pairs:
            return
        rng = random.Random(config.random_seed)

        lr = config.proxy_lr
        rank_w = config.proxy_rank_weight
        margin = config.proxy_rank_margin
        n = len(pairs)

        for _ in range(config.proxy_train_epochs):
            rng.shuffle(pairs)

            # Full-batch gradient on MSE.
            gw1 = [[0.0 for _ in range(self.dim)] for _ in range(self.hidden_dim)]
            gb1 = [0.0 for _ in range(self.hidden_dim)]
            gw2 = [0.0 for _ in range(self.hidden_dim)]
            gb2 = 0.0

            for pair in pairs:
                pre_h, h, pred = self._forward(pair.z)
                err = pred - pair.reward
                d_pred = (2.0 / n) * err

                for j in range(self.hidden_dim):
                    gw2[j] += d_pred * h[j]
                gb2 += d_pred

                for j in range(self.hidden_dim):
                    d_hj = d_pred * self.w2[j]
                    d_pre = d_hj * _relu_grad(pre_h[j])
                    gb1[j] += d_pre
                    row_grad = [d_pre * zi for zi in pair.z]
                    gw1[j] = [a + b for a, b in zip(gw1[j], row_grad)]

            # Pairwise ranking loss on random pairs.
            if rank_w > 0.0 and n >= 2:
                pair_count = min(n * 2, n * (n - 1) // 2)
                for _ in range(pair_count):
                    i = rng.randrange(0, n)
                    j = rng.randrange(0, n)
                    if i == j:
                        continue
                    pi, pj = pairs[i], pairs[j]
                    yi = 1.0 if pi.reward >= pj.reward else -1.0
                    _, hi, predi = self._forward(pi.z)
                    _, hj, predj = self._forward(pj.z)

                    hinge = margin - yi * (predi - predj)
                    if hinge <= 0.0:
                        continue

                    d_pred_i = -rank_w * yi / pair_count
                    d_pred_j = rank_w * yi / pair_count

                    for k in range(self.hidden_dim):
                        gw2[k] += d_pred_i * hi[k] + d_pred_j * hj[k]
                    gb2 += d_pred_i + d_pred_j

                    # Backprop ranking grads to first layer.
                    pre_hi, _, _ = self._forward(pi.z)
                    pre_hj, _, _ = self._forward(pj.z)
                    for k in range(self.hidden_dim):
                        d_pre_i = d_pred_i * self.w2[k] * _relu_grad(pre_hi[k])
                        d_pre_j = d_pred_j * self.w2[k] * _relu_grad(pre_hj[k])
                        gb1[k] += d_pre_i + d_pre_j
                        gw1[k] = [
                            g + d_pre_i * zi + d_pre_j * zj
                            for g, zi, zj in zip(gw1[k], pi.z, pj.z)
                        ]

            # Parameter update.
            for j in range(self.hidden_dim):
                self.w2[j] -= lr * gw2[j]
            self.b2 -= lr * gb2
            for j in range(self.hidden_dim):
                self.b1[j] -= lr * gb1[j]
                self.w1[j] = [w - lr * g for w, g in zip(self.w1[j], gw1[j])]

    def ascent(self, z0: Vector, steps: int, lr: float) -> Vector:
        z = z0[:]
        for _ in range(steps):
            grad = self.gradient_wrt_z(z)
            z = add(z, scale(grad, lr))
        return z


def train_proxy(config: MASConfig, pairs: List[ProxyPair]) -> ProxyModel:
    dim = len(pairs[0].z) if pairs else config.embedding_dim
    proxy = ProxyModel(dim=dim, hidden_dim=config.proxy_hidden_dim, seed=config.random_seed)
    proxy.fit(pairs, config=config)
    return proxy
