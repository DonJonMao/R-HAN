from __future__ import annotations

import math
from typing import Protocol

from .types import DAGState


class ScoreModel(Protocol):
    def score(self, dag: DAGState) -> float:
        """Return larger-is-better score (BIC objective)."""


class BICScorer:
    """BIC scorer placeholder.

    This class is intentionally lightweight. Replace `score` with a real
    dataset-driven BIC implementation when integrating with production data.
    """

    def __init__(self, target_edge_count: int = 6, complexity_penalty: float = 0.55):
        self.target_edge_count = target_edge_count
        self.complexity_penalty = complexity_penalty

    def score(self, dag: DAGState) -> float:
        edge_count = len(dag.edges)
        fit_term = -abs(edge_count - self.target_edge_count)
        complexity_term = self.complexity_penalty * edge_count
        return fit_term - complexity_term


class DiscreteDataBICScorer:
    """Dataset-aware BIC scorer for discrete variables.

    Input format:
    - `data`: list of samples, each sample is a list[int] aligned with `node_order`.
    - each value must be an integer state id in [0, cardinality_i - 1].
    """

    def __init__(
        self,
        data: list[list[int]],
        node_order: list[str],
        cardinalities: list[int] | None = None,
    ):
        if not data:
            raise ValueError("data is empty.")
        if not node_order:
            raise ValueError("node_order is empty.")
        if len(data[0]) != len(node_order):
            raise ValueError("row width does not match node_order length.")

        self.data = data
        self.node_order = node_order
        if cardinalities is None:
            max_vals = [0] * len(node_order)
            for row in data:
                for i, val in enumerate(row):
                    if val > max_vals[i]:
                        max_vals[i] = val
            self.cardinalities = [m + 1 for m in max_vals]
        else:
            self.cardinalities = cardinalities

        self._local_cache: dict[tuple[int, tuple[int, ...]], tuple[float, int]] = {}
        self._idx_by_name = {name: i for i, name in enumerate(node_order)}

    def _local_loglik_and_params(self, child: int, parents: tuple[int, ...]) -> tuple[float, int]:
        key = (child, parents)
        if key in self._local_cache:
            return self._local_cache[key]

        child_card = self.cardinalities[child]
        parent_cards = [self.cardinalities[p] for p in parents]

        # Count N_ijk with dict[parent_config][child_value].
        counts: dict[tuple[int, ...], list[int]] = {}
        for row in self.data:
            parent_key = tuple(row[p] for p in parents)
            if parent_key not in counts:
                counts[parent_key] = [0] * child_card
            counts[parent_key][row[child]] += 1

        loglik = 0.0
        for child_counts in counts.values():
            total = sum(child_counts)
            if total == 0:
                continue
            for n_ijk in child_counts:
                if n_ijk > 0:
                    p = n_ijk / total
                    loglik += n_ijk * math.log(p)

        q_i = 1
        for c in parent_cards:
            q_i *= c
        param_count = (child_card - 1) * q_i

        out = (loglik, param_count)
        self._local_cache[key] = out
        return out

    def score(self, dag: DAGState) -> float:
        n = len(self.data)
        if n <= 1:
            return -1e9

        parents: list[list[int]] = [[] for _ in dag.nodes]
        for src, dst in dag.edges:
            parents[dst].append(src)

        loglik = 0.0
        k_params = 0
        for child_idx in range(len(dag.nodes)):
            ll_local, k_local = self._local_loglik_and_params(child_idx, tuple(sorted(parents[child_idx])))
            loglik += ll_local
            k_params += k_local

        return loglik - 0.5 * k_params * math.log(n)
