from __future__ import annotations

from typing import Callable, List, Tuple

from .graph_utils import apply_op, legal_ops
from .math_utils import cosine
from .scoring import ScoreModel
from .types import DAGState, SampledDAG

Vector = List[float]


def match_nearest_dag(z_star: Vector, sampled_dags: List[SampledDAG]) -> DAGState:
    if not sampled_dags:
        raise ValueError("sampled_dags is empty.")
    best = sampled_dags[0]
    best_sim = cosine(z_star, best.z)
    for item in sampled_dags[1:]:
        sim = cosine(z_star, item.z)
        if sim > best_sim:
            best = item
            best_sim = sim
    return DAGState(nodes=list(best.graph.nodes), edges=list(best.graph.edges), z=best.graph.z, reward=best.graph.reward)


def hill_climb_bic(
    init_dag: DAGState,
    scorer: ScoreModel,
    max_iters: int = 60,
) -> Tuple[DAGState, float]:
    return hill_climb_objective(
        init_dag=init_dag,
        objective_fn=scorer.score,
        max_iters=max_iters,
    )


def hill_climb_objective(
    init_dag: DAGState,
    objective_fn: Callable[[DAGState], float],
    max_iters: int = 60,
) -> Tuple[DAGState, float]:
    cur = DAGState(nodes=list(init_dag.nodes), edges=list(init_dag.edges), z=init_dag.z, reward=init_dag.reward)
    cur_score = objective_fn(cur)

    for _ in range(max_iters):
        improved = False
        best_neighbor = cur
        best_neighbor_score = cur_score

        for op in legal_ops(cur, allow_backtracking=True):
            cand = apply_op(cur, op)
            cand_score = objective_fn(cand)
            if cand_score > best_neighbor_score:
                best_neighbor = cand
                best_neighbor_score = cand_score
                improved = True

        if not improved:
            break
        cur = best_neighbor
        cur_score = best_neighbor_score

    cur.reward = cur_score
    return cur, cur_score
