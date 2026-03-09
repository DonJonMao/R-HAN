from __future__ import annotations

from typing import List, Sequence, Tuple

from .types import DAGState, Edge

GraphOp = Tuple[str, int, int]


def make_empty_dag(nodes: Sequence[str]) -> DAGState:
    return DAGState(nodes=list(nodes), edges=[])


def _has_cycle(node_count: int, edges: List[Edge]) -> bool:
    adj = [[] for _ in range(node_count)]
    for src, dst in edges:
        adj[src].append(dst)

    color = [0] * node_count  # 0 unvisited, 1 visiting, 2 done

    def dfs(u: int) -> bool:
        color[u] = 1
        for v in adj[u]:
            if color[v] == 1:
                return True
            if color[v] == 0 and dfs(v):
                return True
        color[u] = 2
        return False

    for i in range(node_count):
        if color[i] == 0 and dfs(i):
            return True
    return False


def is_acyclic(nodes: Sequence[str], edges: List[Edge]) -> bool:
    return not _has_cycle(len(nodes), edges)


def _edge_set(edges: List[Edge]) -> set[Edge]:
    return set(edges)


def apply_op(dag: DAGState, op: GraphOp) -> DAGState:
    kind, src, dst = op
    edge_set = _edge_set(dag.edges)
    if kind == "add":
        edge_set.add((src, dst))
    elif kind == "del":
        edge_set.discard((src, dst))
    elif kind == "rev":
        edge_set.discard((src, dst))
        edge_set.add((dst, src))
    else:
        raise ValueError(f"Unknown op type: {kind}")
    return DAGState(nodes=list(dag.nodes), edges=sorted(edge_set), z=dag.z, reward=dag.reward)


def legal_ops(
    dag: DAGState,
    allow_backtracking: bool = False,
    allow_reverse: bool | None = None,
) -> List[GraphOp]:
    if allow_reverse is None:
        allow_reverse = allow_backtracking

    node_count = len(dag.nodes)
    current_edges = _edge_set(dag.edges)
    out: List[GraphOp] = []

    for i in range(node_count):
        for j in range(node_count):
            if i == j:
                continue
            edge = (i, j)
            rev_edge = (j, i)

            if edge not in current_edges:
                # add i -> j
                cand = sorted(current_edges | {edge})
                if not _has_cycle(node_count, cand):
                    out.append(("add", i, j))
            else:
                if allow_backtracking:
                    out.append(("del", i, j))

                # reverse i -> j to j -> i
                if allow_reverse and rev_edge not in current_edges:
                    cand = set(current_edges)
                    cand.discard(edge)
                    cand.add(rev_edge)
                    if not _has_cycle(node_count, sorted(cand)):
                        out.append(("rev", i, j))
    return out


def op_to_str(op: GraphOp, nodes: Sequence[str]) -> str:
    kind, src, dst = op
    if kind == "add":
        return f"add {nodes[src]} -> {nodes[dst]}"
    if kind == "del":
        return f"del {nodes[src]} -> {nodes[dst]}"
    if kind == "rev":
        return f"rev {nodes[src]} -> {nodes[dst]}"
    return str(op)
