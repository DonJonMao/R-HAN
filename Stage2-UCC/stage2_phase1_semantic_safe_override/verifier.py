from __future__ import annotations

import heapq
import re
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

from common.answer_contracts import AnswerObject, typed_answer_distance, typed_answer_similarity
from mas_treesearch.evaluator import MultiFidelityEvaluator
from stage2_gcr_plus.code_repair import evaluate_code_candidate

from .artifacts import (
    ArtifactIR,
    answer_text,
    answer_units,
    clamp01,
    cosine_similarity,
    evidence_units,
    lexical_feature_vector,
    pooled_artifact_vector,
    sparsemax,
)


RESIDUAL_KEYS: tuple[str, ...] = (
    "r_parse",
    "r_consistency",
    "r_completeness",
    "r_execution",
    "r_constraint",
    "r_support",
    "r_preserve",
)


@dataclass
class VerifierState:
    residual_vector: Dict[str, float]
    unit_error_map: Dict[str, float]
    support_map: Dict[str, float]
    completeness_score: float
    consistency_score: float
    answer_consistency_score: float
    executability_score: float
    constraint_score: float
    confidence_score: float
    progress_score: float
    preserve_risk: float
    anchor_similarity: float
    answer_similarity: float
    answer_delta: float
    overturn_risk: float
    channel_weights: Dict[str, float] = field(default_factory=dict)
    meta_summary: str = ""
    issues: List[str] = field(default_factory=list)
    missing_requirements: List[str] = field(default_factory=list)
    executor_feedback: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _mean(values: Sequence[float]) -> float:
    values_list = [float(value) for value in values]
    if not values_list:
        return 0.0
    return sum(values_list) / float(len(values_list))


def mean_residual(state: VerifierState) -> float:
    return _mean(list(state.residual_vector.values()))


def _extract_question_keywords(question_text: str) -> List[str]:
    words = re.findall(r"[A-Za-z_]\w+", question_text.lower())
    stop = {
        "the",
        "and",
        "for",
        "with",
        "from",
        "that",
        "this",
        "your",
        "return",
        "write",
        "given",
        "input",
        "output",
        "following",
    }
    seen: List[str] = []
    for word in words:
        if len(word) < 4 or word in stop or word in seen:
            continue
        seen.append(word)
    return seen[:8]


def _option_count(question_text: str, metadata: Optional[Dict[str, Any]]) -> int:
    if isinstance(metadata, dict) and isinstance(metadata.get("options"), list):
        return len(metadata["options"])
    numbers = [int(value) for value in re.findall(r"^\s*(\d+)\)", question_text, flags=re.MULTILINE)]
    return max(numbers) if numbers else 0


def _contract_valid(answer_object: AnswerObject) -> bool:
    return bool(answer_object.fields.get("contract_valid", answer_object.valid))


def _recoverable_valid(answer_object: AnswerObject) -> bool:
    return bool(answer_object.fields.get("recoverable_valid", answer_object.valid))


def _semantic_value(answer_object: AnswerObject) -> Any:
    if "recoverable_object" in answer_object.fields:
        return answer_object.fields.get("recoverable_object")
    if "recoverable_value" in answer_object.fields:
        return answer_object.fields.get("recoverable_value")
    return answer_object.value


def _answer_parse_residual(answer_object: AnswerObject) -> float:
    contract_valid = _contract_valid(answer_object)
    recoverable_valid = _recoverable_valid(answer_object)
    if contract_valid:
        return 0.0
    if recoverable_valid:
        return 0.85
    return 1.0


def _answer_completeness_residual(answer_object: AnswerObject) -> float:
    contract_valid = _contract_valid(answer_object)
    recoverable_valid = _recoverable_valid(answer_object)
    semantic_value = _semantic_value(answer_object)
    if answer_object.kind in {"graph_bool", "graph_scalar", "graph_sequence", "graph_path", "graph_matching", "node_embeddings", "graph_text"} and not contract_valid:
        return 0.85 if recoverable_valid else 1.0
    if not recoverable_valid:
        return 1.0
    if answer_object.kind in {"option", "bool", "numeric", "graph_scalar", "graph_sequence"}:
        return 0.0
    if answer_object.kind == "graph_path":
        path = list((semantic_value or {}).get("path") or [])
        total_weight = (semantic_value or {}).get("total_weight")
        if path and total_weight is not None:
            return 0.0
        if path or total_weight is not None:
            return 0.35
        return 1.0
    if answer_object.kind == "graph_matching":
        matches = list((semantic_value or {}).get("matches") or [])
        count = (semantic_value or {}).get("count")
        if matches and count is not None:
            return 0.0
        if matches or count is not None:
            return 0.30
        return 1.0
    if answer_object.kind == "node_embeddings":
        return 0.0 if dict(semantic_value or {}) else 1.0
    if answer_object.kind == "code":
        if not str(semantic_value or "").strip():
            return 1.0
        if bool(answer_object.fields.get("entry_point")) and not bool(answer_object.fields.get("entry_point_present")):
            return 0.45
        return 0.0
    return 0.0 if str(semantic_value or "").strip() else 1.0


def _support_map(
    artifact: ArtifactIR,
    candidate_entry: Optional[Dict[str, Any]],
) -> Dict[str, float]:
    provenance = artifact.provenance
    base_coverage = clamp01(float(artifact.metadata.get("provenance_coverage", 0.0)))
    reviewer_bonus = clamp01(float((candidate_entry or {}).get("reviewer_mean_trust", 0.5)))
    answer_valid_bonus = 0.10 if bool(artifact.answer_object.valid) else 0.0
    schema_bonus = 0.10 if _contract_valid(artifact.answer_object) else 0.0
    supports: Dict[str, float] = {}
    answer_ids = set(artifact.answer_unit_ids)
    evidence_ids = set(artifact.evidence_unit_ids)
    for unit in artifact.units:
        score = 0.18 * base_coverage + 0.24 * reviewer_bonus + answer_valid_bonus + schema_bonus
        if unit.unit_id in evidence_ids:
            score += 0.18
        if unit.unit_id in answer_ids:
            score += 0.14
        if provenance:
            score += 0.08
        if len(unit.text.strip()) >= 4:
            score += 0.05
        supports[unit.unit_id] = clamp01(score)
    return supports


def _consistency_issues(artifact: ArtifactIR) -> Tuple[List[str], Dict[str, float], float]:
    issues: List[str] = []
    unit_error_map = {unit.unit_id: 0.0 for unit in artifact.units}
    normalized: Dict[str, List[str]] = {}
    for unit in artifact.units:
        text = unit.text.strip().lower()
        if not text:
            unit_error_map[unit.unit_id] = max(unit_error_map[unit.unit_id], 0.4)
            issues.append(f"{unit.unit_id}: empty unit")
            continue
        normalized.setdefault(text, []).append(unit.unit_id)
    duplicates = [unit_ids for unit_ids in normalized.values() if len(unit_ids) > 1]
    for unit_ids in duplicates:
        for unit_id in unit_ids:
            unit_error_map[unit_id] = max(unit_error_map[unit_id], 0.25)
        issues.append(f"duplicate units: {', '.join(unit_ids)}")
    contradiction_score = clamp01(0.25 * len(duplicates))
    answer_ids = set(artifact.answer_unit_ids)
    answer_signatures = {
        re.sub(r"\s+", " ", unit.text.strip().lower())
        for unit in artifact.units
        if unit.unit_id in answer_ids and unit.text.strip()
    }
    if len(answer_signatures) > 1 and artifact.answer_object.kind not in {"code", "graph_path", "graph_matching", "node_embeddings"}:
        contradiction_score = clamp01(contradiction_score + 0.20)
        issues.append("answer units disagree")
        for unit in artifact.units:
            if unit.unit_id in answer_ids:
                unit_error_map[unit.unit_id] = max(unit_error_map[unit.unit_id], 0.35)
    return issues, unit_error_map, contradiction_score


def _generic_completeness_signal(artifact: ArtifactIR, question_text: str) -> Tuple[float, List[str]]:
    keywords = _extract_question_keywords(question_text)
    answer_lower = artifact.rendered_answer.lower()
    missing = [keyword for keyword in keywords if keyword not in answer_lower]
    if not artifact.rendered_answer.strip():
        return 1.0, ["empty answer"]
    residual = clamp01(float(len(missing)) / float(max(1, len(keywords))))
    return residual, missing


def _parse_undirected_edges(question_text: str) -> Dict[Tuple[int, int], int]:
    edges = MultiFidelityEvaluator._parse_nlgraph_edges(question_text)
    return edges


def _node_count(question_text: str, edges: Dict[Tuple[int, int], int]) -> int:
    match = re.search(r"numbered from 0 to (\d+)", question_text, flags=re.IGNORECASE)
    if match:
        return int(match.group(1)) + 1
    if not edges:
        return 0
    return max(max(src, dst) for src, dst in edges.keys()) + 1


def _path_exists(edges: Dict[Tuple[int, int], int], start: int, end: int) -> bool:
    if start == end:
        return True
    graph: Dict[int, List[int]] = defaultdict(list)
    for src, dst in edges.keys():
        graph[src].append(dst)
    queue: deque[int] = deque([start])
    visited = {start}
    while queue:
        node = queue.popleft()
        for nxt in graph.get(node, []):
            if nxt == end:
                return True
            if nxt in visited:
                continue
            visited.add(nxt)
            queue.append(nxt)
    return False


def _has_cycle_undirected(edges: Dict[Tuple[int, int], int]) -> bool:
    graph: Dict[int, List[int]] = defaultdict(list)
    for src, dst in edges.keys():
        graph[src].append(dst)
    visited: set[int] = set()
    for start in list(graph.keys()):
        if start in visited:
            continue
        stack = [(start, -1)]
        while stack:
            node, parent = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            for nxt in graph.get(node, []):
                if nxt == parent:
                    continue
                if nxt in visited:
                    return True
                stack.append((nxt, node))
    return False


def _parse_connectivity_query(question_text: str) -> Tuple[Optional[int], Optional[int]]:
    match = re.search(r"between node (\d+) and node (\d+)", question_text, flags=re.IGNORECASE)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def _parse_flow_edges(question_text: str) -> Dict[Tuple[int, int], int]:
    edges: Dict[Tuple[int, int], int] = {}
    for src, dst, capacity in re.findall(
        r"edge from node (\d+) to node (\d+) with capacity (\d+)",
        question_text,
        flags=re.IGNORECASE,
    ):
        edges[(int(src), int(dst))] = int(capacity)
    return edges


def _parse_flow_query(question_text: str) -> Tuple[Optional[int], Optional[int]]:
    match = re.search(r"maximum flow from node (\d+) to node (\d+)", question_text, flags=re.IGNORECASE)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def _max_flow(capacities: Dict[Tuple[int, int], int], source: int, sink: int) -> int:
    residual: Dict[int, Dict[int, int]] = defaultdict(dict)
    for (src, dst), capacity in capacities.items():
        residual[src][dst] = residual[src].get(dst, 0) + int(capacity)
        residual[dst].setdefault(src, 0)
    total = 0
    while True:
        parent: Dict[int, int] = {source: -1}
        queue: deque[int] = deque([source])
        while queue and sink not in parent:
            node = queue.popleft()
            for nxt, capacity in residual.get(node, {}).items():
                if capacity <= 0 or nxt in parent:
                    continue
                parent[nxt] = node
                queue.append(nxt)
        if sink not in parent:
            break
        bottleneck = None
        node = sink
        while node != source:
            prev = parent[node]
            cap = residual[prev][node]
            bottleneck = cap if bottleneck is None else min(bottleneck, cap)
            node = prev
        flow = int(bottleneck or 0)
        if flow <= 0:
            break
        total += flow
        node = sink
        while node != source:
            prev = parent[node]
            residual[prev][node] -= flow
            residual[node][prev] = residual[node].get(prev, 0) + flow
            node = prev
    return total


def _parse_topology_constraints(question_text: str) -> List[Tuple[int, int]]:
    return [(int(a), int(b)) for a, b in re.findall(r"node (\d+) should be visited before node (\d+)", question_text)]


def _parse_shortest_path_query(question_text: str) -> Tuple[Optional[int], Optional[int]]:
    match = re.search(r"shortest path from node (\d+) to node (\d+)", question_text, flags=re.IGNORECASE)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def _shortest_path_distance(edges: Dict[Tuple[int, int], int], start: int, end: int) -> Optional[int]:
    pq: List[Tuple[int, int]] = [(0, start)]
    best: Dict[int, int] = {start: 0}
    while pq:
        dist, node = heapq.heappop(pq)
        if node == end:
            return dist
        if dist != best.get(node):
            continue
        neighbors = [(dst, weight) for (src, dst), weight in edges.items() if src == node]
        for nxt, weight in neighbors:
            cand = dist + int(weight)
            if cand < best.get(nxt, 10**9):
                best[nxt] = cand
                heapq.heappush(pq, (cand, nxt))
    return None


def _parse_embeddings(question_text: str) -> Dict[str, List[int]]:
    embeddings: Dict[str, List[int]] = {}
    for node, vec in re.findall(r"node\s+(\d+)\s*:\s*\[([^\]]+)\]", question_text, flags=re.IGNORECASE):
        embeddings[str(node)] = [int(token) for token in re.findall(r"-?\d+", vec)]
    return embeddings


def _compute_gnn_one_step(question_text: str) -> Dict[str, List[int]]:
    base = _parse_embeddings(question_text)
    edges = _parse_undirected_edges(question_text)
    neighbors: Dict[str, List[str]] = defaultdict(list)
    for src, dst in edges.keys():
        neighbors[str(src)].append(str(dst))
    updated: Dict[str, List[int]] = {}
    for node, vector in base.items():
        dim = len(vector)
        total = [0 for _ in range(dim)]
        for neighbor in neighbors.get(node, []):
            neighbor_vec = list(base.get(neighbor, [0 for _ in range(dim)]))
            if len(neighbor_vec) != dim:
                continue
            for index, value in enumerate(neighbor_vec):
                total[index] += int(value)
        updated[node] = total
    return updated


def _parse_matching_interests(question_text: str) -> Dict[int, List[int]]:
    interests: Dict[int, List[int]] = defaultdict(list)
    for applicant, job in re.findall(r"Applicant\s+(\d+)\s+is interested in job\s+(\d+)", question_text, flags=re.IGNORECASE):
        interests[int(applicant)].append(int(job))
    return interests


def _max_matching_count(interests: Dict[int, List[int]]) -> int:
    jobs_to_applicant: Dict[int, int] = {}

    def _dfs(applicant: int, seen: set[int]) -> bool:
        for job in interests.get(applicant, []):
            if job in seen:
                continue
            seen.add(job)
            if job not in jobs_to_applicant or _dfs(jobs_to_applicant[job], seen):
                jobs_to_applicant[job] = applicant
                return True
        return False

    matches = 0
    for applicant in sorted(interests):
        if _dfs(applicant, set()):
            matches += 1
    return matches


def _option_surface_features(
    answer_object: AnswerObject,
    *,
    metadata: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if answer_object.kind != "option" or not isinstance(answer_object.value, int):
        return {"value": None, "letter": "", "option_text": ""}
    value = int(answer_object.value)
    options = metadata.get("options") if isinstance(metadata, dict) and isinstance(metadata.get("options"), list) else []
    option_text = ""
    if 1 <= value <= len(options):
        option_text = str(options[value - 1] or "").strip().lower()
    letter = chr(ord("A") + value - 1) if 1 <= value <= 26 else ""
    return {"value": value, "letter": letter, "option_text": option_text}


def _count_option_mentions(blob: str, *, value: Optional[int], letter: str, option_text: str) -> int:
    if not blob:
        return 0
    count = 0
    if value is not None:
        count += len(re.findall(rf"\b{re.escape(str(value))}\b", blob))
    if letter:
        count += len(re.findall(rf"\b{re.escape(letter.lower())}\b", blob))
    if option_text:
        count += len(re.findall(rf"\b{re.escape(option_text)}\b", blob))
    return count


def _mcq_support_features(
    artifact: ArtifactIR,
    *,
    metadata: Optional[Dict[str, Any]],
    contradiction_score: float = 0.0,
    residual_vector: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    answer_object = artifact.answer_object
    answer_blob = answer_text(artifact).lower()
    evidence_blob = "\n".join(unit.text for unit in evidence_units(artifact) if unit.text.strip()).lower()
    features = _option_surface_features(answer_object, metadata=metadata)
    chosen_value = features["value"]
    chosen_letter = features["letter"]
    chosen_text = features["option_text"]
    chosen_in_answer = 1.0 if _count_option_mentions(answer_blob, value=chosen_value, letter=chosen_letter, option_text=chosen_text) > 0 else 0.0
    chosen_in_evidence = float(_count_option_mentions(evidence_blob, value=chosen_value, letter=chosen_letter, option_text=chosen_text))

    competing_mentions = 0.0
    options = metadata.get("options") if isinstance(metadata, dict) and isinstance(metadata.get("options"), list) else []
    if options and isinstance(chosen_value, int):
        for index, option in enumerate(options, start=1):
            if index == chosen_value:
                continue
            option_letter = chr(ord("A") + index - 1) if 1 <= index <= 26 else ""
            competing_mentions += float(
                _count_option_mentions(
                    evidence_blob,
                    value=index,
                    letter=option_letter,
                    option_text=str(option or "").strip().lower(),
                )
            )

    parse_ok = 1.0 - float((residual_vector or {}).get("r_parse", 0.0))
    completeness_ok = 1.0 - float((residual_vector or {}).get("r_completeness", 0.0))
    support_score = clamp01(
        0.26 * chosen_in_answer
        + 0.30 * clamp01(chosen_in_evidence / 2.0)
        + 0.16 * parse_ok
        + 0.12 * completeness_ok
        + 0.16 * clamp01(1.0 - contradiction_score)
        - 0.18 * clamp01(competing_mentions / 3.0)
    )
    return {
        "chosen_in_answer": chosen_in_answer,
        "chosen_in_evidence": chosen_in_evidence,
        "competing_mentions": competing_mentions,
        "support_score": support_score,
    }


def _typed_executor_channel(
    artifact: ArtifactIR,
    *,
    question_text: str,
    metadata: Optional[Dict[str, Any]],
    dataset_name: str,
    task_type: str,
    answer_format: str,
    task_subtype: str,
    timeout_s: float,
    max_failed_examples: int,
) -> Tuple[Dict[str, float], float, Dict[str, Any]]:
    del dataset_name
    if task_type == "code_generation" or answer_format == "python_code":
        feedback = evaluate_code_candidate(
            artifact.rendered_answer,
            metadata,
            timeout_s=timeout_s,
            max_failed_examples=max_failed_examples,
        )
        accuracy = clamp01(feedback.accuracy if feedback.total > 0 else 0.0)
        constraint_ok = 1.0 if feedback.entry_point_ok else 0.0
        parse_ok = 1.0 if feedback.syntax_ok else 0.0
        quality = clamp01(0.25 * parse_ok + 0.25 * constraint_ok + 0.50 * accuracy)
        return (
            {
                "r_parse": clamp01(1.0 - parse_ok),
                "r_execution": clamp01(1.0 - accuracy if feedback.total > 0 else 1.0 - parse_ok),
                "r_constraint": clamp01(1.0 - constraint_ok),
            },
            quality,
            {
                "available": True,
                "channel": "code",
                "syntax_ok": feedback.syntax_ok,
                "entry_point_ok": feedback.entry_point_ok,
                "passed": feedback.passed,
                "total": feedback.total,
                "failure_kind": feedback.failure_kind,
                "failing_examples": list(feedback.failing_examples),
                "stderr": feedback.stderr,
                "exec_error": feedback.exec_error,
                "syntax_error_line": feedback.syntax_error_line,
                "syntax_error_offset": feedback.syntax_error_offset,
                "syntax_error_text": feedback.syntax_error_text,
            },
        )

    answer_object = artifact.answer_object
    if task_type == "mcq" or answer_format == "option":
        max_option = _option_count(question_text, metadata)
        value = answer_object.value if answer_object.kind == "option" else None
        valid_option = isinstance(value, int) and (max_option <= 0 or 1 <= value <= max_option)
        mcq_features = _mcq_support_features(artifact, metadata=metadata)
        support_score = float(mcq_features.get("support_score", 0.0))
        quality = clamp01(0.55 * (1.0 if valid_option else 0.0) + 0.45 * support_score)
        return (
            {
                "r_execution": 0.0,
                "r_constraint": 0.0 if valid_option else 1.0,
            },
            quality,
            {
                "available": True,
                "channel": "mcq",
                "max_option": max_option,
                "valid_option": valid_option,
                "parsed_option": value,
                "mcq_support_score": support_score,
                "mcq_chosen_in_answer": float(mcq_features.get("chosen_in_answer", 0.0)),
                "mcq_chosen_in_evidence": float(mcq_features.get("chosen_in_evidence", 0.0)),
                "mcq_competing_mentions": float(mcq_features.get("competing_mentions", 0.0)),
            },
        )

    if task_type != "graph_reasoning" and answer_format != "graph_json":
        parse_ok = 1.0 if answer_object.valid else 0.0
        return (
            {
                "r_execution": 1.0 - parse_ok,
                "r_constraint": 1.0 - parse_ok,
            },
            parse_ok,
            {"available": False, "channel": "generic"},
        )

    subtype = str(task_subtype or "").strip()
    feedback: Dict[str, Any] = {"available": True, "channel": "graph", "task_subtype": subtype}
    weighted_edges = _parse_undirected_edges(question_text)

    if subtype == "connectivity":
        start, end = _parse_connectivity_query(question_text)
        if start is None or end is None:
            return {"r_execution": 1.0, "r_constraint": 1.0}, 0.0, {**feedback, "query_parse_ok": False}
        expected = "yes" if _path_exists(weighted_edges, start, end) else "no"
        predicted = str(_semantic_value(answer_object) or "").lower()
        correct = predicted == expected
        return (
            {"r_execution": 0.0 if correct else 1.0, "r_constraint": 0.0 if answer_object.valid else 1.0},
            1.0 if correct else 0.0,
            {**feedback, "expected": expected, "predicted": predicted, "query_parse_ok": True},
        )

    if subtype == "cycle":
        expected = "yes" if _has_cycle_undirected(weighted_edges) else "no"
        predicted = str(_semantic_value(answer_object) or "").lower()
        correct = predicted == expected
        return (
            {"r_execution": 0.0 if correct else 1.0, "r_constraint": 0.0 if answer_object.valid else 1.0},
            1.0 if correct else 0.0,
            {**feedback, "expected": expected, "predicted": predicted},
        )

    if subtype == "flow":
        capacities = _parse_flow_edges(question_text)
        source, sink = _parse_flow_query(question_text)
        if not capacities or source is None or sink is None:
            return {"r_execution": 1.0, "r_constraint": 1.0}, 0.0, {**feedback, "query_parse_ok": False}
        expected = float(_max_flow(capacities, source, sink))
        predicted = _semantic_value(answer_object) if answer_object.kind == "graph_scalar" else None
        if predicted is None:
            return {"r_execution": 1.0, "r_constraint": 0.75}, 0.0, {**feedback, "expected": expected, "predicted": None}
        diff = typed_answer_distance(
            AnswerObject(kind="graph_scalar", value=predicted, valid=True, fields={}, signature=""),
            AnswerObject(kind="graph_scalar", value=expected, valid=True, fields={}, signature=""),
        )
        return (
            {"r_execution": diff, "r_constraint": 0.0 if answer_object.valid else 1.0},
            1.0 - diff,
            {**feedback, "expected": expected, "predicted": predicted},
        )

    if subtype == "topology":
        order = list(_semantic_value(answer_object) or [])
        constraints = _parse_topology_constraints(question_text)
        if not order or not constraints:
            return {"r_execution": 1.0, "r_constraint": 1.0}, 0.0, {**feedback, "format_ok": False}
        position = {node: idx for idx, node in enumerate(order)}
        satisfied = sum(1 for left, right in constraints if left in position and right in position and position[left] < position[right])
        ratio = float(satisfied) / float(max(1, len(constraints)))
        valid = ratio >= 0.999 and len(position) == len(order)
        return (
            {"r_execution": 1.0 - ratio, "r_constraint": 0.0 if valid else 1.0 - ratio},
            ratio,
            {**feedback, "constraints_satisfied": satisfied, "constraints_total": len(constraints), "valid": valid},
        )

    if subtype == "hamilton":
        path = list(_semantic_value(answer_object) or [])
        expected_nodes = _node_count(question_text, weighted_edges)
        if not path:
            return {"r_execution": 1.0, "r_constraint": 1.0}, 0.0, {**feedback, "format_ok": False}
        unique_ratio = float(len(set(path))) / float(max(1, expected_nodes))
        edge_ratio = float(sum(1 for src, dst in zip(path, path[1:]) if (src, dst) in weighted_edges)) / float(max(1, len(path) - 1))
        score = clamp01(0.5 * unique_ratio + 0.5 * edge_ratio)
        valid = len(path) == expected_nodes and len(set(path)) == expected_nodes and edge_ratio >= 0.999
        return (
            {"r_execution": 1.0 - score, "r_constraint": 0.0 if valid else 1.0 - score},
            score,
            {**feedback, "path_length": len(path), "expected_nodes": expected_nodes, "edge_ratio": edge_ratio},
        )

    if subtype == "shortest_path":
        path_payload = dict(_semantic_value(answer_object) or {})
        path = list(path_payload.get("path") or [])
        predicted_weight = path_payload.get("total_weight")
        start, end = _parse_shortest_path_query(question_text)
        shortest = _shortest_path_distance(weighted_edges, start, end) if start is not None and end is not None else None
        path_weight = MultiFidelityEvaluator._path_weight(path, weighted_edges) if path else None
        if shortest is None:
            return {"r_execution": 1.0, "r_constraint": 1.0}, 0.0, {**feedback, "query_parse_ok": False}
        valid_path = path_weight is not None
        correct_weight = predicted_weight is not None and abs(float(predicted_weight) - float(shortest)) <= 1e-6
        inferred_weight = path_weight is not None and abs(float(path_weight) - float(shortest)) <= 1e-6
        correct = correct_weight or inferred_weight
        score = 1.0 if correct else (0.45 if valid_path else 0.0)
        return (
            {"r_execution": 1.0 - score, "r_constraint": 0.0 if valid_path else 1.0},
            score,
            {**feedback, "path_weight": path_weight, "predicted_weight": predicted_weight, "shortest_weight": shortest},
        )

    if subtype == "matching":
        payload = dict(_semantic_value(answer_object) or {})
        matches = [tuple(item) for item in payload.get("matches") or []]
        count = payload.get("count")
        interests = _parse_matching_interests(question_text)
        max_count = _max_matching_count(interests) if interests else None
        unique_applicants = len({applicant for applicant, _ in matches}) == len(matches)
        unique_jobs = len({job for _, job in matches}) == len(matches)
        edge_valid = all(job in interests.get(applicant, []) for applicant, job in matches)
        local_valid = bool(matches) and unique_applicants and unique_jobs and edge_valid
        if max_count is None:
            return {"r_execution": 1.0, "r_constraint": 1.0 if not local_valid else 0.0}, 0.0, {**feedback, "matching_parse_ok": False}
        predicted_count = int(count) if isinstance(count, (int, float)) else len(matches)
        count_score = 1.0 if predicted_count == max_count else max(0.0, 1.0 - abs(predicted_count - max_count) / float(max(1, max_count)))
        structure_score = 1.0 if local_valid else 0.0
        score = clamp01(0.65 * count_score + 0.35 * structure_score)
        return (
            {"r_execution": 1.0 - count_score, "r_constraint": 1.0 - structure_score},
            score,
            {**feedback, "predicted_count": predicted_count, "max_count": max_count, "structure_valid": local_valid},
        )

    if subtype == "GNN":
        expected_embeddings = _compute_gnn_one_step(question_text)
        predicted_embeddings = dict(_semantic_value(answer_object) or {})
        if not expected_embeddings:
            return {"r_execution": 1.0, "r_constraint": 1.0}, 0.0, {**feedback, "gnn_parse_ok": False}
        total = len(expected_embeddings)
        matches = sum(1 for node, vector in expected_embeddings.items() if predicted_embeddings.get(node) == vector)
        accuracy = float(matches) / float(max(1, total))
        coverage = float(len(set(predicted_embeddings) & set(expected_embeddings))) / float(max(1, total))
        return (
            {"r_execution": 1.0 - accuracy, "r_constraint": 1.0 - coverage},
            accuracy,
            {**feedback, "matches": matches, "total": total, "coverage": coverage},
        )

    return {"r_execution": 1.0, "r_constraint": 1.0}, 0.0, {**feedback, "unsupported_subtype": subtype}


def _answer_consistency(
    artifact: ArtifactIR,
    *,
    support_map: Dict[str, float],
    contradiction_score: float,
    residual_vector: Dict[str, float],
    metadata: Optional[Dict[str, Any]],
    executor_feedback: Optional[Dict[str, Any]] = None,
) -> float:
    answer_object = artifact.answer_object
    if not _recoverable_valid(answer_object):
        return 0.0
    answer_ids = set(artifact.answer_unit_ids)
    evidence_ids = set(artifact.evidence_unit_ids)
    answer_support = _mean([support_map.get(unit_id, 0.0) for unit_id in answer_ids]) if answer_ids else 0.0
    evidence_support = _mean([support_map.get(unit_id, 0.0) for unit_id in evidence_ids]) if evidence_ids else 0.0
    answer_vec = lexical_feature_vector(answer_text(artifact))
    evidence_text = "\n".join(unit.text for unit in evidence_units(artifact) if unit.text.strip())
    lexical = 0.5
    if evidence_text.strip():
        lexical = clamp01((cosine_similarity(answer_vec, lexical_feature_vector(evidence_text)) + 1.0) * 0.5)

    execution_ok = 1.0 - float(residual_vector.get("r_execution", 1.0))
    constraint_ok = 1.0 - float(residual_vector.get("r_constraint", 1.0))
    parse_ok = 1.0 - float(residual_vector.get("r_parse", 1.0))
    completeness_ok = 1.0 - float(residual_vector.get("r_completeness", 1.0))

    score = (
        0.16
        + 0.20 * answer_support
        + 0.18 * evidence_support
        + 0.12 * lexical
        + 0.10 * parse_ok
        + 0.10 * completeness_ok
        + 0.14 * execution_ok
        + 0.10 * constraint_ok
    )

    if answer_object.kind == "option":
        mcq_features = _mcq_support_features(
            artifact,
            metadata=metadata,
            contradiction_score=contradiction_score,
            residual_vector=residual_vector,
        )
        support_score = float((executor_feedback or {}).get("mcq_support_score", mcq_features.get("support_score", 0.0)))
        chosen_in_answer = float((executor_feedback or {}).get("mcq_chosen_in_answer", mcq_features.get("chosen_in_answer", 0.0)))
        chosen_in_evidence = clamp01(float((executor_feedback or {}).get("mcq_chosen_in_evidence", mcq_features.get("chosen_in_evidence", 0.0))) / 2.0)
        competing_mentions = clamp01(float((executor_feedback or {}).get("mcq_competing_mentions", mcq_features.get("competing_mentions", 0.0))) / 3.0)
        score = (
            0.10
            + 0.22 * support_score
            + 0.18 * chosen_in_answer
            + 0.16 * chosen_in_evidence
            + 0.12 * parse_ok
            + 0.10 * completeness_ok
            + 0.06 * execution_ok
            + 0.06 * constraint_ok
            + 0.08 * answer_support
            - 0.12 * competing_mentions
        )
    if answer_object.kind in {"graph_bool", "graph_scalar", "graph_sequence", "graph_path", "graph_matching", "node_embeddings"}:
        score += 0.08 * parse_ok + 0.10 * constraint_ok + 0.10 * execution_ok
    if answer_object.kind == "code":
        syntax_ok = 1.0 if bool(answer_object.fields.get("syntax_ok")) else 0.0
        entry_ok = 1.0 if bool(answer_object.fields.get("entry_point_present")) else 0.0
        score += 0.10 * syntax_ok + 0.08 * entry_ok

    score -= 0.35 * contradiction_score
    score *= 1.0 - 0.75 * max(float(residual_vector.get("r_execution", 0.0)), float(residual_vector.get("r_constraint", 0.0)))
    return clamp01(score)


def _normalize_unit_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _critical_anchor_unit_ids(anchor_artifact: ArtifactIR, anchor_state: VerifierState) -> List[str]:
    critical = set(anchor_artifact.answer_unit_ids)
    evidence_pairs = sorted(
        (
            (float(anchor_state.support_map.get(unit_id, 0.0)), unit_id)
            for unit_id in anchor_artifact.evidence_unit_ids
        ),
        reverse=True,
    )
    for support, unit_id in evidence_pairs:
        if support >= 0.6:
            critical.add(unit_id)
    for _, unit_id in evidence_pairs[:2]:
        critical.add(unit_id)
    return list(critical)


def _pairwise_preserve_risk(
    artifact: ArtifactIR,
    *,
    anchor_artifact: Optional[ArtifactIR],
    anchor_state: Optional[VerifierState],
) -> Tuple[float, float]:
    if anchor_artifact is None or anchor_state is None or not anchor_artifact.rendered_answer.strip():
        return 0.0, 0.0
    candidate_vec = pooled_artifact_vector(artifact)
    anchor_vec = pooled_artifact_vector(anchor_artifact)
    similarity = clamp01((cosine_similarity(candidate_vec, anchor_vec) + 1.0) * 0.5)

    critical_ids = _critical_anchor_unit_ids(anchor_artifact, anchor_state)
    candidate_units = [_normalize_unit_text(unit.text) for unit in artifact.units if unit.text.strip()]
    total_weight = 0.0
    preserved_weight = 0.0
    answer_similarity = typed_answer_similarity(artifact.answer_object, anchor_artifact.answer_object)
    for unit in anchor_artifact.units:
        if unit.unit_id not in critical_ids:
            continue
        weight = max(0.2, float(anchor_state.support_map.get(unit.unit_id, 0.0)))
        total_weight += weight
        normalized = _normalize_unit_text(unit.text)
        matched = normalized in candidate_units
        if not matched:
            matched = any(
                cosine_similarity(lexical_feature_vector(unit.text), lexical_feature_vector(candidate_unit.text)) >= 0.96
                for candidate_unit in artifact.units
            )
        if not matched and unit.unit_id in set(anchor_artifact.answer_unit_ids) and answer_similarity >= 0.999:
            matched = True
        if matched:
            preserved_weight += weight
    retention = preserved_weight / float(max(1e-6, total_weight))
    return clamp01(1.0 - retention), similarity


def apply_anchor_pairwise_metrics(
    artifact: ArtifactIR,
    verifier_state: VerifierState,
    *,
    anchor_artifact: Optional[ArtifactIR],
    anchor_state: Optional[VerifierState],
) -> VerifierState:
    if anchor_artifact is None or anchor_state is None:
        return replace(
            verifier_state,
            preserve_risk=0.0,
            anchor_similarity=0.0,
            answer_similarity=0.0,
            answer_delta=0.0,
        )
    answer_similarity = typed_answer_similarity(artifact.answer_object, anchor_artifact.answer_object)
    answer_delta = typed_answer_distance(artifact.answer_object, anchor_artifact.answer_object)
    preserve_risk, anchor_similarity = _pairwise_preserve_risk(
        artifact,
        anchor_artifact=anchor_artifact,
        anchor_state=anchor_state,
    )
    return replace(
        verifier_state,
        preserve_risk=preserve_risk,
        anchor_similarity=anchor_similarity,
        answer_similarity=answer_similarity,
        answer_delta=answer_delta,
    )


def verify_artifact(
    artifact: ArtifactIR,
    *,
    question_text: str,
    metadata: Optional[Dict[str, Any]],
    dataset_name: str,
    task_type: str,
    answer_format: str,
    task_subtype: str,
    candidate_entry: Optional[Dict[str, Any]] = None,
    anchor_artifact: Optional[ArtifactIR] = None,
    timeout_s: float = 8.0,
    max_failed_examples: int = 3,
) -> VerifierState:
    del anchor_artifact
    parser_quality = _mean(list(artifact.parse_confidence.values()))
    support_map = _support_map(artifact, candidate_entry)
    issues, unit_error_map, contradiction_score = _consistency_issues(artifact)
    generic_completeness, missing_requirements = _generic_completeness_signal(artifact, question_text)
    typed_parse = _answer_parse_residual(artifact.answer_object)
    typed_completeness = _answer_completeness_residual(artifact.answer_object)

    executor_residuals, executor_quality, executor_feedback = _typed_executor_channel(
        artifact,
        question_text=question_text,
        metadata=metadata,
        dataset_name=dataset_name,
        task_type=task_type,
        answer_format=answer_format,
        task_subtype=task_subtype,
        timeout_s=timeout_s,
        max_failed_examples=max_failed_examples,
    )

    search_quality = _mean(list(support_map.values()))
    symbolic_quality = clamp01(1.0 - contradiction_score)
    channel_names = (
        "parser_channel",
        "executor_channel",
        "constraint_channel",
        "search_channel",
        "symbolic_channel",
    )
    channel_quality_values = (
        clamp01(1.0 - typed_parse),
        executor_quality,
        clamp01(1.0 - executor_residuals.get("r_constraint", 1.0)),
        search_quality,
        symbolic_quality,
    )
    channel_weights = {
        name: weight
        for name, weight in zip(channel_names, sparsemax(channel_quality_values))
    }

    completeness_residual = clamp01(0.35 * generic_completeness + 0.65 * typed_completeness)
    residual_vector = {
        "r_parse": clamp01(executor_residuals.get("r_parse", typed_parse)),
        "r_consistency": contradiction_score,
        "r_completeness": completeness_residual,
        "r_execution": clamp01(executor_residuals.get("r_execution", 1.0)),
        "r_constraint": clamp01(executor_residuals.get("r_constraint", 1.0)),
        "r_support": clamp01(1.0 - search_quality),
        "r_preserve": 0.0,
    }

    if executor_feedback.get("syntax_error_line"):
        line_index = int(executor_feedback["syntax_error_line"]) - 1
        if 0 <= line_index < len(artifact.units):
            unit_id = artifact.units[line_index].unit_id
            unit_error_map[unit_id] = max(unit_error_map.get(unit_id, 0.0), 0.9)
    if executor_feedback.get("failure_kind") == "entry_point_missing" and artifact.units:
        unit_error_map[artifact.units[0].unit_id] = max(unit_error_map.get(artifact.units[0].unit_id, 0.0), 0.85)
    for unit_id, support in support_map.items():
        if support < 0.45:
            unit_error_map[unit_id] = max(unit_error_map.get(unit_id, 0.0), clamp01(0.55 - support))

    answer_consistency_score = _answer_consistency(
        artifact,
        support_map=support_map,
        contradiction_score=contradiction_score,
        residual_vector=residual_vector,
        metadata=metadata,
        executor_feedback=executor_feedback,
    )
    confidence_score = clamp01(
        0.42 * (1.0 - _mean(list(residual_vector.values())))
        + 0.24 * answer_consistency_score
        + 0.16 * search_quality
        + 0.10 * (1.0 - residual_vector["r_parse"])
        + 0.08 * (1.0 - residual_vector["r_constraint"])
    )
    progress_score = clamp01(
        0.38 * confidence_score
        + 0.22 * answer_consistency_score
        + 0.18 * (1.0 - residual_vector["r_support"])
        + 0.22 * (1.0 - residual_vector["r_completeness"])
    )
    meta_summary = (
        f"parse={1.0 - residual_vector['r_parse']:.2f} "
        f"constraint={1.0 - residual_vector['r_constraint']:.2f} "
        f"execution={1.0 - residual_vector['r_execution']:.2f} "
        f"answer_consistency={answer_consistency_score:.2f} "
        f"support={1.0 - residual_vector['r_support']:.2f}"
    )
    return VerifierState(
        residual_vector={key: clamp01(residual_vector.get(key, 0.0)) for key in RESIDUAL_KEYS},
        unit_error_map={unit_id: clamp01(score) for unit_id, score in unit_error_map.items()},
        support_map={unit_id: clamp01(score) for unit_id, score in support_map.items()},
        completeness_score=clamp01(1.0 - residual_vector["r_completeness"]),
        consistency_score=clamp01(1.0 - residual_vector["r_consistency"]),
        answer_consistency_score=answer_consistency_score,
        executability_score=clamp01(1.0 - residual_vector["r_execution"]),
        constraint_score=clamp01(1.0 - residual_vector["r_constraint"]),
        confidence_score=confidence_score,
        progress_score=progress_score,
        preserve_risk=0.0,
        anchor_similarity=0.0,
        answer_similarity=0.0,
        answer_delta=0.0,
        overturn_risk=0.0,
        channel_weights=channel_weights,
        meta_summary=meta_summary,
        issues=issues,
        missing_requirements=list(missing_requirements),
        executor_feedback=executor_feedback,
    )


def compute_overturn_risk(
    artifact: ArtifactIR,
    verifier_state: VerifierState,
    *,
    anchor_artifact: Optional[ArtifactIR],
    anchor_state: Optional[VerifierState],
) -> float:
    if anchor_artifact is None or anchor_state is None:
        return 0.0
    candidate_residual = mean_residual(verifier_state)
    anchor_residual = mean_residual(anchor_state)
    residual_gap = candidate_residual - anchor_residual
    confidence_gap = float(verifier_state.confidence_score) - float(anchor_state.confidence_score)
    support_gap = float(verifier_state.answer_consistency_score) - float(anchor_state.answer_consistency_score)
    risk = (
        0.30 * float(verifier_state.answer_delta)
        + 0.18 * clamp01(residual_gap + 0.5)
        + 0.10 * clamp01(-confidence_gap + 0.5)
        + 0.18 * float(verifier_state.preserve_risk)
        + 0.14 * clamp01(1.0 - verifier_state.answer_consistency_score)
    )
    if artifact.answer_object.kind == "option":
        risk += 0.10 * clamp01(-support_gap + 0.5)
    if artifact.answer_signature == anchor_artifact.answer_signature and artifact.answer_signature:
        risk *= 0.25
    if not artifact.answer_object.valid:
        risk = max(risk, 0.85)
    if verifier_state.executability_score < 0.2 and artifact.schema_features.get("task_type") == "graph_reasoning":
        risk = max(risk, 0.80)
    return clamp01(risk)
