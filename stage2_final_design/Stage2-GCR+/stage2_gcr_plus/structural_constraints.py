from __future__ import annotations

import ast
import hashlib
import heapq
import json
import math
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Literal, Optional, Sequence


STRUCTURAL_DATASETS = {"nlgraph", "knowledge_crosswords"}


@dataclass(frozen=True)
class StructuralProblemIR:
    dataset_name: str
    object_kind: Literal["graph", "knowledge_crossword"]
    task_kind: str
    parse_confidence: str
    raw_question: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class StructuralArtifact:
    raw_text: str
    dataset_name: str
    object_kind: str
    answer: Any
    normalized_answer: str
    witness: dict[str, Any]
    contract_ok: bool
    final_source: str
    parser_confidence: str
    artifact_signature: str


@dataclass(frozen=True)
class StructuralResidual:
    fatal: tuple[str, ...]
    local: tuple[str, ...]
    support: tuple[str, ...]
    repair_locus: str
    verified_constraints: int
    residual_kind: str
    residual_signature: str
    certificate_kind: str
    certificate_ok: bool


@dataclass(frozen=True)
class StructuralEval:
    problem: StructuralProblemIR
    artifact: StructuralArtifact
    residual: StructuralResidual

    @property
    def class_key(self) -> tuple[Any, ...]:
        return structural_class_key(self)

    @property
    def rank_key(self) -> tuple[Any, ...]:
        return structural_rank_key(self)


@dataclass(frozen=True)
class PathCheck:
    fatal: tuple[str, ...]
    local: tuple[str, ...]
    verified: int


@dataclass(frozen=True)
class KCVerifyAllResult:
    status: str
    failed_constraints: tuple[dict[str, Any], ...]
    answers: tuple[str, ...]
    confidence: str
    raw: str


_STRUCTURAL_DETERMINISTIC_FATALS = {
    "invalid_json",
    "invalid_json_list",
    "missing_answer",
    "missing_path",
    "invalid_node",
    "invalid_edge",
    "wrong_endpoint",
    "path_not_shortest",
    "false_positive_connectivity",
    "false_negative_connectivity",
    "order_missing_node",
    "order_invalid_node",
    "order_duplicate_node",
    "order_violation",
    "cycle_invalid",
    "list_length_mismatch",
    "empty_blank_assignment",
    "option_not_allowed",
}


def _strip_hidden_reasoning(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", str(text or ""), flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()


def _jsonable(value: Any) -> Any:
    if value is Ellipsis:
        return "..."
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _hash_json(payload: Any) -> str:
    raw = json.dumps(_jsonable(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _node_sort_key(value: Any) -> tuple[int, Any]:
    text = str(value)
    if re.fullmatch(r"-?\d+", text):
        return (0, int(text))
    return (1, text)


def _node_token(value: Any) -> str:
    return str(value).strip()


def _node_public(value: Any) -> Any:
    text = str(value).strip()
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    return text


def _extract_json_span(text: str) -> str | None:
    raw = _strip_hidden_reasoning(text)
    start = -1
    open_ch = ""
    close_ch = ""
    for idx, ch in enumerate(raw):
        if ch in "{[":
            start = idx
            open_ch = ch
            close_ch = "}" if ch == "{" else "]"
            break
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(raw)):
        ch = raw[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return raw[start : idx + 1]
    return None


def _extract_json_object(text: str) -> tuple[Any, str]:
    raw = _strip_hidden_reasoning(text)
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj, "json_object"
    except Exception:
        pass
    span = _extract_json_span(raw)
    if span and span.strip().startswith("{"):
        try:
            obj = json.loads(span)
            if isinstance(obj, dict):
                return obj, "json_object"
        except Exception:
            try:
                obj = ast.literal_eval(span)
                if isinstance(obj, dict):
                    return obj, "json_object"
            except Exception:
                pass
    return None, "none"


def _extract_json_any(text: str) -> tuple[Any, str]:
    raw = _strip_hidden_reasoning(text)
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj, "json_object"
        if isinstance(obj, list):
            return obj, "json_list"
    except Exception:
        pass
    span = _extract_json_span(raw)
    if span:
        try:
            obj = json.loads(span)
            if isinstance(obj, dict):
                return obj, "json_object"
            if isinstance(obj, list):
                return obj, "json_list"
        except Exception:
            try:
                obj = ast.literal_eval(span)
                if isinstance(obj, dict):
                    return obj, "json_object"
                if isinstance(obj, list):
                    return obj, "json_list"
            except Exception:
                pass
    return None, "none"


def _task_from_metadata(metadata: Optional[dict]) -> str:
    task = str((metadata or {}).get("task") or "").strip().lower()
    return {
        "topology": "topological_sort",
        "topological": "topological_sort",
        "hamilton": "hamilton_path",
        "flow": "maximum_flow",
        "max_flow": "maximum_flow",
    }.get(task, task)


def _detect_graph_task_kind(lower: str) -> str:
    if "shortest path" in lower:
        return "shortest_path"
    if "topological" in lower or "topological order" in lower or "topological sort" in lower:
        return "topological_sort"
    if "should be visited before" in lower or "can all the nodes be visited" in lower:
        return "topological_sort"
    if "hamilton" in lower or "visits every node exactly once" in lower:
        return "hamilton_path"
    if "maximum flow" in lower or "max flow" in lower:
        return "maximum_flow"
    if "cycle" in lower:
        return "cycle"
    if "connected" in lower or "connectivity" in lower or "path between" in lower or "path in this graph" in lower:
        return "connectivity"
    return "unknown"


def _parse_declared_nodes(text: str) -> list[str]:
    nodes: set[str] = set()
    for lo, hi in re.findall(
        r"nodes?\s+(?:are\s+)?numbered\s+from\s+(-?\d+)\s+to\s+(-?\d+)",
        text,
        flags=re.IGNORECASE,
    ):
        start = int(lo)
        stop = int(hi)
        step = 1 if stop >= start else -1
        nodes.update(str(i) for i in range(start, stop + step, step))
    for count in re.findall(r"with\s+(\d+)\s+nodes?\s+numbered\s+from\s+0\s+to\s+\d+", text, flags=re.IGNORECASE):
        if not nodes:
            nodes.update(str(i) for i in range(int(count)))
    return sorted(nodes, key=_node_sort_key)


def _parse_edge_list(text: str, *, directed: bool, weighted: bool) -> list[tuple[str, str, float]]:
    del weighted
    edges: list[tuple[str, str, float]] = []

    def is_placeholder_edge(u: str, v: str, start: int, end: int) -> bool:
        pair = (str(u).strip().lower(), str(v).strip().lower())
        if pair not in {("i", "j"), ("u", "v")}:
            return False
        context = text[max(0, start - 80) : min(len(text), end + 120)].lower()
        return "means" in context or "connected with an undirected edge" in context

    for m in re.finditer(
        r"\(\s*([A-Za-z0-9_\-]+)\s*,\s*([A-Za-z0-9_\-]+)(?:\s*,\s*(-?\d+(?:\.\d+)?))?\s*\)",
        text,
    ):
        u, v = m.group(1), m.group(2)
        if is_placeholder_edge(u, v, *m.span()):
            continue
        w = float(m.group(3)) if m.group(3) is not None else 1.0
        edges.append((u, v, w))

    for m in re.finditer(
        r"node\s+([A-Za-z0-9_\-]+)\s+(?:is\s+)?(?:connected\s+to|adjacent\s+to)\s+node\s+([A-Za-z0-9_\-]+)",
        text,
        flags=re.IGNORECASE,
    ):
        if is_placeholder_edge(m.group(1), m.group(2), *m.span()):
            continue
        edges.append((m.group(1), m.group(2), 1.0))

    for m in re.finditer(
        r"edge\s+(?:between\s+)?node\s+([A-Za-z0-9_\-]+)\s+(?:and|to)\s+node\s+([A-Za-z0-9_\-]+)(?:\s+with\s+weight\s+(-?\d+(?:\.\d+)?))?",
        text,
        flags=re.IGNORECASE,
    ):
        if is_placeholder_edge(m.group(1), m.group(2), *m.span()):
            continue
        w = float(m.group(3)) if m.group(3) is not None else 1.0
        edges.append((m.group(1), m.group(2), w))

    for m in re.finditer(
        r"edge\s+from\s+([A-Za-z0-9_\-]+)\s+to\s+([A-Za-z0-9_\-]+)",
        text,
        flags=re.IGNORECASE,
    ):
        edges.append((m.group(1), m.group(2), 1.0))

    for m in re.finditer(
        r"node\s+([A-Za-z0-9_\-]+)\s+should\s+be\s+visited\s+before\s+node\s+([A-Za-z0-9_\-]+)",
        text,
        flags=re.IGNORECASE,
    ):
        edges.append((m.group(1), m.group(2), 1.0))

    seen: set[tuple[Any, ...]] = set()
    out: list[tuple[str, str, float]] = []
    for u, v, w in edges:
        key: tuple[Any, ...]
        if directed:
            key = (u, v, w)
        else:
            a, b = sorted((u, v), key=_node_sort_key)
            key = (a, b, w)
        if key not in seen:
            seen.add(key)
            out.append((u, v, w))
    return out


def _parse_source_target(text: str) -> tuple[str | None, str | None]:
    patterns = [
        r"from\s+node\s+([A-Za-z0-9_\-]+)\s+to\s+node\s+([A-Za-z0-9_\-]+)",
        r"between\s+node\s+([A-Za-z0-9_\-]+)\s+and\s+node\s+([A-Za-z0-9_\-]+)",
        r"path\s+from\s+([A-Za-z0-9_\-]+)\s+to\s+([A-Za-z0-9_\-]+)",
        r"path\s+between\s+([A-Za-z0-9_\-]+)\s+and\s+([A-Za-z0-9_\-]+)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            s, t = matches[-1]
            return str(s), str(t)
    return None, None


def parse_nlgraph_problem(question_text: str, metadata: dict | None = None) -> StructuralProblemIR:
    text = str(question_text or "")
    lower = text.lower()
    metadata_task = _task_from_metadata(metadata)
    task_kind = metadata_task or _detect_graph_task_kind(lower)
    directed = (
        bool(re.search(r"\bdirected\s+graph\b", lower))
        or "edge from" in lower
        or "should be visited before" in lower
        or task_kind in {"topological_sort", "maximum_flow"}
    )
    weighted = bool(re.search(r"\(\s*[^,\)]+\s*,\s*[^,\)]+\s*,\s*-?\d+(?:\.\d+)?", text)) or bool(
        re.search(r"with\s+weight\s+-?\d", text, flags=re.IGNORECASE)
    )

    edges = _parse_edge_list(text, directed=directed, weighted=weighted)
    declared_nodes = set(_parse_declared_nodes(text))
    source, target = _parse_source_target(text)
    edge_nodes = {u for u, _, _ in edges} | {v for _, v, _ in edges}
    endpoint_nodes = {node for node in (source, target) if node is not None}
    nodes = sorted(declared_nodes | edge_nodes | endpoint_nodes, key=_node_sort_key)

    confidence = "high" if (edges or task_kind == "topological_sort") and task_kind != "unknown" else "medium" if edges else "low"
    return StructuralProblemIR(
        dataset_name="nlgraph",
        object_kind="graph",
        task_kind=task_kind,
        parse_confidence=confidence,
        raw_question=text,
        payload={
            "directed": directed,
            "weighted": weighted,
            "nodes": nodes,
            "edges": edges,
            "source": source,
            "target": target,
        },
    )


def _parse_blank_ids(text: str, metadata: dict | None = None) -> list[str]:
    meta_blanks = (metadata or {}).get("blanks")
    if isinstance(meta_blanks, list) and meta_blanks:
        return [str(item).strip() for item in meta_blanks if str(item).strip()]
    match = re.search(r"Blanks:\s*(\[[^\]]*\])", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        try:
            parsed = ast.literal_eval(match.group(1))
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except Exception:
            pass
    ordered = re.findall(r"^\s*(\d+)\.\s*(blank\s+\d+)\s*$", text, flags=re.IGNORECASE | re.MULTILINE)
    if ordered:
        return [blank for _, blank in sorted(ordered, key=lambda item: int(item[0]))]
    blanks = sorted(set(re.findall(r"blank\s+\d+", text, flags=re.IGNORECASE)), key=_node_sort_key)
    if blanks:
        return blanks
    numbered = re.findall(r"(?:^|\s)(\d+)\s*:\s*\?", text)
    if numbered:
        return [str(item) for item in sorted({int(x) for x in numbered})]
    return []


def _parse_kc_options(text: str, metadata: dict | None = None) -> list[str]:
    meta_options = (metadata or {}).get("options")
    if isinstance(meta_options, dict):
        out: list[str] = []
        for values in meta_options.values():
            if isinstance(values, list):
                out.extend(str(item).strip() for item in values if str(item).strip())
        return out
    if isinstance(meta_options, list):
        return [str(item).strip() for item in meta_options if str(item).strip()]
    match = re.search(r"Options:\s*(\{.*?\})(?:\.|\n)", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        try:
            parsed = ast.literal_eval(match.group(1))
            if isinstance(parsed, dict):
                out: list[str] = []
                for values in parsed.values():
                    if isinstance(values, list):
                        out.extend(str(item).strip() for item in values if str(item).strip())
                return out
        except Exception:
            pass
    simple = re.search(r"Options:\s*([^\n]+)", text, flags=re.IGNORECASE)
    if simple:
        return [part.strip(" .") for part in re.split(r"[;,]", simple.group(1)) if part.strip(" .")]
    return []


def _parse_kc_constraints(text: str) -> list[dict[str, Any]]:
    payload: dict[str, list[Any]] = {}
    for key in ("Sources", "Relations", "Targets"):
        match = re.search(rf"{key}:\s*(\[[^\]]*\])", text, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            continue
        try:
            parsed = ast.literal_eval(match.group(1))
            if isinstance(parsed, list):
                payload[key.lower()] = parsed
        except Exception:
            continue
    sources = payload.get("sources", [])
    relations = payload.get("relations", [])
    targets = payload.get("targets", [])
    count = min(len(sources), len(relations), len(targets))
    return [
        {"constraint_id": str(idx + 1), "source": str(sources[idx]), "relation": str(relations[idx]), "target": str(targets[idx])}
        for idx in range(count)
    ]


def _infer_blank_count_from_metadata(metadata: dict | None) -> int:
    if isinstance((metadata or {}).get("answer_all"), list):
        return len((metadata or {}).get("answer_all") or [])
    if isinstance((metadata or {}).get("golds"), dict):
        return len((metadata or {}).get("golds") or {})
    return 0


def parse_kc_problem(question_text: str, metadata: dict | None = None) -> StructuralProblemIR:
    text = str(question_text or "")
    blanks = _parse_blank_ids(text, metadata)
    options = _parse_kc_options(text, metadata)
    constraints = _parse_kc_constraints(text)
    blank_count = len(blanks) or _infer_blank_count_from_metadata(metadata)
    confidence = "high" if blanks and options else "medium" if blanks else "low"
    return StructuralProblemIR(
        dataset_name="knowledge_crosswords",
        object_kind="knowledge_crossword",
        task_kind="blank_assignment",
        parse_confidence=confidence,
        raw_question=text,
        payload={
            "blank_ids": blanks,
            "options": options,
            "constraints": constraints,
            "blank_count": blank_count,
        },
    )


def parse_structural_problem(
    question_text: str,
    *,
    dataset_profile: Any | None = None,
    metadata: dict | None = None,
    dataset_name: str | None = None,
) -> StructuralProblemIR:
    name = (
        dataset_name
        or (metadata or {}).get("dataset_name")
        or (metadata or {}).get("dataset")
        or (metadata or {}).get("mas_dataset_name")
        or getattr(dataset_profile, "name", None)
        or ""
    )
    normalized_name = str(name).strip().lower()
    task_type = str(getattr(dataset_profile, "task_type", "") or (metadata or {}).get("mas_task_type") or "").strip()
    if normalized_name == "knowledge_crosswords" or task_type == "structured_list":
        return parse_kc_problem(question_text, metadata)
    return parse_nlgraph_problem(question_text, metadata)


def _normalize_yes_no(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"yes", "true", "1"}:
        return "yes"
    if text in {"no", "false", "0"}:
        return "no"
    if re.search(r"\byes\b", text):
        return "yes"
    if re.search(r"\bno\b", text):
        return "no"
    return ""


def _list_from_any(value: Any) -> list[str]:
    if isinstance(value, list):
        return [_node_token(item) for item in value]
    if isinstance(value, tuple):
        return [_node_token(item) for item in value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [_node_token(item) for item in parsed]
        except Exception:
            pass
        return [_node_token(item) for item in re.findall(r"[A-Za-z0-9_\-]+", value)]
    return []


def _extract_graph_answer_fallback(raw: str, task_kind: str) -> Any:
    if task_kind in {"connectivity", "cycle"}:
        return _normalize_yes_no(raw)
    values = re.findall(r"[A-Za-z0-9_\-]+", raw)
    if values:
        return values
    return ""


def _normalize_graph_answer(answer: Any, witness: dict[str, Any], problem: StructuralProblemIR) -> str:
    task = problem.task_kind
    if task in {"connectivity", "cycle"}:
        return _normalize_yes_no(answer)
    if task == "topological_sort":
        order = witness.get("order") or answer
        return json.dumps(_list_from_any(order), ensure_ascii=False, separators=(",", ":"))
    if task in {"shortest_path", "hamilton_path"}:
        path = witness.get("path") or answer
        return json.dumps(_list_from_any(path), ensure_ascii=False, separators=(",", ":"))
    if isinstance(answer, (dict, list)):
        return json.dumps(answer, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return str(answer or "").strip()


def parse_nlgraph_artifact(text: str, problem: StructuralProblemIR) -> StructuralArtifact:
    raw = _strip_hidden_reasoning(text)
    obj, source = _extract_json_object(raw)
    if isinstance(obj, dict):
        answer = obj.get("answer")
        if answer is None:
            if problem.task_kind in {"shortest_path", "hamilton_path"}:
                answer = obj.get("path")
            elif problem.task_kind == "topological_sort":
                answer = obj.get("order")
            elif problem.task_kind == "maximum_flow":
                answer = obj.get("max_flow")
        witness = {
            "path": obj.get("path"),
            "order": obj.get("order"),
            "cycle": obj.get("cycle"),
            "components": obj.get("components"),
            "distance": obj.get("distance", obj.get("total_weight")),
            "total_weight": obj.get("total_weight"),
            "witness_type": obj.get("witness_type"),
            "max_flow": obj.get("max_flow", obj.get("answer")),
        }
        contract_ok = True
        confidence = "high"
    else:
        answer = _extract_graph_answer_fallback(raw, problem.task_kind)
        witness = {}
        source = "fallback_text" if answer else "none"
        contract_ok = False
        confidence = "low" if answer else "none"

    normalized = _normalize_graph_answer(answer, witness, problem)
    signature = _hash_json({"answer": normalized, "witness": witness})
    return StructuralArtifact(
        raw_text=raw,
        dataset_name="nlgraph",
        object_kind="graph",
        answer=answer,
        normalized_answer=normalized,
        witness=witness,
        contract_ok=contract_ok,
        final_source=source,
        parser_confidence=confidence,
        artifact_signature=signature,
    )


def _extract_list_fallback(raw: str) -> list[str] | None:
    match = re.search(r"\[[^\]]*\]", raw, flags=re.DOTALL)
    if match:
        try:
            parsed = ast.literal_eval(match.group(0))
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed]
        except Exception:
            pass
    return None


def _normalize_kc_answers(answers: Any) -> str:
    if not isinstance(answers, list):
        return ""
    return json.dumps([str(item).strip() for item in answers], ensure_ascii=False, separators=(",", ":"))


def parse_kc_artifact(text: str, problem: StructuralProblemIR) -> StructuralArtifact:
    raw = _strip_hidden_reasoning(text)
    obj, source = _extract_json_any(raw)
    answers = None
    witness: dict[str, Any] = {}

    if isinstance(obj, list):
        answers = [str(x).strip() for x in obj]
        source = "json_list"
        contract_ok = True
    elif isinstance(obj, dict):
        raw_answers = obj.get("answers") or obj.get("answer") or obj.get("assignment")
        if isinstance(raw_answers, dict):
            blank_ids = problem.payload.get("blank_ids") or sorted(raw_answers.keys(), key=_node_sort_key)
            answers = []
            for index, blank_id in enumerate(blank_ids, start=1):
                value = (
                    raw_answers.get(str(blank_id))
                    or raw_answers.get(blank_id)
                    or raw_answers.get(str(index))
                    or raw_answers.get(index)
                    or ""
                )
                answers.append(str(value).strip())
        elif isinstance(raw_answers, list):
            answers = [str(x).strip() for x in raw_answers]
        witness = obj
        source = "json_object"
        contract_ok = bool(answers)
    else:
        answers = _extract_list_fallback(raw)
        source = "fallback_list" if answers is not None else "none"
        contract_ok = False

    normalized = _normalize_kc_answers(answers)
    signature = _hash_json({"answers": normalized})
    return StructuralArtifact(
        raw_text=raw,
        dataset_name="knowledge_crosswords",
        object_kind="knowledge_crossword",
        answer=answers,
        normalized_answer=normalized,
        witness=witness,
        contract_ok=contract_ok,
        final_source=source,
        parser_confidence="high" if contract_ok else "low",
        artifact_signature=signature,
    )


def parse_structural_artifact(text: str, problem: StructuralProblemIR) -> StructuralArtifact:
    if problem.dataset_name == "knowledge_crosswords":
        return parse_kc_artifact(text, problem)
    return parse_nlgraph_artifact(text, problem)


def _residual(
    *,
    fatal: Sequence[str],
    local: Sequence[str],
    support: Sequence[str],
    repair_locus: str,
    verified_constraints: int,
    residual_kind: str,
    certificate_kind: str,
    certificate_ok: bool,
) -> StructuralResidual:
    fatal_t = tuple(sorted(dict.fromkeys(str(item) for item in fatal if str(item))))
    local_t = tuple(dict.fromkeys(str(item) for item in local if str(item)))
    support_t = tuple(dict.fromkeys(str(item) for item in support if str(item)))
    signature = _hash_json(
        {
            "fatal": fatal_t,
            "local": local_t,
            "support": support_t,
            "repair_locus": repair_locus,
            "certificate_kind": certificate_kind,
        }
    )
    return StructuralResidual(
        fatal=fatal_t,
        local=local_t,
        support=support_t,
        repair_locus=repair_locus,
        verified_constraints=int(verified_constraints),
        residual_kind=residual_kind,
        residual_signature=signature,
        certificate_kind=certificate_kind,
        certificate_ok=bool(certificate_ok),
    )


def _residual_from_lists(
    fatal: Sequence[str],
    local: Sequence[str],
    support: Sequence[str],
    residual_kind: str,
    *,
    verified: int = 0,
    certificate_kind: str | None = None,
    certificate_ok: bool | None = None,
) -> StructuralResidual:
    locus = str((list(fatal) or list(local) or ["stable"])[0])
    kind = certificate_kind if certificate_kind is not None else (residual_kind if support and not fatal else "none")
    ok = (not fatal and bool(support)) if certificate_ok is None else certificate_ok
    return _residual(
        fatal=fatal,
        local=local,
        support=support,
        repair_locus=locus,
        verified_constraints=verified,
        residual_kind=residual_kind,
        certificate_kind=str(kind or "none"),
        certificate_ok=bool(ok),
    )


def _adjacency(problem: StructuralProblemIR) -> dict[str, list[tuple[str, float]]]:
    directed = bool(problem.payload.get("directed", False))
    adj: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for node in problem.payload.get("nodes") or []:
        adj[str(node)]
    for u, v, w in problem.payload.get("edges") or []:
        adj[str(u)].append((str(v), float(w)))
        if not directed:
            adj[str(v)].append((str(u), float(w)))
    for values in adj.values():
        values.sort(key=lambda item: (_node_sort_key(item[0]), item[1]))
    return dict(adj)


def _edge_weight(problem: StructuralProblemIR, u: str, v: str) -> float | None:
    for nxt, weight in _adjacency(problem).get(str(u), []):
        if nxt == str(v):
            return float(weight)
    return None


def _bfs_path(problem: StructuralProblemIR, source: str | None, target: str | None) -> list[str] | None:
    if source is None or target is None:
        return None
    source = str(source)
    target = str(target)
    adj = _adjacency(problem)
    if source not in adj or target not in adj:
        return None
    queue: deque[str] = deque([source])
    prev: dict[str, str | None] = {source: None}
    while queue:
        node = queue.popleft()
        if node == target:
            break
        for nxt, _ in adj.get(node, []):
            if nxt not in prev:
                prev[nxt] = node
                queue.append(nxt)
    if target not in prev:
        return None
    path: list[str] = []
    cur: str | None = target
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    return list(reversed(path))


def _dijkstra_path(problem: StructuralProblemIR, source: str | None, target: str | None) -> tuple[list[str] | None, float | None]:
    if source is None or target is None:
        return None, None
    source = str(source)
    target = str(target)
    adj = _adjacency(problem)
    if source not in adj or target not in adj:
        return None, None
    heap: list[tuple[float, tuple[Any, ...], str]] = [(0.0, _node_sort_key(source), source)]
    dist: dict[str, float] = {source: 0.0}
    prev: dict[str, str | None] = {source: None}
    while heap:
        cost, _, node = heapq.heappop(heap)
        if cost != dist.get(node):
            continue
        if node == target:
            break
        for nxt, weight in adj.get(node, []):
            new_cost = cost + float(weight)
            if new_cost < dist.get(nxt, math.inf) - 1e-12:
                dist[nxt] = new_cost
                prev[nxt] = node
                heapq.heappush(heap, (new_cost, _node_sort_key(nxt), nxt))
    if target not in dist:
        return None, None
    path: list[str] = []
    cur: str | None = target
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    return list(reversed(path)), dist[target]


def _path_distance(problem: StructuralProblemIR, path: Sequence[Any]) -> float | None:
    nodes = [_node_token(item) for item in path]
    if len(nodes) < 2:
        return 0.0 if nodes else None
    total = 0.0
    for u, v in zip(nodes, nodes[1:]):
        weight = _edge_weight(problem, u, v)
        if weight is None:
            return None
        total += weight
    return total


def _check_path(
    problem: StructuralProblemIR,
    path: Sequence[Any],
    source: str | None = None,
    target: str | None = None,
) -> PathCheck:
    nodes = [_node_token(item) for item in path]
    fatal: list[str] = []
    local: list[str] = []
    verified = 0
    graph_nodes = {str(item) for item in problem.payload.get("nodes") or []}
    if not nodes:
        fatal.append("missing_path")
        return PathCheck(tuple(fatal), tuple(local), verified)
    invalid = [node for node in nodes if graph_nodes and node not in graph_nodes]
    if invalid:
        fatal.append("invalid_node")
        local.append(f"invalid_node:{invalid[0]}")
    if source is not None and target is not None and (nodes[0] != str(source) or nodes[-1] != str(target)):
        fatal.append("wrong_endpoint")
    for u, v in zip(nodes, nodes[1:]):
        verified += 1
        if _edge_weight(problem, u, v) is None:
            fatal.append("invalid_edge")
            local.append(f"invalid_edge:{u}->{v}")
            break
    return PathCheck(tuple(dict.fromkeys(fatal)), tuple(dict.fromkeys(local)), verified)


def _connected_components(problem: StructuralProblemIR) -> list[list[str]]:
    adj = _adjacency(
        StructuralProblemIR(
            dataset_name=problem.dataset_name,
            object_kind=problem.object_kind,
            task_kind=problem.task_kind,
            parse_confidence=problem.parse_confidence,
            raw_question=problem.raw_question,
            payload={**problem.payload, "directed": False},
        )
    )
    seen: set[str] = set()
    comps: list[list[str]] = []
    for node in sorted(adj.keys(), key=_node_sort_key):
        if node in seen:
            continue
        queue = deque([node])
        seen.add(node)
        comp: list[str] = []
        while queue:
            cur = queue.popleft()
            comp.append(cur)
            for nxt, _ in adj.get(cur, []):
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        comps.append(sorted(comp, key=_node_sort_key))
    return comps


def _kahn_topological_order(problem: StructuralProblemIR) -> list[str] | None:
    nodes = [str(item) for item in problem.payload.get("nodes") or []]
    incoming = {node: 0 for node in nodes}
    outgoing: dict[str, list[str]] = {node: [] for node in nodes}
    for u, v, _ in problem.payload.get("edges") or []:
        u_s, v_s = str(u), str(v)
        incoming.setdefault(u_s, 0)
        incoming.setdefault(v_s, 0)
        outgoing.setdefault(u_s, [])
        outgoing.setdefault(v_s, [])
        outgoing[u_s].append(v_s)
        incoming[v_s] += 1
    ready = [node for node, count in incoming.items() if count == 0]
    ready.sort(key=_node_sort_key)
    heap = [(_node_sort_key(node), node) for node in ready]
    heapq.heapify(heap)
    order: list[str] = []
    while heap:
        _, node = heapq.heappop(heap)
        order.append(node)
        for nxt in sorted(outgoing.get(node, []), key=_node_sort_key):
            incoming[nxt] -= 1
            if incoming[nxt] == 0:
                heapq.heappush(heap, (_node_sort_key(nxt), nxt))
    if len(order) != len(incoming):
        return None
    return order


def _find_cycle(problem: StructuralProblemIR) -> list[str] | None:
    directed = bool(problem.payload.get("directed", False))
    adj = _adjacency(problem)
    if directed:
        state: dict[str, int] = {}
        stack: list[str] = []

        def dfs(node: str) -> list[str] | None:
            state[node] = 1
            stack.append(node)
            for nxt, _ in adj.get(node, []):
                if state.get(nxt) == 1:
                    idx = stack.index(nxt)
                    return stack[idx:] + [nxt]
                if state.get(nxt, 0) == 0:
                    found = dfs(nxt)
                    if found:
                        return found
            stack.pop()
            state[node] = 2
            return None

        for node in sorted(adj.keys(), key=_node_sort_key):
            if state.get(node, 0) == 0:
                found = dfs(node)
                if found:
                    return found
        return None

    seen: set[str] = set()

    def dfs_undirected(node: str, parent: str | None, path: list[str]) -> list[str] | None:
        seen.add(node)
        path.append(node)
        for nxt, _ in adj.get(node, []):
            if nxt == parent:
                continue
            if nxt in path:
                idx = path.index(nxt)
                return path[idx:] + [nxt]
            if nxt not in seen:
                found = dfs_undirected(nxt, node, path)
                if found:
                    return found
        path.pop()
        return None

    for node in sorted(adj.keys(), key=_node_sort_key):
        if node not in seen:
            found = dfs_undirected(node, None, [])
            if found:
                return found
    return None


def _verify_connectivity(problem: StructuralProblemIR, artifact: StructuralArtifact) -> StructuralResidual:
    source = problem.payload.get("source")
    target = problem.payload.get("target")
    oracle_path = _bfs_path(problem, source, target)
    answer = _normalize_yes_no(artifact.answer)
    fatal: list[str] = []
    local: list[str] = []
    support: list[str] = []
    verified = 0

    if answer not in {"yes", "no"}:
        fatal.append("missing_yes_no_answer")
    if answer == "yes":
        path = artifact.witness.get("path")
        if not path:
            local.append("missing_path_witness")
        else:
            checked = _check_path(problem, _list_from_any(path), source, target)
            fatal.extend(checked.fatal)
            local.extend(checked.local)
            verified += checked.verified
            if not checked.fatal:
                support.append("valid_path_witness")
    if answer == "no":
        if oracle_path is not None:
            fatal.append("false_negative_connectivity")
        else:
            support.append("oracle_disconnected_certificate")
    if answer == "yes" and oracle_path is None:
        fatal.append("false_positive_connectivity")
    return _residual_from_lists(fatal, local, support, "connectivity", verified=verified)


def _verify_shortest_path(problem: StructuralProblemIR, artifact: StructuralArtifact) -> StructuralResidual:
    source = problem.payload.get("source")
    target = problem.payload.get("target")
    oracle_path, oracle_dist = _dijkstra_path(problem, source, target)
    path = artifact.witness.get("path") or artifact.answer
    fatal: list[str] = []
    local: list[str] = []
    support: list[str] = []
    verified = 0

    path_list = _list_from_any(path)
    if not path_list:
        fatal.append("missing_path")
        return _residual_from_lists(fatal, local, support, "shortest_path")
    checked = _check_path(problem, path_list, source, target)
    fatal.extend(checked.fatal)
    local.extend(checked.local)
    verified += checked.verified
    if not checked.fatal:
        dist = _path_distance(problem, path_list)
        if oracle_path is None:
            fatal.append("path_exists_but_oracle_disconnected")
        elif dist is None or oracle_dist is None or abs(float(dist) - float(oracle_dist)) > 1e-9:
            fatal.append("path_not_shortest")
            if dist is not None:
                local.append(f"candidate_distance:{dist:g}")
            if oracle_dist is not None:
                local.append(f"oracle_distance:{oracle_dist:g}")
        else:
            support.append("shortest_path_certificate")
            verified += 1
    return _residual_from_lists(fatal, local, support, "shortest_path", verified=verified)


def _verify_topological_sort(problem: StructuralProblemIR, artifact: StructuralArtifact) -> StructuralResidual:
    nodes = [str(item) for item in problem.payload.get("nodes") or []]
    order = _list_from_any(artifact.witness.get("order") or artifact.answer)
    fatal: list[str] = []
    local: list[str] = []
    support: list[str] = []

    if not order:
        fatal.append("missing_order")
        return _residual_from_lists(fatal, local, support, "topological_sort")
    if set(order) != set(nodes):
        missing = sorted(set(nodes) - set(order), key=_node_sort_key)
        extra = sorted(set(order) - set(nodes), key=_node_sort_key)
        if missing:
            fatal.append("order_missing_node")
        if extra:
            fatal.append("order_invalid_node")
    if len(order) != len(set(order)):
        fatal.append("order_duplicate_node")
    pos = {node: i for i, node in enumerate(order)}
    for u, v, _ in problem.payload.get("edges") or []:
        if str(u) in pos and str(v) in pos and pos[str(u)] > pos[str(v)]:
            fatal.append("order_violation")
            local.append(f"violating_edge:{u}->{v}")
            break
    if not fatal:
        support.append("topological_order_certificate")
    return _residual_from_lists(
        fatal,
        local,
        support,
        "topological_sort",
        verified=len(problem.payload.get("edges") or []),
        certificate_kind="topological_order" if not fatal else "none",
    )


def _verify_cycle(problem: StructuralProblemIR, artifact: StructuralArtifact) -> StructuralResidual:
    answer = _normalize_yes_no(artifact.answer)
    oracle_cycle = _find_cycle(problem)
    fatal: list[str] = []
    local: list[str] = []
    support: list[str] = []
    verified = 0
    if answer not in {"yes", "no"}:
        fatal.append("missing_yes_no_answer")
    if answer == "yes":
        cycle = _list_from_any(artifact.witness.get("cycle") or [])
        if not cycle:
            local.append("missing_cycle_witness")
        elif len(cycle) < 4 or cycle[0] != cycle[-1]:
            fatal.append("cycle_invalid")
        else:
            checked = _check_path(problem, cycle, cycle[0], cycle[-1])
            fatal.extend("cycle_invalid" if item in {"invalid_edge", "wrong_endpoint"} else item for item in checked.fatal)
            local.extend(checked.local)
            verified += checked.verified
            if not checked.fatal:
                support.append("cycle_certificate")
    if answer == "no" and oracle_cycle is not None:
        fatal.append("cycle_invalid")
        local.append("oracle_cycle_exists")
    if answer == "yes" and oracle_cycle is None:
        fatal.append("cycle_invalid")
        local.append("oracle_acyclic")
    if answer == "no" and oracle_cycle is None:
        support.append("acyclic_certificate")
    return _residual_from_lists(fatal, local, support, "cycle", verified=verified)


def _verify_flow_certificate(problem: StructuralProblemIR, artifact: StructuralArtifact) -> StructuralResidual:
    del problem
    value = artifact.witness.get("max_flow", artifact.answer)
    fatal: list[str] = []
    support: list[str] = []
    if not isinstance(value, (int, float)):
        try:
            float(str(value))
        except Exception:
            fatal.append("missing_answer")
    if not fatal:
        support.append("flow_schema_certificate")
    return _residual_from_lists(fatal, (), support, "maximum_flow", verified=int(not fatal), certificate_kind="flow_schema")


def _verify_hamilton_witness(problem: StructuralProblemIR, artifact: StructuralArtifact) -> StructuralResidual:
    path = _list_from_any(artifact.witness.get("path") or artifact.answer)
    fatal: list[str] = []
    local: list[str] = []
    support: list[str] = []
    checked = _check_path(problem, path)
    fatal.extend(checked.fatal)
    local.extend(checked.local)
    nodes = [str(item) for item in problem.payload.get("nodes") or []]
    if set(path) != set(nodes):
        fatal.append("order_missing_node")
    if len(path) != len(set(path)):
        fatal.append("order_duplicate_node")
    if not fatal:
        support.append("hamilton_path_certificate")
    return _residual_from_lists(fatal, local, support, "hamilton_path", verified=checked.verified)


def verify_nlgraph_artifact(problem: StructuralProblemIR, artifact: StructuralArtifact) -> StructuralResidual:
    task = problem.task_kind
    if problem.parse_confidence == "low":
        return _residual(
            fatal=("graph_parse_failed",),
            local=(),
            support=(),
            repair_locus="problem_parse",
            verified_constraints=0,
            residual_kind="graph_parse_failed",
            certificate_kind="none",
            certificate_ok=False,
        )
    if task == "connectivity":
        return _verify_connectivity(problem, artifact)
    if task == "shortest_path":
        return _verify_shortest_path(problem, artifact)
    if task == "topological_sort":
        return _verify_topological_sort(problem, artifact)
    if task == "cycle":
        return _verify_cycle(problem, artifact)
    if task == "maximum_flow":
        return _verify_flow_certificate(problem, artifact)
    if task == "hamilton_path":
        return _verify_hamilton_witness(problem, artifact)
    return _residual(
        fatal=(),
        local=("unknown_graph_task",),
        support=(),
        repair_locus="task_kind",
        verified_constraints=0,
        residual_kind="unknown_graph_task",
        certificate_kind="none",
        certificate_ok=False,
    )


def _norm_entity(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).lower()


def verify_kc_artifact(problem: StructuralProblemIR, artifact: StructuralArtifact) -> StructuralResidual:
    fatal: list[str] = []
    local: list[str] = []
    support: list[str] = []
    answers = artifact.answer
    expected_len = int(problem.payload.get("blank_count") or len(problem.payload.get("blank_ids") or []))
    if not isinstance(answers, list):
        fatal.append("invalid_json_list")
        return _residual_from_lists(fatal, local, support, "kc_schema", certificate_kind="none")
    if expected_len and len(answers) != expected_len:
        fatal.append("list_length_mismatch")
        local.append(f"expected_len:{expected_len}")
        local.append(f"actual_len:{len(answers)}")
    if any(not str(item).strip() for item in answers):
        fatal.append("empty_blank_assignment")
    options = problem.payload.get("options") or []
    if options:
        option_set = {_norm_entity(item) for item in options}
        for answer in answers:
            if _norm_entity(answer) not in option_set:
                fatal.append("option_not_allowed")
                local.append(f"bad_option:{answer}")
                break
    if not fatal:
        support.append("schema_and_option_certificate")
    local.append("kc_verify_all_needed")
    return _residual_from_lists(
        fatal,
        local,
        support,
        "kc_assignment",
        verified=len(support),
        certificate_kind="schema",
        certificate_ok=not fatal,
    )


def make_structural_eval(problem: StructuralProblemIR, artifact: StructuralArtifact) -> StructuralEval:
    residual = verify_kc_artifact(problem, artifact) if problem.dataset_name == "knowledge_crosswords" else verify_nlgraph_artifact(problem, artifact)
    return StructuralEval(problem=problem, artifact=artifact, residual=residual)


def structural_class_key(eval_obj: StructuralEval) -> tuple[Any, ...]:
    return (
        eval_obj.problem.dataset_name,
        eval_obj.problem.task_kind,
        eval_obj.artifact.normalized_answer,
    )


def structural_rank_key(eval_obj: StructuralEval) -> tuple[Any, ...]:
    return (
        -len(eval_obj.residual.fatal),
        int(bool(eval_obj.residual.certificate_ok)),
        int(bool(eval_obj.artifact.contract_ok)),
        -len(eval_obj.residual.local),
        int(bool(eval_obj.artifact.normalized_answer)),
        int(eval_obj.residual.verified_constraints),
        eval_obj.artifact.artifact_signature,
    )


def _has_structural_deterministic_fatal(eval_obj: StructuralEval) -> bool:
    return bool(set(eval_obj.residual.fatal) & _STRUCTURAL_DETERMINISTIC_FATALS)


def structural_dominates(new: StructuralEval, old: StructuralEval) -> bool:
    same_answer = bool(new.artifact.normalized_answer) and new.artifact.normalized_answer == old.artifact.normalized_answer
    if same_answer:
        if len(new.residual.fatal) < len(old.residual.fatal):
            return True
        if (
            len(new.residual.fatal) == len(old.residual.fatal)
            and new.residual.certificate_ok
            and not old.residual.certificate_ok
        ):
            return True
        return False
    if new.problem.dataset_name == "nlgraph":
        return (
            new.residual.certificate_kind in {"oracle_graph", "shortest_path", "topological_order", "connectivity", "cycle"}
            and new.residual.certificate_ok
            and (
                _has_structural_deterministic_fatal(old)
                or old.artifact.final_source in {"answer_only", "fallback_text", "none"}
            )
        )
    if new.problem.dataset_name == "knowledge_crosswords":
        return False
    return False


def build_nlgraph_oracle_candidate(problem: StructuralProblemIR) -> StructuralArtifact | None:
    if problem.object_kind != "graph" or problem.parse_confidence != "high":
        return None
    task = problem.task_kind
    if task == "connectivity":
        source, target = problem.payload.get("source"), problem.payload.get("target")
        path = _bfs_path(problem, source, target)
        if path:
            obj = {"answer": "yes", "witness_type": "path", "path": path}
        else:
            comps = _connected_components(problem)
            obj = {"answer": "no", "witness_type": "component", "components": comps}
    elif task == "shortest_path":
        source, target = problem.payload.get("source"), problem.payload.get("target")
        path, dist = _dijkstra_path(problem, source, target)
        if path is None:
            obj = {"answer": [], "witness_type": "path", "path": [], "distance": None, "total_weight": None}
        else:
            obj = {"answer": path, "witness_type": "path", "path": path, "distance": dist, "total_weight": dist}
    elif task == "topological_sort":
        order = _kahn_topological_order(problem)
        if order is None:
            obj = {"answer": "no", "witness_type": "cycle"}
        else:
            obj = {"answer": order, "witness_type": "order", "order": order}
    elif task == "cycle":
        cycle = _find_cycle(problem)
        if cycle:
            obj = {"answer": "yes", "witness_type": "cycle", "cycle": cycle}
        else:
            obj = {"answer": "no", "witness_type": "acyclic", "cycle": []}
    else:
        return None
    raw = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    return parse_nlgraph_artifact(raw, problem)


def build_kc_verify_all_probe_prompt(problem: StructuralProblemIR, artifact: StructuralArtifact) -> str:
    assignment = artifact.answer or []
    return (
        "You are verifying a Knowledge Crosswords assignment.\n"
        "Do not assume the current answers are correct.\n"
        "Check every factual constraint against the full assignment.\n"
        "If any constraint fails, propose a corrected full assignment.\n\n"
        f"Problem:\n{problem.raw_question}\n\n"
        f"Current assignment JSON list:\n{json.dumps(assignment, ensure_ascii=False)}\n\n"
        "Return exactly one JSON object:\n"
        "{\n"
        '  "status": "pass" | "fail",\n'
        '  "failed_constraints": [\n'
        '    {"constraint_id": "...", "blank_ids": ["1"], "reason": "..."}\n'
        "  ],\n"
        '  "answers": ["...", "..."],\n'
        '  "confidence": "high" | "medium" | "low"\n'
        "}\n"
    )


def parse_kc_verify_all_result(text: str) -> KCVerifyAllResult:
    obj, _ = _extract_json_object(text)
    if not isinstance(obj, dict):
        return KCVerifyAllResult("fail", (), (), "low", str(text or ""))
    return KCVerifyAllResult(
        status=str(obj.get("status", "fail")).lower(),
        failed_constraints=tuple(item for item in (obj.get("failed_constraints") or ()) if isinstance(item, dict)),
        answers=tuple(str(x).strip() for x in (obj.get("answers") or [])),
        confidence=str(obj.get("confidence", "low")).lower(),
        raw=str(text or ""),
    )


def build_kc_conflict_component_repair_prompt(
    problem: StructuralProblemIR,
    artifact: StructuralArtifact,
    verify_result: KCVerifyAllResult,
) -> str:
    failed_blank_ids = sorted(
        {
            str(blank_id)
            for item in verify_result.failed_constraints
            for blank_id in item.get("blank_ids", [])
        },
        key=_node_sort_key,
    )
    current = list(artifact.answer or [])
    blank_ids = problem.payload.get("blank_ids") or [str(i + 1) for i in range(len(current))]
    failed_set = set(failed_blank_ids)
    frozen = {
        str(blank_id): current[i]
        for i, blank_id in enumerate(blank_ids)
        if str(blank_id) not in failed_set and i < len(current)
    }
    return (
        "You are repairing a Knowledge Crosswords assignment.\n"
        "Only modify blanks involved in failed constraints.\n"
        "Keep all frozen blanks unchanged.\n\n"
        f"Problem:\n{problem.raw_question}\n\n"
        f"Current answers:\n{json.dumps(current, ensure_ascii=False)}\n\n"
        f"Frozen blanks:\n{json.dumps(frozen, ensure_ascii=False)}\n\n"
        f"Failed constraints:\n{json.dumps(list(verify_result.failed_constraints), ensure_ascii=False)}\n\n"
        "Return exactly one JSON object:\n"
        "{\n"
        '  "answers": ["...", "..."],\n'
        '  "changed_blank_ids": ["..."],\n'
        '  "rationale": "short reason"\n'
        "}\n"
    )


def structural_final_answer(eval_obj: StructuralEval) -> str:
    if eval_obj.problem.dataset_name == "nlgraph":
        task = eval_obj.problem.task_kind
        artifact = eval_obj.artifact
        if task in {"connectivity", "cycle"}:
            return json.dumps({"answer": _normalize_yes_no(artifact.answer)}, ensure_ascii=False, separators=(",", ":"))
        if task == "shortest_path":
            path = _list_from_any(artifact.witness.get("path") or artifact.answer)
            distance = artifact.witness.get("total_weight", artifact.witness.get("distance"))
            if distance is None:
                distance = _path_distance(eval_obj.problem, path)
            payload: dict[str, Any] = {"path": [_node_public(node) for node in path]}
            if distance is not None:
                payload["total_weight"] = int(distance) if float(distance).is_integer() else distance
            return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        if task == "topological_sort":
            order = _list_from_any(artifact.witness.get("order") or artifact.answer)
            return json.dumps({"order": [_node_public(node) for node in order]}, ensure_ascii=False, separators=(",", ":"))
        if task == "hamilton_path":
            path = _list_from_any(artifact.witness.get("path") or artifact.answer)
            return json.dumps({"path": [_node_public(node) for node in path]}, ensure_ascii=False, separators=(",", ":"))
        if task == "maximum_flow":
            value = artifact.witness.get("max_flow", artifact.answer)
            return json.dumps({"max_flow": value}, ensure_ascii=False, separators=(",", ":"))
        return artifact.raw_text
    if eval_obj.problem.dataset_name == "knowledge_crosswords":
        return json.dumps(eval_obj.artifact.answer or [], ensure_ascii=False, separators=(",", ":"))
    return eval_obj.artifact.raw_text
