from __future__ import annotations

import ast
import hashlib
import itertools
import json
import re
from dataclasses import dataclass
from typing import Any, Optional, Sequence


@dataclass(frozen=True)
class SlotSpec:
    slot_id: str
    slot_name: str
    options: tuple[str, ...]
    option_labels: tuple[str, ...] = ()
    context: str = ""


@dataclass(frozen=True)
class SlotProblemIR:
    raw_question: str
    slots: tuple[SlotSpec, ...]
    output_kind: str
    parse_confidence: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class SlotArtifact:
    raw_text: str
    assignment: dict[str, str]
    normalized_assignment: str
    answer_source: str
    contract_ok: bool
    parser_confidence: str
    artifact_signature: str


@dataclass(frozen=True)
class SlotResidual:
    fatal: tuple[str, ...]
    local: tuple[str, ...]
    support: tuple[str, ...]
    invalid_slots: tuple[str, ...]
    unstable_slots: tuple[str, ...]
    residual_kind: str
    residual_signature: str


@dataclass(frozen=True)
class SlotEval:
    problem: SlotProblemIR
    artifact: SlotArtifact
    residual: SlotResidual


@dataclass(frozen=True)
class SlotChallenger:
    slot_id: str
    anchor_value: str
    challenger_value: str
    occurrence_count: int
    sink_support: int
    source_count: int
    best_entry_digest: str
    best_entry: dict[str, Any]


@dataclass(frozen=True)
class PairwiseSlotProbeResult:
    winner: str
    anchor_conflict: tuple[str, ...]
    challenger_conflict: tuple[str, ...]
    anchor_support: tuple[str, ...]
    challenger_support: tuple[str, ...]
    discriminator: str
    confidence: str
    raw: str
    question_polarity: str
    target_condition: str
    inverse_condition: str
    anchor_satisfies_target: str
    challenger_satisfies_target: str
    anchor_satisfies_inverse: str
    challenger_satisfies_inverse: str


@dataclass(frozen=True)
class EvidenceAtom:
    slot_id: str
    value: str
    relation: str
    statement: str
    polarity: str
    source: str
    source_id: str = ""
    confidence: float = 1.0


@dataclass(frozen=True)
class SlotCertificate:
    slot_id: str
    anchor_value: str
    challenger_value: str
    target_condition: str
    discriminator: str
    challenger_support: tuple[EvidenceAtom, ...]
    anchor_conflict: tuple[EvidenceAtom, ...]
    challenger_conflict: tuple[EvidenceAtom, ...]
    anchor_target_votes: tuple[str, ...]
    challenger_target_votes: tuple[str, ...]
    score_margin: float
    source_count: int
    certificate_kind: str
    changed_slots: tuple[str, ...] = ()
    anchor_assignment: tuple[tuple[str, str], ...] = ()
    challenger_assignment: tuple[tuple[str, str], ...] = ()
    target_condition_consistency: str = "unknown"
    raw: str = ""


@dataclass(frozen=True)
class KCConstraint:
    subject: str
    predicate: str
    object: str
    weight: float
    source_id: str
    required: bool = True


@dataclass(frozen=True)
class KCAssignmentScore:
    assignment: tuple[tuple[str, str], ...]
    satisfied_edges: tuple[KCConstraint, ...]
    violated_edges: tuple[KCConstraint, ...]
    unsupported_edges: tuple[KCConstraint, ...]
    duplicate_conflicts: tuple[tuple[str, str, str], ...]
    score: float


_MCQ_OPTION_RE = re.compile(r"(?m)^\s*([A-Z]|\d{1,3})[\.\):：]\s*(.+?)\s*$")
_NEGATIVE_CUE_RE = re.compile(
    r"\b(not|except|least|false|incorrect|cannot|can't|never|none of|not\s+an?|not\s+the)\b",
    flags=re.IGNORECASE,
)


def _strip_hidden_reasoning(text: str) -> str:
    cleaned = re.sub(r"(?is)<think>.*?</think>", "", str(text or ""))
    cleaned = re.sub(r"(?im)^\s*</?think>\s*$", "", cleaned)
    return cleaned.strip()


def _jsonable(value: Any) -> Any:
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


def _norm_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).casefold()


def _norm_label(value: Any) -> str:
    return str(value or "").strip().upper().rstrip(".")


def canonicalize_entity(value: Any) -> str:
    text = str(value or "").strip()
    text = text.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"_+", "_", text)
    text = text.strip("_")
    return text


def _assignment_tuple(assignment: dict[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((str(key), str(value)) for key, value in assignment.items()))


def _assignment_dict(items: Sequence[tuple[str, str]]) -> dict[str, str]:
    return {str(key): str(value) for key, value in items}


def _detect_question_polarity(text: str) -> str:
    lower = str(text or "").lower()
    if _NEGATIVE_CUE_RE.search(lower):
        return "negative_or_exception"
    if re.search(r"\b(least|minimum|smallest|lowest)\b", lower):
        return "comparative_low"
    if re.search(r"\b(most|maximum|largest|highest)\b", lower):
        return "comparative_high"
    return "positive_or_unknown"


def _normalize_options(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _extract_json_span(text: str) -> str | None:
    raw = _strip_hidden_reasoning(text)
    start = -1
    open_ch = ""
    close_ch = ""
    for index, ch in enumerate(raw):
        if ch in "{[":
            start = index
            open_ch = ch
            close_ch = "}" if ch == "{" else "]"
            break
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(raw)):
        ch = raw[index]
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
                return raw[start : index + 1]
    return None


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


def _option_labels_for_length(size: int) -> tuple[str, ...]:
    if size <= 0:
        return ()
    return tuple(str(index) for index in range(1, size + 1))


def _slot_context(text: str, slot_id: str) -> str:
    pattern = re.compile(rf"([^\n]{{0,120}}{re.escape(str(slot_id))}[^\n]{{0,120}})", flags=re.IGNORECASE)
    match = pattern.search(text)
    return match.group(1).strip() if match else ""


def _parse_slots_from_metadata(meta: dict[str, Any]) -> list[SlotSpec]:
    for key in ("options_by_blank", "options_by_slot", "candidates_by_blank", "candidate_options_by_slot"):
        value = meta.get(key)
        if isinstance(value, dict):
            slots: list[SlotSpec] = []
            for slot_id, options in value.items():
                normalized = _normalize_options(options)
                if normalized:
                    slots.append(
                        SlotSpec(
                            slot_id=str(slot_id),
                            slot_name=str(slot_id),
                            options=tuple(normalized),
                        )
                    )
            if slots:
                return slots

    direct_options = meta.get("options")
    blanks = meta.get("blanks")
    if isinstance(direct_options, dict):
        ordered_blank_ids: list[str] = []
        if isinstance(blanks, list):
            ordered_blank_ids = [str(item) for item in blanks if str(item).strip()]
        if not ordered_blank_ids:
            ordered_blank_ids = [str(key) for key in direct_options.keys()]
        slots = []
        for index, blank_id in enumerate(ordered_blank_ids, start=1):
            normalized = _normalize_options(direct_options.get(blank_id))
            if normalized:
                slots.append(
                    SlotSpec(
                        slot_id=str(blank_id),
                        slot_name=f"blank_{index}",
                        options=tuple(normalized),
                    )
                )
        if slots:
            return slots

    for key in ("options", "choices", "answer_choices"):
        normalized = _normalize_options(meta.get(key))
        if normalized:
            labels = tuple(str(item).strip() for item in meta.get("option_labels", ()) if str(item).strip())
            if not labels:
                labels = _option_labels_for_length(len(normalized))
            return [
                SlotSpec(
                    slot_id="answer",
                    slot_name="answer",
                    options=tuple(normalized),
                    option_labels=tuple(labels),
                )
            ]

    return []


def _parse_mcq_slots_from_question(text: str) -> list[SlotSpec]:
    matches = _MCQ_OPTION_RE.findall(text)
    if len(matches) < 2:
        return []
    labels: list[str] = []
    options: list[str] = []
    for label, option_text in matches:
        labels.append(str(label).strip())
        options.append(str(option_text).strip())
    return [
        SlotSpec(
            slot_id="answer",
            slot_name="answer",
            options=tuple(options),
            option_labels=tuple(labels),
            context=text,
        )
    ]


def _parse_blank_ids_generic(text: str, meta: dict[str, Any]) -> list[str]:
    meta_blanks = meta.get("blanks")
    if isinstance(meta_blanks, list):
        blanks = [str(item).strip() for item in meta_blanks if str(item).strip()]
        if blanks:
            return blanks
    match = re.search(r"Blanks:\s*(\[[^\]]*\])", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        try:
            parsed = ast.literal_eval(match.group(1))
            if isinstance(parsed, list):
                blanks = [str(item).strip() for item in parsed if str(item).strip()]
                if blanks:
                    return blanks
        except Exception:
            pass
    found = re.findall(r"\bblank(?:\s+|_)(\d+)\b", text, flags=re.IGNORECASE)
    seen: list[str] = []
    for token in found:
        blank_id = f"blank {token}"
        if blank_id not in seen:
            seen.append(blank_id)
    return seen


def _parse_options_by_blank_generic(text: str, meta: dict[str, Any]) -> dict[str, list[str]]:
    for key in ("options_by_blank", "options_by_slot", "candidates_by_blank", "candidate_options_by_slot"):
        value = meta.get(key)
        if isinstance(value, dict):
            parsed = {
                str(slot_id): _normalize_options(options)
                for slot_id, options in value.items()
                if _normalize_options(options)
            }
            if parsed:
                return parsed

    direct_options = meta.get("options")
    if isinstance(direct_options, dict):
        parsed = {
            str(slot_id): _normalize_options(options)
            for slot_id, options in direct_options.items()
            if _normalize_options(options)
        }
        if parsed:
            return parsed

    match = re.search(r"Options:\s*(\{.*?\})(?:[.\n]|$)", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        try:
            parsed = ast.literal_eval(match.group(1))
            if isinstance(parsed, dict):
                return {
                    str(slot_id): _normalize_options(options)
                    for slot_id, options in parsed.items()
                    if _normalize_options(options)
                }
        except Exception:
            pass
    return {}


def _parse_global_options_generic(text: str, meta: dict[str, Any]) -> list[str]:
    for key in ("options", "choices", "answer_choices"):
        normalized = _normalize_options(meta.get(key))
        if normalized and not isinstance(meta.get(key), dict):
            return normalized
    simple = re.search(r"Options:\s*([^\n]+)", text, flags=re.IGNORECASE)
    if simple:
        return [part.strip(" .") for part in re.split(r"[;,]", simple.group(1)) if part.strip(" .")]
    return []


def _parse_blank_slots_from_question(text: str, meta: dict[str, Any]) -> list[SlotSpec]:
    blank_ids = _parse_blank_ids_generic(text, meta)
    if not blank_ids:
        return []
    options_by_blank = _parse_options_by_blank_generic(text, meta)
    global_options = _parse_global_options_generic(text, meta)
    slots: list[SlotSpec] = []
    for index, blank_id in enumerate(blank_ids, start=1):
        options = options_by_blank.get(blank_id) or options_by_blank.get(blank_id.replace(" ", "_")) or global_options
        normalized = _normalize_options(options)
        if not normalized:
            continue
        slots.append(
            SlotSpec(
                slot_id=str(blank_id),
                slot_name=f"blank_{index}",
                options=tuple(normalized),
                context=_slot_context(text, blank_id),
            )
        )
    return slots


def _infer_slot_output_kind(slots: Sequence[SlotSpec], meta: dict[str, Any], text: str) -> str:
    del meta
    del text
    if len(slots) == 1 and slots[0].option_labels:
        return "single_label"
    if len(slots) == 1:
        return "single_value"
    return "json_list"


def parse_slot_problem(
    question_text: str,
    *,
    dataset_profile: Any | None = None,
    metadata: dict | None = None,
) -> SlotProblemIR:
    text = str(question_text or "")
    meta = dict(metadata or {})
    slots = _parse_slots_from_metadata(meta)
    if not slots:
        slots = _parse_mcq_slots_from_question(text)
    if not slots:
        slots = _parse_blank_slots_from_question(text, meta)
    output_kind = _infer_slot_output_kind(slots, meta, text)
    confidence = "high" if slots and all(slot.options for slot in slots) else "low"
    if not slots and getattr(dataset_profile, "answer_format", "") == "option":
        confidence = "low"
    return SlotProblemIR(
        raw_question=text,
        slots=tuple(slots),
        output_kind=output_kind,
        parse_confidence=confidence,
        metadata=meta,
    )


def _normalize_assignment(assignment: dict[str, str], problem: SlotProblemIR) -> str:
    ordered = [[slot.slot_id, str(assignment.get(slot.slot_id, "")).strip()] for slot in problem.slots]
    return json.dumps(ordered, ensure_ascii=False, separators=(",", ":"))


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


def _label_to_option_map(slot: SlotSpec) -> dict[str, str]:
    labels = slot.option_labels or _option_labels_for_length(len(slot.options))
    mapping: dict[str, str] = {}
    for label, option in zip(labels, slot.options):
        mapping[_norm_label(label)] = option
    return mapping


def _option_value_map(slot: SlotSpec) -> dict[str, str]:
    return {_norm_text(option): option for option in slot.options}


def _option_label_for_value(slot: SlotSpec, value: Any) -> str:
    target = _norm_text(value)
    labels = slot.option_labels or _option_labels_for_length(len(slot.options))
    for label, option in zip(labels, slot.options):
        if _norm_text(option) == target:
            return str(label)
    return ""


def _normalize_single_slot_value(value: Any, slot: SlotSpec) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    label_map = _label_to_option_map(slot)
    option_map = _option_value_map(slot)

    token_candidates: list[str] = [text]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    token_candidates.extend(lines)
    patterns = [
        r"OPTION\s*[:：-]?\s*([A-Z]|\d+)",
        r"(?:FINAL|ANSWER)\s*[:：-]?\s*([A-Z]|\d+)",
    ]
    for candidate in token_candidates:
        for pattern in patterns:
            match = re.search(pattern, candidate, flags=re.IGNORECASE)
            if match:
                normalized = label_map.get(_norm_label(match.group(1)))
                if normalized:
                    return normalized
        direct = label_map.get(_norm_label(candidate))
        if direct:
            return direct

    for candidate in token_candidates:
        match = re.search(r"(?:FINAL|ANSWER)\s*[:：-]?\s*(.+)$", candidate, flags=re.IGNORECASE)
        if match:
            candidate = match.group(1).strip()
        normalized = option_map.get(_norm_text(candidate))
        if normalized:
            return normalized

    normalized = option_map.get(_norm_text(text))
    if normalized:
        return normalized
    return ""


def _parse_json_list_assignment(raw: str, problem: SlotProblemIR) -> tuple[dict[str, str], str, bool]:
    obj, source = _extract_json_any(raw)
    values: list[Any] | None = None
    ok = False
    if isinstance(obj, list):
        values = obj
        source = "explicit_json_list"
        ok = True
    elif isinstance(obj, dict):
        raw_values = obj.get("answers") or obj.get("assignment") or obj.get("answer")
        if isinstance(raw_values, list):
            values = raw_values
            source = "explicit_json_object"
            ok = True
        elif isinstance(raw_values, dict):
            values = [raw_values.get(slot.slot_id, raw_values.get(slot.slot_name, "")) for slot in problem.slots]
            source = "explicit_json_object"
            ok = True
    else:
        values = _extract_list_fallback(raw)
        if values is not None:
            source = "fallback_list"

    if not values:
        return {}, source, False
    assignment: dict[str, str] = {}
    for slot, value in zip(problem.slots, values):
        assignment[slot.slot_id] = str(value).strip()
    return assignment, source, bool(ok and len(values) == len(problem.slots))


def _parse_single_slot_assignment(raw: str, problem: SlotProblemIR) -> tuple[dict[str, str], str, bool]:
    slot = problem.slots[0]
    obj, source = _extract_json_any(raw)
    if isinstance(obj, dict):
        value = obj.get("answer") or obj.get("selected") or obj.get("option")
        parsed = _normalize_single_slot_value(value, slot)
        if parsed:
            return {slot.slot_id: parsed}, "explicit_json", True

    for pattern, source_name in (
        (r"(?im)^\s*OPTION\s*[:：-]?\s*([A-Z]|\d+)\s*$", "option_line"),
        (r"(?im)^\s*(?:FINAL|ANSWER)\s*[:：-]?\s*(.+?)\s*$", "answer_line"),
    ):
        match = re.search(pattern, raw)
        if not match:
            continue
        parsed = _normalize_single_slot_value(match.group(1).strip(), slot)
        if parsed:
            return {slot.slot_id: parsed}, source_name, True

    parsed = _normalize_single_slot_value(raw.strip(), slot)
    if parsed:
        return {slot.slot_id: parsed}, "raw_value", False
    return {}, "none", False


def parse_slot_artifact(text: str, problem: SlotProblemIR) -> SlotArtifact:
    raw = _strip_hidden_reasoning(text)
    if problem.output_kind == "json_list":
        assignment, source, ok = _parse_json_list_assignment(raw, problem)
    elif problem.output_kind in {"single_label", "single_value"} and problem.slots:
        assignment, source, ok = _parse_single_slot_assignment(raw, problem)
    else:
        assignment, source, ok = {}, "none", False
    normalized = _normalize_assignment(assignment, problem) if assignment else ""
    return SlotArtifact(
        raw_text=raw,
        assignment=assignment,
        normalized_assignment=normalized,
        answer_source=source,
        contract_ok=ok,
        parser_confidence="high" if ok else ("low" if assignment else "none"),
        artifact_signature=_hash_json({"assignment": normalized}),
    )


def _value_in_slot_options(value: str, slot: SlotSpec) -> bool:
    return _norm_text(value) in _option_value_map(slot)


def _slot_by_id(problem: SlotProblemIR, slot_id: str) -> SlotSpec:
    for slot in problem.slots:
        if slot.slot_id == slot_id:
            return slot
    raise KeyError(slot_id)


def verify_slot_artifact(problem: SlotProblemIR, artifact: SlotArtifact) -> SlotResidual:
    fatal: list[str] = []
    local: list[str] = []
    support: list[str] = []
    invalid_slots: list[str] = []
    unstable_slots: list[str] = []

    if not artifact.assignment:
        fatal.append("missing_assignment")

    for slot in problem.slots:
        value = str(artifact.assignment.get(slot.slot_id, "")).strip()
        if not value:
            fatal.append("missing_slot_value")
            invalid_slots.append(slot.slot_id)
            continue
        if not _value_in_slot_options(value, slot):
            fatal.append("option_not_allowed")
            local.append(f"invalid_slot@{slot.slot_id}:value={value}")
            invalid_slots.append(slot.slot_id)
            continue
        support.append(f"membership_ok@{slot.slot_id}")

    if not fatal:
        local.append("semantic_calibration_needed")
        unstable_slots.extend(slot.slot_id for slot in problem.slots)

    residual_kind = fatal[0] if fatal else "membership_clean"
    signature = _hash_json(
        {
            "fatal": fatal,
            "local": local,
            "support": support,
            "invalid_slots": invalid_slots,
            "unstable_slots": unstable_slots,
            "residual_kind": residual_kind,
        }
    )
    return SlotResidual(
        fatal=tuple(dict.fromkeys(str(item) for item in fatal if str(item))),
        local=tuple(dict.fromkeys(str(item) for item in local if str(item))),
        support=tuple(dict.fromkeys(str(item) for item in support if str(item))),
        invalid_slots=tuple(dict.fromkeys(str(item) for item in invalid_slots if str(item))),
        unstable_slots=tuple(dict.fromkeys(str(item) for item in unstable_slots if str(item))),
        residual_kind=residual_kind,
        residual_signature=signature,
    )


def make_slot_eval(problem: SlotProblemIR, artifact: SlotArtifact) -> SlotEval:
    return SlotEval(
        problem=problem,
        artifact=artifact,
        residual=verify_slot_artifact(problem, artifact),
    )


def slot_update_dominates(new: SlotEval, old: SlotEval) -> bool:
    if new.artifact.normalized_assignment != old.artifact.normalized_assignment:
        return False
    mine = (
        -len(new.residual.fatal),
        -len(new.residual.local),
        int(bool(new.artifact.contract_ok)),
    )
    theirs = (
        -len(old.residual.fatal),
        -len(old.residual.local),
        int(bool(old.artifact.contract_ok)),
    )
    return mine > theirs


def _candidate_support_key(entry: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(entry.get("occurrence_count", 0)),
        int(entry.get("sink_support", 0)),
        float(entry.get("quality_score", entry.get("candidate_model_score", 0.0))),
        int(entry.get("reviewer_event_count", 0)),
        str(entry.get("digest", "")),
    )


def mine_slot_challengers(
    *,
    problem: SlotProblemIR,
    anchor_eval: SlotEval,
    candidate_entries: list[dict[str, Any]],
) -> list[SlotChallenger]:
    buckets: dict[tuple[str, str], dict[str, Any]] = {}
    anchor_assignment = anchor_eval.artifact.assignment

    for entry in candidate_entries:
        eval_obj = entry.get("_slot_eval")
        if not isinstance(eval_obj, SlotEval):
            continue
        if eval_obj.residual.fatal:
            continue
        cand_assignment = eval_obj.artifact.assignment
        for slot in problem.slots:
            slot_id = slot.slot_id
            anchor_value = str(anchor_assignment.get(slot_id, "")).strip()
            cand_value = str(cand_assignment.get(slot_id, "")).strip()
            if not cand_value or _norm_text(cand_value) == _norm_text(anchor_value):
                continue
            key = (slot_id, _norm_text(cand_value))
            bucket = buckets.setdefault(
                key,
                {
                    "slot_id": slot_id,
                    "anchor_value": anchor_value,
                    "challenger_value": cand_value,
                    "occurrence_count": 0,
                    "sink_support": 0,
                    "sources": set(),
                    "best_entry": entry,
                },
            )
            bucket["occurrence_count"] += max(1, int(entry.get("occurrence_count", 1)))
            bucket["sink_support"] += int(entry.get("sink_support", 0))
            source_ids = set()
            for key_name in ("source_node_ids", "source_roles"):
                value = entry.get(key_name)
                if isinstance(value, (list, tuple, set)):
                    source_ids |= {str(item) for item in value if str(item)}
            if not source_ids:
                source_ids.add(str(entry.get("origin_node_id") or entry.get("origin_role") or entry.get("digest") or ""))
            bucket["sources"] |= source_ids
            if _candidate_support_key(entry) > _candidate_support_key(bucket["best_entry"]):
                bucket["best_entry"] = entry

    challengers: list[SlotChallenger] = []
    for bucket in buckets.values():
        challengers.append(
            SlotChallenger(
                slot_id=str(bucket["slot_id"]),
                anchor_value=str(bucket["anchor_value"]),
                challenger_value=str(bucket["challenger_value"]),
                occurrence_count=int(bucket["occurrence_count"]),
                sink_support=int(bucket["sink_support"]),
                source_count=len(bucket["sources"]),
                best_entry_digest=str(bucket["best_entry"].get("digest", "")),
                best_entry=bucket["best_entry"],
            )
        )
    challengers.sort(
        key=lambda item: (
            item.source_count,
            item.occurrence_count,
            item.sink_support,
            _candidate_support_key(item.best_entry),
        ),
        reverse=True,
    )
    return challengers


def mine_slot_challengers_from_mentions(
    *,
    problem: SlotProblemIR,
    anchor_eval: SlotEval,
    candidate_entries: list[dict[str, Any]],
) -> list[SlotChallenger]:
    # Multi-slot mention attribution is ambiguous; keep this recall patch single-slot only.
    if len(problem.slots) != 1:
        return []

    slot = problem.slots[0]
    anchor_value = str(anchor_eval.artifact.assignment.get(slot.slot_id, "")).strip()
    label_map = _label_to_option_map(slot)
    buckets: dict[str, dict[str, Any]] = {}

    for entry in candidate_entries:
        raw = str(entry.get("text", "") or "")
        if not raw:
            continue

        mentioned_values: set[str] = set()
        for match in re.finditer(r"\bOPTION\s*[-:：]?\s*([A-Z]|\d+)\b", raw, flags=re.IGNORECASE):
            value = label_map.get(_norm_label(match.group(1)))
            if value:
                mentioned_values.add(value)

        norm_raw = _norm_text(raw)
        for option in slot.options:
            norm_option = _norm_text(option)
            if norm_option and norm_option in norm_raw:
                mentioned_values.add(option)

        for value in mentioned_values:
            if _norm_text(value) == _norm_text(anchor_value):
                continue

            key = _norm_text(value)
            bucket = buckets.setdefault(
                key,
                {
                    "value": value,
                    "occurrence_count": 0,
                    "sink_support": 0,
                    "sources": set(),
                    "best_entry": entry,
                },
            )
            bucket["occurrence_count"] += max(1, int(entry.get("occurrence_count", 1)))
            bucket["sink_support"] += int(entry.get("sink_support", 0))
            bucket["sources"].add(
                str(entry.get("origin_node_id") or entry.get("origin_role") or entry.get("digest") or "")
            )

            if _candidate_support_key(entry) > _candidate_support_key(bucket["best_entry"]):
                bucket["best_entry"] = entry

    challengers: list[SlotChallenger] = []
    for bucket in buckets.values():
        best_entry = dict(bucket["best_entry"])
        best_entry["candidate_bank_source"] = "slot_text_mention"
        challengers.append(
            SlotChallenger(
                slot_id=slot.slot_id,
                anchor_value=anchor_value,
                challenger_value=str(bucket["value"]),
                occurrence_count=int(bucket["occurrence_count"]),
                sink_support=int(bucket["sink_support"]),
                source_count=len(bucket["sources"]),
                best_entry_digest=str(best_entry.get("digest", "")),
                best_entry=best_entry,
            )
        )

    challengers.sort(
        key=lambda item: (
            item.source_count,
            item.occurrence_count,
            item.sink_support,
            _candidate_support_key(item.best_entry),
        ),
        reverse=True,
    )
    return challengers


def preserves_frozen_slots(
    anchor_assignment: dict[str, str],
    candidate_assignment: dict[str, str],
    mutable_slots: set[str],
) -> bool:
    for slot_id, anchor_value in anchor_assignment.items():
        if slot_id in mutable_slots:
            continue
        if _norm_text(candidate_assignment.get(slot_id, "")) != _norm_text(anchor_value):
            return False
    return True


def fixes_invalid_slots(anchor_eval: SlotEval, candidate_eval: SlotEval, mutable_slots: set[str]) -> bool:
    for slot_id in mutable_slots:
        slot = _slot_by_id(anchor_eval.problem, slot_id)
        anchor_value = anchor_eval.artifact.assignment.get(slot_id, "")
        candidate_value = candidate_eval.artifact.assignment.get(slot_id, "")
        if _value_in_slot_options(anchor_value, slot):
            continue
        if not _value_in_slot_options(candidate_value, slot):
            return False
    return True


def _assignment_edit_count(anchor_assignment: dict[str, str], candidate_assignment: dict[str, str]) -> int:
    slot_ids = set(anchor_assignment) | set(candidate_assignment)
    return sum(1 for slot_id in slot_ids if _norm_text(anchor_assignment.get(slot_id, "")) != _norm_text(candidate_assignment.get(slot_id, "")))


def slot_membership_repair_rank_key(
    entry: dict[str, Any],
    *,
    anchor_eval: SlotEval,
    mutable_slots: set[str],
) -> tuple[Any, ...]:
    eval_obj = entry["_slot_eval"]
    fixed_count = sum(
        1
        for slot_id in mutable_slots
        if _value_in_slot_options(
            eval_obj.artifact.assignment.get(slot_id, ""),
            _slot_by_id(anchor_eval.problem, slot_id),
        )
    )
    edit_count = _assignment_edit_count(anchor_eval.artifact.assignment, eval_obj.artifact.assignment)
    return (
        fixed_count,
        int(not eval_obj.residual.fatal),
        int(entry.get("occurrence_count", 0)),
        int(entry.get("sink_support", 0)),
        float(entry.get("quality_score", entry.get("candidate_model_score", 0.0))),
        -edit_count,
        int(not bool(entry.get("explicit_challenger", False))),
        str(entry.get("digest", "")),
    )


def propose_membership_repair(
    *,
    problem: SlotProblemIR,
    anchor_entry: dict[str, Any],
    anchor_eval: SlotEval,
    candidate_entries: list[dict[str, Any]],
) -> dict[str, Any] | None:
    del problem
    del anchor_entry
    mutable = set(anchor_eval.residual.invalid_slots)
    if not mutable:
        return None
    candidates: list[dict[str, Any]] = []
    for entry in candidate_entries:
        eval_obj = entry.get("_slot_eval")
        if not isinstance(eval_obj, SlotEval):
            continue
        if eval_obj.residual.fatal:
            continue
        if not preserves_frozen_slots(anchor_eval.artifact.assignment, eval_obj.artifact.assignment, mutable):
            continue
        if not fixes_invalid_slots(anchor_eval, eval_obj, mutable):
            continue
        candidates.append(entry)
    if not candidates:
        return None
    candidates.sort(
        key=lambda item: slot_membership_repair_rank_key(item, anchor_eval=anchor_eval, mutable_slots=mutable),
        reverse=True,
    )
    return candidates[0]


def _render_assignment_json(assignment: dict[str, str], problem: SlotProblemIR) -> str:
    if problem.output_kind == "json_list":
        values = [assignment.get(slot.slot_id, "") for slot in problem.slots]
        return json.dumps(values, ensure_ascii=False)
    if len(problem.slots) == 1:
        slot = problem.slots[0]
        return json.dumps({slot.slot_id: assignment.get(slot.slot_id, "")}, ensure_ascii=False)
    return json.dumps(assignment, ensure_ascii=False)


def build_pairwise_slot_probe_prompt(
    *,
    problem: SlotProblemIR,
    artifact: SlotArtifact,
    challenger: SlotChallenger,
    certificate: SlotCertificate | None = None,
) -> str:
    slot = _slot_by_id(problem, challenger.slot_id)
    polarity_hint = _detect_question_polarity(problem.raw_question)
    certificate_block = ""
    if certificate is not None:
        certificate_payload = {
            "certificate_kind": certificate.certificate_kind,
            "target_condition": certificate.target_condition,
            "discriminator": certificate.discriminator,
            "anchor_conflict": [atom.statement for atom in certificate.anchor_conflict],
            "challenger_support": [atom.statement for atom in certificate.challenger_support],
            "challenger_conflict": [atom.statement for atom in certificate.challenger_conflict],
            "score_margin": certificate.score_margin,
            "changed_slots": list(certificate.changed_slots),
            "challenger_assignment": dict(certificate.challenger_assignment),
        }
        certificate_block = (
            f"Proposed structured certificate:\n{json.dumps(certificate_payload, ensure_ascii=False)}\n\n"
            "Audit the certificate instead of discovering a new answer.\n"
            "If discriminator, challenger_support, or anchor_conflict is empty or only restates the option text, return winner=uncertain.\n"
            "The discriminator must be a minimal target-condition contrast where challenger passes and anchor fails.\n\n"
        )
    return (
        "You are calibrating one finite-option slot.\n\n"
        f"Problem:\n{problem.raw_question}\n\n"
        f"Current full assignment:\n{_render_assignment_json(artifact.assignment, problem)}\n\n"
        f"Target slot:\n{slot.slot_id} / {slot.slot_name}\n\n"
        f"Allowed options for this slot:\n{json.dumps(list(slot.options), ensure_ascii=False)}\n\n"
        f"Anchor value:\n{challenger.anchor_value}\n\n"
        f"Challenger value:\n{challenger.challenger_value}\n\n"
        "First identify the target condition for the correct value of this slot.\n"
        "If the question contains NOT, EXCEPT, LEAST, FALSE, INCORRECT, CANNOT, or similar cues, "
        "the target condition must preserve that polarity. Do not reverse it.\n\n"
        f"Polarity hint from parser: {polarity_hint}\n\n"
        "For negative or exception questions, an option that satisfies the excluded property is a conflict, not support.\n\n"
        f"{certificate_block}"
        "Return exactly one JSON object:\n"
        "{\n"
        '  "question_polarity": "positive|negative|exception|comparative|unknown",\n'
        '  "target_condition": "...",\n'
        '  "inverse_condition": "...",\n'
        '  "anchor_satisfies_target": "yes|no|uncertain",\n'
        '  "challenger_satisfies_target": "yes|no|uncertain",\n'
        '  "anchor_satisfies_inverse": "yes|no|uncertain",\n'
        '  "challenger_satisfies_inverse": "yes|no|uncertain",\n'
        '  "winner": "anchor|challenger|uncertain",\n'
        '  "anchor_conflict": ["..."],\n'
        '  "challenger_conflict": ["..."],\n'
        '  "anchor_support": ["..."],\n'
        '  "challenger_support": ["..."],\n'
        '  "discriminator": "the key fact/relation that decides the slot",\n'
        '  "confidence": "high|medium|low"\n'
        "}\n"
    )


def _norm_yes_no_uncertain(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"yes", "true", "y"}:
        return "yes"
    if text in {"no", "false", "n"}:
        return "no"
    return "uncertain"


def parse_slot_probe_result(text: str) -> PairwiseSlotProbeResult:
    obj, _ = _extract_json_any(text)
    if not isinstance(obj, dict):
        return PairwiseSlotProbeResult(
            winner="uncertain",
            anchor_conflict=(),
            challenger_conflict=(),
            anchor_support=(),
            challenger_support=(),
            discriminator="",
            confidence="low",
            raw=str(text or ""),
            question_polarity="unknown",
            target_condition="",
            inverse_condition="",
            anchor_satisfies_target="uncertain",
            challenger_satisfies_target="uncertain",
            anchor_satisfies_inverse="uncertain",
            challenger_satisfies_inverse="uncertain",
        )
    winner = str(obj.get("winner", "uncertain")).strip().lower()
    if winner not in {"anchor", "challenger", "uncertain"}:
        winner = "uncertain"
    confidence = str(obj.get("confidence", "low")).strip().lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = "low"
    return PairwiseSlotProbeResult(
        winner=winner,
        anchor_conflict=tuple(str(item).strip() for item in (obj.get("anchor_conflict") or []) if str(item).strip()),
        challenger_conflict=tuple(
            str(item).strip() for item in (obj.get("challenger_conflict") or []) if str(item).strip()
        ),
        anchor_support=tuple(str(item).strip() for item in (obj.get("anchor_support") or []) if str(item).strip()),
        challenger_support=tuple(
            str(item).strip() for item in (obj.get("challenger_support") or []) if str(item).strip()
        ),
        discriminator=str(obj.get("discriminator", "")).strip(),
        confidence=confidence,
        raw=str(text or ""),
        question_polarity=str(obj.get("question_polarity", "unknown")).strip().lower(),
        target_condition=str(obj.get("target_condition", "")).strip(),
        inverse_condition=str(obj.get("inverse_condition", "")).strip(),
        anchor_satisfies_target=_norm_yes_no_uncertain(obj.get("anchor_satisfies_target")),
        challenger_satisfies_target=_norm_yes_no_uncertain(obj.get("challenger_satisfies_target")),
        anchor_satisfies_inverse=_norm_yes_no_uncertain(obj.get("anchor_satisfies_inverse")),
        challenger_satisfies_inverse=_norm_yes_no_uncertain(obj.get("challenger_satisfies_inverse")),
    )


def accepts_pairwise_slot_update(
    *,
    problem: SlotProblemIR,
    anchor_eval: SlotEval,
    challenger: SlotChallenger,
    probe: PairwiseSlotProbeResult,
    audit: PairwiseSlotProbeResult | None = None,
    anchor_support: tuple[int, int, int] | None = None,
    challenger_support: tuple[int, int, int] | None = None,
) -> bool:
    del anchor_eval
    slot = _slot_by_id(problem, challenger.slot_id)

    if not _value_in_slot_options(challenger.challenger_value, slot):
        return False

    # Candidate-pool support: either challenger has independent support,
    # or it must be confirmed by both probe and audit.
    has_pool_support = challenger.source_count >= 2 or challenger.occurrence_count >= 2

    if not has_pool_support and audit is None:
        return False

    if probe.winner != "challenger":
        return False

    if probe.confidence not in {"high", "medium"}:
        return False

    # New hard evidence requirements.
    if not probe.discriminator:
        return False

    if not probe.challenger_support:
        return False

    if not probe.anchor_conflict:
        return False

    if probe.challenger_conflict:
        return False

    # Target-condition requirements.
    if probe.challenger_satisfies_target != "yes":
        return False

    if probe.anchor_satisfies_target != "no":
        return False

    if probe.challenger_satisfies_inverse == "yes":
        return False

    # Polarity guard.
    parser_polarity = _detect_question_polarity(problem.raw_question)
    if parser_polarity == "negative_or_exception":
        if probe.question_polarity not in {"negative", "exception", "negative_or_exception"}:
            return False
        if not probe.inverse_condition:
            return False

    # Candidate support must not be weaker than anchor support unless audit confirms.
    if anchor_support is not None and challenger_support is not None:
        if challenger_support <= anchor_support and audit is None:
            return False

    # Audit confirmation, if provided.
    if audit is not None:
        if audit.winner != "challenger":
            return False
        if audit.confidence not in {"high", "medium"}:
            return False
        if not audit.challenger_support or not audit.discriminator:
            return False
        if audit.challenger_satisfies_target != "yes":
            return False
        if audit.anchor_satisfies_target != "no":
            return False
        if audit.challenger_satisfies_inverse == "yes":
            return False
        if audit.challenger_conflict:
            return False

    return True


def _slot_contract_payload(problem: SlotProblemIR) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for slot in problem.slots:
        labels = slot.option_labels or _option_labels_for_length(len(slot.options))
        rows.append(
            {
                "slot_id": slot.slot_id,
                "slot_name": slot.slot_name,
                "allowed_option_labels": list(labels),
                "allowed_option_texts": list(slot.options),
            }
        )
    return rows


def _str_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, (list, tuple, set)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    text = str(value or "").strip()
    return (text,) if text else ()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _evidence_atoms(
    *,
    slot_id: str,
    value: str,
    relation: str,
    statements: Sequence[str],
    polarity: str,
    source: str,
    source_id: str = "",
    confidence: float = 1.0,
) -> tuple[EvidenceAtom, ...]:
    return tuple(
        EvidenceAtom(
            slot_id=slot_id,
            value=value,
            relation=relation,
            statement=str(statement).strip(),
            polarity=polarity,
            source=source,
            source_id=source_id,
            confidence=confidence,
        )
        for statement in statements
        if str(statement).strip()
    )


def certificate_assignment_dict(cert: SlotCertificate) -> dict[str, str]:
    if cert.challenger_assignment:
        return _assignment_dict(cert.challenger_assignment)
    if cert.changed_slots:
        return {cert.slot_id: cert.challenger_value}
    return {cert.slot_id: cert.challenger_value}


def challenger_from_certificate(cert: SlotCertificate) -> SlotChallenger:
    return SlotChallenger(
        slot_id=cert.slot_id,
        anchor_value=cert.anchor_value,
        challenger_value=cert.challenger_value,
        occurrence_count=max(1, len(cert.challenger_target_votes)),
        sink_support=max(0, int(round(cert.score_margin))),
        source_count=max(1, cert.source_count),
        best_entry_digest=f"cert_{cert.certificate_kind}_{_hash_json([cert.slot_id, cert.challenger_value, cert.score_margin])}",
        best_entry={
            "candidate_bank_source": cert.certificate_kind,
            "digest": f"cert_{cert.certificate_kind}_{_hash_json([cert.slot_id, cert.challenger_value, cert.score_margin])}",
            "origin_role": "certified_slot_operator",
            "occurrence_count": max(1, len(cert.challenger_target_votes)),
            "sink_support": max(0, int(round(cert.score_margin))),
            "_slot_certificate": cert,
        },
    )


def pairwise_probe_from_certificate(cert: SlotCertificate, problem: SlotProblemIR) -> PairwiseSlotProbeResult:
    parser_polarity = _detect_question_polarity(problem.raw_question)
    question_polarity = "negative" if parser_polarity == "negative_or_exception" else "positive"
    confidence = "high" if cert.score_margin >= 2.0 and cert.source_count >= 2 else "medium"
    return PairwiseSlotProbeResult(
        winner="challenger",
        anchor_conflict=tuple(atom.statement for atom in cert.anchor_conflict),
        challenger_conflict=tuple(atom.statement for atom in cert.challenger_conflict),
        anchor_support=(),
        challenger_support=tuple(atom.statement for atom in cert.challenger_support),
        discriminator=cert.discriminator,
        confidence=confidence,
        raw=cert.raw,
        question_polarity=question_polarity,
        target_condition=cert.target_condition,
        inverse_condition="disqualifies the option from satisfying the target condition",
        anchor_satisfies_target="no",
        challenger_satisfies_target="yes",
        anchor_satisfies_inverse="yes",
        challenger_satisfies_inverse="no",
    )


def build_mmlu_rubric_prompt(*, problem: SlotProblemIR) -> str:
    return (
        "You are deriving an anchor-blind rubric for a multiple-choice problem.\n"
        "Do not choose the final answer.\n\n"
        f"Problem:\n{problem.raw_question}\n\n"
        f"Slot contract:\n{json.dumps(_slot_contract_payload(problem), ensure_ascii=False)}\n\n"
        "Return exactly one compact JSON object:\n"
        "{\n"
        '  "question_type": "conceptual|factual|comparative|legal|medical|scientific|ethical|other",\n'
        '  "polarity": "positive|negative|lowest|highest|exception|cause|definition|recommendation|other",\n'
        '  "target_condition": "one sentence describing the exact test a correct option must satisfy",\n'
        '  "must_have": ["atomic condition"],\n'
        '  "disqualifiers": ["atomic reason an option fails"]\n'
        "}\n"
    )


def parse_mmlu_rubric_result(text: str) -> dict[str, Any]:
    obj, _ = _extract_json_any(text)
    if not isinstance(obj, dict):
        return {
            "question_type": "other",
            "polarity": "other",
            "target_condition": "",
            "must_have": [],
            "disqualifiers": [],
            "raw": str(text or ""),
        }
    return {
        "question_type": str(obj.get("question_type") or "other").strip().lower(),
        "polarity": str(obj.get("polarity") or "other").strip().lower(),
        "target_condition": str(obj.get("target_condition") or "").strip(),
        "must_have": list(_str_tuple(obj.get("must_have"))),
        "disqualifiers": list(_str_tuple(obj.get("disqualifiers"))),
        "raw": str(text or ""),
    }


def build_mmlu_option_matrix_prompt(
    *,
    problem: SlotProblemIR,
    rubric: dict[str, Any],
    sample_index: int = 0,
) -> str:
    return (
        "You are constructing an all-option evidence matrix for a finite-option MCQA problem.\n"
        "Evaluate every allowed option independently against the target condition.\n"
        "Do not choose the final answer directly.\n\n"
        f"Problem:\n{problem.raw_question}\n\n"
        f"Rubric:\n{json.dumps(_jsonable(rubric), ensure_ascii=False)}\n\n"
        f"Slot contract:\n{json.dumps(_slot_contract_payload(problem), ensure_ascii=False)}\n\n"
        f"Independent evidence pass: {sample_index}\n\n"
        "For every option, support must be an atomic fact/relation that is not just a paraphrase of the option text.\n"
        "If support or discriminator is unavailable, mark satisfies_target=uncertain.\n"
        "Preserve NOT, EXCEPT, LEAST, FALSE, INCORRECT, CANNOT, lowest/highest, and comparative cues.\n\n"
        "Return exactly one compact JSON object:\n"
        "{\n"
        '  "target_condition": "...",\n'
        '  "options": [\n'
        "    {\n"
        '      "label": "...",\n'
        '      "option_value": "...",\n'
        '      "satisfies_target": "yes|no|uncertain",\n'
        '      "support": ["atomic support"],\n'
        '      "conflict": ["atomic reason it fails"],\n'
        '      "decisive_relation": "minimal relation/test"\n'
        "    }\n"
        "  ]\n"
        "}\n"
    )


def _parse_mmlu_option_matrix_rows(text: str, problem: SlotProblemIR) -> tuple[str, list[dict[str, Any]]]:
    if len(problem.slots) != 1:
        return "", []
    slot = problem.slots[0]
    raw = _strip_hidden_reasoning(text)
    obj, _ = _extract_json_any(raw)
    if not isinstance(obj, dict):
        return "", []
    target_condition = str(obj.get("target_condition") or "").strip()
    raw_rows = obj.get("options") or obj.get("rows") or obj.get("option_matrix") or []
    if isinstance(raw_rows, dict):
        raw_rows = list(raw_rows.values())
    if not isinstance(raw_rows, list):
        return target_condition, []
    rows: list[dict[str, Any]] = []
    for row in raw_rows:
        if not isinstance(row, dict):
            continue
        raw_value = row.get("option_value") or row.get("value") or row.get("option") or row.get("answer") or row.get("label")
        value = _normalize_single_slot_value(raw_value, slot)
        if not value:
            value = _option_value_map(slot).get(_norm_text(raw_value), "")
        if not value:
            continue
        rows.append(
            {
                "slot_id": slot.slot_id,
                "option_value": value,
                "label": str(row.get("label") or row.get("option_label") or _option_label_for_value(slot, value)).strip(),
                "satisfies_target": _norm_yes_no_uncertain(row.get("satisfies_target")),
                "support": tuple(_str_tuple(row.get("support") or row.get("support_facts"))),
                "conflict": tuple(_str_tuple(row.get("conflict") or row.get("conflict_facts"))),
                "discriminator": str(row.get("decisive_relation") or row.get("discriminator") or "").strip(),
            }
        )
    return target_condition, rows


def build_mmlu_option_matrix_certificates(
    *,
    problem: SlotProblemIR,
    anchor_eval: SlotEval,
    rubric: dict[str, Any],
    matrix_texts: Sequence[str],
) -> list[SlotCertificate]:
    if len(problem.slots) != 1:
        return []
    slot = problem.slots[0]
    anchor_value = str(anchor_eval.artifact.assignment.get(slot.slot_id, "")).strip()
    if not anchor_value:
        return []

    target_condition = str(rubric.get("target_condition") or "").strip()
    buckets: dict[str, dict[str, Any]] = {
        option: {"yes": 0, "no": 0, "uncertain": 0, "support": [], "conflict": [], "discriminator": []}
        for option in slot.options
    }
    for index, raw in enumerate(matrix_texts):
        matrix_target, rows = _parse_mmlu_option_matrix_rows(raw, problem)
        if matrix_target and not target_condition:
            target_condition = matrix_target
        for row in rows:
            value = str(row["option_value"])
            bucket = buckets.setdefault(value, {"yes": 0, "no": 0, "uncertain": 0, "support": [], "conflict": [], "discriminator": []})
            bucket[row["satisfies_target"]] = int(bucket.get(row["satisfies_target"], 0)) + 1
            bucket["support"].extend((index, item) for item in row["support"])
            bucket["conflict"].extend((index, item) for item in row["conflict"])
            if row["discriminator"]:
                bucket["discriminator"].append((index, row["discriminator"]))

    if not target_condition:
        target_condition = "satisfies the question target condition"
    sample_count = max(1, len(matrix_texts))
    vote_threshold = 2 if sample_count >= 2 else 1
    anchor_bucket = buckets.get(anchor_value, {})

    def score(bucket: dict[str, Any]) -> float:
        return (
            2.0 * int(bucket.get("yes", 0))
            - 1.5 * int(bucket.get("no", 0))
            + 0.5 * len({item for _, item in bucket.get("support", [])})
            - 0.8 * len({item for _, item in bucket.get("conflict", [])})
        )

    anchor_score = score(anchor_bucket)
    certificates: list[SlotCertificate] = []
    for option in slot.options:
        if _norm_text(option) == _norm_text(anchor_value):
            continue
        bucket = buckets.get(option, {})
        yes_votes = int(bucket.get("yes", 0))
        no_votes = int(bucket.get("no", 0))
        supports = tuple(dict.fromkeys(str(item) for _, item in bucket.get("support", []) if str(item).strip()))
        conflicts = tuple(dict.fromkeys(str(item) for _, item in bucket.get("conflict", []) if str(item).strip()))
        discriminators = tuple(dict.fromkeys(str(item) for _, item in bucket.get("discriminator", []) if str(item).strip()))
        if yes_votes < vote_threshold:
            continue
        if no_votes > 0 or conflicts:
            continue
        if not supports or not discriminators:
            continue
        anchor_conflicts = tuple(
            dict.fromkeys(str(item) for _, item in anchor_bucket.get("conflict", []) if str(item).strip())
        )
        if not anchor_conflicts and int(anchor_bucket.get("no", 0)) >= vote_threshold:
            anchor_conflicts = (f"anchor does not satisfy target condition: {target_condition}",)
        if not anchor_conflicts and yes_votes <= int(anchor_bucket.get("yes", 0)):
            continue
        margin = score(bucket) - anchor_score
        if margin <= 0:
            continue
        certificates.append(
            SlotCertificate(
                slot_id=slot.slot_id,
                anchor_value=anchor_value,
                challenger_value=option,
                target_condition=target_condition,
                discriminator=discriminators[0],
                challenger_support=_evidence_atoms(
                    slot_id=slot.slot_id,
                    value=option,
                    relation=discriminators[0],
                    statements=supports,
                    polarity="support",
                    source="llm_probe",
                    source_id="mmlu_option_matrix",
                    confidence=0.8,
                ),
                anchor_conflict=_evidence_atoms(
                    slot_id=slot.slot_id,
                    value=anchor_value,
                    relation=discriminators[0],
                    statements=anchor_conflicts,
                    polarity="conflict",
                    source="llm_probe",
                    source_id="mmlu_option_matrix",
                    confidence=0.8,
                ),
                challenger_conflict=(),
                anchor_target_votes=tuple(["no"] * int(anchor_bucket.get("no", 0)) + ["yes"] * int(anchor_bucket.get("yes", 0))),
                challenger_target_votes=tuple(["yes"] * yes_votes + ["no"] * no_votes),
                score_margin=margin,
                source_count=yes_votes,
                certificate_kind="mmlu_option_matrix",
                changed_slots=(slot.slot_id,),
                anchor_assignment=_assignment_tuple(anchor_eval.artifact.assignment),
                challenger_assignment=_assignment_tuple({**anchor_eval.artifact.assignment, slot.slot_id: option}),
                target_condition_consistency="consistent" if target_condition else "unknown",
                raw=json.dumps({"rubric": _jsonable(rubric), "option": option}, ensure_ascii=False),
            )
        )
    certificates.sort(key=lambda item: (item.score_margin, item.source_count, len(item.challenger_support)), reverse=True)
    return certificates


def _literal_list_field(text: str, label: str) -> list[str]:
    match = re.search(rf"{re.escape(label)}\s*:\s*(\[[^\]]*\])", text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return []
    try:
        parsed = ast.literal_eval(match.group(1))
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item).strip() for item in parsed if str(item).strip()]


def _metadata_list_field(meta: dict[str, Any], *keys: str) -> list[str]:
    for key in keys:
        value = meta.get(key)
        if isinstance(value, list):
            parsed = [str(item).strip() for item in value if str(item).strip()]
            if parsed:
                return parsed
    return []


def build_kc_constraints(problem: SlotProblemIR) -> tuple[KCConstraint, ...]:
    meta = dict(problem.metadata or {})
    sources = _metadata_list_field(meta, "sources", "source_entities")
    relations = _metadata_list_field(meta, "relations", "relation_labels")
    targets = _metadata_list_field(meta, "targets", "target_entities")
    if not sources:
        sources = _literal_list_field(problem.raw_question, "Sources")
    if not relations:
        relations = _literal_list_field(problem.raw_question, "Relations")
    if not targets:
        targets = _literal_list_field(problem.raw_question, "Targets")
    constraints: list[KCConstraint] = []
    for index, (source, relation, target) in enumerate(zip(sources, relations, targets)):
        constraints.append(
            KCConstraint(
                subject=str(source).strip(),
                predicate=str(relation).strip(),
                object=str(target).strip(),
                weight=1.0,
                source_id=f"constraint_{index}",
                required=True,
            )
        )
    return tuple(constraints)


def _substitute_constraint(constraint: KCConstraint, assignment: dict[str, str]) -> tuple[str, str, str]:
    return (
        assignment.get(constraint.subject, constraint.subject),
        constraint.predicate,
        assignment.get(constraint.object, constraint.object),
    )


def score_kc_assignment(problem: SlotProblemIR, assignment: dict[str, str]) -> KCAssignmentScore:
    constraints = build_kc_constraints(problem)
    unsupported = tuple(constraints)
    duplicate_conflicts: list[tuple[str, str, str]] = []
    seen: dict[str, str] = {}
    for slot in problem.slots:
        value = canonicalize_entity(assignment.get(slot.slot_id, ""))
        if not value:
            continue
        prior = seen.get(value)
        if prior is not None:
            duplicate_conflicts.append((prior, slot.slot_id, assignment.get(slot.slot_id, "")))
        else:
            seen[value] = slot.slot_id
    score = -2.0 * len(duplicate_conflicts)
    return KCAssignmentScore(
        assignment=_assignment_tuple(assignment),
        satisfied_edges=(),
        violated_edges=(),
        unsupported_edges=unsupported,
        duplicate_conflicts=tuple(duplicate_conflicts),
        score=score,
    )


def kc_candidate_universe_size(problem: SlotProblemIR) -> int:
    size = 1
    for slot in problem.slots:
        size *= max(1, len(slot.options))
    return size


def beam_search_kc_assignments(
    *,
    problem: SlotProblemIR,
    anchor_eval: SlotEval,
    candidate_entries: Sequence[dict[str, Any]],
    max_assignments: int = 64,
) -> list[dict[str, str]]:
    assignments: list[dict[str, str]] = [dict(anchor_eval.artifact.assignment)]
    seen = {_hash_json(assignments[0])}
    for entry in candidate_entries:
        eval_obj = entry.get("_slot_eval")
        if isinstance(eval_obj, SlotEval) and not eval_obj.residual.fatal:
            assignment = dict(eval_obj.artifact.assignment)
            key = _hash_json(assignment)
            if key not in seen:
                assignments.append(assignment)
                seen.add(key)
    universe = kc_candidate_universe_size(problem)
    if universe <= max_assignments:
        for values in itertools.product(*(slot.options for slot in problem.slots)):
            assignment = {slot.slot_id: value for slot, value in zip(problem.slots, values)}
            key = _hash_json(assignment)
            if key not in seen:
                assignments.append(assignment)
                seen.add(key)
    assignments.sort(key=lambda item: score_kc_assignment(problem, item).score, reverse=True)
    return assignments[:max_assignments]


def mine_multislot_mentions_safely(
    *,
    problem: SlotProblemIR,
    anchor_eval: SlotEval,
    candidate_entries: Sequence[dict[str, Any]],
) -> list[SlotChallenger]:
    if len(problem.slots) <= 1:
        return []
    option_to_slots: dict[str, list[SlotSpec]] = {}
    for slot in problem.slots:
        for option in slot.options:
            option_to_slots.setdefault(_norm_text(option), []).append(slot)
    buckets: dict[tuple[str, str], dict[str, Any]] = {}
    for entry in candidate_entries:
        raw = str(entry.get("text", "") or "")
        norm_raw = _norm_text(raw)
        if not norm_raw:
            continue
        for norm_option, slots in option_to_slots.items():
            if len(slots) != 1 or not norm_option or norm_option not in norm_raw:
                continue
            slot = slots[0]
            anchor_value = str(anchor_eval.artifact.assignment.get(slot.slot_id, "")).strip()
            option_value = _option_value_map(slot).get(norm_option, "")
            if not option_value or _norm_text(option_value) == _norm_text(anchor_value):
                continue
            key = (slot.slot_id, norm_option)
            bucket = buckets.setdefault(
                key,
                {
                    "slot": slot,
                    "value": option_value,
                    "anchor": anchor_value,
                    "occurrence_count": 0,
                    "sink_support": 0,
                    "sources": set(),
                    "best_entry": entry,
                },
            )
            bucket["occurrence_count"] += max(1, int(entry.get("occurrence_count", 1)))
            bucket["sink_support"] += int(entry.get("sink_support", 0))
            bucket["sources"].add(str(entry.get("origin_node_id") or entry.get("origin_role") or entry.get("digest") or ""))
            if _candidate_support_key(entry) > _candidate_support_key(bucket["best_entry"]):
                bucket["best_entry"] = entry
    challengers: list[SlotChallenger] = []
    for bucket in buckets.values():
        best_entry = dict(bucket["best_entry"])
        best_entry["candidate_bank_source"] = "safe_multislot_mention"
        challengers.append(
            SlotChallenger(
                slot_id=bucket["slot"].slot_id,
                anchor_value=str(bucket["anchor"]),
                challenger_value=str(bucket["value"]),
                occurrence_count=int(bucket["occurrence_count"]),
                sink_support=int(bucket["sink_support"]),
                source_count=len(bucket["sources"]),
                best_entry_digest=str(best_entry.get("digest", "")),
                best_entry=best_entry,
            )
        )
    challengers.sort(key=lambda item: (item.source_count, item.occurrence_count, item.sink_support), reverse=True)
    return challengers


def build_kc_factor_certificate_prompt(
    *,
    problem: SlotProblemIR,
    anchor_eval: SlotEval,
    candidate_assignments: Sequence[dict[str, str]],
) -> str:
    constraints = [
        {
            "subject": item.subject,
            "predicate": item.predicate,
            "object": item.object,
            "weight": item.weight,
            "source_id": item.source_id,
            "required": item.required,
        }
        for item in build_kc_constraints(problem)
    ]
    scored = [
        {
            "assignment": assignment,
            "deterministic_score": score_kc_assignment(problem, assignment).score,
            "duplicate_conflicts": list(score_kc_assignment(problem, assignment).duplicate_conflicts),
        }
        for assignment in candidate_assignments
    ]
    return (
        "You are constructing auditable factor-graph certificates for a Knowledge Crosswords style slot assignment.\n"
        "Do not use external world knowledge unless it is necessary; prioritize the supplied constraints and assignments.\n"
        "You are NOT free to invent values outside the slot contract.\n\n"
        f"Problem:\n{problem.raw_question}\n\n"
        f"Slot contract:\n{json.dumps(_slot_contract_payload(problem), ensure_ascii=False)}\n\n"
        f"Anchor assignment:\n{_render_assignment_json(anchor_eval.artifact.assignment, problem)}\n\n"
        f"Constraints:\n{json.dumps(constraints, ensure_ascii=False)}\n\n"
        f"Candidate assignments:\n{json.dumps(scored, ensure_ascii=False)}\n\n"
        "For each certificate, include support_edges proving challenger constraints and anchor_conflict_edges showing anchor failures.\n"
        "The discriminator must name the minimal relation/constraint set where challenger passes and anchor fails.\n"
        "Return no certificate if no challenger strictly reduces violations or duplicate conflicts.\n\n"
        "Return exactly one compact JSON object:\n"
        "{\n"
        '  "certificates": [\n'
        "    {\n"
        '      "changed_slots": ["blank 1"],\n'
        '      "challenger_assignment": {"blank 1": "...", "blank 2": "..."},\n'
        '      "target_condition": "satisfies the supplied KG constraints",\n'
        '      "discriminator": "minimal relation/constraint contrast",\n'
        '      "support_edges": ["grounded triple/relation challenger satisfies"],\n'
        '      "anchor_conflict_edges": ["grounded triple/relation anchor violates"],\n'
        '      "new_conflict_edges": [],\n'
        '      "score_margin": 0.0\n'
        "    }\n"
        "  ]\n"
        "}\n"
    )


def parse_kc_factor_certificate_result(
    text: str,
    problem: SlotProblemIR,
    anchor_eval: SlotEval,
) -> list[SlotCertificate]:
    raw = _strip_hidden_reasoning(text)
    obj, _ = _extract_json_any(raw)
    if not isinstance(obj, dict):
        return []
    raw_items = obj.get("certificates") or obj.get("certificate") or []
    if isinstance(raw_items, dict):
        raw_items = [raw_items]
    if not isinstance(raw_items, list):
        return []
    slot_ids = {slot.slot_id for slot in problem.slots}
    certificates: list[SlotCertificate] = []
    anchor_assignment = dict(anchor_eval.artifact.assignment)
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        challenger_assignment_raw = item.get("challenger_assignment") or item.get("assignment") or {}
        if not isinstance(challenger_assignment_raw, dict):
            continue
        challenger_assignment = dict(anchor_assignment)
        legal = True
        for slot in problem.slots:
            raw_value = challenger_assignment_raw.get(slot.slot_id, challenger_assignment_raw.get(slot.slot_name, challenger_assignment.get(slot.slot_id, "")))
            value = _normalize_single_slot_value(raw_value, slot) or _option_value_map(slot).get(_norm_text(raw_value), "")
            if not value:
                legal = False
                break
            challenger_assignment[slot.slot_id] = value
        if not legal:
            continue
        changed_slots = tuple(
            slot_id
            for slot_id in slot_ids
            if _norm_text(anchor_assignment.get(slot_id, "")) != _norm_text(challenger_assignment.get(slot_id, ""))
        )
        if not changed_slots:
            continue
        primary_slot = changed_slots[0]
        cert = SlotCertificate(
            slot_id=primary_slot,
            anchor_value=str(anchor_assignment.get(primary_slot, "")),
            challenger_value=str(challenger_assignment.get(primary_slot, "")),
            target_condition=str(item.get("target_condition") or "satisfies the supplied KG constraints").strip(),
            discriminator=str(item.get("discriminator") or "").strip(),
            challenger_support=_evidence_atoms(
                slot_id=primary_slot,
                value=str(challenger_assignment.get(primary_slot, "")),
                relation=str(item.get("discriminator") or "kc_factor_graph"),
                statements=_str_tuple(item.get("support_edges") or item.get("support")),
                polarity="support",
                source="llm_probe",
                source_id="kc_factor_graph",
                confidence=0.8,
            ),
            anchor_conflict=_evidence_atoms(
                slot_id=primary_slot,
                value=str(anchor_assignment.get(primary_slot, "")),
                relation=str(item.get("discriminator") or "kc_factor_graph"),
                statements=_str_tuple(item.get("anchor_conflict_edges") or item.get("anchor_conflict")),
                polarity="conflict",
                source="llm_probe",
                source_id="kc_factor_graph",
                confidence=0.8,
            ),
            challenger_conflict=_evidence_atoms(
                slot_id=primary_slot,
                value=str(challenger_assignment.get(primary_slot, "")),
                relation=str(item.get("discriminator") or "kc_factor_graph"),
                statements=_str_tuple(item.get("new_conflict_edges") or item.get("challenger_conflict")),
                polarity="conflict",
                source="llm_probe",
                source_id="kc_factor_graph",
                confidence=0.8,
            ),
            anchor_target_votes=("no",),
            challenger_target_votes=("yes",),
            score_margin=_safe_float(item.get("score_margin"), 0.0),
            source_count=1,
            certificate_kind="kc_factor_graph",
            changed_slots=changed_slots,
            anchor_assignment=_assignment_tuple(anchor_assignment),
            challenger_assignment=_assignment_tuple(challenger_assignment),
            target_condition_consistency="consistent",
            raw=raw,
        )
        certificates.append(cert)
    certificates.sort(key=lambda item: (item.score_margin, len(item.changed_slots), len(item.challenger_support)), reverse=True)
    return certificates


def build_kc_factor_certificates(
    *,
    text: str,
    problem: SlotProblemIR,
    anchor_eval: SlotEval,
) -> list[SlotCertificate]:
    return parse_kc_factor_certificate_result(text, problem, anchor_eval)


def _certificate_values_legal(problem: SlotProblemIR, cert: SlotCertificate) -> bool:
    assignment = certificate_assignment_dict(cert)
    for slot in problem.slots:
        value = assignment.get(slot.slot_id, cert.challenger_value if slot.slot_id == cert.slot_id else "")
        if not _value_in_slot_options(value, slot):
            return False
    return True


def accepts_mmlu_certificate(
    *,
    problem: SlotProblemIR,
    cert: SlotCertificate,
    probe: PairwiseSlotProbeResult,
    audit: PairwiseSlotProbeResult | None,
) -> bool:
    if cert.certificate_kind != "mmlu_option_matrix":
        return False
    if not _certificate_values_legal(problem, cert):
        return False
    if not cert.discriminator or not cert.challenger_support or not cert.anchor_conflict or cert.challenger_conflict:
        return False
    if cert.score_margin < 1.0 or cert.source_count < 1:
        return False
    if cert.target_condition_consistency not in {"consistent", "unknown"}:
        return False
    if audit is None:
        return False
    challenger = challenger_from_certificate(cert)
    return accepts_pairwise_slot_update(
        problem=problem,
        anchor_eval=make_slot_eval(problem, parse_slot_artifact(_render_assignment_text(dict(cert.anchor_assignment), problem), problem)),
        challenger=challenger,
        probe=probe,
        audit=audit,
        anchor_support=(1, 1, 0),
        challenger_support=(cert.source_count, len(cert.challenger_support), int(cert.score_margin)),
    )


def accepts_kc_certificate(*, problem: SlotProblemIR, cert: SlotCertificate) -> bool:
    if cert.certificate_kind != "kc_factor_graph":
        return False
    if not _certificate_values_legal(problem, cert):
        return False
    if not cert.changed_slots:
        return False
    if not cert.discriminator or not cert.challenger_support or not cert.anchor_conflict:
        return False
    if cert.challenger_conflict:
        return False
    if cert.score_margin < 0.5:
        return False
    return True


def accepts_certified_slot_update(
    *,
    problem: SlotProblemIR,
    cert: SlotCertificate,
    probe: PairwiseSlotProbeResult,
    audit: PairwiseSlotProbeResult | None,
) -> bool:
    if cert.certificate_kind == "mmlu_option_matrix":
        return accepts_mmlu_certificate(problem=problem, cert=cert, probe=probe, audit=audit)
    if cert.certificate_kind == "kc_factor_graph":
        if not accepts_kc_certificate(problem=problem, cert=cert):
            return False
        if audit is None:
            return False
        challenger = challenger_from_certificate(cert)
        return accepts_pairwise_slot_update(
            problem=problem,
            anchor_eval=make_slot_eval(problem, parse_slot_artifact(_render_assignment_text(dict(cert.anchor_assignment), problem), problem)),
            challenger=challenger,
            probe=probe,
            audit=audit,
            anchor_support=(1, 1, 0),
            challenger_support=(cert.source_count, len(cert.challenger_support), int(cert.score_margin)),
        )
    return False


def build_slot_challenger_proposal_prompt(
    *,
    problem: SlotProblemIR,
    artifact: SlotArtifact,
    max_challengers: int = 2,
) -> str:
    limit = max(1, int(max_challengers))
    if len(problem.slots) > 1:
        limit = min(limit, len(problem.slots))
        proposal_rule = (
            f"For multi-slot assignments, propose at most one challenger per slot "
            f"and at most {limit} challengers total.\n"
        )
    else:
        proposal_rule = (
            f"For a single-slot finite-option problem, propose at most {limit} challengers "
            "for that slot.\n"
        )

    slot_contract = _slot_contract_payload(problem)

    return (
        "You are proposing possible challengers for finite-option slot calibration.\n"
        "You are NOT deciding the final answer.\n"
        "You are only proposing candidates for later pairwise verification and audit.\n\n"
        f"Problem:\n{problem.raw_question}\n\n"
        f"Current assignment:\n{_render_assignment_json(artifact.assignment, problem)}\n\n"
        f"Slot contract:\n{json.dumps(slot_contract, ensure_ascii=False)}\n\n"
        f"{proposal_rule}"
        "For every challenger:\n"
        '- "slot_id" must be exactly one of the slot_id values in Slot contract.\n'
        '- "value" must be either an allowed option label or an exact allowed option text for that slot.\n'
        "- Do not invent values outside the allowed options.\n"
        "- Do not output OPTION - N outside JSON.\n"
        "- Preserve NOT, EXCEPT, LEAST, FALSE, INCORRECT, CANNOT, and other polarity cues.\n\n"
        "Return exactly one JSON object and nothing else:\n"
        "{\n"
        '  "target_condition": "...",\n'
        '  "challengers": [\n'
        '    {"slot_id": "...", "value": "...", "reason": "..."}\n'
        "  ]\n"
        "}\n"
    )


def _proposal_item_to_slot_value(item: Any, problem: SlotProblemIR) -> tuple[str, str, str] | None:
    if isinstance(item, str):
        if len(problem.slots) != 1:
            return None
        slot = problem.slots[0]
        value = _normalize_single_slot_value(item, slot)
        if not value:
            return None
        return slot.slot_id, value, ""

    if not isinstance(item, dict):
        return None

    slot_id = (
        item.get("slot_id")
        or item.get("slot")
        or item.get("slot_name")
        or item.get("target_slot")
        or item.get("blank")
        or item.get("blank_id")
    )
    raw_value = (
        item.get("value")
        or item.get("option")
        or item.get("answer")
        or item.get("label")
        or item.get("option_label")
    )
    reason = str(item.get("reason") or item.get("rationale") or "").strip()

    if not slot_id and len(problem.slots) == 1:
        slot_id = problem.slots[0].slot_id
    if not slot_id or raw_value is None:
        return None

    raw_slot_id = str(slot_id).strip()
    slot: SlotSpec | None = None
    for candidate_slot in problem.slots:
        if (
            candidate_slot.slot_id == raw_slot_id
            or candidate_slot.slot_name == raw_slot_id
            or _norm_text(candidate_slot.slot_name) == _norm_text(raw_slot_id)
        ):
            slot = candidate_slot
            break
    if slot is None:
        return None

    value = _normalize_single_slot_value(raw_value, slot)
    if not value:
        value = _option_value_map(slot).get(_norm_text(raw_value), "")
    if not value:
        return None

    return slot.slot_id, value, reason


def parse_slot_challenger_proposal(text: str, problem: SlotProblemIR) -> tuple[str, list[dict[str, str]]]:
    raw = _strip_hidden_reasoning(text)
    obj, _ = _extract_json_any(raw)

    target_condition = ""
    raw_items: list[Any] = []

    if isinstance(obj, dict):
        target_condition = str(obj.get("target_condition") or "").strip()
        if isinstance(obj.get("challengers"), list):
            raw_items = list(obj.get("challengers") or [])
        elif isinstance(obj.get("challenger"), (dict, str)):
            raw_items = [obj.get("challenger")]
        elif any(key in obj for key in ("slot_id", "slot", "value", "option", "answer", "label")):
            raw_items = [obj]
    elif isinstance(obj, list):
        raw_items = list(obj)

    if not raw_items and obj is None and len(problem.slots) == 1:
        raw_items = [raw]

    proposals: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in raw_items:
        parsed = _proposal_item_to_slot_value(item, problem)
        if not parsed:
            continue
        slot_id, value, reason = parsed
        key = (slot_id, _norm_text(value))
        if key in seen:
            continue
        seen.add(key)
        proposals.append({"slot_id": slot_id, "value": value, "reason": reason})

    return target_condition, proposals


def _render_single_value(value: str, slot: SlotSpec) -> str:
    if slot.option_labels:
        label_map = _label_to_option_map(slot)
        reverse = {_norm_text(option): label for label, option in label_map.items()}
        label = reverse.get(_norm_text(value))
        if label:
            return label
    return value


def _render_assignment_text(assignment: dict[str, str], problem: SlotProblemIR) -> str:
    if problem.output_kind == "json_list":
        values = [assignment.get(slot.slot_id, "") for slot in problem.slots]
        return json.dumps(values, ensure_ascii=False, separators=(",", ":"))
    slot = problem.slots[0]
    value = assignment.get(slot.slot_id, "")
    if problem.output_kind == "single_label":
        return f"FINAL: {_render_single_value(value, slot)}"
    return f"FINAL: {value}"


def apply_slot_update(anchor_eval: SlotEval, challenger: SlotChallenger) -> SlotArtifact:
    new_assignment = dict(anchor_eval.artifact.assignment)
    new_assignment[challenger.slot_id] = challenger.challenger_value
    return SlotArtifact(
        raw_text=_render_assignment_text(new_assignment, anchor_eval.problem),
        assignment=new_assignment,
        normalized_assignment=_normalize_assignment(new_assignment, anchor_eval.problem),
        answer_source="slot_update",
        contract_ok=True,
        parser_confidence="high",
        artifact_signature=_hash_json({"assignment": _normalize_assignment(new_assignment, anchor_eval.problem)}),
    )


def apply_slot_certificate(anchor_eval: SlotEval, cert: SlotCertificate) -> SlotArtifact:
    new_assignment = dict(anchor_eval.artifact.assignment)
    if cert.challenger_assignment:
        new_assignment.update(_assignment_dict(cert.challenger_assignment))
    else:
        new_assignment[cert.slot_id] = cert.challenger_value
    return SlotArtifact(
        raw_text=_render_assignment_text(new_assignment, anchor_eval.problem),
        assignment=new_assignment,
        normalized_assignment=_normalize_assignment(new_assignment, anchor_eval.problem),
        answer_source="slot_certificate",
        contract_ok=True,
        parser_confidence="high",
        artifact_signature=_hash_json({"assignment": _normalize_assignment(new_assignment, anchor_eval.problem)}),
    )


def slot_assignment_final_answer(eval_obj: SlotEval) -> str:
    problem = eval_obj.problem
    assignment = eval_obj.artifact.assignment
    if problem.output_kind == "json_list":
        values = [assignment.get(slot.slot_id, "") for slot in problem.slots]
        return json.dumps(values, ensure_ascii=False, separators=(",", ":"))
    if problem.output_kind == "single_label":
        slot = problem.slots[0]
        return _render_single_value(assignment.get(slot.slot_id, ""), slot)
    if problem.output_kind == "single_value":
        slot = problem.slots[0]
        return assignment.get(slot.slot_id, "")
    return eval_obj.artifact.raw_text
