from __future__ import annotations

import ast
import hashlib
import math
import re
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple


DEDUCTIVE_DATASETS = {"gsm8k", "math"}


@dataclass
class DeductiveStep:
    step_id: int
    text: str
    equations: tuple[str, ...]
    values: tuple[str, ...]
    operation_kind: str | None
    span: tuple[int, int] | None


@dataclass
class DeductiveArtifact:
    raw_text: str
    dataset_name: str
    final_answer: str | None
    normalized_final_answer: str | None
    steps: tuple[DeductiveStep, ...]
    derivation_signature: str
    stable_prefix_signature: str | None
    parser_confidence: str
    final_source: str
    step_source: str
    contract_ok: bool


@dataclass
class DeductiveResidual:
    fatal: tuple[str, ...]
    local: tuple[str, ...]
    support: tuple[str, ...]
    first_bad_step: int | None
    verified_prefix_len: int
    residual_kind: str | None
    repair_locus: str | None
    final_consistent: bool
    residual_signature: str
    checked_equation_count: int = 0


@dataclass
class DeductiveEval:
    artifact: DeductiveArtifact
    residual: DeductiveResidual
    class_key: tuple
    rank_key: tuple


def dataset_name_from(dataset_profile: Any, metadata: Optional[dict]) -> str:
    payload = metadata or {}
    name = (
        payload.get("dataset_name")
        or payload.get("dataset")
        or payload.get("mas_dataset_name")
        or getattr(dataset_profile, "name", None)
        or ""
    )
    return str(name).strip().lower()


def _strip_hidden_reasoning(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", str(text or ""), flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()


def _signature(parts: Sequence[str]) -> str:
    normalized = "\n".join(" ".join(str(part).split()) for part in parts if str(part).strip())
    if not normalized:
        return ""
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]


def _strip_final_lines(text: str) -> str:
    kept: List[str] = []
    for line in str(text or "").splitlines():
        stripped = line.strip()
        if re.match(r"^(?:final|answer|答案)\s*[:：]", stripped, flags=re.IGNORECASE):
            continue
        if stripped.startswith("####"):
            continue
        kept.append(line)
    return "\n".join(kept)


def _extract_boxed_expression(text: str) -> str | None:
    marker = r"\boxed{"
    idx = str(text or "").rfind(marker)
    if idx == -1:
        return None
    start = idx + len(marker)
    depth = 1
    out: List[str] = []
    for ch in text[start:]:
        if ch == "{":
            depth += 1
            out.append(ch)
            continue
        if ch == "}":
            depth -= 1
            if depth == 0:
                value = "".join(out).strip()
                return value or None
            out.append(ch)
            continue
        out.append(ch)
    return None


_EXPLICIT_FINAL_RE = re.compile(
    r"^\s*(?:FINAL|Final|final|答案)\s*[:：]\s*(.+?)\s*$",
    flags=re.MULTILINE,
)

_GSM_HASH_RE = re.compile(r"####\s*([^\n]+)")

_ANSWER_LINE_RE = re.compile(
    r"^\s*(?:answer|Answer|答案)\s*[:：-]\s*(.+?)\s*$",
    flags=re.MULTILINE,
)

_ANSWER_ONLY_RE = re.compile(
    r"^\s*-?\d+(?:,\d{3})*(?:\.\d+)?(?:\s*/\s*-?\d+(?:\.\d+)?)?\s*$"
)


def _extract_final_answer_with_source(text: str, dataset_name: str) -> tuple[str | None, str]:
    cleaned = _strip_hidden_reasoning(text)
    if dataset_name == "math":
        boxed = _extract_boxed_expression(cleaned)
        if boxed:
            return boxed, "boxed"
    final_matches = _EXPLICIT_FINAL_RE.findall(cleaned)
    if final_matches:
        return final_matches[-1].strip().strip("$").strip(), "final_line"
    if dataset_name == "gsm8k":
        gsm_matches = _GSM_HASH_RE.findall(cleaned)
        if gsm_matches:
            return gsm_matches[-1].strip(), "gsm_hash"
        answer_matches = _ANSWER_LINE_RE.findall(cleaned)
        if answer_matches:
            return answer_matches[-1].strip().strip("$").strip(), "answer_line"
        if _ANSWER_ONLY_RE.fullmatch(cleaned.strip()):
            return cleaned.strip(), "answer_only"
        return None, "none"
    if dataset_name == "math":
        lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
        if lines:
            return lines[-1].strip().strip("$").strip(), "last_line"
    return None, "none"


def normalize_numeric_answer(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = text.strip().strip("$").strip()
    text = re.sub(r"[,，]", "", text)
    text = text.rstrip(".;")
    match = re.search(r"-?\d+(?:\.\d+)?(?:\s*/\s*-?\d+(?:\.\d+)?)?", text)
    if not match:
        return None
    token = match.group(0).replace(" ", "")
    try:
        if "/" in token:
            numerator, denominator = token.split("/", 1)
            number = float(numerator) / float(denominator)
        else:
            number = float(token)
    except Exception:
        return token
    if math.isfinite(number) and float(number).is_integer():
        return str(int(number))
    return f"{number:.12g}"


def normalize_math_answer(value: str | None) -> str | None:
    if value is None:
        return None
    expr = str(value).strip()
    if not expr:
        return None
    boxed = _extract_boxed_expression(expr)
    if boxed:
        expr = boxed
    expr = expr.replace("$", "").replace("\\left", "").replace("\\right", "")
    expr = expr.replace("\\dfrac", "\\frac")
    expr = expr.replace("\\!", "").replace(",", "")
    expr = re.sub(r"\s+", "", expr)
    expr = expr.strip(".;")
    return expr or None


def _normalize_final_answer(value: str | None, dataset_name: str) -> str | None:
    if dataset_name == "gsm8k":
        return normalize_numeric_answer(value)
    if dataset_name == "math":
        return normalize_math_answer(value)
    return str(value).strip() if value is not None else None


def _extract_values(text: str, dataset_name: str) -> tuple[str, ...]:
    if dataset_name == "math":
        pattern = r"-?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?|\\frac\{[^{}]+\}\{[^{}]+\}|\\sqrt\{[^{}]+\}|[\[\(][^()\[\]]+[\]\)]"
    else:
        pattern = r"\$?\s*-?\d+(?:,\d{3})*(?:\.\d+)?%?(?:\s*/\s*-?\d+(?:\.\d+)?)?"
    values = [" ".join(item.split()).strip() for item in re.findall(pattern, text)]
    return tuple(item for item in values if item)


def _extract_equations(text: str) -> tuple[str, ...]:
    equations: List[str] = []
    for chunk in re.split(r"[;\n]", text):
        if "=" not in chunk:
            continue
        candidate = chunk.strip()
        candidate = re.sub(r"^[A-Za-z\s,:-]*", "", candidate).strip()
        if candidate.startswith("="):
            candidate = candidate.lstrip("= ").strip()
        candidate = candidate.strip(".。; ")
        if candidate.count("=") >= 2:
            left, rest = candidate.split("=", 1)
            if _safe_arithmetic_eval(left) is None:
                candidate = rest.strip()
        if "=" in candidate:
            equations.append(candidate)
    return tuple(equations)


def _operation_kind(text: str) -> str | None:
    if "=" in text:
        if any(symbol in text for symbol in ("*", "×")):
            return "multiplication"
        if any(symbol in text for symbol in ("/", "÷")):
            return "division"
        if "+" in text:
            return "addition"
        if "-" in text:
            return "subtraction"
        return "equation"
    if any(symbol in text for symbol in ("*", "×", "/", "÷", "+", "-")):
        return "arithmetic"
    return None


def _parse_numbered_steps_with_source(text: str, dataset_name: str) -> tuple[tuple[DeductiveStep, ...], str]:
    cleaned = _strip_hidden_reasoning(text)
    has_solution_header = bool(
        re.search(r"^\s*SOLUTION\s*[:：]?\s*$", cleaned, flags=re.MULTILINE | re.IGNORECASE)
    )
    matches = list(re.finditer(r"(?m)^\s*(\d+)[\.\)]\s+(.+?)\s*$", cleaned))
    steps: List[DeductiveStep] = []
    for match in matches:
        body = match.group(2).strip()
        if re.match(r"^(?:FINAL|Final|final)\s*[:：]", body):
            continue
        steps.append(
            DeductiveStep(
                step_id=int(match.group(1)),
                text=body,
                equations=_extract_equations(body),
                values=_extract_values(body, dataset_name),
                operation_kind=_operation_kind(body),
                span=(match.start(), match.end()),
            )
        )
    if steps:
        return tuple(steps), "solution_numbered" if has_solution_header else "numbered"

    fallback_lines = []
    for line in cleaned.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r"^(?:SOLUTION|FINAL|Final|final|答案)\s*[:：]?", stripped):
            continue
        if stripped.startswith("####"):
            continue
        fallback_lines.append(stripped)
    for index, line in enumerate(fallback_lines, start=1):
        steps.append(
            DeductiveStep(
                step_id=index,
                text=line,
                equations=_extract_equations(line),
                values=_extract_values(line, dataset_name),
                operation_kind=_operation_kind(line),
                span=None,
            )
        )
    if steps:
        return tuple(steps), "fallback_lines"
    return tuple(), "none"


def parse_deductive_artifact(text: str, dataset_name: str) -> DeductiveArtifact:
    dataset = str(dataset_name or "").strip().lower()
    raw_text = _strip_hidden_reasoning(text)
    final_answer, final_source = _extract_final_answer_with_source(raw_text, dataset)
    normalized_final_answer = _normalize_final_answer(final_answer, dataset)
    steps, step_source = _parse_numbered_steps_with_source(raw_text, dataset)
    has_solution_header = bool(
        re.search(r"^\s*SOLUTION\s*[:：]?\s*$", raw_text, flags=re.MULTILINE | re.IGNORECASE)
    )
    has_equation = any(step.equations for step in steps)
    explicit_final = final_source in {"final_line", "gsm_hash", "boxed"}
    contract_ok = bool(has_solution_header and explicit_final and steps)
    if contract_ok and has_equation:
        confidence = "high"
    elif explicit_final and steps and has_equation:
        confidence = "medium"
    elif final_source == "answer_only":
        confidence = "answer_only"
    elif normalized_final_answer:
        confidence = "low"
    else:
        confidence = "low"
    derivation_parts = [step.text for step in steps] or [_strip_final_lines(raw_text)]
    return DeductiveArtifact(
        raw_text=raw_text,
        dataset_name=dataset,
        final_answer=final_answer,
        normalized_final_answer=normalized_final_answer,
        steps=steps,
        derivation_signature=_signature(derivation_parts),
        stable_prefix_signature=None,
        parser_confidence=confidence,
        final_source=final_source,
        step_source=step_source,
        contract_ok=contract_ok,
    )


def _clean_arithmetic_expr(expr: str) -> str | None:
    text = str(expr or "")
    text = text.replace(",", "")
    text = text.replace("$", "")
    text = text.replace("×", "*").replace("÷", "/").replace("^", "**")
    text = text.replace("%", "/100")
    text = text.strip()
    if not text:
        return None
    if re.search(r"[A-Za-z\\{}]", text):
        return None
    if not re.fullmatch(r"[0-9eE\.\+\-\*/\(\)\s]+", text):
        return None
    return text


def _safe_arithmetic_eval(expr: str) -> float | None:
    cleaned = _clean_arithmetic_expr(expr)
    if cleaned is None:
        return None
    try:
        tree = ast.parse(cleaned, mode="eval")
    except Exception:
        return None

    def visit(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
            left = visit(node.left)
            right = visit(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            return left**right
        raise ValueError("unsupported arithmetic expression")

    try:
        value = visit(tree)
    except Exception:
        return None
    return float(value) if math.isfinite(float(value)) else None


def _equation_sides(equation: str) -> tuple[str, str] | None:
    if "=" not in equation:
        return None
    left, right = equation.split("=", 1)
    right = right.split("=", 1)[0]
    left = left.strip()
    right = right.strip()
    if not left or not right:
        return None
    return left, right


def _equation_numeric_values(equation: str) -> tuple[float, float] | None:
    sides = _equation_sides(equation)
    if sides is None:
        return None
    left_value = _safe_arithmetic_eval(sides[0])
    right_value = _safe_arithmetic_eval(sides[1])
    if left_value is None or right_value is None:
        return None
    return left_value, right_value


def _numbers_for_overlap(text: str) -> set[str]:
    values = set()
    for raw in re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", str(text or "")):
        normalized = normalize_numeric_answer(raw)
        if normalized is not None:
            values.add(normalized)
    return values


def _format_number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.12g}"


def _verify_gsm8k_artifact(question_text: str, artifact: DeductiveArtifact) -> DeductiveResidual:
    fatal: List[str] = []
    local: List[str] = []
    support: List[str] = []
    first_bad_step: int | None = None
    verified_prefix_len = 0
    residual_kind: str | None = None
    repair_locus: str | None = None
    final_consistent = bool(artifact.normalized_final_answer)
    last_verified_value: str | None = None
    checked_equation_count = 0

    if artifact.normalized_final_answer:
        support.append(f"final_source:{artifact.final_source}")
    else:
        fatal.append("missing_final_answer")
        residual_kind = "missing_final_answer"
        repair_locus = "final"
        final_consistent = False

    for step_index, step in enumerate(artifact.steps):
        step_checked = False
        mismatch = False
        for equation in step.equations:
            values = _equation_numeric_values(equation)
            if values is None:
                continue
            step_checked = True
            checked_equation_count += 1
            left_value, right_value = values
            if abs(left_value - right_value) > 1e-6:
                mismatch = True
                break
            last_verified_value = _format_number(right_value)
        if mismatch:
            fatal.append("arithmetic_mismatch")
            residual_kind = residual_kind or "arithmetic_mismatch"
            first_bad_step = step.step_id
            repair_locus = f"step_{step.step_id}_suffix"
            verified_prefix_len = step_index
            final_consistent = False
            break
        if step_checked:
            support.append("arithmetic_checked")
            verified_prefix_len = step_index + 1
        elif step.operation_kind or step.values:
            local.append("semantic_unverified_step")

    if not fatal and artifact.normalized_final_answer and last_verified_value is not None:
        if normalize_numeric_answer(last_verified_value) != artifact.normalized_final_answer:
            fatal.append("final_inconsistent_with_derivation")
            residual_kind = "final_inconsistent_with_derivation"
            repair_locus = "final"
            final_consistent = False
        else:
            support.append("final_matches_derivation")
            final_consistent = True

    if artifact.final_source == "answer_only":
        local.append("answer_only_no_derivation")

    if artifact.final_source == "none":
        fatal.append("missing_final_answer")
        residual_kind = residual_kind or "missing_final_answer"
        repair_locus = repair_locus or "final"
        final_consistent = False

    if not fatal and checked_equation_count == 0 and artifact.final_source != "answer_only" and artifact.steps:
        local.append("no_checked_equations")

    if not fatal and artifact.steps:
        question_values = _numbers_for_overlap(question_text)
        artifact_values = _numbers_for_overlap(artifact.raw_text)
        if question_values and not question_values.issubset(artifact_values):
            local.append("unconsumed_question_quantity")
        if not any(step.equations for step in artifact.steps):
            final_value = {artifact.normalized_final_answer} if artifact.normalized_final_answer else set()
            if artifact_values - question_values - final_value:
                local.append("extraneous_quantity")

    if residual_kind is None and local:
        residual_kind = local[0]
        repair_locus = repair_locus or "derivation"
    if residual_kind is None:
        residual_kind = "stable"
        repair_locus = "stable"

    residual_signature = _signature(
        [
            ",".join(sorted(set(fatal))),
            ",".join(sorted(set(local))),
            ",".join(sorted(set(support))),
            str(first_bad_step),
            str(verified_prefix_len),
            str(final_consistent),
            str(checked_equation_count),
            residual_kind,
            repair_locus or "",
        ]
    )
    return DeductiveResidual(
        fatal=tuple(sorted(set(fatal))),
        local=tuple(sorted(set(local))),
        support=tuple(sorted(set(support))),
        first_bad_step=first_bad_step,
        verified_prefix_len=verified_prefix_len,
        residual_kind=residual_kind,
        repair_locus=repair_locus,
        final_consistent=final_consistent,
        residual_signature=residual_signature,
        checked_equation_count=checked_equation_count,
    )


def _try_sympy_equivalent(left: str, right: str) -> bool | None:
    try:
        import sympy as sp  # type: ignore
        from sympy.parsing.sympy_parser import parse_expr  # type: ignore
    except Exception:
        return None

    def clean(expr: str) -> str:
        value = expr.strip().replace("^", "**")
        value = value.replace("\\cdot", "*")
        value = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", value)
        value = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", value)
        value = value.replace("{", "(").replace("}", ")")
        return value

    try:
        left_expr = parse_expr(clean(left), evaluate=False)
        right_expr = parse_expr(clean(right), evaluate=False)
        return bool(sp.simplify(left_expr - right_expr) == 0)
    except Exception:
        return None


def _verify_math_artifact(question_text: str, artifact: DeductiveArtifact) -> DeductiveResidual:
    del question_text
    fatal: List[str] = []
    local: List[str] = []
    support: List[str] = []
    first_bad_step: int | None = None
    verified_prefix_len = 0
    residual_kind: str | None = None
    repair_locus: str | None = None
    final_consistent = bool(artifact.normalized_final_answer)
    saw_parseable_equation = False
    last_expression: str | None = None
    checked_equation_count = 0

    has_boxed = artifact.final_source == "boxed"
    has_final_line = artifact.final_source == "final_line"
    if not artifact.final_answer:
        fatal.append("missing_boxed_answer")
        residual_kind = "missing_boxed_answer"
        repair_locus = "final"
        final_consistent = False
    elif not artifact.normalized_final_answer:
        fatal.append("empty_math_expression")
        residual_kind = "empty_math_expression"
        repair_locus = "final"
        final_consistent = False
    elif has_boxed or has_final_line:
        support.append("final_answer_present")

    for step_index, step in enumerate(artifact.steps):
        mismatch = False
        for equation in step.equations:
            sides = _equation_sides(equation)
            if sides is None:
                continue
            left, right = sides
            last_expression = right
            numeric_values = _equation_numeric_values(equation)
            if numeric_values is not None:
                saw_parseable_equation = True
                checked_equation_count += 1
                if abs(numeric_values[0] - numeric_values[1]) > 1e-6:
                    fatal.append("parseable_algebra_mismatch")
                    fatal.append("algebra_non_equivalence")
                    mismatch = True
                    break
                support.append("parseable_algebra_checked")
                continue
            equivalent = _try_sympy_equivalent(left, right)
            if equivalent is None:
                local.append("semantic_unverified")
                continue
            saw_parseable_equation = True
            checked_equation_count += 1
            if not equivalent:
                fatal.append("parseable_algebra_mismatch")
                fatal.append("algebra_non_equivalence")
                mismatch = True
                break
            support.append("parseable_algebra_checked")
        if mismatch:
            residual_kind = "algebra_non_equivalence"
            first_bad_step = step.step_id
            repair_locus = f"step_{step.step_id}_suffix"
            verified_prefix_len = step_index
            final_consistent = False
            break
        verified_prefix_len = step_index + 1

    if (
        not fatal
        and artifact.normalized_final_answer
        and last_expression is not None
        and artifact.parser_confidence != "low"
    ):
        normalized_last = normalize_math_answer(last_expression)
        if normalized_last and normalized_last != artifact.normalized_final_answer:
            fatal.append("final_inconsistent_with_last_expression")
            residual_kind = "final_inconsistent_with_last_expression"
            repair_locus = "final"
            final_consistent = False
        elif normalized_last:
            support.append("final_matches_last_expression")
            final_consistent = True

    if not fatal and not saw_parseable_equation and artifact.steps:
        local.append("semantic_unverified")

    if residual_kind is None and local:
        residual_kind = local[0]
        repair_locus = repair_locus or "derivation"
    if residual_kind is None:
        residual_kind = "stable"
        repair_locus = "stable"

    residual_signature = _signature(
        [
            ",".join(sorted(set(fatal))),
            ",".join(sorted(set(local))),
            ",".join(sorted(set(support))),
            str(first_bad_step),
            str(verified_prefix_len),
            str(final_consistent),
            str(checked_equation_count),
            residual_kind,
            repair_locus or "",
        ]
    )
    return DeductiveResidual(
        fatal=tuple(sorted(set(fatal))),
        local=tuple(sorted(set(local))),
        support=tuple(sorted(set(support))),
        first_bad_step=first_bad_step,
        verified_prefix_len=verified_prefix_len,
        residual_kind=residual_kind,
        repair_locus=repair_locus,
        final_consistent=final_consistent,
        residual_signature=residual_signature,
        checked_equation_count=checked_equation_count,
    )


def verify_deductive_artifact(
    question_text: str,
    artifact: DeductiveArtifact,
    dataset_profile: Any,
    metadata: dict,
) -> DeductiveResidual:
    del dataset_profile
    dataset = str(artifact.dataset_name or (metadata or {}).get("dataset_name") or "").lower()
    if dataset == "gsm8k":
        return _verify_gsm8k_artifact(question_text, artifact)
    if dataset == "math":
        return _verify_math_artifact(question_text, artifact)
    return DeductiveResidual(
        fatal=("unsupported_deductive_dataset",),
        local=(),
        support=(),
        first_bad_step=None,
        verified_prefix_len=0,
        residual_kind="unsupported_deductive_dataset",
        repair_locus="parse",
        final_consistent=False,
        residual_signature=_signature(["unsupported_deductive_dataset", dataset]),
    )


def deductive_class_key(eval: DeductiveEval) -> tuple:
    return (
        eval.artifact.normalized_final_answer,
        eval.residual.residual_kind,
        eval.residual.first_bad_step,
        eval.artifact.final_source,
        bool(eval.artifact.contract_ok),
    )


def _rank_key(artifact: DeductiveArtifact, residual: DeductiveResidual) -> tuple:
    explicit_final = artifact.final_source in {"final_line", "gsm_hash", "boxed"}
    return (
        -len(residual.fatal),
        int(explicit_final),
        int(artifact.contract_ok),
        int(residual.final_consistent),
        -len(residual.local),
        int(residual.checked_equation_count),
        int(bool(artifact.normalized_final_answer)),
    )


def make_deductive_eval(
    artifact: DeductiveArtifact,
    residual: DeductiveResidual,
    entry: Optional[dict] = None,
) -> DeductiveEval:
    del entry
    stable_prefix = artifact.steps[: max(0, int(residual.verified_prefix_len))]
    artifact = replace(
        artifact,
        stable_prefix_signature=_signature([step.text for step in stable_prefix]) or None,
    )
    placeholder = DeductiveEval(
        artifact=artifact,
        residual=residual,
        class_key=(),
        rank_key=_rank_key(artifact, residual),
    )
    return replace(placeholder, class_key=deductive_class_key(placeholder))


def introduces_earlier_error(new: DeductiveEval, old: DeductiveEval) -> bool:
    new_step = new.residual.first_bad_step
    old_step = old.residual.first_bad_step
    if old_step is None and new_step is not None:
        return True
    if old_step is not None and new_step is not None and int(new_step) < int(old_step):
        return True
    return int(new.residual.verified_prefix_len) < int(old.residual.verified_prefix_len)


_DETERMINISTIC_FATALS = {
    "arithmetic_mismatch",
    "final_inconsistent_with_derivation",
    "final_inconsistent_with_last_expression",
    "parseable_algebra_mismatch",
    "algebra_non_equivalence",
    "missing_final_answer",
    "missing_boxed_answer",
    "empty_math_expression",
}


def _has_deterministic_fatal(eval: DeductiveEval) -> bool:
    return any(item in _DETERMINISTIC_FATALS for item in eval.residual.fatal) or (
        eval.residual.residual_kind in _DETERMINISTIC_FATALS
    )


def _same_answer(new: DeductiveEval, old: DeductiveEval) -> bool:
    return bool(new.artifact.normalized_final_answer) and (
        new.artifact.normalized_final_answer == old.artifact.normalized_final_answer
    )


def _explicit_final(eval: DeductiveEval) -> bool:
    return eval.artifact.final_source in {"final_line", "gsm_hash", "boxed"}


def deductive_dominates(new: DeductiveEval, old: DeductiveEval) -> bool:
    same_answer = _same_answer(new, old)
    old_has_det_fatal = _has_deterministic_fatal(old)

    if same_answer:
        if len(new.residual.fatal) < len(old.residual.fatal):
            return True
        if (
            len(new.residual.fatal) == len(old.residual.fatal)
            and new.residual.final_consistent
            and not introduces_earlier_error(new, old)
            and new.residual.checked_equation_count > old.residual.checked_equation_count
        ):
            return True
        return False

    if not old_has_det_fatal:
        return False

    if not _explicit_final(new):
        return False

    if len(new.residual.fatal) < len(old.residual.fatal) and not introduces_earlier_error(new, old):
        return True

    if (
        old.residual.residual_kind
        in {"missing_final_answer", "final_inconsistent", "final_inconsistent_with_derivation", "final_inconsistent_with_last_expression"}
        and new.residual.final_consistent
        and new.artifact.derivation_signature == old.artifact.derivation_signature
    ):
        return True

    return False


def answer_only_from_artifact(artifact: DeductiveArtifact) -> str:
    if artifact.dataset_name == "gsm8k":
        return artifact.normalized_final_answer or artifact.final_answer or ""
    if artifact.dataset_name == "math":
        return artifact.normalized_final_answer or artifact.final_answer or ""
    return artifact.final_answer or artifact.normalized_final_answer or ""


def verified_prefix_text(artifact: DeductiveArtifact, verified_prefix_len: int) -> str:
    lines = []
    for step in artifact.steps[: max(0, int(verified_prefix_len))]:
        lines.append(f"{step.step_id}. {step.text}")
    return "\n".join(lines)


def bad_step_text(artifact: DeductiveArtifact, first_bad_step: int | None) -> str:
    if first_bad_step is None:
        return ""
    for step in artifact.steps:
        if step.step_id == first_bad_step:
            return f"{step.step_id}. {step.text}"
    return ""


def repair_operator_for_residual(eval: DeductiveEval) -> str | None:
    kind = str(eval.residual.residual_kind or "")
    if kind == "arithmetic_mismatch":
        return "deductive_arithmetic_suffix_patch"
    if kind in {"missing_final_answer", "final_inconsistent", "final_inconsistent_with_derivation", "final_inconsistent_with_last_expression", "missing_boxed_answer", "empty_math_expression"}:
        return "deductive_final_extraction_patch"
    if kind in {"algebra_non_equivalence", "parseable_algebra_mismatch"} and eval.artifact.parser_confidence != "low":
        return "deductive_algebra_suffix_patch"
    if kind in {"unsupported_transition", "extraneous_quantity", "unconsumed_question_quantity"}:
        return "deductive_transition_suffix_patch"
    return None


def build_deductive_repair_prompt(
    *,
    operator_type: str,
    question_text: str,
    artifact: DeductiveArtifact,
    residual: DeductiveResidual,
) -> str:
    prefix = verified_prefix_text(artifact, residual.verified_prefix_len)
    bad_step = bad_step_text(artifact, residual.first_bad_step)
    issue = residual.residual_kind or "none"
    rendered_prefix = prefix or "(none)"
    if operator_type == "deductive_arithmetic_suffix_patch":
        return (
            "You are repairing a mathematical derivation.\n\n"
            f"Problem:\n{question_text}\n\n"
            f"Verified prefix. Do not change these steps:\n{rendered_prefix}\n\n"
            f"First inconsistent step:\n{bad_step or residual.repair_locus or 'unknown'}\n\n"
            f"Verifier residual:\n{issue}\n\n"
            f"Rewrite only from Step {int(residual.verified_prefix_len) + 1} onward.\n"
            "Keep the verified prefix unchanged.\n"
            "Return:\n\n"
            "SOLUTION:\n"
            "1. ...\n"
            "2. ...\n"
            "...\n\n"
            "FINAL: <answer>\n"
        )
    if operator_type == "deductive_final_extraction_patch":
        return (
            "The derivation steps are internally consistent.\n"
            "Only the final answer line is missing or inconsistent.\n\n"
            f"Problem:\n{question_text}\n\n"
            f"Derivation:\n{artifact.raw_text}\n\n"
            "Use the last verified derivation value and produce:\n"
            "FINAL: <answer>\n"
        )
    if operator_type == "deductive_algebra_suffix_patch":
        step_id = residual.first_bad_step if residual.first_bad_step is not None else int(residual.verified_prefix_len) + 1
        return (
            f"The algebraic transformation at Step {step_id} is not equivalent to the previous expression.\n"
            f"Problem:\n{question_text}\n\n"
            f"Keep Steps 1-{max(0, int(step_id) - 1)}:\n{rendered_prefix}\n\n"
            f"Current derivation:\n{artifact.raw_text}\n\n"
            f"Rewrite Step {step_id} onward using equivalent transformations.\n"
            "Return SOLUTION and FINAL.\n"
        )
    return (
        "You are repairing a mathematical derivation with a high-risk local transition issue.\n\n"
        f"Problem:\n{question_text}\n\n"
        f"Verified prefix. Do not change these steps:\n{rendered_prefix}\n\n"
        f"Verifier residual:\n{issue}\n\n"
        "Generate a challenger repair candidate, but preserve all verified steps.\n"
        "Return SOLUTION and FINAL.\n"
    )
