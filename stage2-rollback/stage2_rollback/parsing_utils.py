from __future__ import annotations

import re
from typing import Any, List, Optional


_BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
_FENCED_PY_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_JSON_SNIPPET_RE = re.compile(r"\{.*\}", re.DOTALL)
_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:/\d+)?")


def strip_hidden_reasoning(text: str) -> str:
    cleaned = str(text or "")
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<analysis>.*?</analysis>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()


def extract_boxed(text: str) -> Optional[str]:
    cleaned = strip_hidden_reasoning(text)
    matches = _BOXED_RE.findall(cleaned)
    if not matches:
        return None
    return matches[-1].strip()


def extract_python_code(text: str) -> str:
    cleaned = strip_hidden_reasoning(text)
    matches = _FENCED_PY_RE.findall(cleaned)
    if matches:
        matches = sorted(matches, key=len, reverse=True)
        return matches[0].strip()
    return cleaned.strip()


def safe_json_snippet(text: str) -> Optional[str]:
    cleaned = strip_hidden_reasoning(text)
    snippet = cleaned.strip()
    if snippet.startswith("{") and snippet.endswith("}"):
        return snippet
    match = _JSON_SNIPPET_RE.search(cleaned)
    if match:
        return match.group(0).strip()
    return None


def normalize_yes_no(text: str) -> Optional[str]:
    cleaned = strip_hidden_reasoning(text).strip().lower()
    if not cleaned:
        return None
    if cleaned in {"yes", "true"}:
        return "yes"
    if cleaned in {"no", "false"}:
        return "no"
    if cleaned.startswith("yes"):
        return "yes"
    if cleaned.startswith("no"):
        return "no"
    match = re.search(r"\b(yes|no|true|false)\b", cleaned)
    if not match:
        return None
    token = match.group(1)
    return "yes" if token in {"yes", "true"} else "no"


def extract_last_number(text: str) -> Optional[str]:
    cleaned = strip_hidden_reasoning(text)
    boxed = extract_boxed(cleaned)
    if boxed:
        return boxed
    matches = _NUMBER_RE.findall(cleaned)
    if not matches:
        return None
    return matches[-1]


def extract_sequence_numbers(text: str) -> List[int]:
    cleaned = strip_hidden_reasoning(text)
    return [int(token) for token in re.findall(r"-?\d+", cleaned)]


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
