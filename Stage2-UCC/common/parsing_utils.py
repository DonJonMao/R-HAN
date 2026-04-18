from __future__ import annotations

import json
import re
from typing import List, Optional


def strip_hidden_reasoning(text: str) -> str:
    cleaned = re.sub(r"(?is)<think>.*?</think>", "", str(text or ""))
    cleaned = re.sub(r"(?im)^\s*</?think>\s*$", "", cleaned)
    return cleaned.strip()


def safe_json(text: str) -> Optional[dict]:
    cleaned = strip_hidden_reasoning(text).strip()
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass
    if "{" in cleaned and "}" in cleaned:
        snippet = cleaned[cleaned.find("{") : cleaned.rfind("}") + 1]
        try:
            parsed = json.loads(snippet)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None
    return None


def normalize_yes_no_output(text: str) -> str:
    cleaned = strip_hidden_reasoning(text).strip()
    match = re.search(r"\b(yes|no)\b", cleaned.lower())
    if match:
        return match.group(1)
    return cleaned


def extract_first_number(text: str) -> Optional[float]:
    match = re.search(r"-?\d+(?:\.\d+)?", str(text or "").replace(",", ""))
    if not match:
        return None
    try:
        return float(match.group(0))
    except Exception:
        return None


def extract_last_number(text: str) -> Optional[float]:
    matches = re.findall(r"-?\d+(?:\.\d+)?", str(text or "").replace(",", ""))
    if not matches:
        return None
    try:
        return float(matches[-1])
    except Exception:
        return None


def extract_python_code(text: str) -> str:
    cleaned = strip_hidden_reasoning(text).strip()
    fenced = re.findall(r"```(?:python)?\s*(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        cleaned = max(fenced, key=len).strip()
    return cleaned.strip()


def extract_sequence_numbers(text: str) -> List[int]:
    return [int(token) for token in re.findall(r"-?\d+", str(text or ""))]
