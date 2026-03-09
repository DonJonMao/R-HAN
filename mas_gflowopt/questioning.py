from __future__ import annotations

import hashlib
from typing import Optional, Sequence


def canonicalize_question_text(question_text: str) -> str:
    # Preserve punctuation to avoid collisions like "c++" vs "c",
    # only normalize casing and whitespace.
    return " ".join(question_text.strip().lower().split())


def question_signature(
    question_text: Optional[str],
    question_vector: Optional[Sequence[float]],
    vector_decimals: int = 3,
) -> str:
    if question_text:
        canon = canonicalize_question_text(question_text)
        digest = hashlib.sha256(canon.encode("utf-8")).hexdigest()[:16]
        return f"text::{digest}"

    if question_vector:
        decimals = max(0, int(vector_decimals))
        vec_text = ",".join(f"{x:.{decimals}f}" for x in question_vector)
        digest = hashlib.sha256(vec_text.encode("utf-8")).hexdigest()[:16]
        return f"vec::{len(question_vector)}::{digest}"

    return "none"
