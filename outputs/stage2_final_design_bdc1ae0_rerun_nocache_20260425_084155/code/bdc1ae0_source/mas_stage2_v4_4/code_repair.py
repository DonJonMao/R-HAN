from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from mas_treesearch.evaluator import MultiFidelityEvaluator


@dataclass(frozen=True)
class CodeRepairEval:
    code_text: str
    syntax_ok: bool
    entry_point_ok: bool
    passed: int
    total: int
    failure_kind: str
    failing_examples: Tuple[str, ...]
    stdout: str = ""
    stderr: str = ""
    exec_error: str = ""

    @property
    def fully_passed(self) -> bool:
        return self.total > 0 and self.passed >= self.total

    @property
    def accuracy(self) -> float:
        if self.total <= 0:
            return 0.0
        return float(self.passed) / float(self.total)

    @property
    def dominance_tuple(self) -> Tuple[int, int, int]:
        return (int(self.syntax_ok), int(self.entry_point_ok), int(self.passed))

    @property
    def rank_key(self) -> Tuple[int, int, int, int, int]:
        return (
            int(self.fully_passed),
            int(self.syntax_ok),
            int(self.entry_point_ok),
            int(self.passed),
            -len(self.failing_examples),
        )

    def dominates(self, other: "CodeRepairEval") -> bool:
        mine = self.dominance_tuple
        theirs = other.dominance_tuple
        return all(left >= right for left, right in zip(mine, theirs)) and any(
            left > right for left, right in zip(mine, theirs)
        )


def _entry_point_defined(candidate_code: str, entry_point: str) -> bool:
    if not entry_point:
        return True
    return bool(re.search(rf"def\s+{re.escape(entry_point)}\s*\(", candidate_code))


def _extract_marker_payload(stdout: str, marker: str) -> Optional[Dict[str, Any]]:
    for line in reversed(stdout.splitlines()):
        if line.startswith(marker):
            payload = line[len(marker) :].strip()
            try:
                parsed = json.loads(payload)
            except json.JSONDecodeError:
                return None
            if isinstance(parsed, dict):
                return parsed
            return None
    return None


def evaluate_code_candidate(
    candidate_text: str,
    metadata: Optional[dict],
    *,
    timeout_s: float = 8.0,
    max_failed_examples: int = 3,
) -> CodeRepairEval:
    candidate_code = MultiFidelityEvaluator._extract_python_code(candidate_text)
    if not candidate_code:
        return CodeRepairEval(
            code_text="",
            syntax_ok=False,
            entry_point_ok=False,
            passed=0,
            total=0,
            failure_kind="empty_code",
            failing_examples=(),
        )

    data = dict(metadata or {})
    entry_point = str(data.get("entry_point") or "").strip()
    mbpp_tests = list(data.get("test_list") or [])
    humaneval_test = data.get("test")
    setup_code = str(data.get("test_setup_code") or "")

    syntax_ok = False
    entry_point_ok = False
    exec_error = ""
    try:
        compile(candidate_code, "<candidate>", "exec")
        syntax_ok = True
        entry_point_ok = _entry_point_defined(candidate_code, entry_point)
    except Exception as exc:  # pragma: no cover - parser specific wording
        exec_error = str(exc)

    if not syntax_ok:
        return CodeRepairEval(
            code_text=candidate_code,
            syntax_ok=False,
            entry_point_ok=False,
            passed=0,
            total=0,
            failure_kind="syntax_error",
            failing_examples=((exec_error or "syntax_error"),),
            exec_error=exec_error,
        )
    if not entry_point_ok:
        return CodeRepairEval(
            code_text=candidate_code,
            syntax_ok=True,
            entry_point_ok=False,
            passed=0,
            total=0,
            failure_kind="entry_point_missing",
            failing_examples=((f"missing entry point: {entry_point}" if entry_point else "entry point missing"),),
        )

    marker = "__V43_CODE_REPAIR__"
    script_lines: List[str] = [
        "import json",
        "import math",
        "import itertools",
        "import functools",
        "import collections",
        "import heapq",
        "import bisect",
        candidate_code,
    ]

    if mbpp_tests:
        tests = [str(expr) for expr in mbpp_tests]
        script_lines.append(setup_code)
        script_lines.append("results = []")
        for expr in tests:
            escaped = json.dumps(expr)
            script_lines.extend(
                [
                    f"_expr = {escaped}",
                    "try:",
                    "    exec(_expr, globals(), globals())",
                    "    results.append({'expr': _expr, 'passed': True})",
                    "except Exception as exc:",
                    "    results.append({'expr': _expr, 'passed': False, 'error': repr(exc)})",
                ]
            )
        script_lines.append(f"print('{marker}' + json.dumps({{'results': results}}))")
    elif isinstance(humaneval_test, str) and humaneval_test.strip() and entry_point:
        escaped = json.dumps(humaneval_test)
        script_lines.extend(
            [
                f"_test_code = {escaped}",
                "try:",
                "    exec(_test_code, globals(), globals())",
                f"    check({entry_point})",
                f"    print('{marker}' + json.dumps({{'passed': True}}))",
                "except Exception as exc:",
                f"    print('{marker}' + json.dumps({{'passed': False, 'error': repr(exc)}}))",
            ]
        )
    else:
        return CodeRepairEval(
            code_text=candidate_code,
            syntax_ok=True,
            entry_point_ok=True,
            passed=0,
            total=0,
            failure_kind="no_dataset_tests",
            failing_examples=(),
        )

    try:
        proc = subprocess.run(
            [sys.executable, "-c", "\n".join(script_lines)],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return CodeRepairEval(
            code_text=candidate_code,
            syntax_ok=True,
            entry_point_ok=True,
            passed=0,
            total=max(1, len(mbpp_tests)),
            failure_kind="timeout",
            failing_examples=(f"timeout>{timeout_s:.1f}s",),
        )

    stdout = proc.stdout.strip()
    stderr = proc.stderr.strip()
    payload = _extract_marker_payload(stdout, marker)
    if payload is None:
        return CodeRepairEval(
            code_text=candidate_code,
            syntax_ok=True,
            entry_point_ok=True,
            passed=0,
            total=max(1, len(mbpp_tests)),
            failure_kind="execution_error",
            failing_examples=((stderr or stdout or "execution_error"),),
            stdout=stdout,
            stderr=stderr,
        )

    if "results" in payload:
        results = payload.get("results")
        if not isinstance(results, list):
            results = []
        passed = 0
        failures: List[str] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            if bool(item.get("passed", False)):
                passed += 1
                continue
            expr = str(item.get("expr", "")).strip()
            error = str(item.get("error", "")).strip()
            if len(failures) < max_failed_examples:
                failures.append(f"{expr} -> {error or 'AssertionError'}")
        total = max(1, len(results))
        return CodeRepairEval(
            code_text=candidate_code,
            syntax_ok=True,
            entry_point_ok=True,
            passed=passed,
            total=total,
            failure_kind="" if passed >= total else "visible_test_failure",
            failing_examples=tuple(failures),
            stdout=stdout,
            stderr=stderr,
        )

    passed = bool(payload.get("passed", False))
    error = str(payload.get("error", "")).strip()
    return CodeRepairEval(
        code_text=candidate_code,
        syntax_ok=True,
        entry_point_ok=True,
        passed=1 if passed else 0,
        total=1,
        failure_kind="" if passed else "dataset_test_failure",
        failing_examples=tuple([error] if error else []),
        stdout=stdout,
        stderr=stderr,
    )


def build_failure_summary(
    evaluation: CodeRepairEval,
    *,
    metadata: Optional[dict],
    max_examples: int = 3,
) -> str:
    entry_point = str((metadata or {}).get("entry_point") or "").strip()
    lines = [
        f"syntax_ok={int(evaluation.syntax_ok)}",
        f"entry_point_ok={int(evaluation.entry_point_ok)}",
        f"passed={evaluation.passed}",
        f"total={evaluation.total}",
        f"failure_kind={evaluation.failure_kind or 'none'}",
    ]
    if entry_point:
        lines.append(f"required_entry_point={entry_point}")
    if evaluation.exec_error:
        lines.append(f"exec_error={evaluation.exec_error}")
    for item in evaluation.failing_examples[:max_examples]:
        lines.append(f"failed_test={item}")
    return "\n".join(lines)
