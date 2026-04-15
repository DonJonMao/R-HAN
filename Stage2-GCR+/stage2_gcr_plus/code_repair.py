from __future__ import annotations

import ast
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


def _extract_code_lines(candidate_text: str) -> List[str]:
    code = MultiFidelityEvaluator._extract_python_code(candidate_text)
    return code.splitlines() if code else []


def _parse_traceback_frame(*texts: str) -> Dict[str, Any]:
    joined = "\n".join(str(item or "") for item in texts if str(item or "").strip())
    if not joined:
        return {}
    match = re.search(
        r'File\s+"<candidate>",\s+line\s+(?P<line>\d+)(?:,\s+in\s+(?P<func>[^\n]+))?',
        joined,
        flags=re.IGNORECASE,
    )
    if not match:
        return {}
    line_no = int(match.group("line"))
    function = str(match.group("func") or "").strip()
    code_match = re.search(rf'File\s+"<candidate>",\s+line\s+{line_no}[^\n]*\n(?P<code>[ \t]*.+)', joined)
    return {
        "function": function,
        "line": line_no,
        "code": str(code_match.group("code")).strip() if code_match else "",
    }


def _function_signature_text(node: ast.AST) -> str:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return ""
    parts: List[str] = []
    positional = list(node.args.posonlyargs) + list(node.args.args)
    defaults = [None] * (len(positional) - len(node.args.defaults)) + list(node.args.defaults)
    for arg, default in zip(positional, defaults):
        item = arg.arg
        if arg.annotation is not None:
            item += f": {ast.unparse(arg.annotation)}"
        if default is not None:
            item += f" = {ast.unparse(default)}"
        parts.append(item)
    if node.args.vararg is not None:
        parts.append(f"*{node.args.vararg.arg}")
    elif node.args.kwonlyargs:
        parts.append("*")
    for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults):
        item = arg.arg
        if arg.annotation is not None:
            item += f": {ast.unparse(arg.annotation)}"
        if default is not None:
            item += f" = {ast.unparse(default)}"
        parts.append(item)
    if node.args.kwarg is not None:
        parts.append(f"**{node.args.kwarg.arg}")
    signature = f"{node.name}({', '.join(parts)})"
    if node.returns is not None:
        signature += f" -> {ast.unparse(node.returns)}"
    return signature


def _find_enclosing_function(tree: ast.AST, line_no: int, entry_point: str) -> Optional[ast.AST]:
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and hasattr(node, "lineno") and hasattr(node, "end_lineno")
    ]
    if entry_point:
        for node in functions:
            if node.name == entry_point:
                if line_no <= 0 or (int(node.lineno) <= line_no <= int(node.end_lineno)):
                    return node
    if line_no > 0:
        candidates = [
            node
            for node in functions
            if int(node.lineno) <= line_no <= int(node.end_lineno)
        ]
        if candidates:
            candidates.sort(key=lambda node: (int(node.end_lineno) - int(node.lineno), int(node.lineno)))
            return candidates[0]
    return functions[0] if functions else None


def _statement_for_line(func_node: ast.AST, line_no: int) -> Optional[ast.stmt]:
    statements = [
        node
        for node in ast.walk(func_node)
        if isinstance(node, ast.stmt) and hasattr(node, "lineno") and hasattr(node, "end_lineno")
    ]
    if line_no > 0:
        matches = [
            node
            for node in statements
            if int(node.lineno) <= line_no <= int(node.end_lineno)
        ]
        if matches:
            matches.sort(key=lambda node: (int(node.end_lineno) - int(node.lineno), int(node.lineno)))
            return matches[0]
    body = getattr(func_node, "body", [])
    if body:
        first = body[0]
        if isinstance(first, ast.stmt):
            return first
    return None


def _statement_symbols(node: Optional[ast.AST]) -> List[str]:
    if node is None:
        return []
    names = sorted(
        {
            child.id
            for child in ast.walk(node)
            if isinstance(child, ast.Name) and str(child.id).strip()
        }
    )
    return names[:8]


def _statement_kinds(node: Optional[ast.AST]) -> List[str]:
    if node is None:
        return []
    kinds = sorted(
        {
            child.__class__.__name__
            for child in ast.walk(node)
            if isinstance(child, ast.AST)
        }
    )
    return kinds[:8]


def build_failure_card(
    evaluation: CodeRepairEval,
    *,
    metadata: Optional[dict],
    current_entry: Optional[Dict[str, Any]] = None,
    repair_round: int = -1,
) -> Dict[str, Any]:
    first_failed = str(evaluation.failing_examples[0]).strip() if evaluation.failing_examples else ""
    traceback_frame = _parse_traceback_frame(evaluation.stderr, evaluation.stdout, evaluation.exec_error, first_failed)
    current = dict(current_entry or {})
    return {
        "entry_point": str((metadata or {}).get("entry_point") or "").strip(),
        "verifier": {
            "syntax_ok": bool(evaluation.syntax_ok),
            "entry_point_ok": bool(evaluation.entry_point_ok),
            "passed": int(evaluation.passed),
            "total": int(evaluation.total),
            "failure_kind": str(evaluation.failure_kind or ""),
        },
        "symptom": {
            "first_failed_test": first_failed,
            "exec_error": str(evaluation.exec_error or ""),
            "top_traceback_frame": traceback_frame,
        },
        "candidate_state": {
            "current_code_digest": str(current.get("digest", "")),
            "parent_digest": str(current.get("parent_candidate_digest", current.get("repair_parent_digest", ""))),
            "repair_round": int(repair_round if repair_round >= 0 else current.get("repair_round", -1)),
        },
    }


def build_locus_card(
    candidate_text: str,
    evaluation: CodeRepairEval,
    *,
    metadata: Optional[dict],
) -> Dict[str, Any]:
    code = MultiFidelityEvaluator._extract_python_code(candidate_text)
    lines = code.splitlines() if code else []
    entry_point = str((metadata or {}).get("entry_point") or "").strip()
    traceback_frame = _parse_traceback_frame(evaluation.stderr, evaluation.stdout, evaluation.exec_error, *(evaluation.failing_examples[:1]))
    line_no = int(traceback_frame.get("line", 0) or 0)
    evidence: List[str] = []
    if traceback_frame:
        evidence.append(
            f"top traceback hits line {line_no}"
            + (f" in {traceback_frame.get('function')}" if traceback_frame.get("function") else "")
        )
    if evaluation.failing_examples:
        evidence.append(f"first failing path: {evaluation.failing_examples[0]}")
    if not code:
        return {
            "scope_kind": "file_span",
            "function": entry_point,
            "line_start": 1,
            "line_end": 1,
            "ast_kind": [],
            "involved_symbols": [],
            "evidence": evidence or ["missing candidate code"],
            "editable_region_id": f"{entry_point or 'module'}:1-1",
        }
    try:
        tree = ast.parse(code)
    except SyntaxError:
        end_line = min(max(len(lines), 1), 3)
        return {
            "scope_kind": "file_span",
            "function": entry_point,
            "line_start": 1,
            "line_end": end_line,
            "ast_kind": [],
            "involved_symbols": [],
            "evidence": evidence or ["syntax error prevents AST localization"],
            "editable_region_id": f"{entry_point or 'module'}:1-{end_line}",
        }

    function_node = _find_enclosing_function(tree, line_no, entry_point)
    statement = _statement_for_line(function_node, line_no) if function_node is not None else None
    if statement is not None:
        start_line = int(statement.lineno)
        end_line = int(statement.end_lineno)
    elif function_node is not None:
        start_line = int(function_node.lineno)
        end_line = int(function_node.end_lineno)
    else:
        start_line = 1
        end_line = min(max(len(lines), 1), 3)
    function_name = ""
    if isinstance(function_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        function_name = function_node.name
    elif entry_point:
        function_name = entry_point
    else:
        function_name = "module"
    if not evidence:
        evidence.append(f"defaulted to {function_name}:{start_line}-{end_line}")
    return {
        "scope_kind": "line_span",
        "function": function_name,
        "line_start": start_line,
        "line_end": end_line,
        "ast_kind": _statement_kinds(statement or function_node),
        "involved_symbols": _statement_symbols(statement or function_node),
        "evidence": evidence,
        "editable_region_id": f"{function_name}:{start_line}-{end_line}",
    }


def build_preserve_card(
    candidate_text: str,
    evaluation: CodeRepairEval,
    *,
    metadata: Optional[dict],
) -> Dict[str, Any]:
    code = MultiFidelityEvaluator._extract_python_code(candidate_text)
    entry_point = str((metadata or {}).get("entry_point") or "").strip()
    imports: List[str] = []
    helper_defs: List[str] = []
    signature = entry_point
    if code:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            tree = None
        if tree is not None:
            for node in tree.body:
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    rendered = ast.get_source_segment(code, node) or ast.unparse(node)
                    imports.append(str(rendered).strip())
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name == entry_point:
                        signature = _function_signature_text(node) or signature
                    else:
                        helper_defs.append(node.name)
    behavior_invariants = [
        f"visible test #{index + 1} remains passing"
        for index in range(max(0, int(evaluation.passed)))
    ]
    frozen_regions: List[str] = []
    if imports:
        frozen_regions.append("imports")
    for helper_name in helper_defs:
        frozen_regions.append(f"helper {helper_name}")
    if entry_point:
        frozen_regions.append(f"{entry_point} signature")
    return {
        "entry_contract": {
            "name": entry_point,
            "signature": signature,
            "must_return_kind": str((metadata or {}).get("return_kind") or "unknown"),
        },
        "must_keep": {
            "imports": imports,
            "helper_defs": helper_defs,
            "function_name": entry_point,
        },
        "behavior_invariants": behavior_invariants,
        "frozen_regions": frozen_regions,
        "disallowed_changes": [
            "rename entry point",
            "delete imports",
            "change return type contract",
        ],
    }


def apply_local_edit_artifact(
    candidate_text: str,
    artifact: Dict[str, Any],
    *,
    locus_card: Dict[str, Any],
    preserve_card: Dict[str, Any],
    metadata: Optional[dict],
) -> Tuple[Optional[str], str]:
    code = MultiFidelityEvaluator._extract_python_code(candidate_text)
    if not code:
        return None, "missing_candidate_code"
    if not isinstance(artifact, dict):
        return None, "invalid_artifact"
    if str(artifact.get("edit_op", "")).strip() != "replace_span":
        return None, "unsupported_edit_op"

    edit_scope = artifact.get("edit_scope")
    if not isinstance(edit_scope, dict):
        return None, "missing_edit_scope"
    editable_region_id = str(edit_scope.get("editable_region_id", "")).strip()
    if editable_region_id != str(locus_card.get("editable_region_id", "")).strip():
        return None, "scope_mismatch"

    start_line = int(edit_scope.get("line_start", 0) or 0)
    end_line = int(edit_scope.get("line_end", 0) or 0)
    locus_start = int(locus_card.get("line_start", 0) or 0)
    locus_end = int(locus_card.get("line_end", 0) or 0)
    if start_line < locus_start or end_line > locus_end or start_line <= 0 or end_line < start_line:
        return None, "scope_out_of_bounds"

    replacement = artifact.get("replacement_code")
    if isinstance(replacement, str):
        replacement_lines = replacement.splitlines()
    elif isinstance(replacement, list) and all(isinstance(item, str) for item in replacement):
        replacement_lines = list(replacement)
    else:
        return None, "invalid_replacement_code"

    original_lines = code.splitlines()
    patched_lines = original_lines[: start_line - 1] + replacement_lines + original_lines[end_line:]
    patched_code = "\n".join(patched_lines).strip("\n") + "\n"

    entry_point = str(((metadata or {}).get("entry_point")) or preserve_card.get("entry_contract", {}).get("name") or "").strip()
    if entry_point and not _entry_point_defined(patched_code, entry_point):
        return None, "entry_point_missing_after_patch"
    for import_stmt in preserve_card.get("must_keep", {}).get("imports", ()):
        if str(import_stmt).strip() and str(import_stmt).strip() not in patched_code:
            return None, "import_removed"
    for helper_name in preserve_card.get("must_keep", {}).get("helper_defs", ()):
        if str(helper_name).strip() and not re.search(rf"def\s+{re.escape(str(helper_name).strip())}\s*\(", patched_code):
            return None, "helper_removed"
    try:
        compile(patched_code, "<patched>", "exec")
    except Exception:
        return None, "patched_code_not_executable"
    return patched_code, ""
