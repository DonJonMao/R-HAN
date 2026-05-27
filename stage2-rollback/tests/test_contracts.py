from __future__ import annotations

from stage2_rollback.contracts import parse_answer_object


def test_parse_option_answer() -> None:
    obj = parse_answer_object(
        "Reasoning...\nOPTION - 8",
        dataset_name="mmlu_pro",
        answer_format="option",
        task_subtype="",
        metadata={"options": ["a"] * 10},
    )
    assert obj.valid is True
    assert obj.value == 8
    assert obj.signature == "option::8"


def test_parse_graph_connectivity_answer() -> None:
    obj = parse_answer_object(
        '{"answer":"no"}',
        dataset_name="nlgraph",
        answer_format="graph_json",
        task_subtype="connectivity",
        metadata={"task": "connectivity"},
    )
    assert obj.valid is True
    assert obj.value == "no"
    assert obj.signature == "graph_bool::answer::no"


def test_parse_code_answer() -> None:
    obj = parse_answer_object(
        "```python\ndef foo(x):\n    return x + 1\n```",
        dataset_name="mbpp",
        answer_format="python_code",
        task_subtype="",
        metadata={"entry_point": "foo"},
    )
    assert obj.valid is True
    assert obj.kind == "code"
    assert obj.fields["entry_point"] == "foo"


def test_parse_math_expression_answer() -> None:
    obj = parse_answer_object(
        "We get \\boxed{\\frac{5}{7}}",
        dataset_name="math",
        answer_format="math_expression",
        task_subtype="",
        metadata={},
    )
    assert obj.valid is True
    assert obj.signature.startswith("math_expression::")
