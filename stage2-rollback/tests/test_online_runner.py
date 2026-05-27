from __future__ import annotations

from mas_treesearch.agents import default_agent_pool

from stage2_rollback.online import build_online_evaluator, build_online_messages, build_online_samples, resolve_profile


def test_build_online_samples_smoke() -> None:
    samples = build_online_samples(
        dataset_name="mmlu_pro",
        items=[
            {
                "id": "toy",
                "question": "Which option is correct?\n1) A\n2) B\n3) C\n4) D",
                "answer": "OPTION - 2",
                "metadata": {"options": ["A", "B", "C", "D"]},
            }
        ],
    )
    assert len(samples) == 1
    sample = samples[0]
    assert sample.anchor_candidate.boundary_index == 0
    assert len(sample.candidates) == 1
    assert sample.answer_format == "option"


def test_build_online_messages_includes_core_fields() -> None:
    sample = build_online_samples(
        dataset_name="mmlu_pro",
        items=[
            {
                "id": "toy",
                "question": "Which option is correct?\n1) A\n2) B\n3) C\n4) D",
                "answer": "OPTION - 2",
                "metadata": {"options": ["A", "B", "C", "D"]},
            }
        ],
    )[0]
    evaluator = build_online_evaluator(chat_api_base="http://127.0.0.1:8041")
    profile = resolve_profile("mmlu_pro")
    messages = build_online_messages(
        sample=sample,
        request={
            "boundary_index": 1,
            "origin_node_id": "fallback",
            "rerun_subgraph_node_ids": ["fallback"],
        },
        profile=profile,
        evaluator=evaluator,
        pool=default_agent_pool(),
    )
    assert len(messages) == 2
    assert "Stage1 anchor" in messages[1]["content"]
    assert "Rollback boundary index" in messages[1]["content"]
    assert "Output contract" in messages[1]["content"]
