from __future__ import annotations

from stage2_rollback.artifacts import canonicalize_candidate
from stage2_rollback.pipeline import RollbackPreparedStage1Artifact
from stage2_rollback.runtime import RollbackRuntime
from stage2_rollback.trace_ir import build_rollback_trace
from stage2_rollback.train_bank import RollbackTeacherCandidate, RollbackTeacherSample


def test_runtime_analyze_smoke() -> None:
    prepared = RollbackPreparedStage1Artifact(
        base_prepared=None,
        anchor_artifact={},
        rollback_trace={},
        anchor_replay_cache={},
    )
    anchor_text = "Reasoning...\nOPTION - 2"
    candidate_text = "Updated support.\nOPTION - 3"
    trace = build_rollback_trace(
        prepared,
        question_text="Which option is correct?",
        candidate_text=anchor_text,
        dataset_name="mmlu_pro",
        answer_format="option",
        task_subtype="",
        metadata={"options": ["a", "b", "c", "d"]},
    )
    anchor_artifact = canonicalize_candidate(
        question_text="Which option is correct?",
        candidate_text=anchor_text,
        dataset_name="mmlu_pro",
        answer_format="option",
        task_subtype="",
        metadata={"options": ["a", "b", "c", "d"]},
    )
    candidate_artifact = canonicalize_candidate(
        question_text="Which option is correct?",
        candidate_text=candidate_text,
        dataset_name="mmlu_pro",
        answer_format="option",
        task_subtype="",
        metadata={"options": ["a", "b", "c", "d"]},
    )
    sample = RollbackTeacherSample(
        id="toy",
        dataset="mmlu_pro",
        answer_format="option",
        task_subtype="",
        question="Which option is correct?",
        reference_answer="OPTION - 3",
        metadata={"options": ["a", "b", "c", "d"]},
        prepared=prepared,
        trace=trace,
        anchor_candidate=RollbackTeacherCandidate(
            candidate_id="toy::b0",
            boundary_index=0,
            origin_kind="anchor",
            emitted_node_id="__null__",
            output=anchor_text,
            artifact=anchor_artifact,
            eval_success=0.0,
            eval_task_score=0.0,
        ),
        candidates=[
            RollbackTeacherCandidate(
                candidate_id="toy::b0",
                boundary_index=0,
                origin_kind="anchor",
                emitted_node_id="__null__",
                output=anchor_text,
                artifact=anchor_artifact,
                eval_success=0.0,
                eval_task_score=0.0,
            ),
            RollbackTeacherCandidate(
                candidate_id="toy::b1",
                boundary_index=1,
                origin_kind="oracle",
                emitted_node_id="fallback",
                output=candidate_text,
                artifact=candidate_artifact,
                eval_success=1.0,
                eval_task_score=1.0,
                rerun_subgraph_node_ids=["fallback"],
                q_sup_target={"fallback": 1.0},
                q_ans_target={"fallback": 1.0},
            ),
        ],
        teacher_boundary=1,
        phat_bank={"toy::b0": 0.2, "toy::b1": 0.8},
        rerun_needed=1,
    )
    result = RollbackRuntime().analyze(sample, train_mode=True)
    assert result.boundary_output.pi_tilde_rb.shape[0] == len(trace.steps) + 1
    assert result.diffusion_output.answer.alpha_tilde.shape[0] >= 1
    assert result.emitter_output.gamma_tilde_plus.shape[0] == len(result.diffusion_output.node_ids) + 1
    assert result.selector_output.probabilities.shape[0] == 2
