from __future__ import annotations

import json
from typing import Any, Dict, Optional

from mas_treesearch.prompting import render_question_text


PROMPT_ARTIFACT_IR = """You are a universal solution canonicalizer.

Input:
- problem statement
- candidate answer
- optional provenance summary

Task:
Convert the candidate into a unified editable representation.

Requirements:
1. Split the answer into minimal editable units.
2. Preserve order and dependency when possible.
3. Produce four views if possible:
   - surface_view
   - step_view
   - struct_view
   - exec_view
4. If a view is not available, set its confidence clue to low and explain why.
5. Do not solve the task again.

Return JSON only:
{
  "units": [...],
  "dependencies": [...],
  "views": {...},
  "parse_clues": {...}
}
"""


PROMPT_META_VERIFY = """You are a task-agnostic verifier.

Input:
- problem statement
- canonicalized ArtifactIR

Task:
Evaluate the artifact without assuming any task type.

Check only:
- completeness
- consistency
- unsupported units
- preservation risk
- obvious malformed structure

Return JSON only:
{
  "global_issues": [...],
  "unit_issues": [{"unit_id":"...", "issue":"...", "severity":0~1}],
  "supported_units": [...],
  "missing_requirements": [...],
  "meta_summary": "..."
}
"""


PROMPT_CRITIQUE = """You are a universal correction critic.

Input:
- problem statement
- ArtifactIR
- VerifierState summary
- unit heatmap
- preserve heatmap

Task:
1. Explain the highest-residual region.
2. Identify a minimal editable region.
3. Explain what must be preserved.
4. Predict which residual dimensions should decrease after a successful edit.

Return JSON only:
{
  "target_units": [...],
  "preserve_units": [...],
  "main_residual_causes": [...],
  "expected_delta": {
    "should_drop": ["r_consistency", "r_constraint"],
    "must_not_increase": ["r_preserve"]
  },
  "edit_rationale": "..."
}
"""


PROMPT_ARTIFACT_PROPOSAL = """You are a universal local editor.

Input:
- problem statement
- ArtifactIR
- critique JSON

Task:
Produce exactly one local edit artifact.

Requirements:
1. Edit only target_units.
2. Preserve preserve_units.
3. Use only one of:
   replace / insert_before / insert_after / delete / reorder
4. Keep the edit minimal.
5. Do not rewrite the whole answer.
6. The output must remain renderable.

Return JSON only:
{
  "target_units": [...],
  "operation": "...",
  "new_units": [...],
  "preserve_units": [...],
  "expected_delta": {...},
  "rationale": "..."
}
"""


PROMPT_DELTA = """You are a delta predictor.

Input:
- original ArtifactIR
- edited ArtifactIR
- original VerifierState
- correction artifact

Task:
Predict what will change after re-verification.

Return JSON only:
{
  "expected_residual_drop": {
    "r_parse": 0.0,
    "r_consistency": 0.0,
    "r_completeness": 0.0,
    "r_execution": 0.0,
    "r_constraint": 0.0,
    "r_support": 0.0,
    "r_preserve": 0.0
  },
  "expected_confidence_gain": 0.0,
  "expected_preserve_risk": 0.0,
  "expected_frontier_shift": 0.0
}
"""


def build_prompt_payload(
    *,
    question_text: str,
    metadata: Optional[Dict[str, Any]],
    payload: Dict[str, Any],
) -> str:
    rendered_question = render_question_text(question_text, metadata=metadata)
    return f"Problem Statement:\n{rendered_question}\n\nPayload:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
