from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from .boundary import BoundaryOutput, RollbackBoundaryModel, build_boundary_features
from .diffusion import DualDiffusionOutput, RollbackDiffusionModel
from .emitter import EmitterOutput, RollbackEmitterModel, build_rerun_request
from .selector import RollbackSelectorModel, SelectorOutput
from .train_bank import RollbackTeacherSample
from .verifier import VerifierState, build_candidate_artifact_and_state


@dataclass
class RuntimeSampleResult:
    anchor_state: VerifierState
    candidate_states: List[VerifierState]
    boundary_output: BoundaryOutput
    diffusion_output: DualDiffusionOutput
    emitter_output: EmitterOutput
    selector_output: SelectorOutput
    requests: List[Dict[str, Any]]


class RollbackRuntime:
    def __init__(
        self,
        *,
        boundary_model: RollbackBoundaryModel | None = None,
        diffusion_model: RollbackDiffusionModel | None = None,
        emitter_model: RollbackEmitterModel | None = None,
        selector_model: RollbackSelectorModel | None = None,
    ) -> None:
        self.boundary_model = boundary_model or RollbackBoundaryModel()
        self.diffusion_model = diffusion_model or RollbackDiffusionModel()
        self.emitter_model = emitter_model or RollbackEmitterModel()
        self.selector_model = selector_model or RollbackSelectorModel()

    def analyze(self, sample: RollbackTeacherSample, *, train_mode: bool) -> RuntimeSampleResult:
        union_graph = sample.prepared.base_prepared.union_graph if sample.prepared.base_prepared is not None else None
        anchor_state = build_candidate_artifact_and_state(
            question_text=sample.question,
            candidate_text=sample.anchor_candidate.output,
            dataset_name=sample.dataset,
            answer_format=sample.answer_format,
            task_subtype=sample.task_subtype,
            metadata=sample.metadata,
            trace=sample.trace,
            union_graph=union_graph,
        )
        candidate_states: List[VerifierState] = []
        non_anchor_candidates = [candidate for candidate in sample.candidates if candidate.boundary_index > 0]
        for candidate in non_anchor_candidates:
            candidate_states.append(
                build_candidate_artifact_and_state(
                    question_text=sample.question,
                    candidate_text=candidate.output,
                    dataset_name=sample.dataset,
                    answer_format=sample.answer_format,
                    task_subtype=sample.task_subtype,
                    metadata=sample.metadata,
                    trace=sample.trace,
                    union_graph=union_graph,
                )
            )
        boundary_features = build_boundary_features(anchor_state, sample.trace)
        boundary_output = self.boundary_model(boundary_features)
        diffusion_output = self.diffusion_model(
            verifier_state=anchor_state,
            boundary_output=boundary_output,
            union_graph=union_graph,
            train_mode=train_mode,
        )
        emitter_output = self.emitter_model(
            diffusion_output=diffusion_output,
            boundary_index=boundary_output.predicted_boundary,
            anchor_typed_support=anchor_state.typed_support_score,
            train_mode=train_mode,
        )
        selector_output = self.selector_model(anchor_state, candidate_states)
        requests: List[Dict[str, Any]] = []
        trace_suffix_briefs = [step.brief_key for step in sample.trace.steps if step.index > boundary_output.predicted_boundary]
        rerun_nodes = [node_id for node_id, mass in zip(diffusion_output.node_ids, diffusion_output.answer.alpha_bar.tolist()) if mass > 0.0]
        for node_id in emitter_output.selected_node_ids:
            request = build_rerun_request(
                question_text=sample.question,
                boundary_index=boundary_output.predicted_boundary,
                origin_node_id=node_id,
                rerun_subgraph_node_ids=rerun_nodes,
                trace_suffix_briefs=trace_suffix_briefs,
                anchor_replay_cache=sample.prepared.anchor_replay_cache,
            )
            requests.append(
                {
                    "question_text": request.question_text,
                    "boundary_index": request.boundary_index,
                    "origin_node_id": request.origin_node_id,
                    "rerun_subgraph_node_ids": request.rerun_subgraph_node_ids,
                    "trace_suffix_briefs": request.trace_suffix_briefs,
                }
            )
        return RuntimeSampleResult(
            anchor_state=anchor_state,
            candidate_states=candidate_states,
            boundary_output=boundary_output,
            diffusion_output=diffusion_output,
            emitter_output=emitter_output,
            selector_output=selector_output,
            requests=requests,
        )
