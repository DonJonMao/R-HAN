from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch

from mas_treesearch.agents import AgentPool, default_agent_pool
from mas_treesearch.config import TieredEvalConfig
from mas_treesearch.evaluator import MultiFidelityEvaluator
from mas_treesearch.prompting import build_system_prompt, render_question_text
from mas_treesearch.profiles import DatasetProfile, resolve_dataset_profile
from mas_treesearch.types import PromptSlots

from .runtime import RollbackRuntime
from .train_bank import RollbackTeacherSample, build_online_rollout_samples
from .verifier import VerifierState, build_candidate_artifact_and_state


def _prompt_slots_from_node(node: Any) -> PromptSlots:
    payload = dict(getattr(node, "metadata", {}).get("prompt_slots", {}) if getattr(node, "metadata", None) else {})
    return PromptSlots(
        reasoning_mode=str(payload.get("reasoning_mode", PromptSlots().reasoning_mode)),
        upstream_usage=str(payload.get("upstream_usage", PromptSlots().upstream_usage)),
        output_style=str(payload.get("output_style", PromptSlots().output_style)),
        verification_mode=str(payload.get("verification_mode", PromptSlots().verification_mode)),
        finalization=str(payload.get("finalization", PromptSlots().finalization)),
    )


def _fallback_agent_id(answer_format: str) -> str:
    if answer_format == "python_code":
        return "coder"
    if answer_format in {"math_expression", "numeric", "option"}:
        return "reasoner"
    return "summarizer"


def _resolve_agent(pool: AgentPool, *, preferred_agent_id: str, answer_format: str) -> Any:
    by_id = pool.by_id()
    if preferred_agent_id in by_id:
        return by_id[preferred_agent_id]
    fallback_id = _fallback_agent_id(answer_format)
    if fallback_id in by_id:
        return by_id[fallback_id]
    if "reasoner" in by_id:
        return by_id["reasoner"]
    return pool.agents[0]


def load_online_models(
    *,
    trained_root: str,
    dataset_name: str,
) -> RollbackRuntime:
    runtime = RollbackRuntime()
    root = trained_root
    boundary_payload = torch.load(f"{root}/{dataset_name}/phase_s2_boundary.pt", map_location="cpu")
    diffusion_payload = torch.load(f"{root}/{dataset_name}/phase_s34_diffusion.pt", map_location="cpu")
    selector_payload = torch.load(f"{root}/{dataset_name}/phase_s45_selector.pt", map_location="cpu")
    runtime.boundary_model.load_state_dict(boundary_payload["state_dict"], strict=False)
    runtime.diffusion_model.load_state_dict(diffusion_payload["state_dict"], strict=False)
    runtime.emitter_model.load_state_dict(selector_payload["emitter_state"], strict=False)
    runtime.selector_model.load_state_dict(selector_payload["selector_state"], strict=False)
    runtime.boundary_model.eval()
    runtime.diffusion_model.eval()
    runtime.emitter_model.eval()
    runtime.selector_model.eval()
    return runtime


def build_online_messages(
    *,
    sample: RollbackTeacherSample,
    request: Dict[str, Any],
    profile: DatasetProfile,
    evaluator: MultiFidelityEvaluator,
    pool: AgentPool,
) -> List[Dict[str, str]]:
    union_graph = sample.prepared.base_prepared.union_graph if sample.prepared.base_prepared is not None else None
    origin_node = None
    if union_graph is not None:
        origin_node = union_graph.nodes.get(request["origin_node_id"])
    agent = _resolve_agent(
        pool,
        preferred_agent_id=str(getattr(origin_node, "agent_id", "")),
        answer_format=sample.answer_format,
    )
    slots = _prompt_slots_from_node(origin_node) if origin_node is not None else PromptSlots(
        reasoning_mode="critique_then_answer",
        upstream_usage="summary",
        output_style="raw",
        verification_mode="strict",
        finalization="answer_only",
    )
    system_prompt = build_system_prompt(
        agent,
        slots,
        extra_role_hint=f"rollback_rerun::{getattr(origin_node, 'role', 'fallback')}",
    )
    answer_contract = evaluator._output_contract(
        sample.question,
        reference_answer=sample.reference_answer,
        metadata=sample.metadata,
    )
    suffix_steps = [
        step
        for step in sample.trace.steps
        if int(step.index) > int(request["boundary_index"])
    ]
    suffix_blocks: List[str] = []
    for step in suffix_steps[:8]:
        residual = step.contract_residual
        suffix_blocks.append(
            f"- step={step.index} node={step.node_id} role={step.brief.get('role','')} "
            f"reasoning={step.brief.get('reasoning_mode','')} output={step.output_summary[:160]} "
            f"ctr(parse={residual.get('parse',0.0):.2f},comp={residual.get('completeness',0.0):.2f},"
            f"const={residual.get('constraint',0.0):.2f},exec={residual.get('execution',0.0):.2f})"
        )
    rerun_nodes = ", ".join(request.get("rerun_subgraph_node_ids", [])[:12]) or "(none)"
    user_prompt = (
        "You are executing an online stage2 rollback rerun.\n"
        "The stage1 graph and anchor are frozen. Recompute only the suffix after the rollback boundary.\n"
        "Use the rollback context to improve the final answer if you can; otherwise keep the best valid answer.\n"
        "Return only the final answer that satisfies the output contract.\n\n"
        f"Question:\n{render_question_text(sample.question, metadata=sample.metadata)}\n\n"
        f"Stage1 anchor:\n{sample.anchor_candidate.output}\n\n"
        f"Rollback boundary index: {request['boundary_index']}\n"
        f"Origin node id: {request['origin_node_id']}\n"
        f"Rerun subgraph nodes: {rerun_nodes}\n\n"
        "Suffix trace brief:\n"
        f"{chr(10).join(suffix_blocks) if suffix_blocks else '(empty suffix)'}\n\n"
        f"Output contract:\n{answer_contract}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def generate_online_candidate(
    *,
    sample: RollbackTeacherSample,
    request: Dict[str, Any],
    evaluator: MultiFidelityEvaluator,
    profile: DatasetProfile,
    pool: AgentPool,
) -> str:
    messages = build_online_messages(
        sample=sample,
        request=request,
        profile=profile,
        evaluator=evaluator,
        pool=pool,
    )
    runtime = evaluator._resolve_runtime("tier2", profile)
    raw = evaluator._cached_chat(messages, runtime=runtime, cacheable=False)
    return evaluator._sanitize_final_output(
        sample.question,
        raw,
        reference_answer=sample.reference_answer,
        metadata=sample.metadata,
    )


@dataclass
class OnlineSampleDecision:
    sample_id: str
    boundary_index: int
    null_emission: bool
    request_count: int
    generated_candidates: List[str]
    winner_index: int
    final_output: str
    anchor_task_score: float
    anchor_success: float
    final_task_score: float
    final_success: float


def build_online_candidate_state(
    *,
    sample: RollbackTeacherSample,
    candidate_text: str,
) -> VerifierState:
    union_graph = sample.prepared.base_prepared.union_graph if sample.prepared.base_prepared is not None else None
    return build_candidate_artifact_and_state(
        question_text=sample.question,
        candidate_text=candidate_text,
        dataset_name=sample.dataset,
        answer_format=sample.answer_format,
        task_subtype=sample.task_subtype,
        metadata=sample.metadata,
        trace=sample.trace,
        union_graph=union_graph,
    )


def build_online_samples(
    *,
    dataset_name: str,
    items: Sequence[Dict[str, Any]],
) -> List[RollbackTeacherSample]:
    return build_online_rollout_samples(dataset_name=dataset_name, items=items)


def build_online_evaluator(
    *,
    chat_api_base: str,
) -> MultiFidelityEvaluator:
    config = TieredEvalConfig()
    config.chat.api_base = str(chat_api_base)
    config.enable_chat_cache = True
    return MultiFidelityEvaluator(config, default_agent_pool())


def resolve_profile(dataset_name: str) -> DatasetProfile:
    return resolve_dataset_profile(dataset_name)
