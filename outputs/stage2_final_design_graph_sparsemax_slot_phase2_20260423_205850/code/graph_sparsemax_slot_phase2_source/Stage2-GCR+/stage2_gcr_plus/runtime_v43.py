from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .runtime_v2 import Stage2RuntimeV2
from mas_stage2.types import Stage2RunResult, TurnTrace
from .runtime_v42 import Stage2RuntimeV42
from mas_treesearch.prompting import build_system_prompt, render_question_text
from mas_treesearch.profiles import DEFAULT_PROFILE, DatasetProfile
from mas_treesearch.types import PromptSlots, UnionGraph

from .code_repair import CodeRepairEval, build_failure_summary, evaluate_code_candidate
from .config import Stage2V43Config


class Stage2RuntimeV43(Stage2RuntimeV42):
    def __init__(self, config: Stage2V43Config, evaluator, agent_pool, embedder):
        super().__init__(config, evaluator, agent_pool, embedder)
        self.config = config
        self._last_v4_3_selection: Dict[str, Any] = {}

    @staticmethod
    def _repair_slots() -> PromptSlots:
        return PromptSlots(
            reasoning_mode="stepwise",
            upstream_usage="summary",
            output_style="raw",
            verification_mode="off",
            finalization="answer_only",
        )

    def _ensure_v4_3_entry_fields(self, entry: Dict[str, Any]) -> None:
        self._ensure_v4_2_entry_fields(entry)
        entry.setdefault("repair_branch", False)
        entry.setdefault("repair_agent_id", "")
        entry.setdefault("repair_round", -1)
        entry.setdefault("repair_parent_digest", "")
        entry.setdefault("code_repair_level", 0)
        entry.setdefault("code_repair_passed", 0)
        entry.setdefault("code_repair_total", 0)
        entry.setdefault("code_repair_failure_kind", "")
        entry.setdefault("code_repair_failing_examples", [])

    def _candidate_source_label(self, entry: Dict[str, Any]) -> str:
        if entry.get("repair_branch"):
            return "code_repair_branch"
        return super()._candidate_source_label(entry)

    def _serialize_candidate_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        payload = super()._serialize_candidate_entry(entry)
        payload.update(
            {
                "repair_branch": bool(entry.get("repair_branch", False)),
                "repair_agent_id": str(entry.get("repair_agent_id", "")),
                "repair_round": int(entry.get("repair_round", -1)),
                "repair_parent_digest": str(entry.get("repair_parent_digest", "")),
                "code_repair_level": int(entry.get("code_repair_level", 0)),
                "code_repair_passed": int(entry.get("code_repair_passed", 0)),
                "code_repair_total": int(entry.get("code_repair_total", 0)),
                "code_repair_failure_kind": str(entry.get("code_repair_failure_kind", "")),
                "code_repair_failing_examples": list(entry.get("code_repair_failing_examples", ())),
            }
        )
        return payload

    def _select_repair_agents(self) -> List[str]:
        selected: List[str] = []
        for agent_id in self.config.repair_agent_ids:
            if agent_id in self._by_id and agent_id not in selected:
                selected.append(agent_id)
        if selected:
            return selected
        for fallback in ("coder", "planner", "debater_a", "reasoner", "researcher"):
            if fallback in self._by_id and fallback not in selected:
                selected.append(fallback)
        return selected[:3]

    @staticmethod
    def _code_eval_level(feedback: CodeRepairEval) -> int:
        if not feedback.syntax_ok:
            return 0
        if not feedback.entry_point_ok:
            return 1
        if feedback.fully_passed:
            return 3
        return 2

    def _attach_code_feedback(self, entry: Dict[str, Any], feedback: CodeRepairEval) -> None:
        self._ensure_v4_3_entry_fields(entry)
        entry["parse_ok"] = bool(feedback.syntax_ok)
        entry["entry_point_ok"] = bool(feedback.entry_point_ok)
        entry["code_repair_level"] = int(self._code_eval_level(feedback))
        entry["code_repair_passed"] = int(feedback.passed)
        entry["code_repair_total"] = int(feedback.total)
        entry["code_repair_failure_kind"] = str(feedback.failure_kind)
        entry["code_repair_failing_examples"] = list(feedback.failing_examples)

    def _verified_rank_key(self, entry: Dict[str, Any], feedback: CodeRepairEval) -> Tuple[Any, ...]:
        return (
            feedback.rank_key,
            int(bool(entry.get("stage1_anchor", False))),
            -int(entry.get("line_count", 0)),
            str(entry.get("digest", "")),
        )

    def _prepare_verified_entry(
        self,
        text: str,
        *,
        metadata: Optional[dict],
        parent: Optional[Dict[str, Any]] = None,
        repair_agent_id: str = "",
        repair_round: int = -1,
        recovery_subgraph_node_ids: Optional[Sequence[str]] = None,
        recovery_subgraph_edge_ids: Optional[Sequence[str]] = None,
        trigger_verifier_snapshot: Optional[Dict[str, Any]] = None,
        repair_operator_type: str = "",
    ) -> Tuple[Dict[str, Any], CodeRepairEval]:
        entry = self._init_candidate_entry(text)
        if parent is not None:
            entry["stage1_anchor"] = bool(parent.get("stage1_anchor", False))
            entry["source_roles"] = set(parent.get("source_roles", set()))
            entry["source_node_ids"] = set(parent.get("source_node_ids", set()))
            entry["turn_indices"] = set(parent.get("turn_indices", set()))
            entry["candidate_model_score"] = float(parent.get("candidate_model_score", 0.5))
            entry["candidate_model_uncertainty"] = float(parent.get("candidate_model_uncertainty", 0.2))
            entry["support_score"] = float(parent.get("support_score", 0.5))
            entry["sink_support"] = int(parent.get("sink_support", 0))
            entry["occurrence_count"] = int(parent.get("occurrence_count", 0))
            entry["feedback_pass"] = float(parent.get("feedback_pass", 0.0))
            entry["feedback_challenge"] = float(parent.get("feedback_challenge", 0.0))
            entry["feedback_uncertain"] = float(parent.get("feedback_uncertain", 0.0))
            entry["feedback_pass_calibrated"] = float(parent.get("feedback_pass_calibrated", 0.0))
            entry["feedback_challenge_calibrated"] = float(parent.get("feedback_challenge_calibrated", 0.0))
            entry["feedback_uncertain_calibrated"] = float(parent.get("feedback_uncertain_calibrated", 0.0))
            entry["reviewer_event_count"] = float(parent.get("reviewer_event_count", 0.0))
            entry["reviewer_mean_trust"] = float(parent.get("reviewer_mean_trust", 0.5))
        self._ensure_v4_3_entry_fields(entry)
        if parent is not None:
            entry["origin_node_id"] = str(parent.get("origin_node_id", ""))
            entry["origin_turn_index"] = int(parent.get("origin_turn_index", -1))
            entry["origin_role"] = str(parent.get("origin_role", ""))
            entry["verifier_snapshot"] = dict(parent.get("verifier_snapshot", {}))
            entry["candidate_bank_source"] = str(parent.get("candidate_bank_source", ""))
            entry["provenance"] = [dict(item) for item in parent.get("provenance", ()) if isinstance(item, dict)]
            entry["parent_candidate_digest"] = str(parent.get("parent_candidate_digest", ""))
            entry["repair_operator_type"] = str(parent.get("repair_operator_type", ""))
            entry["recovery_subgraph_node_ids"] = list(parent.get("recovery_subgraph_node_ids", ()))
            entry["recovery_subgraph_edge_ids"] = list(parent.get("recovery_subgraph_edge_ids", ()))
            entry["trigger_verifier_snapshot"] = dict(parent.get("trigger_verifier_snapshot", {}))
            entry["recovery_reinserted"] = bool(parent.get("recovery_reinserted", False))
        if repair_agent_id:
            entry["repair_branch"] = True
            entry["repair_agent_id"] = repair_agent_id
            entry["repair_round"] = repair_round
            if parent is not None:
                entry["repair_parent_digest"] = str(parent.get("digest", ""))
                entry["parent_candidate_digest"] = str(parent.get("digest", ""))
            entry["repair_operator_type"] = repair_operator_type or "code_repair_patch"
            entry["recovery_subgraph_node_ids"] = list(recovery_subgraph_node_ids or ())
            entry["recovery_subgraph_edge_ids"] = list(recovery_subgraph_edge_ids or ())
            entry["trigger_verifier_snapshot"] = dict(trigger_verifier_snapshot or {})
            entry["recovery_reinserted"] = False
        feedback = evaluate_code_candidate(
            text,
            metadata,
            timeout_s=float(self.config.repair_timeout_s),
            max_failed_examples=int(self.config.repair_max_failed_examples),
        )
        self._attach_code_feedback(entry, feedback)
        return entry, feedback

    def _verified_seed_pool(
        self,
        *,
        anchor: Optional[Dict[str, Any]],
        candidates: Sequence[Dict[str, Any]],
        metadata: Optional[dict],
    ) -> Tuple[List[Tuple[Dict[str, Any], CodeRepairEval]], Optional[Tuple[Dict[str, Any], CodeRepairEval]]]:
        pool: List[Tuple[Dict[str, Any], CodeRepairEval]] = []
        seen: set[str] = set()

        ordered: List[Dict[str, Any]] = []
        if anchor is not None:
            ordered.append(anchor)
        ordered.extend(list(candidates))
        for entry in ordered:
            digest = str(entry.get("digest", ""))
            if digest in seen:
                continue
            seen.add(digest)
            verified, feedback = self._prepare_verified_entry(str(entry.get("text", "")), metadata=metadata, parent=entry)
            pool.append((verified, feedback))

        pool.sort(key=lambda item: self._verified_rank_key(item[0], item[1]), reverse=True)
        top_k = max(1, int(self.config.repair_seed_top_k))
        kept = list(pool[:top_k])

        anchor_pair = None
        if anchor is not None:
            anchor_digest = str(anchor.get("digest", ""))
            for item in pool:
                if str(item[0].get("digest", "")) == anchor_digest:
                    anchor_pair = item
                    break
            if anchor_pair is not None and all(str(item[0].get("digest", "")) != anchor_digest for item in kept):
                kept.append(anchor_pair)
        kept.sort(key=lambda item: self._verified_rank_key(item[0], item[1]), reverse=True)
        return kept, anchor_pair

    def _build_repair_prompt(
        self,
        *,
        question_text: str,
        metadata: Optional[dict],
        current_entry: Dict[str, Any],
        feedback: CodeRepairEval,
    ) -> str:
        entry_point = str((metadata or {}).get("entry_point") or "").strip()
        failure_summary = build_failure_summary(
            feedback,
            metadata=metadata,
            max_examples=int(self.config.repair_max_failed_examples),
        )
        prompt = (
            "You are repairing a Python solution after executable verification failed.\n\n"
            f"Question:\n{render_question_text(question_text, metadata=metadata)}\n\n"
            f"Current checkpoint code:\n{str(current_entry.get('text', '')).strip()}\n\n"
            f"Verification feedback:\n{failure_summary}\n\n"
            "Requirements:\n"
            "- Return only executable Python code.\n"
            "- Preserve the required function signature.\n"
            "- Prefer a local patch over rewriting unrelated logic.\n"
            "- Fix the failing constraints first.\n"
        )
        if entry_point:
            prompt += f"- The required entry point is `{entry_point}`.\n"
        return prompt

    def _generate_repair_branches(
        self,
        *,
        question_text: str,
        metadata: Optional[dict],
        dataset_profile: DatasetProfile,
        current_entry: Dict[str, Any],
        feedback: CodeRepairEval,
        repair_round: int,
        recovery_subgraph_node_ids: Optional[Sequence[str]] = None,
        recovery_subgraph_edge_ids: Optional[Sequence[str]] = None,
        trigger_verifier_snapshot: Optional[Dict[str, Any]] = None,
        repair_operator_type: str = "",
    ) -> List[Tuple[Dict[str, Any], CodeRepairEval]]:
        branches: List[Tuple[Dict[str, Any], CodeRepairEval]] = []
        seen: set[str] = {str(current_entry.get("text", "")).strip()}

        for agent_id in self._select_repair_agents():
            agent = self._by_id[agent_id]
            system_prompt = build_system_prompt(agent, self._repair_slots(), extra_role_hint="code_repair")
            user_prompt = self._build_repair_prompt(
                question_text=question_text,
                metadata=metadata,
                current_entry=current_entry,
                feedback=feedback,
            )
            raw_output = self.evaluator._cached_chat(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                runtime=self.evaluator._resolve_runtime("tier2", dataset_profile),
            )
            repaired = self._sanitize_candidate(question_text, raw_output, metadata=metadata)
            if not repaired or repaired in seen:
                continue
            seen.add(repaired)
            verified, verified_feedback = self._prepare_verified_entry(
                repaired,
                metadata=metadata,
                parent=current_entry,
                repair_agent_id=agent_id,
                repair_round=repair_round,
                recovery_subgraph_node_ids=recovery_subgraph_node_ids,
                recovery_subgraph_edge_ids=recovery_subgraph_edge_ids,
                trigger_verifier_snapshot=trigger_verifier_snapshot,
                repair_operator_type=repair_operator_type,
            )
            branches.append((verified, verified_feedback))
        return branches

    def _select_code_repair_against_anchor(
        self,
        *,
        question_text: str,
        metadata: Optional[dict],
        dataset_profile: DatasetProfile,
        candidates: Sequence[Dict[str, Any]],
        anchor: Optional[Dict[str, Any]],
    ) -> Tuple[Optional[Dict[str, Any]], str, Dict[str, Any]]:
        seed_pool, anchor_pair = self._verified_seed_pool(anchor=anchor, candidates=candidates, metadata=metadata)
        if not seed_pool:
            return anchor, "v4_3_code_empty_bank", {
                "v4_3_protocol_family": "code_repair",
                "v4_3_verified_seed_count": 0,
            }

        selected_entry, selected_feedback = seed_pool[0]
        selected_reason = "v4_3_code_override_verified_stage2_seed"
        if anchor_pair is not None:
            anchor_entry, anchor_feedback = anchor_pair
            if not selected_feedback.dominates(anchor_feedback):
                selected_entry, selected_feedback = anchor_entry, anchor_feedback
                selected_reason = (
                    "v4_3_code_preserve_verified_anchor"
                    if anchor_feedback.fully_passed
                    else "v4_3_code_preserve_no_seed_improvement"
                )
        else:
            anchor_entry = None
            anchor_feedback = None

        repair_rounds_run = 0
        repair_branch_count = 0
        repair_improvement_count = 0
        restart_count = 0
        explored_seed_digests: set[str] = set()

        seed_index = 0
        while repair_rounds_run < max(1, int(self.config.repair_rounds)):
            if selected_feedback.fully_passed:
                break
            repair_rounds_run += 1
            explored_seed_digests.add(str(selected_entry.get("digest", "")))
            branches = self._generate_repair_branches(
                question_text=question_text,
                metadata=metadata,
                dataset_profile=dataset_profile,
                current_entry=selected_entry,
                feedback=selected_feedback,
                repair_round=repair_rounds_run,
            )
            repair_branch_count += len(branches)
            improved = [
                (entry, feedback)
                for entry, feedback in branches
                if feedback.dominates(selected_feedback)
            ]
            if improved:
                improved.sort(key=lambda item: self._verified_rank_key(item[0], item[1]), reverse=True)
                selected_entry, selected_feedback = improved[0]
                selected_reason = "v4_3_code_override_repair_improvement"
                repair_improvement_count += 1
                continue
            if seed_index + 1 < len(seed_pool):
                seed_index += 1
                selected_entry, selected_feedback = seed_pool[seed_index]
                restart_count += 1
                selected_reason = "v4_3_code_restart_to_verified_seed"
                continue
            if anchor_entry is not None and str(selected_entry.get("digest", "")) == str(anchor_entry.get("digest", "")):
                selected_reason = "v4_3_code_preserve_no_repair_improvement"
            else:
                selected_reason = "v4_3_code_override_verified_stage2_seed"
            break

        if (
            anchor_entry is not None
            and anchor_feedback is not None
            and str(selected_entry.get("digest", "")) != str(anchor_entry.get("digest", ""))
            and not selected_feedback.dominates(anchor_feedback)
        ):
            selected_entry, selected_feedback = anchor_entry, anchor_feedback
            selected_reason = "v4_3_code_preserve_anchor_superior"

        if anchor_entry is not None and str(selected_entry.get("digest", "")) == str(anchor_entry.get("digest", "")):
            anchor_used = True
        else:
            anchor_used = False

        extra = {
            "v4_3_protocol_family": "code_repair",
            "v4_3_stage1_anchor_present": bool(anchor is not None),
            "v4_3_stage1_anchor_used": bool(anchor_used),
            "v4_3_candidate_count": int(len(candidates)),
            "v4_3_verified_seed_count": int(len(seed_pool)),
            "v4_3_selected_candidate_digest": str(selected_entry.get("digest", "")),
            "v4_3_selected_candidate_source": self._candidate_source_label(selected_entry),
            "v4_3_selected_quality_score": float(self._quality_score(selected_entry)),
            "v4_3_selected_model_uncertainty": float(selected_entry.get("candidate_model_uncertainty", 0.0)),
            "v4_3_selected_is_repair_branch": bool(selected_entry.get("repair_branch", False)),
            "v4_3_selected_visible_tests_passed": int(selected_feedback.passed),
            "v4_3_selected_visible_tests_total": int(selected_feedback.total),
            "v4_3_selected_failure_kind": str(selected_feedback.failure_kind),
            "v4_3_repair_rounds_run": int(repair_rounds_run),
            "v4_3_repair_branch_count": int(repair_branch_count),
            "v4_3_repair_improvement_count": int(repair_improvement_count),
            "v4_3_restart_count": int(restart_count),
            "v4_3_explored_seed_count": int(max(len(explored_seed_digests), 1 if seed_pool else 0)),
            "v4_3_checkpoint_count": int(1 + repair_improvement_count),
            "v4_3_top_verified_seeds": [
                {
                    "digest": str(entry.get("digest", "")),
                    "source": self._candidate_source_label(entry),
                    "passed": int(feedback.passed),
                    "total": int(feedback.total),
                    "failure_kind": str(feedback.failure_kind),
                    "repair_branch": bool(entry.get("repair_branch", False)),
                }
                for entry, feedback in seed_pool
            ],
        }
        if anchor_entry is not None and anchor_feedback is not None:
            extra.update(
                {
                    "v4_3_stage1_anchor_digest": str(anchor_entry.get("digest", "")),
                    "v4_3_stage1_anchor_visible_tests_passed": int(anchor_feedback.passed),
                    "v4_3_stage1_anchor_visible_tests_total": int(anchor_feedback.total),
                    "v4_3_stage1_anchor_failure_kind": str(anchor_feedback.failure_kind),
                }
            )
        return selected_entry, selected_reason, extra

    def _record_selection_metadata(
        self,
        *,
        task_type: str,
        candidates: Sequence[Dict[str, Any]],
        selected: Optional[Dict[str, Any]],
        anchor: Optional[Dict[str, Any]],
        strategy: str,
        selection_extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        if task_type != "code_generation":
            super()._record_selection_metadata(
                task_type=task_type,
                candidates=candidates,
                selected=selected,
                anchor=anchor,
                strategy=strategy,
                selection_extra=selection_extra,
            )
            self._last_v4_3_selection = dict(self._last_v4_2_selection)
            self._last_v4_3_selection["stage2_version"] = "v4.3"
            return

        top = [dict(item) for item in list(candidates)[: self.config.max_logged_candidates]]
        for item in top:
            item.pop("text", None)
        extra = dict(selection_extra or {})
        self._last_v4_3_selection = {
            "stage2_version": "v4.3",
            "v4_3_task_type": task_type,
            "v4_3_selection_reason": strategy,
            "v4_3_candidate_count": int(len(candidates)),
            "v4_3_stage1_anchor_present": bool(anchor is not None),
            "v4_3_stage1_anchor_used": bool(anchor is not None and selected is not None and anchor.get("digest") == selected.get("digest")),
            "v4_3_selected_candidate_digest": str((selected or {}).get("digest", "")),
            "v4_3_selected_candidate_source": self._candidate_source_label(selected or {}),
            "v4_3_selected_quality_score": self._quality_score(selected or {}),
            "v4_3_selected_model_uncertainty": float((selected or {}).get("candidate_model_uncertainty", 0.0)),
            "v4_3_selected_is_repair_branch": bool((selected or {}).get("repair_branch", False)),
            "v4_3_top_candidates": top,
        }
        self._last_v4_3_selection.update(extra)

    def _finalize_answer(
        self,
        *,
        question_text: str,
        controller_state,
        sink_outputs: Dict[str, str],
        turn_traces: Sequence[TurnTrace],
        metadata: Optional[dict],
        reference_answer: Optional[str],
        dataset_profile: DatasetProfile,
    ) -> Tuple[str, str]:
        bundle = self._candidate_bank_bundle(
            question_text=question_text,
            turn_traces=turn_traces,
            metadata=metadata,
            dataset_profile=dataset_profile,
        )
        bundle["question_text"] = question_text
        bundle["metadata"] = metadata
        self._last_candidate_bundle = bundle
        candidates = bundle["candidates"]
        candidates_serialized = bundle["candidates_serialized"]
        anchor = bundle["anchor"]
        anchor_serialized = bundle["anchor_serialized"]

        if dataset_profile.task_type != "code_generation":
            return super()._finalize_answer(
                question_text=question_text,
                controller_state=controller_state,
                sink_outputs=sink_outputs,
                turn_traces=turn_traces,
                metadata=metadata,
                reference_answer=reference_answer,
                dataset_profile=dataset_profile,
            )

        selected, strategy, extra = self._select_code_repair_against_anchor(
            question_text=question_text,
            metadata=metadata,
            dataset_profile=dataset_profile,
            candidates=candidates,
            anchor=anchor,
        )

        selected_serialized = None
        if selected is not None:
            for item in candidates_serialized:
                if item.get("digest") == selected.get("digest"):
                    selected_serialized = dict(item)
                    break
            if selected_serialized is None:
                selected_serialized = self._serialize_candidate_entry(selected)
                candidates_serialized.append(dict(selected_serialized))

        final_answer = str((selected or {}).get("text", "")) if selected is not None else ""
        self._record_selection_metadata(
            task_type=dataset_profile.task_type,
            candidates=candidates_serialized,
            selected=selected_serialized,
            anchor=anchor_serialized,
            strategy=strategy,
            selection_extra=extra,
        )
        return final_answer, strategy

    def run(
        self,
        graph: UnionGraph,
        *,
        question_text: str,
        metadata: Optional[dict] = None,
        reference_answer: Optional[str] = None,
        dataset_profile: DatasetProfile = DEFAULT_PROFILE,
        replay_dir: Optional[str] = None,
        learn: bool = False,
    ) -> Stage2RunResult:
        self._last_v4_3_selection = {}
        self._last_candidate_bundle = {}
        result = Stage2RuntimeV2.run(
            self,
            graph,
            question_text=question_text,
            metadata=metadata,
            reference_answer=reference_answer,
            dataset_profile=dataset_profile,
            replay_dir=replay_dir,
            learn=learn,
        )
        if result.signature.startswith("stage2_v2|"):
            result.signature = "stage2_v4_3|" + result.signature[len("stage2_v2|") :]
        result.metadata.update(self._last_v4_3_selection)
        result.metadata["stage2_version"] = "v4.3"
        result.metadata["v4_3_all_task_nodes_each_turn"] = True
        return result

    def save_replay_bundle(
        self,
        result: Stage2RunResult,
        replay_dir: str,
        *,
        question_text: str,
        metadata: Optional[dict],
    ) -> None:
        Stage2RuntimeV2.save_replay_bundle(self, result, replay_dir, question_text=question_text, metadata=metadata)
        with open(os.path.join(replay_dir, "v4_3_selection.json"), "w", encoding="utf-8") as handle:
            json.dump(self._last_v4_3_selection, handle, ensure_ascii=False, indent=2)
        with open(os.path.join(replay_dir, "final_answer.json"), "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "final_answer": result.final_answer,
                    "signature": result.signature,
                    "final_controller_state": asdict(result.final_controller_state),
                    "memory_record_counts": result.memory_record_counts,
                    "metadata": result.metadata | self._last_v4_3_selection,
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )
