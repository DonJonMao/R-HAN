from __future__ import annotations

import json
import math
import os
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from mas_stage2.runtime_v2 import Stage2RuntimeV2
from mas_stage2.types import ExportedMemoryMessage, Stage2RunResult, TurnTrace
from mas_stage2_v3_1.runtime import Stage2RuntimeV31
from mas_treesearch.prompting import build_system_prompt, render_question_text
from mas_treesearch.profiles import DEFAULT_PROFILE, DatasetProfile
from mas_treesearch.types import PromptSlots, UnionGraph

from .config import Stage2V41Config


class Stage2RuntimeV41(Stage2RuntimeV31):
    def __init__(self, config: Stage2V41Config, evaluator, agent_pool, embedder):
        super().__init__(config, evaluator, agent_pool, embedder)
        self.config = config
        self._last_v4_1_selection: Dict[str, Any] = {}

    @staticmethod
    def _challenger_slots() -> PromptSlots:
        return PromptSlots(
            reasoning_mode="critique_then_answer",
            upstream_usage="summary",
            output_style="raw",
            verification_mode="strict",
            finalization="answer_only",
        )

    @staticmethod
    def _inspector_slots() -> PromptSlots:
        return PromptSlots(
            reasoning_mode="critique_then_answer",
            upstream_usage="summary",
            output_style="raw",
            verification_mode="strict",
            finalization="answer_only",
        )

    @staticmethod
    def _ensure_v4_entry_fields(entry: Dict[str, Any]) -> None:
        entry.setdefault("explicit_challenger", False)
        entry.setdefault("challenger_agent_id", "")
        entry.setdefault("challenger_rationale", "")
        entry.setdefault("inspector_decision", "")
        entry.setdefault("inspector_rationale", "")

    @staticmethod
    def _normalize_vector(values: Sequence[float]) -> List[float]:
        norm = math.sqrt(sum(float(value) * float(value) for value in values))
        if norm <= 1e-9:
            return [0.0 for _ in values]
        return [float(value) / norm for value in values]

    @staticmethod
    def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0
        return float(sum(float(a) * float(b) for a, b in zip(left, right)))

    def _dar_retain_messages(
        self,
        messages: Sequence[ExportedMemoryMessage],
        *,
        max_items: int,
    ) -> List[ExportedMemoryMessage]:
        if len(messages) <= max_items:
            return list(messages)
        vectors = [self._normalize_vector(message.latent_vector) for message in messages]
        if not vectors or not vectors[0]:
            ranked = sorted(messages, key=lambda item: (item.confidence, item.node_id), reverse=True)
            return ranked[:max_items]

        centroid = [sum(values) / float(len(vectors)) for values in zip(*vectors)]
        centroid = self._normalize_vector(centroid)
        if centroid and any(abs(value) > 1e-9 for value in centroid):
            consensus_index = max(
                range(len(messages)),
                key=lambda index: (
                    self._cosine_similarity(vectors[index], centroid),
                    messages[index].confidence,
                    messages[index].node_id,
                ),
            )
        else:
            consensus_index = max(
                range(len(messages)),
                key=lambda index: (messages[index].confidence, messages[index].node_id),
            )

        selected = [consensus_index]
        while len(selected) < max_items:
            remaining = [index for index in range(len(messages)) if index not in selected]
            if not remaining:
                break
            next_index = max(
                remaining,
                key=lambda index: (
                    min(1.0 - self._cosine_similarity(vectors[index], vectors[chosen]) for chosen in selected),
                    messages[index].confidence,
                    messages[index].node_id,
                ),
            )
            selected.append(next_index)
        return [messages[index] for index in selected]

    def _neighbour_exports(
        self,
        node_id: str,
        active_edges,
        previous_exports: Dict[str, ExportedMemoryMessage],
    ) -> List[ExportedMemoryMessage]:
        messages = [
            previous_exports[activation.src]
            for activation in active_edges
            if activation.active and activation.dst == node_id and activation.src in previous_exports
        ]
        messages.sort(key=lambda item: (item.confidence, item.node_id), reverse=True)
        limit = max(1, int(self.config.memory.max_neighbour_exports))
        return self._dar_retain_messages(messages, max_items=limit)

    def _candidate_source_label(self, entry: Dict[str, Any]) -> str:
        if entry.get("explicit_challenger"):
            return "explicit_challenger"
        return super()._candidate_source_label(entry)

    def _serialize_candidate_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        payload = super()._serialize_candidate_entry(entry)
        payload.update(
            {
                "explicit_challenger": bool(entry.get("explicit_challenger", False)),
                "challenger_agent_id": str(entry.get("challenger_agent_id", "")),
                "challenger_rationale": str(entry.get("challenger_rationale", "")),
                "inspector_decision": str(entry.get("inspector_decision", "")),
                "inspector_rationale": str(entry.get("inspector_rationale", "")),
            }
        )
        return payload

    @staticmethod
    def _parse_tagged_output(raw_output: str) -> Dict[str, str]:
        parsed: Dict[str, str] = {}
        for raw_line in raw_output.splitlines():
            line = raw_line.strip()
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            parsed[key.strip().lower()] = value.strip()
        return parsed

    def _score_bank(
        self,
        bank: Dict[str, Dict[str, Any]],
        *,
        dataset_profile: DatasetProfile,
        total_turns: int,
        metadata: Optional[dict],
    ) -> Dict[str, Any]:
        for entry in bank.values():
            self._ensure_v4_entry_fields(entry)
            if dataset_profile.task_type == "code_generation" and entry.get("parse_ok") is None:
                parse_ok, entry_point_ok = self._code_candidate_checks(entry["text"], metadata=metadata)
                entry["parse_ok"] = parse_ok
                entry["entry_point_ok"] = entry_point_ok
            if float(entry.get("reviewer_event_count", 0.0)) > 0.0:
                entry["reviewer_mean_trust"] = float(entry.get("reviewer_trust_sum", 0.0)) / float(entry["reviewer_event_count"])
            entry["candidate_features"] = self._candidate_entry_features(
                entry,
                dataset_profile=dataset_profile,
                total_turns=total_turns,
            )
            model_score, uncertainty = self._candidate_model.predict(entry["candidate_features"])
            entry["candidate_model_score"] = float(self._clamp01(model_score))
            entry["candidate_model_uncertainty"] = float(uncertainty)
            entry["support_score"] = float(entry["candidate_model_score"])

        candidates = [dict(entry) for entry in bank.values()]
        candidates.sort(key=self._candidate_sort_key(dataset_profile), reverse=True)
        serialized = [self._serialize_candidate_entry(entry) for entry in candidates]
        return {
            "candidates": candidates,
            "candidates_serialized": serialized,
            "anchor": self._find_anchor_candidate(candidates),
            "anchor_serialized": self._find_anchor_candidate(serialized),
        }

    def _select_explicit_challenger_agents(self) -> List[str]:
        selected: List[str] = []
        for agent_id in self.config.explicit_challenger_agents:
            if agent_id in self._by_id and agent_id not in selected:
                selected.append(agent_id)
        if selected:
            return selected
        for fallback in ("skeptic", "debater_b", "verifier", "reasoner"):
            if fallback in self._by_id and fallback not in selected:
                selected.append(fallback)
        return selected

    def _generate_explicit_challengers(
        self,
        *,
        question_text: str,
        metadata: Optional[dict],
        dataset_profile: DatasetProfile,
        anchor: Dict[str, Any],
        candidates: Sequence[Dict[str, Any]],
        total_turns: int,
    ) -> List[Dict[str, Any]]:
        anchor_text = str(anchor.get("text", "")).strip()
        if not anchor_text:
            return []

        evidence_candidates = [
            entry
            for entry in candidates
            if str(entry.get("digest", "")) != str(anchor.get("digest", ""))
        ][: self.config.max_logged_candidates]
        evidence_lines = [
            f"[{index}] source={self._candidate_source_label(entry)} quality={self._quality_score(entry):.3f} "
            f"text={entry.get('text_preview', entry.get('text', ''))}"
            for index, entry in enumerate(evidence_candidates, start=1)
        ]
        evidence_block = "\n".join(evidence_lines) if evidence_lines else "No non-anchor candidates were produced by stage2."
        answer_contract = self.evaluator._output_contract(
            question_text,
            reference_answer=None,
            metadata=metadata,
        )
        entries: List[Dict[str, Any]] = []

        for agent_id in self._select_explicit_challenger_agents():
            agent = self._by_id[agent_id]
            system_prompt = build_system_prompt(agent, self._challenger_slots(), extra_role_hint="challenger")
            user_prompt = (
                f"Question:\n{render_question_text(question_text, metadata=metadata)}\n\n"
                f"Stage1 anchor:\n{anchor_text}\n\n"
                f"Observed non-anchor candidates:\n{evidence_block}\n\n"
                "Your task is to explicitly test whether the stage1 anchor should be overturned.\n"
                "If you cannot produce a materially better answer, preserve the anchor.\n"
                "Return exactly three lines:\n"
                "VERDICT: challenge|preserve\n"
                "FLAW: <the strongest flaw in the anchor, or none>\n"
                "ANSWER: <your final answer>\n"
            )
            if answer_contract:
                user_prompt += f"\nOutput contract:\n{answer_contract}\n"
            raw_output = self.evaluator._cached_chat(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                runtime=self.evaluator._resolve_runtime("tier2", dataset_profile),
            )
            parsed = self._parse_tagged_output(raw_output)
            verdict = str(parsed.get("verdict", "")).strip().lower()
            answer = self._sanitize_candidate(
                question_text,
                parsed.get("answer", ""),
                metadata=metadata,
            )
            if verdict != "challenge" or not answer or answer == anchor_text:
                continue

            entry = self._init_candidate_entry(answer)
            self._ensure_v4_entry_fields(entry)
            entry["occurrence_count"] = 1
            entry["sink_support"] = 1
            entry["source_node_ids"].add(f"challenger::{agent_id}")
            entry["source_roles"].add("challenger")
            entry["turn_indices"].add(total_turns)
            entry["explicit_challenger"] = True
            entry["challenger_agent_id"] = agent_id
            entry["challenger_rationale"] = str(parsed.get("flaw", "")).strip()
            entries.append(entry)
        return entries

    def _merge_candidate_entry(
        self,
        bank: Dict[str, Dict[str, Any]],
        incoming: Dict[str, Any],
    ) -> None:
        text = str(incoming.get("text", "")).strip()
        if not text:
            return
        existing = bank.get(text)
        if existing is None:
            self._ensure_v4_entry_fields(incoming)
            bank[text] = incoming
            return

        self._ensure_v4_entry_fields(existing)
        existing["occurrence_count"] += int(incoming.get("occurrence_count", 0))
        existing["sink_support"] += int(incoming.get("sink_support", 0))
        existing["source_node_ids"].update(incoming.get("source_node_ids", ()))
        existing["source_roles"].update(incoming.get("source_roles", ()))
        existing["turn_indices"].update(incoming.get("turn_indices", ()))
        existing["explicit_challenger"] = bool(existing.get("explicit_challenger", False) or incoming.get("explicit_challenger", False))
        if not existing.get("challenger_agent_id"):
            existing["challenger_agent_id"] = str(incoming.get("challenger_agent_id", ""))
        if not existing.get("challenger_rationale"):
            existing["challenger_rationale"] = str(incoming.get("challenger_rationale", ""))

    @classmethod
    def _checker_positive_labels(cls) -> set[str]:
        return {"pass", "preserve", "keep", "approve"}

    @classmethod
    def _checker_hard_veto_labels(cls) -> set[str]:
        return {"reject", "conflict"}

    @classmethod
    def _proposal_candidate_roles(cls) -> set[str]:
        return set(cls._candidate_roles()) - {"aggregator"}

    def _checker_snapshot(self, events) -> Dict[str, Any]:
        checker_events = [event for event in events if str(event.source_kind) in self._review_roles()]
        labels = [str(event.event_type) for event in checker_events]
        positive = any(label in self._checker_positive_labels() for label in labels)
        hard_veto = any(label in self._checker_hard_veto_labels() for label in labels)
        return {
            "labels": list(labels),
            "positive": bool(positive),
            "hard_veto": bool(hard_veto),
            "positive_without_veto": bool(positive and not hard_veto),
            "event_count": int(len(checker_events)),
        }

    @staticmethod
    def _is_recovery_output(node_trace) -> bool:
        metadata = dict(getattr(node_trace, "metadata", {}) or {})
        return bool(metadata.get("repair_branch") or metadata.get("recovery_output"))

    def _candidate_bank_admission(
        self,
        *,
        role: str,
        is_sink: bool,
        checker_snapshot: Dict[str, Any],
        is_recovery_output: bool,
    ) -> Tuple[bool, str]:
        if is_sink:
            return True, "sink_output"
        if is_recovery_output:
            return True, "recovery_output"
        if role == "aggregator" and checker_snapshot.get("positive_without_veto", False):
            return True, "checker_positive_aggregator"
        if role in self._proposal_candidate_roles() and checker_snapshot.get("positive_without_veto", False):
            return True, "checker_approved_proposal"
        return False, "filtered"

    def _candidate_bank_bundle(
        self,
        *,
        question_text: str,
        turn_traces: Sequence[TurnTrace],
        metadata: Optional[dict],
        dataset_profile: DatasetProfile,
    ) -> Dict[str, Any]:
        bank: Dict[str, Dict[str, Any]] = {}
        occurrence_map: Dict[Tuple[int, str], Dict[str, Any]] = {}
        feedback_by_occurrence = self._feedback_events_by_target_occurrence(turn_traces)
        stage1_output = str((metadata or {}).get("stage1_anchor_output", "")).strip()
        total_turns = max(1, len(turn_traces))

        def ensure_entry(text: str) -> Optional[Dict[str, Any]]:
            normalized = self._sanitize_candidate(question_text, text, metadata=metadata)
            if not normalized:
                return None
            entry = bank.get(normalized)
            if entry is None:
                entry = self._init_candidate_entry(normalized)
                self._ensure_v4_entry_fields(entry)
                bank[normalized] = entry
            return entry

        for turn_trace in turn_traces:
            sink_node_ids = set(turn_trace.sink_outputs)
            for node_trace in turn_trace.node_traces:
                if node_trace.role not in self._candidate_roles() and node_trace.node_id not in sink_node_ids:
                    continue
                normalized = self._sanitize_candidate(question_text, node_trace.output, metadata=metadata)
                if not normalized:
                    continue
                parse_ok = None
                entry_point_ok = None
                if dataset_profile.task_type == "code_generation":
                    parse_ok, entry_point_ok = self._code_candidate_checks(normalized, metadata=metadata)
                occurrence_key = (turn_trace.turn_index, node_trace.node_id)
                checker_snapshot = self._checker_snapshot(feedback_by_occurrence.get(occurrence_key, ()))
                admitted, admission_source = self._candidate_bank_admission(
                    role=node_trace.role,
                    is_sink=node_trace.node_id in sink_node_ids,
                    checker_snapshot=checker_snapshot,
                    is_recovery_output=self._is_recovery_output(node_trace),
                )
                occurrence_map[occurrence_key] = {
                    "digest": self._candidate_digest(normalized),
                    "role": node_trace.role,
                    "is_sink": node_trace.node_id in sink_node_ids,
                    "parse_ok": parse_ok,
                    "entry_point_ok": entry_point_ok,
                    "admitted_to_candidate_bank": bool(admitted),
                    "candidate_bank_source": admission_source,
                    "checker_labels": list(checker_snapshot["labels"]),
                    "checker_positive": bool(checker_snapshot["positive"]),
                    "checker_hard_veto": bool(checker_snapshot["hard_veto"]),
                }
                if not admitted:
                    continue
                entry = ensure_entry(normalized)
                if entry is None:
                    continue
                if dataset_profile.task_type == "code_generation":
                    entry["parse_ok"] = parse_ok
                    entry["entry_point_ok"] = entry_point_ok
                feedback_stats = self._aggregate_occurrence_feedback(
                    feedback_by_occurrence.get(occurrence_key, ()),
                    dataset_profile=dataset_profile,
                    target_role=node_trace.role,
                    target_is_sink=node_trace.node_id in sink_node_ids,
                    parse_ok=parse_ok,
                    entry_point_ok=entry_point_ok,
                )
                entry["occurrence_count"] += 1
                if node_trace.node_id in sink_node_ids:
                    entry["sink_support"] += 1
                entry["source_node_ids"].add(node_trace.node_id)
                entry["source_roles"].add(node_trace.role)
                entry["turn_indices"].add(turn_trace.turn_index)
                entry["feedback_pass"] += feedback_stats["pass_count"]
                entry["feedback_challenge"] += feedback_stats["challenge_count"]
                entry["feedback_uncertain"] += feedback_stats["uncertain_count"]
                entry["feedback_pass_calibrated"] += feedback_stats["pass_weight"]
                entry["feedback_challenge_calibrated"] += feedback_stats["challenge_weight"]
                entry["feedback_uncertain_calibrated"] += feedback_stats["uncertain_weight"]
                entry["reviewer_event_count"] += feedback_stats["event_count"]
                entry["reviewer_trust_sum"] += feedback_stats["trust_sum"]

        if stage1_output:
            anchor_entry = ensure_entry(stage1_output)
            if anchor_entry is not None:
                anchor_entry["stage1_anchor"] = True

        base_bundle = self._score_bank(
            bank,
            dataset_profile=dataset_profile,
            total_turns=total_turns,
            metadata=metadata,
        )
        anchor = base_bundle.get("anchor")
        if isinstance(anchor, dict):
            challengers = self._generate_explicit_challengers(
                question_text=question_text,
                metadata=metadata,
                dataset_profile=dataset_profile,
                anchor=anchor,
                candidates=base_bundle["candidates"],
                total_turns=total_turns,
            )
            for challenger in challengers:
                self._merge_candidate_entry(bank, challenger)

        scored_bundle = self._score_bank(
            bank,
            dataset_profile=dataset_profile,
            total_turns=total_turns,
            metadata=metadata,
        )
        scored_bundle["occurrences"] = occurrence_map
        return scored_bundle

    def _inspector_compare(
        self,
        *,
        question_text: str,
        metadata: Optional[dict],
        dataset_profile: DatasetProfile,
        anchor_text: str,
        challenger_text: str,
    ) -> Tuple[str, str]:
        inspector_id = self.config.override_inspector_agent_id
        if inspector_id not in self._by_id:
            inspector_id = "verifier" if "verifier" in self._by_id else next(iter(self._by_id))
        agent = self._by_id[inspector_id]
        answer_contract = self.evaluator._output_contract(
            question_text,
            reference_answer=None,
            metadata=metadata,
        )
        system_prompt = build_system_prompt(agent, self._inspector_slots(), extra_role_hint="inspector")
        user_prompt = (
            f"Question:\n{render_question_text(question_text, metadata=metadata)}\n\n"
            f"Stage1 anchor:\n{anchor_text}\n\n"
            f"Explicit challenger:\n{challenger_text}\n\n"
            "Decide which answer should survive as the final answer.\n"
            "Return exactly two lines:\n"
            "DECISION: anchor|challenger|uncertain\n"
            "RATIONALE: <short justification>\n"
        )
        if answer_contract:
            user_prompt += f"\nOutput contract:\n{answer_contract}\n"
        raw_output = self.evaluator._cached_chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            runtime=self.evaluator._resolve_runtime("tier2", dataset_profile),
        )
        parsed = self._parse_tagged_output(raw_output)
        decision = str(parsed.get("decision", "")).strip().lower()
        if decision not in {"anchor", "challenger", "uncertain"}:
            decision = "uncertain"
        return decision, str(parsed.get("rationale", "")).strip()

    def _select_against_anchor(
        self,
        candidates: Sequence[Dict[str, Any]],
        anchor: Optional[Dict[str, Any]],
        *,
        dataset_profile: DatasetProfile,
    ) -> Tuple[Optional[Dict[str, Any]], str, Dict[str, Any]]:
        if not candidates:
            reason = "v4_1_empty_bank_use_stage1" if anchor is not None else "v4_1_empty_bank"
            return anchor, reason, {
                "v4_1_pairwise_candidate_count": 0,
                "v4_1_explicit_challenger_count": 0,
                "v4_1_top_pairwise_candidates": [],
                **self._pairwise_extra(None, prefix="v4_1_best_pairwise"),
            }

        best_quality = dict(candidates[0])
        if anchor is None:
            return best_quality, "v4_1_no_stage1_anchor", {
                "v4_1_pairwise_candidate_count": 0,
                "v4_1_explicit_challenger_count": 0,
                "v4_1_top_pairwise_candidates": [],
                **self._pairwise_extra(None, prefix="v4_1_best_pairwise"),
            }

        comparisons = self._pairwise_candidate_comparisons(candidates, anchor, dataset_profile=dataset_profile)
        top_pairwise = [self._pairwise_public_view(item) for item in comparisons[: self.config.max_logged_candidates]]
        best_pairwise = comparisons[0] if comparisons else None
        explicit = [
            item
            for item in comparisons
            if bool(item["candidate"].get("explicit_challenger", False))
            and (dataset_profile.task_type != "code_generation" or bool(item.get("is_valid_code", False)))
        ]
        extra = {
            "v4_1_pairwise_candidate_count": int(len(comparisons)),
            "v4_1_explicit_challenger_count": int(len(explicit)),
            "v4_1_top_pairwise_candidates": top_pairwise,
            **self._pairwise_extra(best_pairwise, prefix="v4_1_best_pairwise"),
        }
        if not explicit:
            return anchor, "v4_1_preserve_no_explicit_challenger", extra

        chosen_item = explicit[0]
        inspector_decision, inspector_rationale = self._inspector_compare(
            question_text=self._last_candidate_bundle.get("question_text", ""),
            metadata=self._last_candidate_bundle.get("metadata"),
            dataset_profile=dataset_profile,
            anchor_text=str(anchor.get("text", "")),
            challenger_text=str(chosen_item["candidate"].get("text", "")),
        )
        chosen_item["candidate"]["inspector_decision"] = inspector_decision
        chosen_item["candidate"]["inspector_rationale"] = inspector_rationale
        extra.update(self._pairwise_extra(chosen_item, prefix="v4_1_decision_pairwise"))
        extra["v4_1_inspector_decision"] = inspector_decision
        extra["v4_1_inspector_rationale"] = inspector_rationale
        extra["v4_1_override_eligible"] = bool(inspector_decision == "challenger")
        if float(chosen_item.get("pairwise_probability", 0.0)) > 0.5 and inspector_decision == "challenger":
            return dict(chosen_item["candidate"]), "v4_1_override_explicit_challenger", extra
        if float(chosen_item.get("pairwise_probability", 0.0)) <= 0.5:
            return anchor, "v4_1_preserve_pairwise", extra
        return anchor, "v4_1_preserve_inspector", extra

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
        top = [dict(item) for item in list(candidates)[: self.config.max_logged_candidates]]
        for item in top:
            item.pop("text", None)
        extra = dict(selection_extra or {})
        self._last_v4_1_selection = {
            "stage2_version": "v4.1",
            "v4_1_task_type": task_type,
            "v4_1_candidate_count": int(len(candidates)),
            "v4_1_stage1_anchor_present": bool(anchor is not None),
            "v4_1_stage1_anchor_used": bool(anchor is not None and selected is not None and anchor.get("digest") == selected.get("digest")),
            "v4_1_selection_reason": strategy,
            "v4_1_selected_candidate_digest": str((selected or {}).get("digest", "")),
            "v4_1_selected_candidate_source": self._candidate_source_label(selected or {}),
            "v4_1_selected_is_explicit_challenger": bool((selected or {}).get("explicit_challenger", False)),
            "v4_1_selected_quality_score": self._quality_score(selected or {}),
            "v4_1_selected_support_score": float((selected or {}).get("support_score", 0.0)),
            "v4_1_selected_model_uncertainty": float((selected or {}).get("candidate_model_uncertainty", 0.0)),
            "v4_1_selected_sink_support": int((selected or {}).get("sink_support", 0)),
            "v4_1_selected_sink_ratio": self._sink_ratio(selected or {}),
            "v4_1_selected_source_diversity": self._source_diversity(selected or {}),
            "v4_1_selected_review_advantage": float(self._review_advantage(selected or {})),
            "v4_1_selected_review_consensus": float(self._review_consensus(selected or {})),
            "v4_1_stage1_anchor_digest": str((anchor or {}).get("digest", "")),
            "v4_1_stage1_anchor_quality_score": self._quality_score(anchor or {}),
            "v4_1_stage1_support_score": float((anchor or {}).get("support_score", 0.0)),
            "v4_1_stage1_model_uncertainty": float((anchor or {}).get("candidate_model_uncertainty", 0.0)),
            "v4_1_stage1_sink_support": int((anchor or {}).get("sink_support", 0)),
            "v4_1_stage1_sink_ratio": self._sink_ratio(anchor or {}),
            "v4_1_stage1_source_diversity": self._source_diversity(anchor or {}),
            "v4_1_stage1_review_advantage": float(self._review_advantage(anchor or {})),
            "v4_1_stage1_review_consensus": float(self._review_consensus(anchor or {})),
            "v4_1_candidate_model_steps": float(self._candidate_model.steps),
            "v4_1_pairwise_model_steps": float(self._pairwise_model.steps),
            "v4_1_reviewer_model_steps": float(self._reviewer_model.steps),
            "v4_1_top_candidates": top,
        }
        self._last_v4_1_selection.update(extra)

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
        del controller_state
        del sink_outputs
        del reference_answer
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
        selected, strategy, extra = self._select_against_anchor(candidates, anchor, dataset_profile=dataset_profile)
        selected_serialized = None
        if selected is not None:
            for item in candidates_serialized:
                if item.get("digest") == selected.get("digest"):
                    selected_serialized = dict(item)
                    break
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
        self._last_v4_1_selection = {}
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
            result.signature = "stage2_v4_1|" + result.signature[len("stage2_v2|") :]
        result.metadata.update(self._last_v4_1_selection)
        result.metadata["stage2_version"] = "v4.1"
        result.metadata["v4_1_dar_enabled"] = True
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
        with open(os.path.join(replay_dir, "v4_1_selection.json"), "w", encoding="utf-8") as handle:
            json.dump(self._last_v4_1_selection, handle, ensure_ascii=False, indent=2)
        with open(os.path.join(replay_dir, "final_answer.json"), "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "final_answer": result.final_answer,
                    "signature": result.signature,
                    "final_controller_state": asdict(result.final_controller_state),
                    "memory_record_counts": result.memory_record_counts,
                    "metadata": result.metadata | self._last_v4_1_selection,
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )
