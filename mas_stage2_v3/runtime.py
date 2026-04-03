from __future__ import annotations

import ast
import hashlib
import json
import math
import os
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from mas_stage2.learning import OnlineLinearModel
from mas_stage2.runtime import _truncate
from mas_stage2.runtime_v2 import Stage2RuntimeV2
from mas_stage2.types import FeedbackEvent, NodeTurnTrace, Stage2RunResult, TurnTrace
from mas_treesearch.profiles import DEFAULT_PROFILE, DatasetProfile
from mas_treesearch.types import EvalSummary, UnionGraph, UnionNode

from .config import Stage2V3Config


class Stage2RuntimeV3(Stage2RuntimeV2):
    def __init__(self, config: Stage2V3Config, evaluator, agent_pool, embedder):
        super().__init__(config, evaluator, agent_pool, embedder)
        self.config = config
        self._candidate_model = OnlineLinearModel(learning_rate=config.learning.selector_learning_rate)
        self._override_model = OnlineLinearModel(learning_rate=config.learning.edge_learning_rate)
        self._reviewer_model = OnlineLinearModel(learning_rate=config.learning.controller_learning_rate)
        self._last_v3_selection: Dict[str, Any] = {}
        self._last_candidate_bundle: Dict[str, Any] = {}

    def state_dict(self) -> dict:
        payload = super().state_dict()
        payload.pop("edge_model", None)
        payload.pop("controller_model", None)
        payload.update(
            {
                "v3_candidate_model": self._candidate_model.state_dict(),
                "v3_override_model": self._override_model.state_dict(),
                "v3_reviewer_model": self._reviewer_model.state_dict(),
            }
        )
        return payload

    def load_state_dict(self, state: dict) -> None:
        super().load_state_dict(state)
        candidate_state = state.get("v3_candidate_model")
        if isinstance(candidate_state, dict):
            self._candidate_model.load_state_dict(candidate_state)
        override_state = state.get("v3_override_model")
        if isinstance(override_state, dict):
            self._override_model.load_state_dict(override_state)
        reviewer_state = state.get("v3_reviewer_model")
        if isinstance(reviewer_state, dict):
            self._reviewer_model.load_state_dict(reviewer_state)

    @staticmethod
    def _candidate_roles() -> set[str]:
        return {"solver", "solver_a", "solver_b", "generator", "reviser", "aggregator"}

    @staticmethod
    def _review_roles() -> set[str]:
        return {"critic", "verifier", "judge"}

    @staticmethod
    def _candidate_digest(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]

    @staticmethod
    def _line_count(text: str) -> int:
        return max(1, len([line for line in text.splitlines() if line.strip()]))

    @staticmethod
    def _extract_verdict(text: str) -> str:
        for raw in text.splitlines():
            line = raw.strip()
            if line.lower().startswith("verdict:"):
                return line.split(":", 1)[1].strip().lower()
        return ""

    def _sanitize_candidate(
        self,
        question_text: str,
        raw_text: str,
        *,
        metadata: Optional[dict],
    ) -> str:
        return self.evaluator._sanitize_final_output(
            question_text,
            raw_text,
            reference_answer=None,
            metadata=metadata,
        )

    def _role_answer_contract(
        self,
        graph: UnionGraph,
        node: UnionNode,
        *,
        question_text: str,
        reference_answer: Optional[str],
        metadata: Optional[dict],
        dataset_profile: DatasetProfile,
    ) -> str:
        del reference_answer
        if dataset_profile.task_type == "code_generation":
            if node.role in self._review_roles():
                return (
                    "Do not output Python code.\n"
                    "Output exactly three lines:\n"
                    "VERDICT: pass|challenge|uncertain\n"
                    "ISSUES: <short text>\n"
                    "FIX: <short text or none>"
                )
            if node.role == "router":
                return (
                    "Do not output Python code.\n"
                    "Output exactly two lines:\n"
                    "FOCUS: <short text>\n"
                    "PLAN: <short text>"
                )
            if node.role in self._candidate_roles():
                return self.evaluator._output_contract(
                    question_text,
                    reference_answer=None,
                    metadata=metadata,
                )
        if node.role in {"aggregator", "reviser", "verifier", "judge"} or node.node_id in graph.sink_node_ids:
            return self.evaluator._output_contract(
                question_text,
                reference_answer=None,
                metadata=metadata,
            )
        return ""

    def _postprocess_node_output(
        self,
        node: UnionNode,
        raw_output: str,
        *,
        question_text: str,
        reference_answer: Optional[str],
        metadata: Optional[dict],
        dataset_profile: DatasetProfile,
        answer_contract: str,
    ) -> str:
        del reference_answer
        if dataset_profile.task_type == "code_generation":
            if node.role in self._candidate_roles():
                return self._sanitize_candidate(question_text, raw_output, metadata=metadata)
            return self.evaluator._strip_hidden_reasoning(raw_output)
        if answer_contract:
            return self._sanitize_candidate(question_text, raw_output, metadata=metadata)
        return self.evaluator._strip_hidden_reasoning(raw_output)

    def _run_task_node(
        self,
        graph: UnionGraph,
        node: UnionNode,
        *,
        question_text: str,
        metadata: Optional[dict],
        reference_answer: Optional[str],
        dataset_profile: DatasetProfile,
        controller_state,
        active_edges,
        previous_exports,
        episode_id: str,
        current_turn: int,
        prepared_state: Optional[Dict[str, object]] = None,
    ) -> Tuple[NodeTurnTrace, Any, Any]:
        trace, output_record, export_message = super()._run_task_node(
            graph,
            node,
            question_text=question_text,
            metadata=metadata,
            reference_answer=reference_answer,
            dataset_profile=dataset_profile,
            controller_state=controller_state,
            active_edges=active_edges,
            previous_exports=previous_exports,
            episode_id=episode_id,
            current_turn=current_turn,
            prepared_state=prepared_state,
        )
        typed_meta = dict(trace.metadata)
        if node.role in self._candidate_roles() or node.node_id in graph.sink_node_ids:
            candidate = self._sanitize_candidate(question_text, trace.output, metadata=metadata)
            typed_meta["typed_output_kind"] = "candidate" if candidate else "empty_candidate"
            typed_meta["candidate_digest"] = self._candidate_digest(candidate) if candidate else ""
            typed_meta["candidate_preview"] = _truncate(candidate, 200) if candidate else ""
            if dataset_profile.task_type == "code_generation" and candidate:
                parse_ok, entry_point_ok = self._code_candidate_checks(candidate, metadata=metadata)
                typed_meta["candidate_parse_ok"] = parse_ok
                typed_meta["candidate_entry_point_ok"] = entry_point_ok
        elif node.role in self._review_roles():
            typed_meta["typed_output_kind"] = "review"
            typed_meta["review_verdict"] = self._extract_verdict(trace.output)
        else:
            typed_meta["typed_output_kind"] = "analysis"
        trace.metadata = typed_meta
        return trace, output_record, export_message

    def _active_task_nodes_v2(
        self,
        graph: UnionGraph,
        task_nodes: Sequence[UnionNode],
        active_edges,
    ) -> List[UnionNode]:
        del graph
        del active_edges
        return list(task_nodes)

    def _feedback_events_by_target_occurrence(
        self,
        turn_traces: Sequence[TurnTrace],
    ) -> Dict[Tuple[int, str], List[FeedbackEvent]]:
        grouped: Dict[Tuple[int, str], List[FeedbackEvent]] = {}
        for turn_trace in turn_traces:
            for event in turn_trace.feedback_events:
                grouped.setdefault((event.turn_index, event.target_node_id), []).append(event)
        return grouped

    def _reviewer_features(
        self,
        event: FeedbackEvent,
        *,
        dataset_profile: DatasetProfile,
        target_role: str,
        target_is_sink: bool,
        parse_ok: Optional[bool],
        entry_point_ok: Optional[bool],
    ) -> Dict[str, float]:
        return {
            "bias": 1.0,
            f"task::{dataset_profile.task_type}": 1.0,
            f"reviewer::{event.source_kind}": 1.0,
            f"event::{event.event_type}": 1.0,
            f"target_role::{target_role}": 1.0,
            "confidence": float(event.confidence),
            "target_is_sink": 1.0 if target_is_sink else 0.0,
            "detail_length": min(1.5, math.log1p(len(event.detail or "")) / 5.5),
            "parse_ok": 1.0 if bool(parse_ok) else 0.0,
            "entry_point_ok": 1.0 if bool(entry_point_ok) else 0.0,
        }

    def _candidate_entry_features(
        self,
        entry: Dict[str, Any],
        *,
        dataset_profile: DatasetProfile,
        total_turns: int,
    ) -> Dict[str, float]:
        review_balance = self._review_advantage(entry)
        review_volume = (
            float(entry["feedback_pass_calibrated"])
            + float(entry["feedback_challenge_calibrated"])
            + float(entry["feedback_uncertain_calibrated"])
        )
        features = {
            "bias": 1.0,
            f"task::{dataset_profile.task_type}": 1.0,
            "occurrence_count": math.log1p(float(entry["occurrence_count"])),
            "sink_support": math.log1p(float(entry["sink_support"])),
            "turn_coverage": len(entry["turn_indices"]) / max(1.0, float(total_turns)),
            "feedback_pass_raw": float(entry["feedback_pass"]),
            "feedback_challenge_raw": float(entry["feedback_challenge"]),
            "feedback_uncertain_raw": float(entry["feedback_uncertain"]),
            "feedback_pass_calibrated": float(entry["feedback_pass_calibrated"]),
            "feedback_challenge_calibrated": float(entry["feedback_challenge_calibrated"]),
            "feedback_uncertain_calibrated": float(entry["feedback_uncertain_calibrated"]),
            "review_balance": float(review_balance),
            "review_volume": float(review_volume),
            "reviewer_event_count": math.log1p(float(entry["reviewer_event_count"])),
            "reviewer_mean_trust": float(entry["reviewer_mean_trust"]),
            "text_length": min(1.5, math.log1p(float(entry["text_length"])) / 6.5),
            "line_count": min(1.5, math.log1p(float(entry["line_count"])) / 4.5),
            "stage1_anchor": 1.0 if entry.get("stage1_anchor") else 0.0,
            "parse_ok": 1.0 if bool(entry.get("parse_ok")) else 0.0,
            "entry_point_ok": 1.0 if bool(entry.get("entry_point_ok")) else 0.0,
        }
        for role in entry["source_roles"]:
            features[f"source_role::{role}"] = 1.0
        return features

    def _override_features(
        self,
        challenger: Dict[str, Any],
        anchor: Dict[str, Any],
        *,
        dataset_profile: DatasetProfile,
    ) -> Dict[str, float]:
        challenger_score = float(challenger.get("support_score", 0.0))
        anchor_score = float(anchor.get("support_score", 0.0))
        challenger_review = self._review_advantage(challenger)
        anchor_review = self._review_advantage(anchor)
        return {
            "bias": 1.0,
            f"task::{dataset_profile.task_type}": 1.0,
            "challenger_score": challenger_score,
            "anchor_score": anchor_score,
            "score_margin": challenger_score - anchor_score,
            "challenger_occurrence": math.log1p(float(challenger["occurrence_count"])),
            "anchor_occurrence": math.log1p(float(anchor["occurrence_count"])),
            "occurrence_margin": math.log1p(float(challenger["occurrence_count"])) - math.log1p(float(anchor["occurrence_count"])),
            "challenger_sink_support": math.log1p(float(challenger["sink_support"])),
            "anchor_sink_support": math.log1p(float(anchor["sink_support"])),
            "sink_margin": math.log1p(float(challenger["sink_support"])) - math.log1p(float(anchor["sink_support"])),
            "challenger_review_balance": float(challenger_review),
            "anchor_review_balance": float(anchor_review),
            "review_margin": float(challenger_review - anchor_review),
            "challenger_pass": float(challenger["feedback_pass_calibrated"]),
            "anchor_pass": float(anchor["feedback_pass_calibrated"]),
            "challenger_challenge": float(challenger["feedback_challenge_calibrated"]),
            "anchor_challenge": float(anchor["feedback_challenge_calibrated"]),
            "challenger_stage1_anchor": 1.0 if challenger.get("stage1_anchor") else 0.0,
            "anchor_stage1_anchor": 1.0 if anchor.get("stage1_anchor") else 0.0,
            "challenger_parse_ok": 1.0 if bool(challenger.get("parse_ok")) else 0.0,
            "anchor_parse_ok": 1.0 if bool(anchor.get("parse_ok")) else 0.0,
            "challenger_entry_point_ok": 1.0 if bool(challenger.get("entry_point_ok")) else 0.0,
            "anchor_entry_point_ok": 1.0 if bool(anchor.get("entry_point_ok")) else 0.0,
            "challenger_model_uncertainty": float(challenger.get("candidate_model_uncertainty", 0.0)),
            "anchor_model_uncertainty": float(anchor.get("candidate_model_uncertainty", 0.0)),
        }

    def _reviewer_weight(self, features: Dict[str, float], confidence: float) -> Tuple[float, float]:
        trust, uncertainty = self._reviewer_model.predict(features)
        return float(confidence) * (0.5 + float(trust)), float(trust)

    def _aggregate_occurrence_feedback(
        self,
        events: Sequence[FeedbackEvent],
        *,
        dataset_profile: DatasetProfile,
        target_role: str,
        target_is_sink: bool,
        parse_ok: Optional[bool],
        entry_point_ok: Optional[bool],
    ) -> Dict[str, float]:
        stats = {
            "pass_count": 0.0,
            "challenge_count": 0.0,
            "uncertain_count": 0.0,
            "pass_weight": 0.0,
            "challenge_weight": 0.0,
            "uncertain_weight": 0.0,
            "event_count": 0.0,
            "trust_sum": 0.0,
        }
        for event in events:
            features = self._reviewer_features(
                event,
                dataset_profile=dataset_profile,
                target_role=target_role,
                target_is_sink=target_is_sink,
                parse_ok=parse_ok,
                entry_point_ok=entry_point_ok,
            )
            weight, trust = self._reviewer_weight(features, event.confidence)
            stats["event_count"] += 1.0
            stats["trust_sum"] += float(trust)
            if event.event_type in {"pass", "preserve"}:
                stats["pass_count"] += 1.0
                stats["pass_weight"] += float(weight)
            elif event.event_type in {"challenge", "reject", "conflict"}:
                stats["challenge_count"] += 1.0
                stats["challenge_weight"] += float(weight)
            else:
                stats["uncertain_count"] += 1.0
                stats["uncertain_weight"] += float(weight)
        return stats

    def _init_candidate_entry(self, text: str) -> Dict[str, Any]:
        return {
            "text": text,
            "digest": self._candidate_digest(text),
            "text_preview": _truncate(text, 320),
            "text_length": len(text),
            "line_count": self._line_count(text),
            "occurrence_count": 0,
            "sink_support": 0,
            "source_node_ids": set(),
            "source_roles": set(),
            "turn_indices": set(),
            "feedback_pass": 0.0,
            "feedback_challenge": 0.0,
            "feedback_uncertain": 0.0,
            "feedback_pass_calibrated": 0.0,
            "feedback_challenge_calibrated": 0.0,
            "feedback_uncertain_calibrated": 0.0,
            "reviewer_event_count": 0.0,
            "reviewer_trust_sum": 0.0,
            "reviewer_mean_trust": 0.5,
            "stage1_anchor": False,
            "parse_ok": None,
            "entry_point_ok": None,
            "candidate_features": {},
            "candidate_model_score": 0.5,
            "candidate_model_uncertainty": 0.2,
            "support_score": 0.5,
        }

    @staticmethod
    def _serialize_candidate_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "text": entry["text"],
            "digest": entry["digest"],
            "text_preview": entry["text_preview"],
            "text_length": int(entry["text_length"]),
            "line_count": int(entry["line_count"]),
            "occurrence_count": int(entry["occurrence_count"]),
            "sink_support": int(entry["sink_support"]),
            "source_node_ids": sorted(entry["source_node_ids"]),
            "source_roles": sorted(entry["source_roles"]),
            "turn_indices": sorted(entry["turn_indices"]),
            "feedback_pass": float(entry["feedback_pass"]),
            "feedback_challenge": float(entry["feedback_challenge"]),
            "feedback_uncertain": float(entry["feedback_uncertain"]),
            "feedback_pass_calibrated": float(entry["feedback_pass_calibrated"]),
            "feedback_challenge_calibrated": float(entry["feedback_challenge_calibrated"]),
            "feedback_uncertain_calibrated": float(entry["feedback_uncertain_calibrated"]),
            "reviewer_event_count": float(entry["reviewer_event_count"]),
            "reviewer_mean_trust": float(entry["reviewer_mean_trust"]),
            "stage1_anchor": bool(entry["stage1_anchor"]),
            "parse_ok": entry["parse_ok"],
            "entry_point_ok": entry["entry_point_ok"],
            "candidate_model_score": float(entry["candidate_model_score"]),
            "candidate_model_uncertainty": float(entry["candidate_model_uncertainty"]),
            "support_score": float(entry["support_score"]),
        }

    @staticmethod
    def _find_anchor_candidate(candidates: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        for entry in candidates:
            if entry.get("stage1_anchor"):
                return dict(entry)
        return None

    @staticmethod
    def _review_advantage(entry: Dict[str, Any]) -> float:
        if "feedback_pass_calibrated" in entry:
            return float(entry.get("feedback_pass_calibrated", 0.0)) - float(entry.get("feedback_challenge_calibrated", 0.0))
        return float(entry.get("feedback_pass", 0.0)) - float(entry.get("feedback_challenge", 0.0))

    @staticmethod
    def _candidate_source_label(entry: Dict[str, Any]) -> str:
        if entry.get("stage1_anchor") and entry.get("source_roles"):
            return "stage1+stage2"
        if entry.get("stage1_anchor"):
            return "stage1"
        return "stage2"

    def _code_candidate_checks(self, text: str, *, metadata: Optional[dict]) -> Tuple[bool, bool]:
        parse_ok = True
        try:
            tree = ast.parse(text)
        except Exception:
            parse_ok = False
            tree = None
        entry_point = str((metadata or {}).get("entry_point") or "").strip()
        if not entry_point or not self.config.code_require_entry_point:
            return parse_ok, parse_ok
        if not parse_ok or tree is None:
            return False, False
        entry_ok = any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == entry_point
            for node in ast.walk(tree)
        )
        return True, bool(entry_ok)

    def _candidate_sort_key(self, dataset_profile: DatasetProfile):
        if dataset_profile.task_type == "code_generation":
            return lambda entry: (
                1 if entry.get("parse_ok") else 0,
                1 if entry.get("entry_point_ok") else 0,
                float(entry.get("candidate_model_score", entry.get("support_score", 0.0))),
                float(self._review_advantage(entry)),
                int(entry["sink_support"]),
                int(entry["occurrence_count"]),
                -int(entry["line_count"]),
                str(entry["digest"]),
            )
        return lambda entry: (
            float(entry.get("candidate_model_score", entry.get("support_score", 0.0))),
            float(self._review_advantage(entry)),
            int(entry["sink_support"]),
            int(entry["occurrence_count"]),
            -int(entry["text_length"]),
            str(entry["digest"]),
        )

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
                bank[normalized] = entry
            return entry

        for turn_trace in turn_traces:
            sink_node_ids = set(turn_trace.sink_outputs)
            for node_trace in turn_trace.node_traces:
                if node_trace.role not in self._candidate_roles() and node_trace.node_id not in sink_node_ids:
                    continue
                entry = ensure_entry(node_trace.output)
                if entry is None:
                    continue
                parse_ok = None
                entry_point_ok = None
                if dataset_profile.task_type == "code_generation":
                    parse_ok, entry_point_ok = self._code_candidate_checks(entry["text"], metadata=metadata)
                    entry["parse_ok"] = parse_ok
                    entry["entry_point_ok"] = entry_point_ok
                occurrence_key = (turn_trace.turn_index, node_trace.node_id)
                occurrence_map[occurrence_key] = {
                    "digest": entry["digest"],
                    "role": node_trace.role,
                    "is_sink": node_trace.node_id in sink_node_ids,
                    "parse_ok": parse_ok,
                    "entry_point_ok": entry_point_ok,
                }
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

        for entry in bank.values():
            if dataset_profile.task_type == "code_generation" and entry["parse_ok"] is None:
                parse_ok, entry_point_ok = self._code_candidate_checks(entry["text"], metadata=metadata)
                entry["parse_ok"] = parse_ok
                entry["entry_point_ok"] = entry_point_ok
            if entry["reviewer_event_count"] > 0:
                entry["reviewer_mean_trust"] = float(entry["reviewer_trust_sum"]) / float(entry["reviewer_event_count"])
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
        anchor = self._find_anchor_candidate(candidates)
        anchor_serialized = self._find_anchor_candidate(serialized)
        return {
            "candidates": candidates,
            "candidates_serialized": serialized,
            "anchor": anchor,
            "anchor_serialized": anchor_serialized,
            "occurrences": occurrence_map,
        }

    def _override_probability(
        self,
        challenger: Dict[str, Any],
        anchor: Dict[str, Any],
        *,
        dataset_profile: DatasetProfile,
    ) -> Tuple[float, float, Dict[str, float]]:
        features = self._override_features(challenger, anchor, dataset_profile=dataset_profile)
        probability, uncertainty = self._override_model.predict(features)
        return float(self._clamp01(probability)), float(uncertainty), features

    @staticmethod
    def _is_valid_code_candidate(entry: Dict[str, Any]) -> bool:
        return bool(entry.get("parse_ok")) and bool(entry.get("entry_point_ok"))

    def _select_numeric_candidate(
        self,
        candidates: Sequence[Dict[str, Any]],
        anchor: Optional[Dict[str, Any]],
        *,
        dataset_profile: DatasetProfile,
    ) -> Tuple[Optional[Dict[str, Any]], str, Dict[str, Any]]:
        if not candidates:
            return anchor, "v3_numeric_empty_bank_use_stage1" if anchor is not None else "v3_numeric_empty_bank", {}
        best = dict(candidates[0])
        if anchor is None:
            return best, "v3_numeric_no_stage1_anchor", {}
        if best["digest"] == anchor["digest"]:
            return anchor, "v3_numeric_keep_stage1_anchor", {}
        probability, uncertainty, _ = self._override_probability(best, anchor, dataset_profile=dataset_profile)
        extra = {
            "v3_override_probability": probability,
            "v3_override_uncertainty": uncertainty,
            "v3_override_candidate_digest": best["digest"],
        }
        if probability > 0.5:
            return best, "v3_numeric_override_stage1_learned", extra
        return anchor, "v3_numeric_preserve_stage1_gate", extra

    def _select_code_candidate(
        self,
        candidates: Sequence[Dict[str, Any]],
        anchor: Optional[Dict[str, Any]],
        *,
        dataset_profile: DatasetProfile,
    ) -> Tuple[Optional[Dict[str, Any]], str, Dict[str, Any]]:
        if not candidates:
            return anchor, "v3_code_empty_bank_use_stage1" if anchor is not None else "v3_code_empty_bank", {}
        best = dict(candidates[0])
        if anchor is None:
            return best, "v3_code_no_stage1_anchor", {}
        if best["digest"] == anchor["digest"]:
            return anchor, "v3_code_keep_stage1_anchor", {}
        best_valid = self._is_valid_code_candidate(best)
        anchor_valid = self._is_valid_code_candidate(anchor)
        if not best_valid:
            return anchor, "v3_code_preserve_stage1_invalid_override", {}
        if not anchor_valid:
            return best, "v3_code_override_invalid_stage1", {}
        probability, uncertainty, _ = self._override_probability(best, anchor, dataset_profile=dataset_profile)
        extra = {
            "v3_override_probability": probability,
            "v3_override_uncertainty": uncertainty,
            "v3_override_candidate_digest": best["digest"],
        }
        if probability > 0.5:
            return best, "v3_code_override_stage1_learned", extra
        return anchor, "v3_code_preserve_stage1_gate", extra

    def _select_generic_candidate(
        self,
        candidates: Sequence[Dict[str, Any]],
        anchor: Optional[Dict[str, Any]],
        *,
        dataset_profile: DatasetProfile,
    ) -> Tuple[Optional[Dict[str, Any]], str, Dict[str, Any]]:
        if not candidates:
            return anchor, "v3_generic_empty_bank_use_stage1" if anchor is not None else "v3_generic_empty_bank", {}
        best = dict(candidates[0])
        if anchor is None:
            return best, "v3_generic_no_stage1_anchor", {}
        if best["digest"] == anchor["digest"]:
            return anchor, "v3_generic_keep_stage1_anchor", {}
        probability, uncertainty, _ = self._override_probability(best, anchor, dataset_profile=dataset_profile)
        extra = {
            "v3_override_probability": probability,
            "v3_override_uncertainty": uncertainty,
            "v3_override_candidate_digest": best["digest"],
        }
        if probability > 0.5:
            return best, "v3_generic_override_stage1_learned", extra
        return anchor, "v3_generic_preserve_stage1_gate", extra

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
        self._last_v3_selection = {
            "stage2_version": "v3",
            "v3_task_type": task_type,
            "v3_candidate_count": int(len(candidates)),
            "v3_stage1_anchor_present": bool(anchor is not None),
            "v3_stage1_anchor_used": bool(anchor is not None and selected is not None and anchor["digest"] == selected["digest"]),
            "v3_selection_reason": strategy,
            "v3_selected_candidate_digest": str((selected or {}).get("digest", "")),
            "v3_selected_candidate_source": self._candidate_source_label(selected or {}),
            "v3_selected_support_score": float((selected or {}).get("support_score", 0.0)),
            "v3_selected_model_uncertainty": float((selected or {}).get("candidate_model_uncertainty", 0.0)),
            "v3_selected_sink_support": int((selected or {}).get("sink_support", 0)),
            "v3_selected_review_advantage": float(self._review_advantage(selected or {})),
            "v3_stage1_anchor_digest": str((anchor or {}).get("digest", "")),
            "v3_stage1_support_score": float((anchor or {}).get("support_score", 0.0)),
            "v3_stage1_model_uncertainty": float((anchor or {}).get("candidate_model_uncertainty", 0.0)),
            "v3_stage1_sink_support": int((anchor or {}).get("sink_support", 0)),
            "v3_candidate_model_steps": float(self._candidate_model.steps),
            "v3_override_model_steps": float(self._override_model.steps),
            "v3_reviewer_model_steps": float(self._reviewer_model.steps),
            "v3_top_candidates": top,
        }
        self._last_v3_selection.update(extra)

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
        self._last_candidate_bundle = bundle
        candidates = bundle["candidates"]
        candidates_serialized = bundle["candidates_serialized"]
        anchor = bundle["anchor"]
        anchor_serialized = bundle["anchor_serialized"]
        if dataset_profile.task_type in {"numeric", "math_expression"}:
            selected, strategy, extra = self._select_numeric_candidate(candidates, anchor, dataset_profile=dataset_profile)
        elif dataset_profile.task_type == "code_generation":
            selected, strategy, extra = self._select_code_candidate(candidates, anchor, dataset_profile=dataset_profile)
        else:
            selected, strategy, extra = self._select_generic_candidate(candidates, anchor, dataset_profile=dataset_profile)
        selected_serialized = None
        if selected is not None:
            for item in candidates_serialized:
                if item["digest"] == selected["digest"]:
                    selected_serialized = dict(item)
                    break
        if selected is None:
            final_answer = ""
        else:
            final_answer = str(selected.get("text", ""))
        self._record_selection_metadata(
            task_type=dataset_profile.task_type,
            candidates=candidates_serialized,
            selected=selected_serialized,
            anchor=anchor_serialized,
            strategy=strategy,
            selection_extra=extra,
        )
        return final_answer, strategy

    def _candidate_target_from_summary(self, summary: EvalSummary) -> float:
        return self._summary_target(summary, reward_target=None)

    def _evaluate_candidate_targets(
        self,
        candidates: Sequence[Dict[str, Any]],
        *,
        question_text: str,
        reference_answer: Optional[str],
        metadata: Optional[dict],
        dataset_profile: DatasetProfile,
    ) -> Dict[str, float]:
        targets: Dict[str, float] = {}
        for entry in candidates:
            summary = self.evaluator.evaluate_output(
                question_text,
                entry["text"],
                tier="tier2",
                reference_answer=reference_answer,
                metadata=metadata,
                dataset_profile=dataset_profile,
            )
            targets[entry["digest"]] = self._candidate_target_from_summary(summary)
        return targets

    @staticmethod
    def _reviewer_target(event_type: str, candidate_target: float) -> float:
        if event_type in {"pass", "preserve"}:
            return float(candidate_target)
        if event_type in {"challenge", "reject", "conflict"}:
            return float(1.0 - candidate_target)
        return max(0.0, 1.0 - abs(float(candidate_target) - 0.5) * 2.0)

    def learn_from_run(
        self,
        graph: UnionGraph,
        result: Stage2RunResult,
        *,
        dataset_profile: DatasetProfile,
        summary: Optional[EvalSummary] = None,
        reward_target: Optional[float] = None,
        question_text: Optional[str] = None,
        reference_answer: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> Dict[str, float]:
        stats = dict(super().learn_from_run(
            graph,
            result,
            dataset_profile=dataset_profile,
            summary=summary,
            reward_target=reward_target,
        ))
        if not self.config.learning.enabled or not question_text:
            stats.update(
                {
                    "candidate_updates": 0.0,
                    "override_updates": 0.0,
                    "reviewer_updates": 0.0,
                    "candidate_steps": float(self._candidate_model.steps),
                    "override_steps": float(self._override_model.steps),
                    "reviewer_steps": float(self._reviewer_model.steps),
                }
            )
            return stats

        bundle = self._last_candidate_bundle or self._candidate_bank_bundle(
            question_text=question_text,
            turn_traces=result.turn_traces,
            metadata=metadata,
            dataset_profile=dataset_profile,
        )
        ranked_candidates = list(bundle.get("candidates", []))
        if not ranked_candidates:
            stats.update(
                {
                    "candidate_updates": 0.0,
                    "override_updates": 0.0,
                    "reviewer_updates": 0.0,
                    "candidate_steps": float(self._candidate_model.steps),
                    "override_steps": float(self._override_model.steps),
                    "reviewer_steps": float(self._reviewer_model.steps),
                }
            )
            return stats

        learning_candidates: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for entry in ranked_candidates:
            digest = str(entry["digest"])
            if digest in seen:
                continue
            learning_candidates.append(entry)
            seen.add(digest)
            if len(learning_candidates) >= self.config.max_logged_candidates:
                break
        anchor = bundle.get("anchor")
        if isinstance(anchor, dict) and anchor.get("digest") not in seen:
            learning_candidates.append(anchor)

        candidate_targets = self._evaluate_candidate_targets(
            learning_candidates,
            question_text=question_text,
            reference_answer=reference_answer,
            metadata=metadata,
            dataset_profile=dataset_profile,
        )

        candidate_updates = 0
        for entry in learning_candidates:
            features = entry.get("candidate_features")
            if not isinstance(features, dict) or not features:
                continue
            target = candidate_targets.get(entry["digest"])
            if target is None:
                continue
            self._candidate_model.update({str(name): float(value) for name, value in features.items()}, target)
            candidate_updates += 1

        override_updates = 0
        if isinstance(anchor, dict):
            anchor_target = candidate_targets.get(anchor["digest"])
            if anchor_target is not None:
                for challenger in learning_candidates:
                    if challenger["digest"] == anchor["digest"]:
                        continue
                    challenger_target = candidate_targets.get(challenger["digest"])
                    if challenger_target is None:
                        continue
                    override_target = 1.0 if challenger_target > anchor_target else 0.0
                    if dataset_profile.task_type == "code_generation":
                        challenger_valid = self._is_valid_code_candidate(challenger)
                        anchor_valid = self._is_valid_code_candidate(anchor)
                        if not challenger_valid:
                            override_target = 0.0
                        elif not anchor_valid and challenger_valid and challenger_target > 0.0:
                            override_target = 1.0
                    features = self._override_features(challenger, anchor, dataset_profile=dataset_profile)
                    self._override_model.update(features, override_target)
                    override_updates += 1

        reviewer_updates = 0
        occurrences = bundle.get("occurrences", {})
        candidate_by_digest = {str(entry["digest"]): entry for entry in learning_candidates}
        for turn_trace in result.turn_traces:
            for event in turn_trace.feedback_events:
                occurrence = occurrences.get((event.turn_index, event.target_node_id))
                if not isinstance(occurrence, dict):
                    continue
                digest = str(occurrence.get("digest", ""))
                target = candidate_targets.get(digest)
                candidate_entry = candidate_by_digest.get(digest)
                if target is None or candidate_entry is None:
                    continue
                features = self._reviewer_features(
                    event,
                    dataset_profile=dataset_profile,
                    target_role=str(occurrence.get("role", "")),
                    target_is_sink=bool(occurrence.get("is_sink", False)),
                    parse_ok=occurrence.get("parse_ok"),
                    entry_point_ok=occurrence.get("entry_point_ok"),
                )
                reviewer_target = self._reviewer_target(event.event_type, target)
                self._reviewer_model.update(features, reviewer_target)
                reviewer_updates += 1

        stats.update(
            {
                "candidate_updates": float(candidate_updates),
                "override_updates": float(override_updates),
                "reviewer_updates": float(reviewer_updates),
                "candidate_steps": float(self._candidate_model.steps),
                "override_steps": float(self._override_model.steps),
                "reviewer_steps": float(self._reviewer_model.steps),
            }
        )
        return stats

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
        self._last_v3_selection = {}
        self._last_candidate_bundle = {}
        result = super().run(
            graph,
            question_text=question_text,
            metadata=metadata,
            reference_answer=reference_answer,
            dataset_profile=dataset_profile,
            replay_dir=replay_dir,
            learn=learn,
        )
        if result.signature.startswith("stage2_v2|"):
            result.signature = "stage2_v3|" + result.signature[len("stage2_v2|") :]
        result.metadata.update(self._last_v3_selection)
        result.metadata["stage2_version"] = "v3"
        result.metadata["v3_all_task_nodes_each_turn"] = True
        return result

    def save_replay_bundle(
        self,
        result: Stage2RunResult,
        replay_dir: str,
        *,
        question_text: str,
        metadata: Optional[dict],
    ) -> None:
        super().save_replay_bundle(result, replay_dir, question_text=question_text, metadata=metadata)
        with open(os.path.join(replay_dir, "v3_selection.json"), "w", encoding="utf-8") as handle:
            json.dump(self._last_v3_selection, handle, ensure_ascii=False, indent=2)
        with open(os.path.join(replay_dir, "final_answer.json"), "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "final_answer": result.final_answer,
                    "signature": result.signature,
                    "final_controller_state": asdict(result.final_controller_state),
                    "memory_record_counts": result.memory_record_counts,
                    "metadata": result.metadata | self._last_v3_selection,
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )
