from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .runtime import _truncate
from .runtime_v2 import Stage2RuntimeV2
from mas_stage2.types import Stage2RunResult, TurnTrace
from .runtime_v41 import Stage2RuntimeV41
from mas_treesearch.prompting import build_system_prompt, render_question_text
from mas_treesearch.profiles import DEFAULT_PROFILE, DatasetProfile
from mas_treesearch.types import PromptSlots, UnionGraph

from .config import Stage2V42Config


class Stage2RuntimeV42(Stage2RuntimeV41):
    def __init__(self, config: Stage2V42Config, evaluator, agent_pool, embedder):
        super().__init__(config, evaluator, agent_pool, embedder)
        self.config = config
        self._last_v4_2_selection: Dict[str, Any] = {}

    @staticmethod
    def _auditor_slots() -> PromptSlots:
        return PromptSlots(
            reasoning_mode="critique_then_answer",
            upstream_usage="summary",
            output_style="raw",
            verification_mode="strict",
            finalization="answer_only",
        )

    @staticmethod
    def _adjudicator_slots() -> PromptSlots:
        return PromptSlots(
            reasoning_mode="critique_then_answer",
            upstream_usage="summary",
            output_style="raw",
            verification_mode="strict",
            finalization="answer_only",
        )

    @staticmethod
    def _calibration_slots() -> PromptSlots:
        return PromptSlots(
            reasoning_mode="critique_then_answer",
            upstream_usage="summary",
            output_style="raw",
            verification_mode="strict",
            finalization="answer_only",
        )

    def _ensure_v4_2_entry_fields(self, entry: Dict[str, Any]) -> None:
        self._ensure_v4_entry_fields(entry)
        entry.setdefault("provisional_challenger", bool(entry.get("explicit_challenger", False)))
        entry.setdefault(
            "promotion_source",
            "explicit_challenger" if bool(entry.get("explicit_challenger", False)) else "",
        )
        entry.setdefault("promotion_rationale", "")
        entry.setdefault("semantic_distance_to_anchor", 0.0)
        entry.setdefault("anomaly_support_count", 0)
        entry.setdefault("auditor_support_agents", set())
        entry.setdefault("auditor_findings", [])
        entry.setdefault("local_consistency_decision", "")

    def _candidate_source_label(self, entry: Dict[str, Any]) -> str:
        if entry.get("explicit_challenger"):
            return "explicit_challenger"
        if entry.get("provisional_challenger"):
            return "provisional_challenger"
        return super()._candidate_source_label(entry)

    def _serialize_candidate_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        payload = super()._serialize_candidate_entry(entry)
        support_agents = entry.get("auditor_support_agents", ())
        if isinstance(support_agents, set):
            support_agents = sorted(support_agents)
        payload.update(
            {
                "provisional_challenger": bool(entry.get("provisional_challenger", False)),
                "promotion_source": str(entry.get("promotion_source", "")),
                "promotion_rationale": str(entry.get("promotion_rationale", "")),
                "semantic_distance_to_anchor": float(entry.get("semantic_distance_to_anchor", 0.0)),
                "anomaly_support_count": int(entry.get("anomaly_support_count", 0)),
                "auditor_support_agents": list(support_agents),
                "auditor_findings": list(entry.get("auditor_findings", ())),
                "local_consistency_decision": str(entry.get("local_consistency_decision", "")),
            }
        )
        return payload

    def _semantic_distance(self, left_text: str, right_text: str) -> float:
        left = self.embedder.embed(left_text)
        right = self.embedder.embed(right_text)
        return float(1.0 - self._cosine_similarity(left, right))

    def _select_auditor_agents(self) -> List[str]:
        selected: List[str] = []
        for agent_id in self.config.auditor_agent_ids:
            if agent_id in self._by_id and agent_id not in selected:
                selected.append(agent_id)
        if selected:
            return selected
        for fallback in ("verifier", "skeptic", "reasoner", "critic"):
            if fallback in self._by_id and fallback not in selected:
                selected.append(fallback)
        return selected[:2]

    def _select_single_agent(self, configured_id: str, fallbacks: Sequence[str]) -> str:
        if configured_id in self._by_id:
            return configured_id
        for agent_id in fallbacks:
            if agent_id in self._by_id:
                return agent_id
        return next(iter(self._by_id))

    def _candidate_evidence_summary(
        self,
        entry: Dict[str, Any],
        *,
        anchor: Dict[str, Any],
        pairwise_probability: float,
    ) -> str:
        return (
            f"digest={entry.get('digest', '')}\n"
            f"source={self._candidate_source_label(entry)}\n"
            f"quality={self._quality_score(entry):.3f}\n"
            f"review_advantage={self._review_advantage(entry):.3f}\n"
            f"review_consensus={self._review_consensus(entry):.3f}\n"
            f"sink_ratio={self._sink_ratio(entry):.3f}\n"
            f"source_diversity={self._source_diversity(entry):.3f}\n"
            f"model_uncertainty={float(entry.get('candidate_model_uncertainty', 0.0)):.3f}\n"
            f"pairwise_vs_anchor={pairwise_probability:.3f}\n"
            f"semantic_distance_to_anchor={float(entry.get('semantic_distance_to_anchor', 0.0)):.3f}\n"
            f"anchor_quality={self._quality_score(anchor):.3f}\n"
            f"anchor_review_consensus={self._review_consensus(anchor):.3f}\n"
            f"challenger_rationale={str(entry.get('challenger_rationale', '')).strip() or 'none'}"
        )

    def _audit_candidate_for_promotion(
        self,
        *,
        question_text: str,
        metadata: Optional[dict],
        dataset_profile: DatasetProfile,
        anchor: Dict[str, Any],
        candidate: Dict[str, Any],
        pairwise_probability: float,
    ) -> Dict[str, Any]:
        anchor_text = str(anchor.get("text", "")).strip()
        candidate_text = str(candidate.get("text", "")).strip()
        evidence_summary = self._candidate_evidence_summary(
            candidate,
            anchor=anchor,
            pairwise_probability=pairwise_probability,
        )
        findings: List[Dict[str, str]] = []

        for agent_id in self._select_auditor_agents():
            agent = self._by_id[agent_id]
            system_prompt = build_system_prompt(agent, self._auditor_slots(), extra_role_hint="auditor")
            user_prompt = (
                "You are auditing whether a non-anchor hypothesis contains anomaly cues that justify preserving it "
                "as a provisional challenger.\n\n"
                f"Question:\n{render_question_text(question_text, metadata=metadata)}\n\n"
                f"Stage1 anchor:\n{anchor_text}\n\n"
                f"Candidate hypothesis:\n{candidate_text}\n\n"
                f"Compact evidence summary:\n{evidence_summary}\n\n"
                "Use only the local discrepancy and consistency cues above.\n"
                "Return exactly four lines:\n"
                "SUPPORT: anomaly|none\n"
                "CONSISTENCY: consistent|inconsistent|uncertain\n"
                "EVIDENCE: <short anomaly cue or none>\n"
                "RATIONALE: <short justification>\n"
            )
            raw_output = self.evaluator._cached_chat(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                runtime=self.evaluator._resolve_runtime("tier2", dataset_profile),
            )
            parsed = self._parse_tagged_output(raw_output)
            support = str(parsed.get("support", "")).strip().lower()
            if support not in {"anomaly", "none"}:
                support = "none"
            consistency = str(parsed.get("consistency", "")).strip().lower()
            if consistency not in {"consistent", "inconsistent", "uncertain"}:
                consistency = "uncertain"
            findings.append(
                {
                    "agent_id": agent_id,
                    "support": support,
                    "consistency": consistency,
                    "evidence": str(parsed.get("evidence", "")).strip(),
                    "rationale": str(parsed.get("rationale", "")).strip(),
                }
            )

        support_count = sum(1 for item in findings if item["support"] == "anomaly")
        support_agents = {item["agent_id"] for item in findings if item["support"] == "anomaly"}
        if any(item["consistency"] == "inconsistent" for item in findings):
            local_consistency = "inconsistent"
        elif any(item["consistency"] == "consistent" for item in findings):
            local_consistency = "consistent"
        elif findings:
            local_consistency = "uncertain"
        else:
            local_consistency = ""

        promote = support_count > 0 and local_consistency != "inconsistent"
        rationales = [
            item["rationale"]
            for item in findings
            if item["support"] == "anomaly" and item["rationale"]
        ]
        evidences = [
            item["evidence"]
            for item in findings
            if item["support"] == "anomaly" and item["evidence"]
        ]
        return {
            "promote": promote,
            "support_count": support_count,
            "support_agents": support_agents,
            "local_consistency": local_consistency,
            "findings": findings,
            "promotion_rationale": " | ".join(rationales[:2]),
            "promotion_evidence": " | ".join(evidences[:2]),
        }

    def _promote_provisional_challengers(
        self,
        *,
        question_text: str,
        metadata: Optional[dict],
        dataset_profile: DatasetProfile,
        anchor: Dict[str, Any],
        candidates: Sequence[Dict[str, Any]],
        comparisons: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        anchor_digest = str(anchor.get("digest", ""))
        anchor_text = str(anchor.get("text", "")).strip()
        comparison_by_digest = {str(item.get("digest", "")): item for item in comparisons}

        promotable: List[Dict[str, Any]] = []
        for candidate in candidates:
            self._ensure_v4_2_entry_fields(candidate)
            if str(candidate.get("digest", "")) == anchor_digest:
                continue
            if dataset_profile.task_type == "code_generation" and not self._is_valid_code_candidate(candidate):
                continue
            candidate["semantic_distance_to_anchor"] = self._semantic_distance(
                anchor_text,
                str(candidate.get("text", "")),
            )
            if bool(candidate.get("explicit_challenger", False)):
                candidate["provisional_challenger"] = True
                candidate["promotion_source"] = "explicit_challenger"
                candidate["local_consistency_decision"] = "consistent"
                promotable.append(candidate)
                continue
            promotable.append(candidate)

        non_explicit = [entry for entry in promotable if not bool(entry.get("explicit_challenger", False))]
        non_explicit.sort(
            key=lambda entry: (
                float(entry.get("semantic_distance_to_anchor", 0.0)),
                float(comparison_by_digest.get(str(entry.get("digest", "")), {}).get("pairwise_probability", 0.0)),
                self._quality_score(entry),
                self._review_consensus(entry),
                str(entry.get("digest", "")),
            ),
            reverse=True,
        )

        for entry in non_explicit[: self.config.max_logged_candidates]:
            pairwise_probability = float(
                comparison_by_digest.get(str(entry.get("digest", "")), {}).get("pairwise_probability", 0.0)
            )
            audit = self._audit_candidate_for_promotion(
                question_text=question_text,
                metadata=metadata,
                dataset_profile=dataset_profile,
                anchor=anchor,
                candidate=entry,
                pairwise_probability=pairwise_probability,
            )
            entry["anomaly_support_count"] = int(audit["support_count"])
            entry["auditor_support_agents"] = set(audit["support_agents"])
            entry["auditor_findings"] = list(audit["findings"])
            entry["local_consistency_decision"] = str(audit["local_consistency"])
            if bool(audit["promote"]):
                entry["provisional_challenger"] = True
                entry["promotion_source"] = "auditor_soft_promotion"
                entry["promotion_rationale"] = str(audit["promotion_rationale"] or audit["promotion_evidence"])

        provisional = [
            entry
            for entry in promotable
            if bool(entry.get("provisional_challenger", False))
            and str(entry.get("digest", "")) != anchor_digest
        ]
        provisional.sort(
            key=lambda entry: (
                float(comparison_by_digest.get(str(entry.get("digest", "")), {}).get("pairwise_probability", 0.0)),
                self._quality_score(entry),
                self._review_consensus(entry),
                float(entry.get("semantic_distance_to_anchor", 0.0)),
                str(entry.get("digest", "")),
            ),
            reverse=True,
        )
        return provisional

    def _hypothesis_block(
        self,
        label: str,
        entry: Dict[str, Any],
        *,
        anchor_digest: str,
        comparison: Optional[Dict[str, Any]],
    ) -> str:
        role = "anchor" if str(entry.get("digest", "")) == anchor_digest else self._candidate_source_label(entry)
        findings = list(entry.get("auditor_findings", ()))
        finding_text = " | ".join(
            f"{item.get('agent_id', '')}:{item.get('support', '')}/{item.get('consistency', '')}:{item.get('evidence', '')}"
            for item in findings[:2]
        ) or "none"
        pairwise_probability = float((comparison or {}).get("pairwise_probability", 0.0))
        return (
            f"{label} [{role}]\n"
            f"answer={str(entry.get('text', '')).strip()}\n"
            f"quality={self._quality_score(entry):.3f}\n"
            f"review_advantage={self._review_advantage(entry):.3f}\n"
            f"review_consensus={self._review_consensus(entry):.3f}\n"
            f"sink_ratio={self._sink_ratio(entry):.3f}\n"
            f"source_diversity={self._source_diversity(entry):.3f}\n"
            f"pairwise_vs_anchor={pairwise_probability:.3f}\n"
            f"explicit_challenger={int(bool(entry.get('explicit_challenger', False)))}\n"
            f"provisional_challenger={int(bool(entry.get('provisional_challenger', False)))}\n"
            f"promotion_source={str(entry.get('promotion_source', '')) or 'none'}\n"
            f"promotion_rationale={str(entry.get('promotion_rationale', '')) or 'none'}\n"
            f"anomaly_support_count={int(entry.get('anomaly_support_count', 0))}\n"
            f"local_consistency={str(entry.get('local_consistency_decision', '')) or 'none'}\n"
            f"auditor_findings={finding_text}\n"
        )

    def _adjudicate_hypotheses(
        self,
        *,
        question_text: str,
        metadata: Optional[dict],
        dataset_profile: DatasetProfile,
        anchor: Dict[str, Any],
        provisional: Sequence[Dict[str, Any]],
        comparisons: Sequence[Dict[str, Any]],
    ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        if not provisional:
            return anchor, {
                "winner": "anchor",
                "winner_digest": str(anchor.get("digest", "")),
                "decision": "anchor",
                "pressure": "",
                "rationale": "",
                "hypothesis_count": 1,
            }

        agent_id = self._select_single_agent(self.config.adjudicator_agent_id, ("verifier", "skeptic", "reasoner"))
        comparison_by_digest = {str(item.get("digest", "")): item for item in comparisons}
        anchor_digest = str(anchor.get("digest", ""))
        hypotheses = [anchor] + list(provisional[: self.config.max_logged_candidates])
        label_to_entry: Dict[str, Dict[str, Any]] = {}
        blocks: List[str] = []
        for index, entry in enumerate(hypotheses):
            label = f"H{index}"
            label_to_entry[label] = entry
            blocks.append(
                self._hypothesis_block(
                    label,
                    entry,
                    anchor_digest=anchor_digest,
                    comparison=comparison_by_digest.get(str(entry.get("digest", ""))),
                )
            )

        agent = self._by_id[agent_id]
        system_prompt = build_system_prompt(agent, self._adjudicator_slots(), extra_role_hint="adjudicator")
        user_prompt = (
            "You are the final adjudicator for stage2 v4.2.\n"
            "Follow a structured ACH-style protocol: compare the competing hypotheses, identify the strongest "
            "evidence against the current leader, and only then choose the surviving hypothesis.\n\n"
            f"Question:\n{render_question_text(question_text, metadata=metadata)}\n\n"
            f"Hypotheses and evidence:\n\n{chr(10).join(blocks)}\n"
            "Return exactly four lines:\n"
            "WINNER: H0|H1|H2|...|uncertain\n"
            "DECISION: anchor|challenger|uncertain\n"
            "PRESSURE: <the strongest contrary evidence against the winner>\n"
            "RATIONALE: <short justification>\n"
        )
        raw_output = self.evaluator._cached_chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            runtime=self.evaluator._resolve_runtime("tier2", dataset_profile),
        )
        parsed = self._parse_tagged_output(raw_output)
        winner_label = str(parsed.get("winner", "")).strip().upper()
        decision = str(parsed.get("decision", "")).strip().lower()
        if decision not in {"anchor", "challenger", "uncertain"}:
            decision = "uncertain"
        pressure = str(parsed.get("pressure", "")).strip()
        rationale = str(parsed.get("rationale", "")).strip()

        winner_entry = label_to_entry.get(winner_label)
        if winner_label == "UNCERTAIN" or decision == "uncertain" or winner_entry is None:
            return None, {
                "winner": "uncertain",
                "winner_digest": "",
                "decision": "uncertain",
                "pressure": pressure,
                "rationale": rationale,
                "hypothesis_count": len(hypotheses),
            }
        if str(winner_entry.get("digest", "")) == anchor_digest or decision == "anchor":
            return anchor, {
                "winner": "anchor",
                "winner_digest": anchor_digest,
                "decision": "anchor",
                "pressure": pressure,
                "rationale": rationale,
                "hypothesis_count": len(hypotheses),
            }
        return dict(winner_entry), {
            "winner": winner_label,
            "winner_digest": str(winner_entry.get("digest", "")),
            "decision": "challenger",
            "pressure": pressure,
            "rationale": rationale,
            "hypothesis_count": len(hypotheses),
        }

    @staticmethod
    def _calibration_label_pairs() -> List[Tuple[str, str]]:
        return [("A", "B"), ("X", "Y"), ("LEFT", "RIGHT"), ("ONE", "TWO")]

    def _calibrate_challenger_against_anchor(
        self,
        *,
        question_text: str,
        metadata: Optional[dict],
        dataset_profile: DatasetProfile,
        anchor: Dict[str, Any],
        challenger: Dict[str, Any],
    ) -> Dict[str, Any]:
        agent_id = self._select_single_agent(self.config.calibration_agent_id, ("verifier", "judge", "skeptic"))
        agent = self._by_id[agent_id]
        rounds = max(1, int(self.config.calibration_rounds))
        votes = {"anchor": 0, "challenger": 0, "uncertain": 0}
        traces: List[Dict[str, Any]] = []

        for round_index in range(rounds):
            left_label, right_label = self._calibration_label_pairs()[round_index % len(self._calibration_label_pairs())]
            if round_index % 2 == 0:
                left_entry, right_entry = anchor, challenger
                left_kind, right_kind = "anchor", "challenger"
            else:
                left_entry, right_entry = challenger, anchor
                left_kind, right_kind = "challenger", "anchor"
            system_prompt = build_system_prompt(agent, self._calibration_slots(), extra_role_hint="calibrator")
            user_prompt = (
                "You are performing a position-debiased pairwise calibration before override.\n\n"
                f"Question:\n{render_question_text(question_text, metadata=metadata)}\n\n"
                f"Candidate {left_label}:\n{str(left_entry.get('text', '')).strip()}\n\n"
                f"Candidate {right_label}:\n{str(right_entry.get('text', '')).strip()}\n\n"
                "Return exactly two lines:\n"
                f"DECISION: {left_label}|{right_label}|uncertain\n"
                "RATIONALE: <short justification>\n"
            )
            raw_output = self.evaluator._cached_chat(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                runtime=self.evaluator._resolve_runtime("tier2", dataset_profile),
            )
            parsed = self._parse_tagged_output(raw_output)
            decision = str(parsed.get("decision", "")).strip().upper()
            if decision == left_label:
                mapped = left_kind
            elif decision == right_label:
                mapped = right_kind
            else:
                mapped = "uncertain"
            votes[mapped] += 1
            traces.append(
                {
                    "round": round_index + 1,
                    "left_label": left_label,
                    "right_label": right_label,
                    "left_kind": left_kind,
                    "right_kind": right_kind,
                    "mapped_decision": mapped,
                    "rationale": str(parsed.get("rationale", "")).strip(),
                }
            )

        probability = float(votes["challenger"]) / float(rounds)
        return {
            "rounds": rounds,
            "challenger_votes": int(votes["challenger"]),
            "anchor_votes": int(votes["anchor"]),
            "uncertain_votes": int(votes["uncertain"]),
            "probability": probability,
            "override": probability > 0.5,
            "traces": traces,
        }

    def _select_against_anchor(
        self,
        candidates: Sequence[Dict[str, Any]],
        anchor: Optional[Dict[str, Any]],
        *,
        dataset_profile: DatasetProfile,
    ) -> Tuple[Optional[Dict[str, Any]], str, Dict[str, Any]]:
        if not candidates:
            reason = "v4_2_empty_bank_use_stage1" if anchor is not None else "v4_2_empty_bank"
            return anchor, reason, {
                "v4_2_pairwise_candidate_count": 0,
                "v4_2_explicit_challenger_count": 0,
                "v4_2_provisional_challenger_count": 0,
                "v4_2_soft_promotion_count": 0,
                "v4_2_top_pairwise_candidates": [],
                **self._pairwise_extra(None, prefix="v4_2_best_pairwise"),
            }

        best_quality = dict(candidates[0])
        if anchor is None:
            return best_quality, "v4_2_no_stage1_anchor", {
                "v4_2_pairwise_candidate_count": 0,
                "v4_2_explicit_challenger_count": 0,
                "v4_2_provisional_challenger_count": 0,
                "v4_2_soft_promotion_count": 0,
                "v4_2_top_pairwise_candidates": [],
                **self._pairwise_extra(None, prefix="v4_2_best_pairwise"),
            }

        for entry in candidates:
            self._ensure_v4_2_entry_fields(entry)
        self._ensure_v4_2_entry_fields(anchor)

        comparisons = self._pairwise_candidate_comparisons(candidates, anchor, dataset_profile=dataset_profile)
        comparison_by_digest = {str(item.get("digest", "")): item for item in comparisons}
        top_pairwise = [self._pairwise_public_view(item) for item in comparisons[: self.config.max_logged_candidates]]
        best_pairwise = comparisons[0] if comparisons else None

        provisional = self._promote_provisional_challengers(
            question_text=self._last_candidate_bundle.get("question_text", ""),
            metadata=self._last_candidate_bundle.get("metadata"),
            dataset_profile=dataset_profile,
            anchor=anchor,
            candidates=candidates,
            comparisons=comparisons,
        )
        explicit_count = sum(1 for entry in candidates if bool(entry.get("explicit_challenger", False)))
        provisional_count = len(provisional)
        soft_count = sum(1 for entry in provisional if str(entry.get("promotion_source", "")) == "auditor_soft_promotion")

        extra = {
            "v4_2_pairwise_candidate_count": int(len(comparisons)),
            "v4_2_explicit_challenger_count": int(explicit_count),
            "v4_2_provisional_challenger_count": int(provisional_count),
            "v4_2_soft_promotion_count": int(soft_count),
            "v4_2_top_pairwise_candidates": top_pairwise,
            **self._pairwise_extra(best_pairwise, prefix="v4_2_best_pairwise"),
        }
        if not provisional:
            return anchor, "v4_2_preserve_no_provisional_challenger", extra

        winner, adjudication = self._adjudicate_hypotheses(
            question_text=self._last_candidate_bundle.get("question_text", ""),
            metadata=self._last_candidate_bundle.get("metadata"),
            dataset_profile=dataset_profile,
            anchor=anchor,
            provisional=provisional,
            comparisons=comparisons,
        )
        extra.update(
            {
                "v4_2_adjudication_winner": str(adjudication.get("winner", "")),
                "v4_2_adjudication_winner_digest": str(adjudication.get("winner_digest", "")),
                "v4_2_adjudication_decision": str(adjudication.get("decision", "")),
                "v4_2_adjudication_pressure": str(adjudication.get("pressure", "")),
                "v4_2_adjudication_rationale": str(adjudication.get("rationale", "")),
                "v4_2_hypothesis_count": int(adjudication.get("hypothesis_count", 0)),
            }
        )
        if winner is None:
            return anchor, "v4_2_preserve_adjudication_uncertain", extra
        if str(winner.get("digest", "")) == str(anchor.get("digest", "")):
            return anchor, "v4_2_preserve_adjudication_anchor", extra

        winning_pairwise = comparison_by_digest.get(str(winner.get("digest", "")))
        extra.update(self._pairwise_extra(winning_pairwise, prefix="v4_2_decision_pairwise"))
        calibration = self._calibrate_challenger_against_anchor(
            question_text=self._last_candidate_bundle.get("question_text", ""),
            metadata=self._last_candidate_bundle.get("metadata"),
            dataset_profile=dataset_profile,
            anchor=anchor,
            challenger=winner,
        )
        extra.update(
            {
                "v4_2_calibration_rounds": int(calibration["rounds"]),
                "v4_2_calibration_challenger_votes": int(calibration["challenger_votes"]),
                "v4_2_calibration_anchor_votes": int(calibration["anchor_votes"]),
                "v4_2_calibration_uncertain_votes": int(calibration["uncertain_votes"]),
                "v4_2_calibration_probability": float(calibration["probability"]),
                "v4_2_calibration_override": bool(calibration["override"]),
                "v4_2_calibration_traces": list(calibration["traces"]),
            }
        )
        if bool(calibration["override"]):
            return dict(winner), "v4_2_override_calibrated_challenger", extra
        return anchor, "v4_2_preserve_calibration", extra

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
        self._last_v4_2_selection = {
            "stage2_version": "v4.2",
            "v4_2_task_type": task_type,
            "v4_2_candidate_count": int(len(candidates)),
            "v4_2_stage1_anchor_present": bool(anchor is not None),
            "v4_2_stage1_anchor_used": bool(anchor is not None and selected is not None and anchor.get("digest") == selected.get("digest")),
            "v4_2_selection_reason": strategy,
            "v4_2_selected_candidate_digest": str((selected or {}).get("digest", "")),
            "v4_2_selected_candidate_source": self._candidate_source_label(selected or {}),
            "v4_2_selected_is_explicit_challenger": bool((selected or {}).get("explicit_challenger", False)),
            "v4_2_selected_is_provisional_challenger": bool((selected or {}).get("provisional_challenger", False)),
            "v4_2_selected_quality_score": self._quality_score(selected or {}),
            "v4_2_selected_support_score": float((selected or {}).get("support_score", 0.0)),
            "v4_2_selected_model_uncertainty": float((selected or {}).get("candidate_model_uncertainty", 0.0)),
            "v4_2_selected_sink_support": int((selected or {}).get("sink_support", 0)),
            "v4_2_selected_sink_ratio": self._sink_ratio(selected or {}),
            "v4_2_selected_source_diversity": self._source_diversity(selected or {}),
            "v4_2_selected_review_advantage": float(self._review_advantage(selected or {})),
            "v4_2_selected_review_consensus": float(self._review_consensus(selected or {})),
            "v4_2_stage1_anchor_digest": str((anchor or {}).get("digest", "")),
            "v4_2_stage1_anchor_quality_score": self._quality_score(anchor or {}),
            "v4_2_stage1_support_score": float((anchor or {}).get("support_score", 0.0)),
            "v4_2_stage1_model_uncertainty": float((anchor or {}).get("candidate_model_uncertainty", 0.0)),
            "v4_2_stage1_sink_support": int((anchor or {}).get("sink_support", 0)),
            "v4_2_stage1_sink_ratio": self._sink_ratio(anchor or {}),
            "v4_2_stage1_source_diversity": self._source_diversity(anchor or {}),
            "v4_2_stage1_review_advantage": float(self._review_advantage(anchor or {})),
            "v4_2_stage1_review_consensus": float(self._review_consensus(anchor or {})),
            "v4_2_candidate_model_steps": float(self._candidate_model.steps),
            "v4_2_pairwise_model_steps": float(self._pairwise_model.steps),
            "v4_2_reviewer_model_steps": float(self._reviewer_model.steps),
            "v4_2_top_candidates": top,
        }
        self._last_v4_2_selection.update(extra)

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
        anchor = bundle["anchor"]
        selected, strategy, extra = self._select_against_anchor(candidates, anchor, dataset_profile=dataset_profile)
        candidates_serialized = [self._serialize_candidate_entry(entry) for entry in candidates]
        anchor_serialized = None
        if anchor is not None:
            for item in candidates_serialized:
                if item.get("digest") == anchor.get("digest"):
                    anchor_serialized = dict(item)
                    break
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
        self._last_v4_2_selection = {}
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
            result.signature = "stage2_v4_2|" + result.signature[len("stage2_v2|") :]
        result.metadata.update(self._last_v4_2_selection)
        result.metadata["stage2_version"] = "v4.2"
        result.metadata["v4_2_all_task_nodes_each_turn"] = True
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
        with open(os.path.join(replay_dir, "v4_2_selection.json"), "w", encoding="utf-8") as handle:
            json.dump(self._last_v4_2_selection, handle, ensure_ascii=False, indent=2)
        with open(os.path.join(replay_dir, "final_answer.json"), "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "final_answer": result.final_answer,
                    "signature": result.signature,
                    "final_controller_state": asdict(result.final_controller_state),
                    "memory_record_counts": result.memory_record_counts,
                    "metadata": result.metadata | self._last_v4_2_selection,
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )
