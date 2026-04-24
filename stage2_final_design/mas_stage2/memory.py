from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from mas_treesearch.clients import CachedEmbedder
from mas_treesearch.gating import cosine
from mas_treesearch.types import UnionNode

from .config import Stage2MemoryConfig
from .learning import OnlineLinearModel, selector_features
from .types import (
    ControllerState,
    ExportedMemoryMessage,
    LocalLatentMemory,
    MemoryRecord,
    SelectedMemoryItem,
)


def _truncate(text: str, limit: int) -> str:
    cleaned = " ".join((text or "").split()).strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(0, limit - 3)].rstrip() + "..."


@dataclass
class PrivateEpisodeMemoryStore:
    config: Stage2MemoryConfig

    def __post_init__(self) -> None:
        self._records: Dict[str, List[MemoryRecord]] = {}

    def add(self, record: MemoryRecord) -> None:
        bucket = self._records.setdefault(record.owner_node_id, [])
        bucket.append(record)
        if len(bucket) > self.config.max_private_records_per_agent:
            del bucket[: len(bucket) - self.config.max_private_records_per_agent]

    def get(self, owner_node_id: str) -> List[MemoryRecord]:
        return list(self._records.get(owner_node_id, ()))

    def counts(self) -> Dict[str, int]:
        return {owner: len(records) for owner, records in self._records.items()}

    def total_token_estimate(self) -> int:
        return sum(record.token_estimate for records in self._records.values() for record in records)


class RoleAwareMemorySelector:
    def __init__(
        self,
        config: Stage2MemoryConfig,
        embedder: CachedEmbedder,
        *,
        learned_model: Optional[OnlineLinearModel] = None,
        learned_weight: float = 0.0,
    ):
        self.config = config
        self.embedder = embedder
        self.learned_model = learned_model
        self.learned_weight = learned_weight

    @staticmethod
    def _feedback_bias(record: MemoryRecord) -> float:
        mapping = {
            "pass": 0.12,
            "preserve": 0.10,
            "challenge": 0.14,
            "reject": 0.16,
            "conflict": 0.12,
            "revise": 0.10,
            "uncertain": 0.04,
            "unresolved": 0.02,
        }
        return mapping.get(record.feedback_type, 0.0)

    def _score_record(
        self,
        query_vec: Sequence[float],
        node: UnionNode,
        controller_state: ControllerState,
        record: MemoryRecord,
        *,
        current_turn: int,
    ) -> tuple[float, Dict[str, float], float, float]:
        query_similarity = cosine(query_vec, record.embedding)
        recency = 1.0 / max(1.0, 1.0 + (current_turn - record.turn_index))
        role_bonus = 0.08 if record.role == node.role else 0.0
        heuristic_score = (
            0.62 * query_similarity
            + 0.14 * recency
            + 0.12 * self._feedback_bias(record)
            + role_bonus
            - 0.02 * min(1.0, record.token_estimate / 120.0)
        )
        features = selector_features(
            node,
            record,
            controller_state,
            current_turn=current_turn,
            query_similarity=query_similarity,
        )
        learned_score = 0.5
        if self.learned_model is not None:
            learned_score, _ = self.learned_model.predict(features)
        final_score = heuristic_score + self.learned_weight * (learned_score - 0.5)
        return final_score, features, heuristic_score, learned_score

    def select(
        self,
        node: UnionNode,
        question_text: str,
        controller_state: ControllerState,
        records: Sequence[MemoryRecord],
        *,
        current_turn: int,
    ) -> List[SelectedMemoryItem]:
        if not records:
            return []
        query_text = _truncate(
            "\n".join(
                [
                    question_text,
                    f"role={node.role}",
                    f"turn={current_turn}",
                    f"global_state={_truncate(controller_state.summary, 240)}",
                    f"uncertainty={controller_state.uncertainty:.3f}",
                    controller_state.summary,
                ]
            ),
            self.config.query_max_chars,
        )
        query_vec = self.embedder.embed(query_text)
        selected_ids: List[str] = []
        selected: List[SelectedMemoryItem] = []
        scored_lookup: Dict[str, tuple[Dict[str, float], float, float]] = {}

        def add_record(record: MemoryRecord, rationale: str, score: float) -> None:
            if record.record_id in selected_ids or len(selected) >= self.config.max_selected_records:
                return
            selected_ids.append(record.record_id)
            features, heuristic_score, learned_score = scored_lookup.get(record.record_id, ({}, score, 0.5))
            selected.append(
                SelectedMemoryItem(
                    record_id=record.record_id,
                    score=score,
                    rationale=rationale,
                    metadata={
                        "features": dict(features),
                        "heuristic_score": float(heuristic_score),
                        "learned_score": float(learned_score),
                    },
                )
            )

        ordered = sorted(records, key=lambda item: (item.turn_index, item.record_id), reverse=True)
        score_bundles = [
            self._score_record(query_vec, node, controller_state, record, current_turn=current_turn)
            for record in records
        ]
        for (_, features, heuristic_score, learned_score), record in zip(score_bundles, records):
            scored_lookup[record.record_id] = (features, heuristic_score, learned_score)

        if self.config.keep_latest_self_output:
            latest_output = next((item for item in ordered if item.record_type == "self_output"), None)
            if latest_output is not None:
                add_record(latest_output, "latest_self_output", 1.0)
        if self.config.keep_latest_feedback:
            latest_feedback = next((item for item in ordered if item.record_type == "feedback"), None)
            if latest_feedback is not None:
                add_record(latest_feedback, "latest_feedback", 1.0)

        scored = [
            (bundle[0], record)
            for bundle, record in zip(score_bundles, records)
            if record.record_id not in selected_ids
        ]
        scored.sort(key=lambda item: (item[0], item[1].turn_index, item[1].record_id), reverse=True)

        failure_types = {"challenge", "reject", "conflict", "revise"}
        success_types = {"pass", "preserve"}
        if self.config.include_failure_memory:
            failure_pick = next((item for item in scored if item[1].feedback_type in failure_types), None)
            if failure_pick is not None:
                add_record(failure_pick[1], "failure_signal", failure_pick[0])
        if self.config.include_success_memory:
            success_pick = next((item for item in scored if item[1].feedback_type in success_types), None)
            if success_pick is not None:
                add_record(success_pick[1], "stable_signal", success_pick[0])
        for score, record in scored:
            if len(selected) >= self.config.max_selected_records:
                break
            add_record(record, "top_scored", score)
        return selected


class LocalMemoryComposer:
    def __init__(self, config: Stage2MemoryConfig, embedder: CachedEmbedder):
        self.config = config
        self.embedder = embedder

    def compose(
        self,
        node: UnionNode,
        records_by_id: Dict[str, MemoryRecord],
        selected_items: Sequence[SelectedMemoryItem],
        controller_state: ControllerState,
        *,
        current_turn: int,
    ) -> LocalLatentMemory:
        failure_signals: List[str] = []
        stable_signals: List[str] = []
        excerpts: List[str] = []
        ranked_items = sorted(selected_items, key=lambda item: (item.score, item.record_id), reverse=True)
        for item in ranked_items:
            record = records_by_id[item.record_id]
            excerpt = _truncate(record.text, self.config.max_record_chars)
            excerpts.append(f"[{record.record_type}|{record.feedback_type}] {excerpt}")
            if record.feedback_type in {"challenge", "reject", "conflict", "revise"}:
                failure_signals.append(excerpt)
            elif record.feedback_type in {"pass", "preserve"}:
                stable_signals.append(excerpt)
        parts: List[str] = [
            f"Role={node.role}.",
            f"Global state: {_truncate(controller_state.summary, self.config.max_record_chars // 2)}",
        ]
        if stable_signals:
            parts.append("Keep using:")
            parts.extend(f"- {signal}" for signal in stable_signals[:2])
        if failure_signals:
            parts.append("Avoid repeating:")
            parts.extend(f"- {signal}" for signal in failure_signals[:2])
        if not stable_signals and not failure_signals and excerpts:
            parts.append("Recent carryover:")
            parts.extend(f"- {text}" for text in excerpts[:2])
        summary = "\n".join(parts)
        return LocalLatentMemory(
            node_id=node.node_id,
            turn_index=current_turn,
            selected_items=list(ranked_items),
            summary=summary,
            latent_vector=self.embedder.embed(summary),
            stable_signals=stable_signals[:2],
            failure_signals=failure_signals[:2],
            metadata={"selected_record_ids": [item.record_id for item in ranked_items]},
        )


class ExportMessageBuilder:
    def __init__(self, config: Stage2MemoryConfig, embedder: CachedEmbedder):
        self.config = config
        self.embedder = embedder

    def build(
        self,
        node: UnionNode,
        latent_memory: LocalLatentMemory,
        output_text: str,
    ) -> ExportedMemoryMessage:
        lines = [f"{node.role} update:"]
        if latent_memory.failure_signals:
            lines.append(f"Risk: {_truncate(latent_memory.failure_signals[0], self.config.max_export_chars)}")
        if latent_memory.stable_signals:
            lines.append(f"Keep: {_truncate(latent_memory.stable_signals[0], self.config.max_export_chars)}")
        if output_text.strip():
            lines.append(f"Current output: {_truncate(output_text, self.config.max_export_chars)}")
        summary = "\n".join(lines)
        return ExportedMemoryMessage(
            node_id=node.node_id,
            turn_index=latent_memory.turn_index,
            summary=summary,
            latent_vector=self.embedder.embed(summary),
            provenance_record_ids=[item.record_id for item in latent_memory.selected_items],
            confidence=min(1.0, 0.45 + 0.10 * len(latent_memory.selected_items)),
            metadata={"role": node.role},
        )


class MemoryBriefVerbalizer:
    def __init__(self, config: Stage2MemoryConfig):
        self.config = config

    def build(
        self,
        local_latent: LocalLatentMemory,
        neighbour_exports: Sequence[ExportedMemoryMessage],
        controller_state: ControllerState,
    ) -> str:
        parts: List[str] = [
            "[Global State]",
            _truncate(controller_state.summary, self.config.max_brief_chars // 3),
            "",
            "[Local Memory]",
            _truncate(local_latent.summary, self.config.max_brief_chars // 3),
        ]
        if neighbour_exports:
            parts.extend(["", "[Neighbour Signals]"])
            for message in neighbour_exports[: self.config.max_neighbour_exports]:
                parts.append(f"- {message.node_id}: {_truncate(message.summary, self.config.max_export_chars)}")
        return _truncate("\n".join(parts), self.config.max_brief_chars)
