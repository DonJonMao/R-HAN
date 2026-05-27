from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from mas_treesearch.clients import CachedEmbedder
from mas_treesearch.gating import cosine
from mas_treesearch.types import UnionNode

from mas_stage2.config import Stage2MemoryConfig
from mas_stage2.learning import OnlineLinearModel, selector_features, slot_features
from mas_stage2.types import (
    ControllerState,
    ExportedMemoryMessage,
    LocalLatentMemory,
    MemoryRecord,
    SelectedMemoryItem,
)

PHYSICAL_MEMORY_BUCKETS = ("self_output", "feedback", "class_summary", "repair_trace")
FEEDBACK_STABLE_TYPES = frozenset({"pass", "preserve", "keep", "approve"})
FEEDBACK_FAILURE_TYPES = frozenset({"challenge", "reject", "conflict", "revise"})
SPARSEMAX_EPSILON = 1e-8


def _truncate(text: str, limit: int) -> str:
    cleaned = " ".join((text or "").split()).strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(0, limit - 3)].rstrip() + "..."


def _physical_bucket_name(record: MemoryRecord) -> str:
    bucket_name = str(record.record_type or "").strip()
    if bucket_name not in PHYSICAL_MEMORY_BUCKETS:
        raise ValueError(f"Unsupported memory bucket: {bucket_name}")
    return bucket_name


def _feedback_view_labels(view_name: str) -> frozenset[str]:
    normalized = str(view_name or "").strip()
    if normalized == "stable_view":
        return FEEDBACK_STABLE_TYPES
    if normalized == "failure_view":
        return FEEDBACK_FAILURE_TYPES
    raise ValueError(f"Unsupported feedback view: {view_name}")


def _average_embedding(records: Sequence[MemoryRecord]) -> List[float]:
    if not records:
        return []
    vectors = [list(record.embedding or []) for record in records if record.embedding]
    if not vectors:
        return []
    width = len(vectors[0])
    if any(len(vector) != width for vector in vectors):
        return list(vectors[0])
    return [sum(vector[index] for vector in vectors) / float(len(vectors)) for index in range(width)]


def _sparsemax(scores: Sequence[float]) -> List[float]:
    """Project scores onto the probability simplex with sparse support."""
    values = [float(score) for score in scores]
    if not values:
        return []
    if len(values) == 1:
        return [1.0]
    sorted_values = sorted(values, reverse=True)
    cumulative = 0.0
    support_size = 0
    for index, value in enumerate(sorted_values, start=1):
        cumulative += value
        if 1.0 + index * value > cumulative:
            support_size = index
    if support_size <= 0:
        winner = max(range(len(values)), key=lambda idx: (values[idx], -idx))
        return [1.0 if idx == winner else 0.0 for idx in range(len(values))]
    tau = (sum(sorted_values[:support_size]) - 1.0) / float(support_size)
    weights = [max(score - tau, 0.0) for score in values]
    total = sum(weights)
    if total <= 0.0:
        winner = max(range(len(values)), key=lambda idx: (values[idx], -idx))
        return [1.0 if idx == winner else 0.0 for idx in range(len(values))]
    return [weight / total for weight in weights]


def _standardize_scores(scores: Sequence[float]) -> List[float]:
    values = [float(score) for score in scores]
    if len(values) <= 1:
        return values
    mean = sum(values) / float(len(values))
    variance = sum((score - mean) ** 2 for score in values) / float(len(values))
    scale = variance ** 0.5
    if scale <= 1e-8:
        return [0.0 for _ in values]
    return [(score - mean) / (scale + 1e-8) for score in values]


def _summary_view_record(records: Sequence[MemoryRecord], view_name: str) -> Optional[MemoryRecord]:
    if not records:
        return None
    ordered = sorted(records, key=lambda item: (item.turn_index, item.record_id), reverse=True)
    if view_name == "checker_verdict_summary":
        stable = sum(1 for record in ordered if record.feedback_type in FEEDBACK_STABLE_TYPES)
        failure = sum(1 for record in ordered if record.feedback_type in FEEDBACK_FAILURE_TYPES)
        head = f"checker verdict summary: stable={stable}, failure={failure}"
        excerpts = [
            _truncate(f"{record.feedback_type}: {record.text}", 120)
            for record in ordered[:2]
        ]
        summary_text = "\n".join([head] + excerpts)
        source_bucket = "feedback"
    elif view_name == "recovery_summary":
        head = f"recovery summary: branches={len(ordered)}"
        excerpts = [_truncate(record.text, 120) for record in ordered[:2]]
        summary_text = "\n".join([head] + excerpts)
        source_bucket = "repair_trace"
    else:
        raise ValueError(f"Unsupported summary view: {view_name}")
    owner = ordered[0].owner_node_id
    return MemoryRecord(
        record_id=f"{owner}::{view_name}",
        episode_id=ordered[0].episode_id,
        turn_index=max(record.turn_index for record in ordered),
        owner_node_id=owner,
        agent_id=ordered[0].agent_id,
        role=ordered[0].role,
        record_type=source_bucket,
        text=summary_text,
        embedding=_average_embedding(ordered),
        token_estimate=max(1, len(summary_text.split())),
        feedback_type=view_name,
        confidence=max(float(record.confidence) for record in ordered),
        source_node_id=ordered[0].source_node_id,
        metadata={
            "view_name": view_name,
            "source_record_ids": [record.record_id for record in ordered[:4]],
            "summary_view": True,
        },
    )


@dataclass
class PrivateEpisodeMemoryStore:
    config: Stage2MemoryConfig

    def __post_init__(self) -> None:
        self._records: Dict[str, List[MemoryRecord]] = {}

    def add(self, record: MemoryRecord) -> None:
        _physical_bucket_name(record)
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

    def get_bucket(self, owner_node_id: str, bucket_name: str) -> List[MemoryRecord]:
        normalized = str(bucket_name or "").strip()
        if normalized not in PHYSICAL_MEMORY_BUCKETS:
            raise ValueError(f"Unsupported memory bucket: {bucket_name}")
        return [record for record in self.get(owner_node_id) if _physical_bucket_name(record) == normalized]

    def get_feedback_view(self, owner_node_id: str, view_name: str) -> List[MemoryRecord]:
        labels = _feedback_view_labels(view_name)
        return [
            record
            for record in self.get_bucket(owner_node_id, "feedback")
            if str(record.feedback_type or "").strip() in labels
        ]

    def get_view(self, owner_node_id: str, view_name: str) -> List[MemoryRecord]:
        normalized = str(view_name or "").strip()
        if normalized in {"stable_view", "failure_view"}:
            return self.get_feedback_view(owner_node_id, normalized)
        if normalized == "checker_verdict_summary":
            summary = _summary_view_record(self.get_bucket(owner_node_id, "feedback"), normalized)
            return [summary] if summary is not None else []
        if normalized == "recovery_summary":
            summary = _summary_view_record(self.get_bucket(owner_node_id, "repair_trace"), normalized)
            return [summary] if summary is not None else []
        raise ValueError(f"Unsupported memory view: {view_name}")


@dataclass(frozen=True)
class SlotSpec:
    name: str
    legacy_cap: int
    core: bool
    learnable: bool = True


class RoleAwareMemorySelector:
    def __init__(
        self,
        config: Stage2MemoryConfig,
        embedder: CachedEmbedder,
        *,
        learned_model: Optional[OnlineLinearModel] = None,
        learned_weight: float = 0.0,
        slot_gate_model: Optional[OnlineLinearModel] = None,
        slot_gate_weight: float = 1.0,
    ):
        self.config = config
        self.embedder = embedder
        self.learned_model = learned_model
        self.learned_weight = learned_weight
        self.slot_gate_model = slot_gate_model
        self.slot_gate_weight = slot_gate_weight

    @staticmethod
    def _proposal_slots() -> List[SlotSpec]:
        return [
            SlotSpec("self_output", 1, core=True),
            SlotSpec("feedback", 1, core=True),
            SlotSpec("failure_view", 1, core=False),
            SlotSpec("stable_view", 1, core=False),
        ]

    @staticmethod
    def _checker_slots() -> List[SlotSpec]:
        return [
            SlotSpec("class_summary", 2, core=True),
            SlotSpec("feedback", 1, core=True),
            SlotSpec("failure_view", 1, core=False),
        ]

    @staticmethod
    def _aggregator_slots() -> List[SlotSpec]:
        return [
            SlotSpec("class_summary", 1, core=True),
            SlotSpec("checker_verdict_summary", 1, core=True),
            SlotSpec("recovery_summary", 1, core=False),
        ]

    @staticmethod
    def _role_compat_runtime_node_type(node: UnionNode) -> str:
        role = str(node.role)
        if role in {"tester", "verifier", "critic", "judge", "checker"}:
            return "checker"
        if role in {"aggregator", "router"}:
            return "aggregator"
        return "proposal"

    @classmethod
    def _resolved_runtime_node_type(cls, node: UnionNode) -> str:
        metadata = dict(node.metadata or {})
        runtime_node_type = str(metadata.get("runtime_node_type", "")).strip().lower()
        if runtime_node_type in {"sink", "checker", "aggregator", "proposal"}:
            if metadata.get("runtime_node_type_source") != "compat_fallback":
                metadata["runtime_node_type_source"] = "runtime_annotation"
            node.metadata = metadata
            return runtime_node_type
        if bool(metadata.get("is_sink_runtime", False)):
            fallback = "sink"
        else:
            fallback = cls._role_compat_runtime_node_type(node)
        metadata["runtime_node_type"] = fallback
        metadata["runtime_node_type_source"] = "compat_fallback"
        metadata["runtime_node_type_fallback_role"] = str(node.role)
        node.metadata = metadata
        return fallback

    @classmethod
    def _node_slot_plan(cls, node: UnionNode) -> List[SlotSpec]:
        runtime_node_type = cls._resolved_runtime_node_type(node)
        if runtime_node_type == "sink":
            return cls._aggregator_slots()
        if runtime_node_type == "checker":
            return cls._checker_slots()
        if runtime_node_type == "aggregator":
            return cls._aggregator_slots()
        return cls._proposal_slots()

    @staticmethod
    def _records_for_slot(records: Sequence[MemoryRecord], slot_name: str) -> List[MemoryRecord]:
        if slot_name in PHYSICAL_MEMORY_BUCKETS:
            return [record for record in records if record.record_type == slot_name]
        if slot_name in {"stable_view", "failure_view"}:
            labels = _feedback_view_labels(slot_name)
            return [record for record in records if record.record_type == "feedback" and record.feedback_type in labels]
        if slot_name == "checker_verdict_summary":
            summary = _summary_view_record(
                [record for record in records if record.record_type == "feedback"],
                slot_name,
            )
            return [summary] if summary is not None else []
        if slot_name == "recovery_summary":
            summary = _summary_view_record(
                [record for record in records if record.record_type == "repair_trace"],
                slot_name,
            )
            return [summary] if summary is not None else []
        raise ValueError(f"Unsupported memory slot: {slot_name}")

    @staticmethod
    def _slot_evidence_counts(records: Sequence[MemoryRecord]) -> tuple[int, int]:
        stable = sum(1 for record in records if record.feedback_type in FEEDBACK_STABLE_TYPES)
        failure = sum(1 for record in records if record.feedback_type in FEEDBACK_FAILURE_TYPES)
        return stable, failure

    def _slot_gate(
        self,
        node: UnionNode,
        controller_state: ControllerState,
        spec: SlotSpec,
        *,
        current_turn: int,
        runtime_node_type: str,
        candidate_count: int,
        stable_count: int,
        failure_count: int,
    ) -> tuple[float, Dict[str, float]]:
        features = slot_features(
            node,
            controller_state,
            current_turn=current_turn,
            slot_name=spec.name,
            runtime_node_type=runtime_node_type,
            candidate_count=candidate_count,
            stable_count=stable_count,
            failure_count=failure_count,
            core_slot=spec.core,
            learnable_slot=spec.learnable,
        )
        if spec.core and str(self.config.slot_mask_core_policy) == "always_on":
            return 1.0, features
        if str(self.config.slot_mask_mode).lower() != "soft" or not spec.learnable:
            return 1.0, features
        if self.slot_gate_model is None:
            return 1.0, features
        predicted, _ = self.slot_gate_model.predict(features)
        gate = float(predicted)
        if self.slot_gate_weight < 1.0:
            gate = 1.0 + float(self.slot_gate_weight) * (gate - 1.0)
        gate = max(float(self.config.slot_mask_min_gate), min(1.0, gate))
        return gate, features

    def _slot_query_vector(
        self,
        node: UnionNode,
        question_text: str,
        controller_state: ControllerState,
        *,
        current_turn: int,
        slot_name: str,
    ) -> List[float]:
        query_text = _truncate(
            "\n".join(
                [
                    question_text,
                    f"role={node.role}",
                    f"turn={current_turn}",
                    f"slot={slot_name}",
                    f"global_state={_truncate(controller_state.summary, 240)}",
                    f"uncertainty={controller_state.uncertainty:.3f}",
                ]
            ),
            self.config.query_max_chars,
        )
        return self.embedder.embed(query_text)

    def _score_record(
        self,
        query_vec: Sequence[float],
        node: UnionNode,
        controller_state: ControllerState,
        record: MemoryRecord,
        *,
        current_turn: int,
        slot_name: str,
    ) -> tuple[float, Dict[str, float], float, float, float]:
        query_similarity = cosine(query_vec, record.embedding)
        features = selector_features(
            node,
            record,
            controller_state,
            current_turn=current_turn,
            query_similarity=query_similarity,
        )
        features["query_similarity"] = float(query_similarity)
        features[f"slot={slot_name}"] = 1.0
        features[f"slot_head::{slot_name}::bias"] = 1.0
        features[f"slot_head::{slot_name}::query_similarity"] = float(query_similarity)
        learned_delta = 0.0
        if self.learned_model is not None:
            learned_delta = float(self.learned_model.bias) - 0.5
            learned_delta += sum(
                float(self.learned_model.weights.get(name, 0.0)) * float(value)
                for name, value in features.items()
            )
        score = float(query_similarity) + float(self.learned_weight) * learned_delta
        return float(score), features, query_similarity, learned_delta, float(query_similarity)

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
        selected_ids: List[str] = []
        selected: List[SelectedMemoryItem] = []
        slot_supports: List[Tuple[float, int, str, List[SelectedMemoryItem]]] = []

        runtime_node_type = self._resolved_runtime_node_type(node)
        for slot_index, spec in enumerate(self._node_slot_plan(node)):
            slot_name = spec.name
            slot_records = self._records_for_slot(records, slot_name)
            stable_count, failure_count = self._slot_evidence_counts(slot_records)
            slot_gate, gate_features = self._slot_gate(
                node,
                controller_state,
                spec,
                current_turn=current_turn,
                runtime_node_type=runtime_node_type,
                candidate_count=len(slot_records),
                stable_count=stable_count,
                failure_count=failure_count,
            )
            if not slot_records:
                continue
            query_vec = self._slot_query_vector(
                node,
                question_text,
                controller_state,
                current_turn=current_turn,
                slot_name=slot_name,
            )
            scored: List[Tuple[float, MemoryRecord, Dict[str, float], float, float, float]] = []
            for record in slot_records:
                score, features, similarity_score, learned_score, heuristic_score = self._score_record(
                    query_vec,
                    node,
                    controller_state,
                    record,
                    current_turn=current_turn,
                    slot_name=slot_name,
                )
                scored.append((score, record, features, similarity_score, learned_score, heuristic_score))
            if not scored:
                continue
            normalized_scores = _standardize_scores([item[0] for item in scored])
            weights = _sparsemax(normalized_scores)
            support: List[Tuple[float, float, MemoryRecord, Dict[str, float], float, float, float, float]] = []
            for normalized_score, weight, (score, record, features, similarity_score, learned_score, heuristic_score) in zip(normalized_scores, weights, scored):
                if weight <= SPARSEMAX_EPSILON:
                    continue
                support.append((float(weight), score, record, features, similarity_score, learned_score, heuristic_score, float(normalized_score)))
            if not support:
                best = max(scored, key=lambda item: (item[0], item[1].turn_index, item[1].record_id))
                score, record, features, similarity_score, learned_score, heuristic_score = best
                support.append((1.0, score, record, features, similarity_score, learned_score, heuristic_score, 0.0))
            support.sort(key=lambda item: (item[0], item[1], item[2].turn_index, item[2].record_id), reverse=True)
            slot_items: List[SelectedMemoryItem] = []
            support_size = len(support)
            for rank, (weight, raw_logit, record, features, similarity_score, learned_delta, heuristic_score, normalized_score) in enumerate(support):
                effective_weight = float(slot_gate) * float(weight)
                slot_items.append(
                    SelectedMemoryItem(
                        record_id=record.record_id,
                        score=effective_weight,
                        rationale=f"slot:{slot_name}",
                        metadata={
                            "features": dict(features),
                            "slot_gate_features": dict(gate_features),
                            "slot_name": slot_name,
                            "runtime_node_type": runtime_node_type,
                            "slot_schema_cap": int(spec.legacy_cap),
                            "legacy_slot_cap": int(spec.legacy_cap),
                            "slot_core": bool(spec.core),
                            "slot_learnable": bool(spec.learnable),
                            "slot_candidate_count": int(len(slot_records)),
                            "slot_support_size": int(support_size),
                            "slot_gate_weight": float(slot_gate),
                            "raw_logit": float(raw_logit),
                            "normalized_logit": float(normalized_score),
                            "query_similarity": float(similarity_score),
                            "slot_similarity": float(similarity_score),
                            "support_weight": float(weight),
                            "slot_support_weight": float(weight),
                            "effective_weight": float(effective_weight),
                            "slot_sparsemax_weight": float(weight),
                            "support_rank_in_slot": int(rank),
                            "slot_rank": int(rank),
                            "slot_learned_delta": float(learned_delta),
                            "slot_heuristic_score": float(heuristic_score),
                        },
                    )
                )
            slot_supports.append((float(slot_gate), slot_index, slot_name, slot_items))
        slot_supports.sort(key=lambda item: (-item[0], item[1], item[2]))
        slot_merge_order = {slot_name: index for index, (_gate, _slot_index, slot_name, _items) in enumerate(slot_supports)}
        round_index = 0
        while len(selected) < self.config.max_selected_records:
            changed = False
            for slot_gate, _slot_index, slot_name, slot_items in slot_supports:
                if round_index >= len(slot_items):
                    continue
                item = slot_items[round_index]
                if item.record_id in selected_ids:
                    continue
                metadata = dict(item.metadata)
                metadata["slot_merge_order"] = int(slot_merge_order.get(slot_name, 0))
                metadata["selection_round"] = int(round_index)
                metadata["selection_order"] = len(selected)
                selected.append(
                    SelectedMemoryItem(
                        record_id=item.record_id,
                        score=item.score,
                        rationale=item.rationale,
                        metadata=metadata,
                    )
                )
                selected_ids.append(item.record_id)
                changed = True
                if len(selected) >= self.config.max_selected_records:
                    break
            if not changed:
                break
            round_index += 1
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
        ordered_items = sorted(
            selected_items,
            key=lambda item: (int(item.metadata.get("selection_order", 0)), item.record_id),
        )
        for item in ordered_items:
            record = records_by_id[item.record_id]
            excerpt = _truncate(record.text, self.config.max_record_chars)
            support_weight = float(item.metadata.get("support_weight", item.score))
            slot_name = str(item.metadata.get("slot_name", "unknown_slot"))
            weighted_excerpt = f"[{slot_name} | alpha={support_weight:.3f}] {excerpt}"
            excerpts.append(f"[{slot_name}|alpha={support_weight:.3f}|{record.record_type}|{record.feedback_type}] {excerpt}")
            if record.feedback_type in FEEDBACK_FAILURE_TYPES:
                failure_signals.append(weighted_excerpt)
            elif record.feedback_type in FEEDBACK_STABLE_TYPES:
                stable_signals.append(weighted_excerpt)
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
            selected_items=list(ordered_items),
            summary=summary,
            latent_vector=self.embedder.embed(summary),
            stable_signals=stable_signals[:2],
            failure_signals=failure_signals[:2],
            metadata={
                "selected_record_ids": [item.record_id for item in ordered_items],
                "active_slot_names": list(dict.fromkeys(str(item.metadata.get("slot_name", "")) for item in ordered_items)),
                "slot_gate_weights": {
                    str(item.metadata.get("slot_name", "")): float(item.metadata.get("slot_gate_weight", 1.0))
                    for item in ordered_items
                },
                "slot_support_sizes": {
                    str(item.metadata.get("slot_name", "")): int(item.metadata.get("slot_support_size", 0))
                    for item in ordered_items
                },
                "slot_candidate_counts": {
                    str(item.metadata.get("slot_name", "")): int(item.metadata.get("slot_candidate_count", 0))
                    for item in ordered_items
                },
                "slot_merge_order": {
                    str(item.metadata.get("slot_name", "")): int(item.metadata.get("slot_merge_order", 0))
                    for item in ordered_items
                },
                "slot_support_weights": {
                    item.record_id: float(item.metadata.get("support_weight", item.score))
                    for item in ordered_items
                },
            },
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
