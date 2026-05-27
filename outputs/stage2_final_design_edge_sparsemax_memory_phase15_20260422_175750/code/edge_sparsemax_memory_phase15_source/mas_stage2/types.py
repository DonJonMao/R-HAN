from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from mas_treesearch.types import Vector


@dataclass
class FeedbackEvent:
    event_id: str
    turn_index: int
    source_node_id: str
    target_node_id: str
    source_kind: str
    event_type: str
    confidence: float
    detail: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryRecord:
    record_id: str
    episode_id: str
    turn_index: int
    owner_node_id: str
    agent_id: str
    role: str
    record_type: str
    text: str
    embedding: Vector
    token_estimate: int
    feedback_type: str = "unresolved"
    confidence: float = 0.0
    source_node_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SelectedMemoryItem:
    record_id: str
    score: float
    rationale: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LocalLatentMemory:
    node_id: str
    turn_index: int
    selected_items: List[SelectedMemoryItem]
    summary: str
    latent_vector: Vector
    stable_signals: List[str] = field(default_factory=list)
    failure_signals: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExportedMemoryMessage:
    node_id: str
    turn_index: int
    summary: str
    latent_vector: Vector
    provenance_record_ids: List[str]
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ControllerState:
    turn_index: int
    mode: str
    focus: str
    uncertainty: float
    role_weights: Dict[str, float]
    summary: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EdgeActivation:
    edge_id: str
    src: str
    dst: str
    score: float
    active: bool
    reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NodeTurnTrace:
    node_id: str
    agent_id: str
    role: str
    active_incoming_edge_ids: List[str]
    selected_records: List[SelectedMemoryItem]
    neighbour_sources: List[str]
    memory_brief: str
    output: str
    local_latent_summary: str
    exported_summary: str
    prompt_excerpt: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TurnTrace:
    turn_index: int
    controller_state: ControllerState
    active_edges: List[EdgeActivation]
    node_traces: List[NodeTurnTrace]
    feedback_events: List[FeedbackEvent]
    sink_outputs: Dict[str, str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Stage2RunResult:
    final_answer: str
    final_controller_state: ControllerState
    turn_traces: List[TurnTrace]
    memory_record_counts: Dict[str, int]
    signature: str
    metadata: Dict[str, Any] = field(default_factory=dict)
