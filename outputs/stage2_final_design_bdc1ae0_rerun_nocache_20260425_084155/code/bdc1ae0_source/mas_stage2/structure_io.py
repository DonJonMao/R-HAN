from __future__ import annotations

import json
from pathlib import Path
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

from mas_treesearch import resolve_result_output
from mas_treesearch.types import (
    EvalSummary,
    SearchResult,
    StructureMetrics,
    StructureSummary,
    TaskEvaluation,
    UnionEdge,
    UnionGraph,
    UnionNode,
)


@dataclass
class PreparedStage1Artifact:
    question_text: str
    union_graph: UnionGraph
    stage1_signature: str
    stage1_output: str
    dataset_name: str = ""
    stage1_summary: Optional[EvalSummary] = None
    structure_summary: Optional[StructureSummary] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def prepared_stage1_from_search_result(
    result: SearchResult,
    *,
    dataset_name: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> PreparedStage1Artifact:
    if result.union_graph is None:
        raise RuntimeError("Stage-1 search result does not contain a union graph.")
    artifact_metadata = dict(metadata or {})
    artifact_metadata.setdefault("source", "live_stage1_search")
    return PreparedStage1Artifact(
        question_text=result.question_text,
        union_graph=result.union_graph,
        stage1_signature=result.final_signature or result.best_node.compiled.signature(),
        stage1_output=resolve_result_output(result),
        dataset_name=str(dataset_name or ""),
        stage1_summary=result.best_node.tier2,
        structure_summary=result.structure_summary,
        metadata=artifact_metadata,
    )


def _task_evaluation_from_dict(payload: Dict[str, Any]) -> TaskEvaluation:
    return TaskEvaluation(
        task_score=float(payload.get("task_score", 0.0)),
        success=float(payload.get("success", 0.0)),
        latency=float(payload.get("latency", 0.0)),
        token_cost=float(payload.get("token_cost", 0.0)),
        safety_penalty=float(payload.get("safety_penalty", 0.0)),
        raw_output=str(payload.get("raw_output", "")),
        trace=list(payload.get("trace", [])),
        custom_metrics=dict(payload.get("custom_metrics", {})),
        debug_info=dict(payload.get("debug_info", {})),
    )


def _eval_summary_from_dict(payload: Optional[Dict[str, Any]]) -> Optional[EvalSummary]:
    if not isinstance(payload, dict):
        return None
    return EvalSummary(
        tier=str(payload.get("tier", "")),
        mean_reward=float(payload.get("mean_reward", 0.0)),
        reward_std=float(payload.get("reward_std", 0.0)),
        mean_task_score=float(payload.get("mean_task_score", 0.0)),
        mean_success=float(payload.get("mean_success", 0.0)),
        mean_latency=float(payload.get("mean_latency", 0.0)),
        mean_token_cost=float(payload.get("mean_token_cost", 0.0)),
        mean_safety_penalty=float(payload.get("mean_safety_penalty", 0.0)),
        evaluations=[_task_evaluation_from_dict(item) for item in payload.get("evaluations", [])],
    )


def _structure_summary_from_dict(payload: Optional[Dict[str, Any]]) -> Optional[StructureSummary]:
    if not isinstance(payload, dict):
        return None
    metrics_payload = dict(payload.get("metrics", {}))
    metrics = StructureMetrics(
        coverage=float(metrics_payload.get("coverage", 0.0)),
        complementarity=float(metrics_payload.get("complementarity", 0.0)),
        redundancy_quality=float(metrics_payload.get("redundancy_quality", 0.0)),
        structural_faithfulness=float(metrics_payload.get("structural_faithfulness", 0.0)),
        runtime_affordability=float(metrics_payload.get("runtime_affordability", 0.0)),
        topology_quality=float(metrics_payload.get("topology_quality", 0.0)),
        diversity_quality=float(metrics_payload.get("diversity_quality", 0.0)),
        deployability=float(metrics_payload.get("deployability", 0.0)),
        execution_probe=float(metrics_payload.get("execution_probe", 0.0)),
        total_reward=float(metrics_payload.get("total_reward", 0.0)),
        metadata=dict(metrics_payload.get("metadata", {})),
    )
    return StructureSummary(
        mode=str(payload.get("mode", "")),
        signature=str(payload.get("signature", "")),
        selected_topology_signatures=list(payload.get("selected_topology_signatures", [])),
        selected_topology_scores=[float(score) for score in payload.get("selected_topology_scores", [])],
        metrics=metrics,
        metadata=dict(payload.get("metadata", {})),
    )


def _union_graph_from_dict(payload: Dict[str, Any]) -> UnionGraph:
    nodes_payload = dict(payload.get("nodes", {}))
    nodes = {
        node_id: UnionNode(
            node_id=str(node_payload.get("node_id", node_id)),
            agent_id=str(node_payload.get("agent_id", "")),
            role=str(node_payload.get("role", "")),
            node_type=str(node_payload.get("node_type", "")),
            source_graph_ids=list(node_payload.get("source_graph_ids", [])),
            support_count=int(node_payload.get("support_count", 0)),
            avg_graph_score=float(node_payload.get("avg_graph_score", 0.0)),
            root_frequency=float(node_payload.get("root_frequency", 0.0)),
            sink_frequency=float(node_payload.get("sink_frequency", 0.0)),
            topo_level_mean=float(node_payload.get("topo_level_mean", 0.0)),
            topo_level_var=float(node_payload.get("topo_level_var", 0.0)),
            state_vector=list(node_payload.get("state_vector", [])),
            metadata=dict(node_payload.get("metadata", {})),
        )
        for node_id, node_payload in nodes_payload.items()
    }
    edges = [
        UnionEdge(
            src=str(edge_payload.get("src", "")),
            dst=str(edge_payload.get("dst", "")),
            edge_type=str(edge_payload.get("edge_type", "")),
            source_graph_ids=list(edge_payload.get("source_graph_ids", [])),
            support_count=int(edge_payload.get("support_count", 0)),
            support_ratio=float(edge_payload.get("support_ratio", 0.0)),
            avg_parent_score=float(edge_payload.get("avg_parent_score", 0.0)),
            best_parent_score=float(edge_payload.get("best_parent_score", 0.0)),
            initial_keep_logit=float(edge_payload.get("initial_keep_logit", 0.0)),
            dynamic_keep_weight=float(edge_payload.get("dynamic_keep_weight", 0.0)),
            level_delta_mean=float(edge_payload.get("level_delta_mean", 1.0)),
            latency_prior=float(edge_payload.get("latency_prior", 0.0)),
            token_cost_prior=float(edge_payload.get("token_cost_prior", 0.0)),
            metadata=dict(edge_payload.get("metadata", {})),
        )
        for edge_payload in payload.get("edges", [])
    ]
    return UnionGraph(
        nodes=nodes,
        edges=edges,
        source_topology_signatures=list(payload.get("source_topology_signatures", [])),
        root_node_ids=list(payload.get("root_node_ids", [])),
        sink_node_ids=list(payload.get("sink_node_ids", [])),
        metadata=dict(payload.get("metadata", {})),
    )


def prepared_stage1_from_dict(payload: Dict[str, Any]) -> PreparedStage1Artifact:
    return PreparedStage1Artifact(
        question_text=str(payload.get("question_text", "")),
        union_graph=_union_graph_from_dict(dict(payload.get("union_graph", {}))),
        stage1_signature=str(payload.get("stage1_signature", "")),
        stage1_output=str(payload.get("stage1_output", "")),
        dataset_name=str(payload.get("dataset_name", "")),
        stage1_summary=_eval_summary_from_dict(payload.get("stage1_summary")),
        structure_summary=_structure_summary_from_dict(payload.get("structure_summary")),
        metadata=dict(payload.get("metadata", {})),
    )


def save_prepared_stage1_artifact(artifact: PreparedStage1Artifact, path: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(artifact), ensure_ascii=False, indent=2), encoding="utf-8")


def load_prepared_stage1_artifact(path: str) -> PreparedStage1Artifact:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return prepared_stage1_from_dict(dict(payload))
