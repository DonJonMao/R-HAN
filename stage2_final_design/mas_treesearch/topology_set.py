from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Set, Tuple

from .config import SearchConfig, UnionRuntimeConfig
from .profiles import DEFAULT_PROFILE, DatasetProfile
from .reward import risk_adjusted_score, squash_probe_score, structure_reward_from_components
from .types import SearchNode, StructureSummary, UnionGraph


def _edge_signature(node: SearchNode) -> Set[Tuple[str, str]]:
    role_by_agent = {agent_id: role for role, agent_id in node.state.role_to_agent.items()}
    return {
        (f"{role_by_agent.get(src, src)}@{src}", f"{role_by_agent.get(dst, dst)}@{dst}")
        for src, dst in node.compiled.edges
    }


def _agent_signature(node: SearchNode) -> Set[str]:
    return {f"{role}@{agent_id}" for role, agent_id in node.state.role_to_agent.items()}


@dataclass(frozen=True)
class RankedTopology:
    node: SearchNode
    quality: float
    quality_score: float
    structure_score: float
    readiness: float
    affordability: float
    score: float


class TopologySetScorer:
    def __init__(self, search_config: SearchConfig, union_config: UnionRuntimeConfig):
        self.search_config = search_config
        self.union_config = union_config

    def _quality(self, node: SearchNode) -> float:
        if node.tier2 is not None:
            return risk_adjusted_score(node.tier2, self.search_config.risk_std_penalty)
        if node.tier1 is not None:
            return node.tier1.mean_reward
        return node.proxy_score if node.proxy_score is not None else float("-inf")

    def _rank_node(self, node: SearchNode, dataset_profile: DatasetProfile) -> RankedTopology:
        quality = self._quality(node)
        quality_score = self._normalize_quality(quality)
        structure_score, readiness, affordability = self._structure_score(node, dataset_profile)
        prior = dataset_profile.structure_prior
        total = max(1e-6, prior.selector_quality_weight + prior.selector_structure_weight)
        quality_weight = prior.selector_quality_weight / total
        structure_weight = prior.selector_structure_weight / total
        score = quality_weight * quality_score + structure_weight * structure_score
        return RankedTopology(
            node=node,
            quality=quality,
            quality_score=quality_score,
            structure_score=structure_score,
            readiness=readiness,
            affordability=affordability,
            score=score,
        )

    @staticmethod
    def _normalize_quality(value: float) -> float:
        if math.isinf(value):
            return 0.0
        return squash_probe_score(value)

    @staticmethod
    def _critical_role_groups(dataset_profile: DatasetProfile) -> List[Set[str]]:
        task_type = dataset_profile.task_type
        if task_type == "code_generation":
            return [
                {"solver", "generator", "solver_a", "solver_b"},
                {"critic", "verifier", "judge"},
                {"reviser", "aggregator"},
                {"router"},
            ]
        if task_type in {"numeric", "math_expression"}:
            return [
                {"solver", "generator", "solver_a", "solver_b"},
                {"critic", "verifier", "judge"},
                {"reviser", "aggregator"},
            ]
        return [
            {"solver", "generator", "solver_a", "solver_b"},
            {"critic", "verifier", "judge"},
            {"reviser", "aggregator"},
        ]

    def _role_group_coverage(self, roles: Set[str], dataset_profile: DatasetProfile) -> float:
        groups = self._critical_role_groups(dataset_profile)
        if not groups:
            return 0.0
        hits = sum(1 for group in groups if roles & group)
        return hits / len(groups)

    def _readiness(self, node: SearchNode, dataset_profile: DatasetProfile) -> float:
        roles = set(node.state.role_to_agent)
        readiness = 0.0
        if "verifier" in roles or "critic" in roles or "judge" in roles:
            readiness += 0.03
        if "router" in roles:
            readiness += 0.02
        if len(node.compiled.sinks) == 1:
            readiness += 0.01
        if dataset_profile.task_type == "code_generation":
            active = set(node.state.active_agents())
            if "coder" in active:
                readiness += 0.03
            if "planner" in active:
                readiness += 0.02
            if "verifier" in active:
                readiness += 0.03
        readiness += 0.04 * self._role_group_coverage(roles, dataset_profile)
        return max(0.0, min(1.0, readiness))

    def _affordability(self, node: SearchNode, dataset_profile: DatasetProfile) -> float:
        prior = dataset_profile.structure_prior
        active_count = len(node.state.active_agents())
        edge_count = len(node.compiled.edges)
        target_nodes = max(2, prior.target_task_nodes)
        target_edges = max(2, prior.target_task_edges)
        node_score = max(0.0, 1.0 - max(0, active_count - target_nodes) / max(2, target_nodes))
        edge_score = max(0.0, 1.0 - max(0, edge_count - target_edges) / max(3, target_edges))
        preferred_sinks = max(1, prior.preferred_sink_count)
        sink_delta = abs(len(node.compiled.sinks) - preferred_sinks)
        sink_score = max(0.4, 1.0 - 0.25 * sink_delta)
        return max(0.0, min(1.0, 0.45 * node_score + 0.35 * edge_score + 0.20 * sink_score))

    def _structure_score(self, node: SearchNode, dataset_profile: DatasetProfile) -> tuple[float, float, float]:
        readiness = self._readiness(node, dataset_profile)
        affordability = self._affordability(node, dataset_profile)
        structure_score = max(0.0, min(1.0, 0.60 * readiness + 0.40 * affordability))
        return structure_score, readiness, affordability

    def _complementarity(self, left: SearchNode, right: SearchNode) -> float:
        left_edges = _edge_signature(left)
        right_edges = _edge_signature(right)
        union_edges = left_edges | right_edges
        edge_diversity = 0.0 if not union_edges else 1.0 - (len(left_edges & right_edges) / len(union_edges))

        left_agents = _agent_signature(left)
        right_agents = _agent_signature(right)
        union_agents = left_agents | right_agents
        agent_diversity = 0.0 if not union_agents else 1.0 - (len(left_agents & right_agents) / len(union_agents))

        template_bonus = 0.15 if left.state.template != right.state.template else 0.0
        sink_bonus = 0.08 if set(left.compiled.sinks) != set(right.compiled.sinks) else 0.0
        return max(0.0, min(1.0, 0.45 * edge_diversity + 0.30 * agent_diversity + template_bonus + sink_bonus))

    def _ranked_candidates(
        self,
        candidates: Iterable[SearchNode],
        dataset_profile: DatasetProfile,
    ) -> List[RankedTopology]:
        ranked: List[RankedTopology] = []
        seen: set[str] = set()
        for node in candidates:
            signature = node.compiled.signature()
            if signature in seen:
                continue
            seen.add(signature)
            ranked_node = self._rank_node(node, dataset_profile)
            if ranked_node.quality < self.union_config.topology_quality_floor:
                continue
            ranked.append(ranked_node)
        ranked.sort(key=lambda item: (item.score, item.quality, item.node.compiled.signature()), reverse=True)
        return ranked[: max(self.union_config.candidate_pool_k, self.union_config.selected_topology_k)]

    def select(
        self,
        candidates: Iterable[SearchNode],
        *,
        dataset_profile: DatasetProfile = DEFAULT_PROFILE,
        fallback_best: SearchNode,
    ) -> List[SearchNode]:
        ranked = self._ranked_candidates(candidates, dataset_profile)
        if not ranked:
            return [fallback_best]

        selected: List[RankedTopology] = [ranked[0]]
        remaining = ranked[1:]
        target_k = max(1, self.union_config.selected_topology_k)

        while remaining and len(selected) < target_k:
            best_idx = 0
            best_score = float("-inf")
            diversity_weight = max(
                0.0,
                min(1.0, self.union_config.topology_diversity_weight * dataset_profile.structure_prior.selector_diversity_scale),
            )
            for idx, candidate in enumerate(remaining):
                diversity = 0.0
                if selected:
                    diversity = sum(self._complementarity(candidate.node, cur.node) for cur in selected) / len(selected)
                blended = (
                    (1.0 - diversity_weight) * candidate.score
                    + diversity_weight * diversity
                    + self.union_config.topology_union_bonus * min(1.0, diversity)
                )
                if blended > best_score:
                    best_score = blended
                    best_idx = idx
            selected.append(remaining.pop(best_idx))

        result = [item.node for item in selected]
        if fallback_best.compiled.signature() not in {node.compiled.signature() for node in result}:
            result[0] = fallback_best
        return result[:target_k]

    def summarize(
        self,
        selected: Sequence[SearchNode],
        union_graph: UnionGraph,
        *,
        dataset_profile: DatasetProfile = DEFAULT_PROFILE,
        mode: str,
    ) -> StructureSummary:
        selected_nodes = list(selected)
        selected_signatures = [node.compiled.signature() for node in selected_nodes]
        selected_scores = [self._rank_node(node, dataset_profile).score for node in selected_nodes]

        task_nodes = [node for node in union_graph.nodes.values() if node.node_type == "task"]
        task_edges = [edge for edge in union_graph.edges if edge.edge_type == "task"]
        selected_count = max(1, len(selected_nodes))

        templates = {node.state.template.value for node in selected_nodes}
        role_coverage = self._role_group_coverage({node.role for node in task_nodes}, dataset_profile)
        template_spread = min(1.0, len(templates) / max(1, min(3, selected_count)))
        coverage = max(0.0, min(1.0, 0.70 * role_coverage + 0.30 * template_spread))

        if len(selected_nodes) <= 1:
            complementarity = 0.0
        else:
            pair_scores: List[float] = []
            for idx, left in enumerate(selected_nodes):
                for right in selected_nodes[idx + 1 :]:
                    pair_scores.append(self._complementarity(left, right))
            complementarity = sum(pair_scores) / max(1, len(pair_scores))

        graph_count = max(1, len(union_graph.source_topology_signatures))
        if task_nodes:
            shared_node_ratio = sum(1 for node in task_nodes if node.support_count > 1) / len(task_nodes)
            support_strength = sum(node.support_count / graph_count for node in task_nodes) / len(task_nodes)
            node_balance = max(0.0, 1.0 - abs(shared_node_ratio - 0.5) * 2.0)
        else:
            support_strength = 0.0
            node_balance = 0.0
        if task_edges:
            shared_edge_ratio = sum(1 for edge in task_edges if edge.support_count > 1) / len(task_edges)
            edge_balance = max(0.0, 1.0 - abs(shared_edge_ratio - 0.5) * 2.0)
        else:
            edge_balance = 0.0
        redundancy_quality = max(0.0, min(1.0, 0.40 * node_balance + 0.20 * edge_balance + 0.40 * support_strength))

        prior = dataset_profile.structure_prior
        root_clarity = 1.0 / (1.0 + max(0, len(union_graph.root_node_ids) - max(1, prior.max_root_count)))
        sink_clarity = 1.0 / (1.0 + abs(len(union_graph.sink_node_ids) - max(1, prior.preferred_sink_count)))
        if task_nodes:
            level_coherence = sum(1.0 / (1.0 + node.topo_level_var) for node in task_nodes) / len(task_nodes)
        else:
            level_coherence = 0.0
        structural_faithfulness = max(
            0.0,
            min(
                1.0,
                0.30 * root_clarity + 0.25 * sink_clarity + 0.20 * level_coherence + 0.25 * role_coverage,
            ),
        )

        target_nodes = max(2, prior.target_task_nodes)
        target_edges = max(2, prior.target_task_edges)
        node_score = max(0.0, 1.0 - max(0, len(task_nodes) - target_nodes) / max(2, target_nodes))
        edge_score = max(0.0, 1.0 - max(0, len(task_edges) - target_edges) / max(4, target_edges))
        runtime_affordability = max(0.0, min(1.0, 0.55 * node_score + 0.45 * edge_score))

        if selected_nodes:
            execution_probe = sum(self._normalize_quality(self._quality(node)) for node in selected_nodes) / len(selected_nodes)
        else:
            execution_probe = 0.0

        metrics = structure_reward_from_components(
            coverage=coverage,
            complementarity=complementarity,
            redundancy_quality=redundancy_quality,
            structural_faithfulness=structural_faithfulness,
            runtime_affordability=runtime_affordability,
            execution_probe=execution_probe,
            topology_quality_weight=prior.topology_quality_weight,
            diversity_quality_weight=prior.diversity_quality_weight,
            deployability_weight=prior.deployability_weight,
            execution_probe_weight=prior.execution_probe_weight,
            coverage_mix=prior.coverage_mix,
            complementarity_mix=prior.complementarity_mix,
            metadata={
                "template_count": len(templates),
                "task_node_count": len(task_nodes),
                "task_edge_count": len(task_edges),
                "root_count": len(union_graph.root_node_ids),
                "sink_count": len(union_graph.sink_node_ids),
            },
        )
        signature = "structure|" + "||".join(selected_signatures)
        return StructureSummary(
            mode=mode,
            signature=signature,
            selected_topology_signatures=selected_signatures,
            selected_topology_scores=selected_scores,
            metrics=metrics,
            metadata={
                "templates": sorted(templates),
                "graph_count": graph_count,
                "root_node_ids": list(union_graph.root_node_ids),
                "sink_node_ids": list(union_graph.sink_node_ids),
            },
        )
