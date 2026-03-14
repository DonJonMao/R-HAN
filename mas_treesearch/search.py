from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

from .agents import AgentPool
from .compiler import compile_architecture
from .config import SearchConfig
from .evaluator import MultiFidelityEvaluator
from .gating import AgentSelection
from .learning import FeatureBuilder, LearnableEditPrior, LearnableValueModel
from .operators import TEMPLATE_SPECS, roles_for_template
from .profiles import DatasetProfile, DEFAULT_PROFILE
from .proxy import StaticProxyScorer
from .reward import risk_adjusted_score
from .types import (
    ArchitectureState,
    EditAction,
    PromptSlots,
    SearchNode,
    SearchRecord,
    SearchResult,
    WorkflowTemplate,
)


_PROMPT_SLOT_VALUES: Dict[str, Sequence[str]] = {
    "reasoning_mode": ("direct", "stepwise", "critique_then_answer"),
    "upstream_usage": ("summary", "quote_then_reason"),
    "output_style": ("raw", "bullet", "json"),
    "verification_mode": ("off", "light", "strict"),
    "finalization": ("answer_only", "answer_with_rationale"),
}


def _mean(values: Sequence[float]) -> float:
    return sum(values) / max(1, len(values))


@dataclass
class RootTemplateBuilder:
    agent_pool: AgentPool

    @staticmethod
    def _role_requires_unique_agent(role: str) -> bool:
        return role in {
            "solver",
            "solver_a",
            "solver_b",
            "generator",
            "critic",
            "reviser",
            "verifier",
            "aggregator",
            "judge",
            "router",
        }

    def _pick(
        self,
        candidate_ids: Sequence[str],
        preferred_agent_ids: Sequence[str],
        preferred_capabilities: Sequence[str],
        used: Sequence[str],
    ) -> str:
        by_id = self.agent_pool.by_id()
        for aid in preferred_agent_ids:
            if aid in candidate_ids and aid not in used:
                return aid
        for cap in preferred_capabilities:
            for aid in candidate_ids:
                if aid in used:
                    continue
                if cap in by_id[aid].capabilities:
                    return aid
        for aid in candidate_ids:
            if aid not in used:
                return aid
        return candidate_ids[0]

    def build(
        self,
        selection: AgentSelection,
        template_name: str,
        profile: DatasetProfile = DEFAULT_PROFILE,
    ) -> ArchitectureState:
        template = WorkflowTemplate(template_name)
        roles = roles_for_template(template)
        assigned: Dict[str, str] = {}
        used: List[str] = []
        prefs = {
            "solver": ("reasoning", "math", "analysis"),
            "solver_a": ("reasoning", "math", "debate"),
            "solver_b": ("reasoning", "debate", "analysis"),
            "generator": ("reasoning", "analysis"),
            "critic": ("verification", "critique"),
            "reviser": ("synthesis", "reasoning"),
            "verifier": ("verification", "format_check"),
            "aggregator": ("aggregation", "synthesis"),
            "judge": ("verification", "aggregation"),
            "router": ("routing", "planning"),
        }
        for role in roles:
            aid = self._pick(
                selection.candidate_agent_ids,
                profile.role_agent_preferences.get(role, ()),
                prefs.get(role, ("reasoning",)),
                used,
            )
            assigned[role] = aid
            used.append(aid)
        for required_agent_id in profile.required_agent_ids:
            if required_agent_id in assigned.values() or required_agent_id not in selection.candidate_agent_ids:
                continue
            for role in roles:
                if required_agent_id in profile.role_agent_preferences.get(role, ()):
                    if self._role_requires_unique_agent(role) and required_agent_id in used:
                        continue
                    assigned[role] = required_agent_id
                    if required_agent_id not in used:
                        used.append(required_agent_id)
                    break
        prompts = {role: profile.prompt_for_role(role, PromptSlots()) for role in roles}
        if template == WorkflowTemplate.SOLVE_VERIFY and "verifier" in prompts:
            prompts["verifier"] = prompts["verifier"].with_slot("verification_mode", "strict")
        return ArchitectureState(
            template=template,
            role_to_agent=assigned,
            role_to_prompt=prompts,
        )


class TreeSearchEngine:
    def __init__(
        self,
        config: SearchConfig,
        agent_pool: AgentPool,
        proxy_scorer: StaticProxyScorer,
        evaluator: MultiFidelityEvaluator,
        feature_builder: FeatureBuilder,
        edit_prior: Optional[LearnableEditPrior] = None,
        value_model: Optional[LearnableValueModel] = None,
    ):
        self.config = config
        self.agent_pool = agent_pool
        self.proxy_scorer = proxy_scorer
        self.evaluator = evaluator
        self.feature_builder = feature_builder
        self.edit_prior = edit_prior
        self.value_model = value_model
        self.rng = random.Random(config.random_seed)
        self._nodes: Dict[str, SearchNode] = {}
        self._records: List[SearchRecord] = []
        self._roots: List[str] = []
        self._candidate_agent_ids: List[str] = []
        self._allowed_templates: List[str] = [template.value for template in WorkflowTemplate]
        self._dataset_profile: DatasetProfile = DEFAULT_PROFILE
        self._prompt_edit_cooldown: int = config.prompt_edit_cooldown
        self._max_prompt_edits_per_state: int = config.max_prompt_edits_per_state

    @staticmethod
    def _role_requires_unique_agent(role: str) -> bool:
        return RootTemplateBuilder._role_requires_unique_agent(role)

    def _signature(self, state: ArchitectureState) -> str:
        return compile_architecture(state).signature()

    def _ensure_node(
        self,
        state: ArchitectureState,
        parent_signature: Optional[str],
        action: str,
    ) -> SearchNode:
        signature = self._signature(state)
        existing = self._nodes.get(signature)
        if existing is not None:
            return existing
        compiled = compile_architecture(state)
        node = SearchNode(
            state=state,
            compiled=compiled,
            parent_signature=parent_signature,
            action_from_parent=action,
        )
        node.unexpanded_actions = self._enumerate_actions(state)
        self._nodes[signature] = node
        return node

    def _enumerate_actions(self, state: ArchitectureState) -> List[EditAction]:
        actions: List[EditAction] = []
        for template_name in self._allowed_templates:
            template = WorkflowTemplate(template_name)
            if template != state.template:
                actions.append(EditAction(kind="change_template", payload={"template": template.value}))
        for role, aid in sorted(state.role_to_agent.items()):
            for candidate in self._candidate_agent_ids:
                if candidate == aid:
                    continue
                if self._role_requires_unique_agent(role):
                    occupied_elsewhere = any(
                        other_role != role and other_agent == candidate
                        for other_role, other_agent in state.role_to_agent.items()
                    )
                    if occupied_elsewhere:
                        continue
                actions.append(EditAction(kind="swap_agent", payload={"role": role, "agent_id": candidate}))
        if (
            state.prompt_edit_count < self._max_prompt_edits_per_state
            and state.non_prompt_steps_since_edit >= self._prompt_edit_cooldown
        ):
            for role, slots in sorted(state.role_to_prompt.items()):
                for slot_name, values in _PROMPT_SLOT_VALUES.items():
                    current_val = getattr(slots, slot_name)
                    for value in values:
                        if value != current_val:
                            actions.append(
                                EditAction(
                                    kind="set_prompt_slot",
                                    payload={"role": role, "slot": slot_name, "value": value},
                                )
                            )
        elif state.prompt_edit_count == 0:
            for role, slots in sorted(state.role_to_prompt.items()):
                for slot_name, values in _PROMPT_SLOT_VALUES.items():
                    current_val = getattr(slots, slot_name)
                    for value in values:
                        if value != current_val:
                            actions.append(
                                EditAction(
                                    kind="set_prompt_slot",
                                    payload={"role": role, "slot": slot_name, "value": value},
                                )
                            )
        actions.append(EditAction(kind="stop", payload={}))
        return actions

    def _rank_actions(
        self,
        state: ArchitectureState,
        actions: List[EditAction],
        question_vector: Sequence[float],
    ) -> List[EditAction]:
        if not self.config.enable_learned_edit_prior or self.edit_prior is None:
            return actions
        scored: List[tuple[float, EditAction]] = []
        for action in actions:
            feats = self.feature_builder.action_features(state, action, question_vector)
            score = self.edit_prior.score(feats)
            scored.append((score, action))
        scored.sort(key=lambda item: (item[0], item[1].describe()), reverse=True)
        return [action for _, action in scored]

    def _apply_action(self, state: ArchitectureState, action: EditAction) -> ArchitectureState:
        if action.kind == "stop":
            return state
        if action.kind == "change_template":
            template = WorkflowTemplate(action.payload["template"])
            current_agents = state.active_agents()
            candidate_pool = current_agents + [aid for aid in self._candidate_agent_ids if aid not in current_agents]
            builder = RootTemplateBuilder(self.agent_pool)
            rebuilt = builder.build(
                AgentSelection(
                    question_vector=[],
                    candidate_agent_ids=candidate_pool,
                    core_agent_ids=[],
                    explore_agent_ids=[],
                    scores={},
                ),
                template.value,
                profile=self._dataset_profile,
            )
            return ArchitectureState(
                template=template,
                role_to_agent=rebuilt.role_to_agent,
                role_to_prompt=rebuilt.role_to_prompt,
                prompt_edit_count=0,
                non_prompt_steps_since_edit=state.non_prompt_steps_since_edit + 1,
            )
        if action.kind == "swap_agent":
            role_to_agent = dict(state.role_to_agent)
            target_agent = action.payload["agent_id"]
            role = action.payload["role"]
            if self._role_requires_unique_agent(role):
                occupied_elsewhere = any(
                    other_role != role and other_agent == target_agent
                    for other_role, other_agent in role_to_agent.items()
                )
                if occupied_elsewhere:
                    return state
            role_to_agent[role] = target_agent
            return ArchitectureState(
                template=state.template,
                role_to_agent=role_to_agent,
                role_to_prompt=dict(state.role_to_prompt),
                prompt_edit_count=state.prompt_edit_count,
                non_prompt_steps_since_edit=state.non_prompt_steps_since_edit + 1,
            )
        if action.kind == "set_prompt_slot":
            role_to_prompt = dict(state.role_to_prompt)
            role = action.payload["role"]
            role_to_prompt[role] = role_to_prompt[role].with_slot(
                action.payload["slot"],
                action.payload["value"],
            )
            return ArchitectureState(
                template=state.template,
                role_to_agent=dict(state.role_to_agent),
                role_to_prompt=role_to_prompt,
                prompt_edit_count=state.prompt_edit_count + 1,
                non_prompt_steps_since_edit=0,
            )
        raise KeyError(f"Unsupported action kind: {action.kind}")

    def _progressive_limit(self, visits: int) -> int:
        return self.config.progressive_widening_base + int(
            self.config.progressive_widening_alpha * math.sqrt(max(1, visits))
        )

    def _select_parent(self) -> SearchNode:
        if self.rng.random() < self.config.root_restart_prob:
            return self._nodes[self.rng.choice(self._roots)]
        scored = sorted(
            self._nodes.values(),
            key=lambda node: (
                node.stats.q_mean if node.stats.visits > 0 else float("-inf"),
                node.proxy_score if node.proxy_score is not None else float("-inf"),
            ),
            reverse=True,
        )
        candidates = scored[: max(1, self.config.top_k_selection)]
        weights = []
        for node in candidates:
            prior = node.proxy_score if node.proxy_score is not None else 0.1
            exploit = node.stats.q_mean
            bonus = self.config.puct_c * prior * math.sqrt(max(1, node.stats.visits + 1)) / (1 + len(node.children))
            weights.append(max(1e-6, exploit + bonus + 1.0))
        total = sum(weights)
        draw = self.rng.random() * total
        acc = 0.0
        for node, weight in zip(candidates, weights):
            acc += weight
            if acc >= draw:
                return node
        return candidates[-1]

    def _evaluate_proxy(self, node: SearchNode, question_vector: Sequence[float]) -> None:
        proxy = self.proxy_scorer.score(node.compiled, list(question_vector), profile=self._dataset_profile)
        score = proxy.score
        uncertainty = proxy.uncertainty
        if self.config.enable_learned_value_model and self.value_model is not None:
            feats = self.feature_builder.state_features(node.compiled, question_vector)
            learned_mean, learned_uncertainty = self.value_model.predict(feats)
            alpha = max(0.0, min(1.0, self.config.learned_value_weight))
            score = (1.0 - alpha) * score + alpha * learned_mean
            uncertainty = (1.0 - alpha) * uncertainty + alpha * learned_uncertainty
        node.proxy_score = score
        node.proxy_uncertainty = uncertainty
        node.stats.proxy_mean = score

    def _update_stats(self, node: SearchNode, reward: float) -> None:
        node.stats.visits += 1
        node.stats.q_mean += (reward - node.stats.q_mean) / node.stats.visits
        node.stats.q_max = max(node.stats.q_max, reward)

    def _backpropagate(self, node: SearchNode, reward: float) -> None:
        cursor: Optional[SearchNode] = node
        while cursor is not None:
            self._update_stats(cursor, reward)
            if cursor.parent_signature is None:
                break
            cursor = self._nodes.get(cursor.parent_signature)

    def _tier_filter(self, nodes: List[SearchNode], fraction: float) -> List[SearchNode]:
        if not nodes:
            return []
        limit = max(1, int(math.ceil(len(nodes) * fraction)))
        ranked = sorted(
            nodes,
            key=lambda n: (
                n.tier1.mean_reward if n.tier1 is not None else (n.proxy_score if n.proxy_score is not None else float("-inf")),
                -(n.proxy_uncertainty if n.proxy_uncertainty is not None else 0.0),
            ),
            reverse=True,
        )
        return ranked[:limit]

    def search(
        self,
        question_text: str,
        selection: AgentSelection,
        *,
        reference_answer: Optional[str] = None,
        metadata: Optional[dict] = None,
        dataset_profile: DatasetProfile = DEFAULT_PROFILE,
        learn: bool = True,
    ) -> SearchResult:
        self._nodes = {}
        self._records = []
        self._roots = []
        self._candidate_agent_ids = list(selection.candidate_agent_ids)
        self._dataset_profile = dataset_profile
        self._allowed_templates = list(dataset_profile.allowed_templates or [template.value for template in WorkflowTemplate])
        self._prompt_edit_cooldown = (
            dataset_profile.search_overrides.prompt_edit_cooldown or self.config.prompt_edit_cooldown
        )
        self._max_prompt_edits_per_state = (
            dataset_profile.search_overrides.max_prompt_edits_per_state or self.config.max_prompt_edits_per_state
        )
        search_iterations = dataset_profile.search_overrides.search_iterations or self.config.search_iterations
        tier1_fraction = dataset_profile.search_overrides.tier1_top_fraction or self.config.tier1_top_fraction
        tier2_fraction = dataset_profile.search_overrides.tier2_top_fraction or self.config.tier2_top_fraction
        final_top_k = dataset_profile.search_overrides.final_top_k or self.config.final_top_k
        builder = RootTemplateBuilder(self.agent_pool)
        root_templates = dataset_profile.root_templates or self.config.root_templates
        for template_name in root_templates:
            state = builder.build(selection, template_name, profile=dataset_profile)
            node = self._ensure_node(state, parent_signature=None, action="root")
            node.unexpanded_actions = self._rank_actions(node.state, node.unexpanded_actions, selection.question_vector)
            self._evaluate_proxy(node, selection.question_vector)
            signature = node.compiled.signature()
            self._roots.append(signature)

        for _ in range(search_iterations):
            parent = self._select_parent()
            parent.unexpanded_actions = self._rank_actions(parent.state, parent.unexpanded_actions, selection.question_vector)
            limit = self._progressive_limit(parent.stats.visits)
            expanded: List[SearchNode] = []
            while parent.unexpanded_actions and len(expanded) < limit:
                action = parent.unexpanded_actions.pop(0)
                child_state = self._apply_action(parent.state, action)
                child = self._ensure_node(
                    child_state,
                    parent_signature=parent.compiled.signature(),
                    action=action.describe(),
                )
                child.unexpanded_actions = self._rank_actions(child.state, child.unexpanded_actions, selection.question_vector)
                if child.compiled.signature() not in parent.children:
                    parent.children.append(child.compiled.signature())
                self._evaluate_proxy(child, selection.question_vector)
                self._records.append(
                    SearchRecord(
                        state_signature=child.compiled.signature(),
                        parent_signature=parent.compiled.signature(),
                        action=action.describe(),
                        proxy_score=child.proxy_score or 0.0,
                        tier1_score=None,
                        tier2_score=None,
                    )
                )
                expanded.append(child)
                if action.kind == "stop":
                    break
            tier1_nodes = self._tier_filter(expanded, tier1_fraction)
            for node in tier1_nodes:
                node.tier1 = self.evaluator.evaluate(
                    node.compiled,
                    question_text,
                    tier="tier1",
                    reference_answer=reference_answer,
                    metadata=metadata,
                    dataset_profile=dataset_profile,
                )
                node.stats.tier1_mean = node.tier1.mean_reward
                for record in self._records:
                    if record.state_signature == node.compiled.signature():
                        record.tier1_score = node.tier1.mean_reward
            tier2_nodes = self._tier_filter(tier1_nodes, tier2_fraction)
            for node in tier2_nodes:
                node.tier2 = self.evaluator.evaluate(
                    node.compiled,
                    question_text,
                    tier="tier2",
                    reference_answer=reference_answer,
                    metadata=metadata,
                    dataset_profile=dataset_profile,
                )
                node.stats.tier2_mean = node.tier2.mean_reward
                node.stats.tier2_std = node.tier2.reward_std
                if learn and self.config.enable_learned_value_model and self.value_model is not None:
                    state_feats = self.feature_builder.state_features(node.compiled, selection.question_vector)
                    self.value_model.update(state_feats, node.tier2.mean_reward)
                if learn and self.config.enable_learned_edit_prior and self.edit_prior is not None and node.parent_signature:
                    parent_node = self._nodes.get(node.parent_signature)
                    if parent_node is not None:
                        action_desc = node.action_from_parent
                        action = self._action_from_description(parent_node.state, action_desc)
                        if action is not None:
                            action_feats = self.feature_builder.action_features(
                                parent_node.state,
                                action,
                                selection.question_vector,
                            )
                            parent_baseline = parent_node.proxy_score if parent_node.proxy_score is not None else 0.0
                            target = 1.0 if node.tier2.mean_reward > parent_baseline + self.config.score_improvement_epsilon else 0.0
                            self.edit_prior.update(action_feats, target)
                for record in self._records:
                    if record.state_signature == node.compiled.signature():
                        record.tier2_score = node.tier2.mean_reward
                reward = risk_adjusted_score(node.tier2, self.config.risk_std_penalty)
                self._backpropagate(node, reward)

        ranked = sorted(
            self._nodes.values(),
            key=lambda node: (
                risk_adjusted_score(node.tier2, self.config.risk_std_penalty) if node.tier2 else float("-inf"),
                node.stats.q_mean,
                node.proxy_score if node.proxy_score is not None else float("-inf"),
            ),
            reverse=True,
        )
        finalists = ranked[: max(1, final_top_k)]
        if not finalists:
            finalists = [self._nodes[self._roots[0]]]
        for node in finalists:
            if node.tier2 is None:
                node.tier2 = self.evaluator.evaluate(
                    node.compiled,
                    question_text,
                    tier="tier2",
                    reference_answer=reference_answer,
                    metadata=metadata,
                    dataset_profile=dataset_profile,
                )
                node.stats.tier2_mean = node.tier2.mean_reward
                node.stats.tier2_std = node.tier2.reward_std
        finalists = sorted(
            finalists,
            key=lambda node: risk_adjusted_score(node.tier2, self.config.risk_std_penalty) if node.tier2 else float("-inf"),
            reverse=True,
        )
        return SearchResult(
            question_text=question_text,
            selected_agents=selection.candidate_agent_ids,
            root_signatures=list(self._roots),
            best_node=finalists[0],
            top_nodes=finalists,
            records=list(self._records),
            nodes=dict(self._nodes),
        )

    def _action_from_description(self, state: ArchitectureState, description: str) -> Optional[EditAction]:
        for action in self._enumerate_actions(state):
            if action.describe() == description:
                return action
        return None
