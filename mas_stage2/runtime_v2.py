"""Stage2 Runtime V2: 复用 V1 执行壳并接入 latent memory 闭环。"""

from __future__ import annotations

import os
import time
from dataclasses import asdict
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from mas_treesearch.agents import AgentPool
from mas_treesearch.evaluator import MultiFidelityEvaluator
from mas_treesearch.profiles import DEFAULT_PROFILE, DatasetProfile
from mas_treesearch.prompting import build_system_prompt, render_question_text
from mas_treesearch.types import EvalSummary, UnionGraph, UnionNode

from .composer import SimpleMemoryComposer, tokenize_texts
from .config_v2 import Stage2V2Config
from .global_node import GlobalContextNode
from .gnn import LightweightGNN
from .lmpo import LMPOTrainer
from .memory import PrivateEpisodeMemoryStore
from .runtime import Stage2Runtime, _truncate
from .types import (
    ControllerState,
    ExportedMemoryMessage,
    MemoryRecord,
    NodeTurnTrace,
    Stage2RunResult,
    TurnTrace,
)


class Stage2RuntimeV2(Stage2Runtime):
    """正式版 V2 运行时。

    设计原则：
    1. 保留 V1 已经验证可运行的多轮执行骨架；
    2. 用 latent composer 替换 V1 的纯文本 local composer；
    3. 用 GNN 聚合邻居 latent，并通过 global node 汇总轮次级信息；
    4. 通过 latent-guided verbalizer 将 latent state 真正作用到 prompt；
    5. 对 verbalizer 的离散选择做 REINFORCE，而不是对占位 tensor 做伪优化。
    """

    def __init__(
        self,
        config: Stage2V2Config,
        evaluator: MultiFidelityEvaluator,
        agent_pool: AgentPool,
        embedder,
    ):
        self.v2_config = config
        self.embed_dim = int(getattr(embedder.config, "dim", 4096))
        super().__init__(config.to_runtime_config(), evaluator, agent_pool, embedder)
        hidden_dim = self.v2_config.resolved_hidden_dim(self.embed_dim)
        self.composer = SimpleMemoryComposer(
            self.v2_config.get_composer_config(self.embed_dim),
            vocab_size=self.v2_config.composer_vocab_size,
        )
        self.gnn = LightweightGNN(self.v2_config.get_gnn_config(self.embed_dim)) if self.v2_config.gnn_enabled else None
        self.global_node = (
            GlobalContextNode(self.v2_config.get_global_node_config(self.embed_dim))
            if self.v2_config.global_node_enabled
            else None
        )
        self.latent_to_embed = nn.Linear(hidden_dim, self.embed_dim)
        if not self.v2_config.latent_bridge_trainable:
            for param in self.latent_to_embed.parameters():
                param.requires_grad = False
        self.lmpo_trainer = LMPOTrainer([self.composer, self.latent_to_embed], self.v2_config.get_lmpo_config())
        self._global_text_summary = ""
        self._pending_policy_log_probs: List[torch.Tensor] = []
        self._pending_policy_entropies: List[torch.Tensor] = []
        self._current_learn = False

    def state_dict(self) -> dict:
        payload = dict(super().state_dict())
        payload["v2_runtime"] = {
            "global_text_summary": self._global_text_summary,
            "lmpo_stats": self.lmpo_trainer.get_stats(),
        }
        return payload

    def load_state_dict(self, state: dict) -> None:
        super().load_state_dict(state)
        runtime_state = state.get("v2_runtime")
        if isinstance(runtime_state, dict):
            self._global_text_summary = str(runtime_state.get("global_text_summary", ""))

    def save_binary_state(self, path: str) -> None:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        torch.save(
            {
                "composer": self.composer.state_dict(),
                "latent_to_embed": self.latent_to_embed.state_dict(),
                "gnn": self.gnn.state_dict() if self.gnn is not None else None,
                "global_node": self.global_node.state_dict() if self.global_node is not None else None,
            },
            path,
        )

    def load_binary_state(self, path: str) -> None:
        payload = torch.load(path, map_location="cpu")
        if isinstance(payload, dict):
            if "composer" in payload:
                self.composer.load_state_dict(payload["composer"])
            if "latent_to_embed" in payload:
                self.latent_to_embed.load_state_dict(payload["latent_to_embed"])
            if self.gnn is not None and payload.get("gnn") is not None:
                self.gnn.load_state_dict(payload["gnn"])
            if self.global_node is not None and payload.get("global_node") is not None:
                self.global_node.load_state_dict(payload["global_node"])

    @staticmethod
    def _vector_tensor(vector: Sequence[float]) -> torch.Tensor:
        return torch.tensor(list(vector), dtype=torch.float32)

    def _hidden_to_embed(self, hidden: torch.Tensor) -> torch.Tensor:
        projected = self.latent_to_embed(hidden)
        return F.normalize(projected, dim=-1)

    def _latent_query(self, latent: torch.Tensor) -> torch.Tensor:
        pooled = latent.mean(dim=0)
        return self._hidden_to_embed(pooled)

    def _latent_sequence_from_export(self, message: ExportedMemoryMessage) -> Optional[torch.Tensor]:
        payload = message.metadata.get("latent_sequence")
        if isinstance(payload, list) and payload and isinstance(payload[0], list):
            return torch.tensor(payload, dtype=torch.float32)
        return None

    def _composer_texts(
        self,
        node: UnionNode,
        question_text: str,
        controller_state: ControllerState,
        records: Sequence[MemoryRecord],
    ) -> List[str]:
        texts = [
            f"role={node.role}",
            f"question={_truncate(question_text, 480)}",
            f"controller_mode={controller_state.mode}",
            f"controller_focus={_truncate(controller_state.focus, 220)}",
        ]
        if records:
            for record in records:
                texts.append(f"[{record.record_type}|{record.feedback_type}] {_truncate(record.text, self.v2_config.memory.max_record_chars)}")
        else:
            texts.append("no_private_memory")
        return texts

    def _compose_latent(
        self,
        node: UnionNode,
        question_text: str,
        controller_state: ControllerState,
        selected_items,
        records_by_id: Dict[str, MemoryRecord],
    ) -> torch.Tensor:
        raw_records = [records_by_id[item.record_id] for item in selected_items if item.record_id in records_by_id]
        input_ids, attention_mask = tokenize_texts(
            self._composer_texts(node, question_text, controller_state, raw_records),
            vocab_size=self.v2_config.composer_vocab_size,
            max_length=self.v2_config.composer_max_input_length,
        )
        latent = self.composer(input_ids, attention_mask).squeeze(0)
        return latent

    def _aggregate_v2_neighbors(
        self,
        node: UnionNode,
        self_latent: torch.Tensor,
        active_edges,
        previous_exports: Dict[str, ExportedMemoryMessage],
    ) -> torch.Tensor:
        if self.gnn is None:
            return self_latent
        neighbor_latents: List[torch.Tensor] = []
        edge_scores: List[float] = []
        for activation in active_edges:
            if not activation.active or activation.dst != node.node_id:
                continue
            export_message = previous_exports.get(activation.src)
            if export_message is None:
                continue
            latent_seq = self._latent_sequence_from_export(export_message)
            if latent_seq is None:
                continue
            neighbor_latents.append(latent_seq)
            edge_scores.append(float(activation.score))
        if not neighbor_latents:
            return self_latent
        edge_weight_tensor = torch.tensor(edge_scores, dtype=torch.float32)
        edge_weights = torch.softmax(edge_weight_tensor, dim=0).tolist()
        return self.gnn(self_latent, neighbor_latents, edge_weights)

    def _integrate_global_context(self, latent: torch.Tensor, controller_state: ControllerState) -> torch.Tensor:
        if self.global_node is None:
            return latent
        global_ctx = self.global_node.get_context(1).squeeze(0)
        strength = 0.20 + 0.20 * (1.0 - float(controller_state.uncertainty))
        return latent + strength * global_ctx.unsqueeze(0)

    def _candidate_scores(self, query: torch.Tensor, vectors: Sequence[Sequence[float]]) -> torch.Tensor:
        matrix = torch.stack([self._vector_tensor(vector) for vector in vectors])
        matrix = F.normalize(matrix, dim=-1)
        return matrix @ query

    def _choose_indices(
        self,
        scores: torch.Tensor,
        *,
        max_items: int,
        sample: bool,
    ) -> Tuple[List[int], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if scores.numel() == 0 or max_items <= 0:
            return [], None, None
        max_items = min(int(max_items), int(scores.numel()))
        if not sample:
            values, indices = torch.topk(scores, k=max_items)
            del values
            return sorted(indices.tolist()), None, None
        masked = scores.clone()
        chosen: List[int] = []
        log_probs: List[torch.Tensor] = []
        entropies: List[torch.Tensor] = []
        temperature = max(1e-3, float(self.v2_config.lmpo_sampling_temperature))
        for _ in range(max_items):
            probs = torch.softmax(masked / temperature, dim=0)
            index = torch.multinomial(probs, 1).item()
            chosen.append(index)
            log_probs.append(torch.log(probs[index] + 1e-9))
            entropies.append(-(probs * torch.log(probs + 1e-9)).sum())
            masked[index] = -1e9
        return sorted(chosen), torch.stack(log_probs).sum(), torch.stack(entropies).sum()

    def _build_memory_brief(
        self,
        node: UnionNode,
        selected_items,
        records_by_id: Dict[str, MemoryRecord],
        neighbour_exports: Sequence[ExportedMemoryMessage],
        enhanced_latent: torch.Tensor,
    ) -> Tuple[str, Optional[torch.Tensor], Optional[torch.Tensor], Dict[str, object]]:
        query = self._latent_query(enhanced_latent)
        local_candidates = []
        for item in selected_items:
            record = records_by_id.get(item.record_id)
            if record is None:
                continue
            local_candidates.append(
                {
                    "record_id": item.record_id,
                    "text": f"[{record.record_type}|{record.feedback_type}] {_truncate(record.text, self.v2_config.memory.max_record_chars)}",
                    "embedding": record.embedding,
                    "feedback_type": record.feedback_type,
                }
            )
        neighbour_candidates = [
            {
                "node_id": message.node_id,
                "text": f"[{message.node_id}] {_truncate(message.summary, self.v2_config.memory.max_export_chars)}",
                "embedding": message.latent_vector,
            }
            for message in neighbour_exports
        ]

        local_indices: List[int] = []
        local_log_prob: Optional[torch.Tensor] = None
        local_entropy: Optional[torch.Tensor] = None
        if local_candidates:
            local_scores = self._candidate_scores(query, [item["embedding"] for item in local_candidates])
            local_indices, local_log_prob, local_entropy = self._choose_indices(
                local_scores,
                max_items=self.v2_config.latent_prompt_max_local,
                sample=self._current_learn,
            )

        neighbour_indices: List[int] = []
        neighbour_log_prob: Optional[torch.Tensor] = None
        neighbour_entropy: Optional[torch.Tensor] = None
        if neighbour_candidates:
            neighbour_scores = self._candidate_scores(query, [item["embedding"] for item in neighbour_candidates])
            neighbour_indices, neighbour_log_prob, neighbour_entropy = self._choose_indices(
                neighbour_scores,
                max_items=self.v2_config.latent_prompt_max_neighbor,
                sample=self._current_learn,
            )

        local_lines = [local_candidates[index]["text"] for index in local_indices]
        neighbour_lines = [neighbour_candidates[index]["text"] for index in neighbour_indices]

        parts: List[str] = [
            "[Global Guidance]",
            _truncate(self._global_text_summary or "No accumulated global summary yet.", self.v2_config.memory.max_brief_chars // 4),
            "",
            "[Controller]",
            _truncate(node.role + " | " + self._controller_summary(node.role), self.v2_config.memory.max_brief_chars // 4),
        ]
        if local_lines:
            parts.extend(["", "[Latent-Selected Private Memory]"])
            parts.extend(f"- {line}" for line in local_lines)
        else:
            parts.extend(["", "[Latent-Selected Private Memory]", "- No strong private memory selected."])
        if neighbour_lines:
            parts.extend(["", "[Graph-Mediated Neighbour Signals]"])
            parts.extend(f"- {line}" for line in neighbour_lines)
        brief = _truncate("\n".join(parts), self.v2_config.memory.max_brief_chars)

        combined_log_prob = None
        combined_entropy = None
        log_prob_terms = [term for term in (local_log_prob, neighbour_log_prob) if term is not None]
        entropy_terms = [term for term in (local_entropy, neighbour_entropy) if term is not None]
        if log_prob_terms:
            combined_log_prob = torch.stack(log_prob_terms).sum()
        if entropy_terms:
            combined_entropy = torch.stack(entropy_terms).sum()
        return brief, combined_log_prob, combined_entropy, {
            "selected_local_ids": [local_candidates[index]["record_id"] for index in local_indices],
            "selected_neighbour_ids": [neighbour_candidates[index]["node_id"] for index in neighbour_indices],
        }

    def _controller_summary(self, role: str) -> str:
        weights = getattr(self, "_latest_controller_state", None)
        if weights is None:
            return "Controller not initialized."
        role_weight = float(weights.role_weights.get(role, 1.0))
        return f"mode={weights.mode}; focus={weights.focus}; role_weight={role_weight:.2f}; uncertainty={weights.uncertainty:.2f}"

    def _build_v2_export(
        self,
        node: UnionNode,
        latent: torch.Tensor,
        output_text: str,
        selected_items,
        records_by_id: Dict[str, MemoryRecord],
        *,
        current_turn: int,
    ) -> Tuple[ExportedMemoryMessage, str]:
        query = self._latent_query(latent)
        carried_lines: List[str] = []
        for item in selected_items[:2]:
            record = records_by_id.get(item.record_id)
            if record is not None:
                carried_lines.append(f"[{record.feedback_type}] {_truncate(record.text, self.v2_config.memory.max_export_chars)}")
        summary_lines = [f"{node.role} update:"]
        if carried_lines:
            summary_lines.append(f"Memory carry: {carried_lines[0]}")
        if output_text.strip():
            summary_lines.append(f"Current output: {_truncate(output_text, self.v2_config.memory.max_export_chars)}")
        export_summary = "\n".join(summary_lines)
        return (
            ExportedMemoryMessage(
                node_id=node.node_id,
                turn_index=current_turn,
                summary=export_summary,
                latent_vector=query.detach().cpu().tolist(),
                provenance_record_ids=[item.record_id for item in selected_items],
                confidence=min(1.0, 0.45 + 0.08 * len(selected_items)),
                metadata={
                    "role": node.role,
                    "latent_sequence": latent.detach().cpu().tolist(),
                },
            ),
            "\n".join(carried_lines) if carried_lines else "No carried private memory.",
        )

    def _run_task_node(
        self,
        graph: UnionGraph,
        node: UnionNode,
        *,
        question_text: str,
        metadata: Optional[dict],
        reference_answer: Optional[str],
        dataset_profile: DatasetProfile,
        controller_state: ControllerState,
        active_edges,
        previous_exports: Dict[str, ExportedMemoryMessage],
        episode_id: str,
        current_turn: int,
    ) -> Tuple[NodeTurnTrace, MemoryRecord, ExportedMemoryMessage]:
        local_records = self._memory_store.get(node.node_id)
        records_by_id = {record.record_id: record for record in local_records}
        selected_items = self._selector.select(
            node,
            question_text,
            controller_state,
            local_records,
            current_turn=current_turn,
        )
        local_latent = self._compose_latent(node, question_text, controller_state, selected_items, records_by_id)
        aggregated_latent = self._aggregate_v2_neighbors(node, local_latent, active_edges, previous_exports)
        enhanced_latent = self._integrate_global_context(aggregated_latent, controller_state)
        neighbour_exports = self._neighbour_exports(node.node_id, active_edges, previous_exports)
        memory_brief, policy_log_prob, entropy_term, selection_debug = self._build_memory_brief(
            node,
            selected_items,
            records_by_id,
            neighbour_exports,
            enhanced_latent,
        )
        if policy_log_prob is not None:
            self._pending_policy_log_probs.append(policy_log_prob)
        if entropy_term is not None:
            self._pending_policy_entropies.append(entropy_term)
        task_context = self.evaluator._task_context(
            question_text,
            dataset_profile=dataset_profile,
            metadata=metadata,
            role=node.role,
        )
        answer_contract = self._role_answer_contract(
            graph,
            node,
            question_text=question_text,
            reference_answer=reference_answer,
            metadata=metadata,
            dataset_profile=dataset_profile,
        )
        agent = self._by_id[node.agent_id]
        system_prompt = build_system_prompt(agent, self._prompt_slots(node), extra_role_hint=node.role)
        user_prompt = (
            f"Question:\n{render_question_text(question_text, metadata=metadata)}\n\n"
            f"Latent-guided memory brief:\n{memory_brief}\n\n"
            f"Current task:\n{self._role_instruction(node, turn_index=current_turn, total_turns=self.config.graph.turn_count, dataset_profile=dataset_profile, controller_state=controller_state, final_turn=current_turn + 1 == self.config.graph.turn_count)}"
        )
        if task_context:
            user_prompt += f"\n\nTask context:\n{task_context}"
        if answer_contract:
            user_prompt += f"\n\nOutput contract:\n{answer_contract}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        raw_output = self.evaluator._cached_chat(messages, runtime=self.evaluator._resolve_runtime("tier2", dataset_profile))
        output = self._postprocess_node_output(
            node,
            raw_output,
            question_text=question_text,
            reference_answer=reference_answer,
            metadata=metadata,
            dataset_profile=dataset_profile,
            answer_contract=answer_contract,
        )
        output_record = MemoryRecord(
            record_id=f"{node.node_id}::turn{current_turn}::output",
            episode_id=episode_id,
            turn_index=current_turn,
            owner_node_id=node.node_id,
            agent_id=node.agent_id,
            role=node.role,
            record_type="self_output",
            text=output,
            embedding=self.embedder.embed(output),
            token_estimate=max(1, len(output.split())),
            feedback_type="unresolved",
            confidence=min(1.0, 0.45 + 0.10 * len(selected_items)),
            source_node_id=node.node_id,
            metadata={
                "selected_record_ids": [item.record_id for item in selected_items],
                "neighbour_sources": [message.node_id for message in neighbour_exports],
                "stage2_version": "v2",
            },
        )
        export_message, latent_summary = self._build_v2_export(
            node,
            enhanced_latent,
            output,
            selected_items,
            records_by_id,
            current_turn=current_turn,
        )
        trace = NodeTurnTrace(
            node_id=node.node_id,
            agent_id=node.agent_id,
            role=node.role,
            active_incoming_edge_ids=[edge.edge_id for edge in active_edges if edge.active and edge.dst == node.node_id],
            selected_records=list(selected_items),
            neighbour_sources=[message.node_id for message in neighbour_exports],
            memory_brief=memory_brief,
            output=output,
            local_latent_summary=latent_summary,
            exported_summary=export_message.summary if self.config.replay.save_exports else "",
            prompt_excerpt=_truncate(user_prompt, self.config.replay.max_prompt_chars) if self.config.replay.save_prompts else "",
            metadata={
                "selected_record_count": len(selected_items),
                "stage2_version": "v2",
                "latent_norm": float(enhanced_latent.norm().detach().cpu().item()),
                "selected_local_ids": selection_debug["selected_local_ids"],
                "selected_neighbour_ids": selection_debug["selected_neighbour_ids"],
            },
        )
        return trace, output_record, export_message

    def _update_global_state(
        self,
        node_traces: Sequence[NodeTurnTrace],
        current_exports: Dict[str, ExportedMemoryMessage],
        controller_state: ControllerState,
    ) -> None:
        if self.global_node is None:
            self._global_text_summary = controller_state.summary
            return
        updates: List[Tuple[str, torch.Tensor]] = []
        for trace in node_traces:
            export = current_exports.get(trace.node_id)
            if export is None:
                continue
            latent_seq = self._latent_sequence_from_export(export)
            if latent_seq is None:
                continue
            updates.append((trace.node_id, latent_seq.mean(dim=0)))
        self.global_node.update(updates, [trace.output for trace in node_traces])
        if not node_traces:
            self._global_text_summary = controller_state.summary
            return
        query = self._hidden_to_embed(self.global_node.get_context(1).squeeze(0))
        ranked = []
        for trace in node_traces:
            embedding = self._vector_tensor(self.embedder.embed(trace.output))
            score = float(torch.dot(query, F.normalize(embedding, dim=0)).detach().cpu().item())
            ranked.append((score, trace))
        ranked.sort(key=lambda item: (item[0], item[1].node_id), reverse=True)
        chosen = ranked[: self.v2_config.global_summary_max_nodes]
        parts = [controller_state.summary]
        for _, trace in chosen:
            parts.append(f"[{trace.node_id}] {_truncate(trace.output, 220)}")
        self._global_text_summary = _truncate("\n".join(parts), self.v2_config.memory.max_brief_chars // 2)

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
        if self.config.allow_cross_agent_raw_memory:
            raise ValueError("Stage2RuntimeV2 requires graph-mediated memory access; cross-agent raw memory is disabled.")
        self._memory_store = PrivateEpisodeMemoryStore(self.config.memory)
        self._pending_policy_log_probs = []
        self._pending_policy_entropies = []
        self._current_learn = bool(learn)
        self._global_text_summary = ""
        if self.global_node is not None:
            self.global_node.reset()
        episode_id = str((metadata or {}).get("question_id") or (metadata or {}).get("id") or abs(hash(question_text)))
        start = time.perf_counter()
        task_nodes = self._task_nodes(graph)
        controller_state = self._controller.bootstrap(
            graph,
            dataset_profile,
            role_adjustments=self._controller_role_adjustments(
                dataset_profile,
                None,
                turn_index=0,
                total_turns=self.config.graph.turn_count,
                feedback_events=[],
            ),
        )
        self._latest_controller_state = controller_state
        previous_feedback = []
        previous_exports: Dict[str, ExportedMemoryMessage] = {}
        turn_traces: List[TurnTrace] = []
        final_sink_outputs: Dict[str, str] = {}
        turn_token_estimates: List[int] = []
        turn_token_costs: List[float] = []

        for turn_index in range(self.config.graph.turn_count):
            active_edges = self._activate_edges(graph, controller_state, previous_feedback, turn_index=turn_index)
            active_task_nodes = self._active_task_nodes(
                graph,
                task_nodes,
                active_edges,
                controller_state,
                previous_feedback,
                turn_index=turn_index,
            )
            active_node_ids = {node.node_id for node in active_task_nodes}
            skipped_node_ids = [node.node_id for node in task_nodes if node.node_id not in active_node_ids]
            node_traces: List[NodeTurnTrace] = []
            current_exports: Dict[str, ExportedMemoryMessage] = {}
            sink_outputs: Dict[str, str] = {}
            turn_token_estimate = 0
            for node in active_task_nodes:
                trace, output_record, export_message = self._run_task_node(
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
                    current_turn=turn_index,
                )
                self._memory_store.add(output_record)
                turn_token_estimate += int(output_record.token_estimate)
                current_exports[node.node_id] = export_message
                node_traces.append(trace)
                if node.node_id in graph.sink_node_ids:
                    sink_outputs[node.node_id] = trace.output
            feedback_events = self._extract_feedback_events(graph, node_traces, turn_index=turn_index)
            feedback_records = self._feedback_records(graph, feedback_events, episode_id=episode_id)
            for record in feedback_records:
                self._memory_store.add(record)
                turn_token_estimate += int(record.token_estimate)
            turn_token_cost = self._token_cost_from_estimate(turn_token_estimate)
            turn_token_estimates.append(turn_token_estimate)
            turn_token_costs.append(turn_token_cost)
            self._update_global_state(node_traces, current_exports, controller_state)
            controller_state = self._controller.update(
                controller_state,
                turn_index=turn_index,
                total_turns=self.config.graph.turn_count,
                feedback_events=feedback_events,
                sink_outputs=sink_outputs,
                role_adjustments=self._controller_role_adjustments(
                    dataset_profile,
                    controller_state,
                    turn_index=turn_index,
                    total_turns=self.config.graph.turn_count,
                    feedback_events=feedback_events,
                ),
            )
            self._latest_controller_state = controller_state
            turn_traces.append(
                TurnTrace(
                    turn_index=turn_index,
                    controller_state=controller_state,
                    active_edges=active_edges,
                    node_traces=node_traces,
                    feedback_events=feedback_events,
                    sink_outputs=sink_outputs,
                    metadata={
                        "elapsed_s": time.perf_counter() - start,
                        "active_node_ids": sorted(active_node_ids),
                        "skipped_node_ids": skipped_node_ids,
                        "global_summary": self._global_text_summary,
                        "stage2_version": "v2",
                        "turn_token_estimate": turn_token_estimate,
                        "turn_token_cost": turn_token_cost,
                    },
                )
            )
            previous_feedback = feedback_events
            previous_exports = current_exports
            final_sink_outputs = sink_outputs

        final_answer, finalizer_strategy = self._finalize_answer(
            question_text=question_text,
            controller_state=controller_state,
            sink_outputs=final_sink_outputs,
            turn_traces=turn_traces,
            metadata=metadata,
            reference_answer=reference_answer,
            dataset_profile=dataset_profile,
        )
        result = Stage2RunResult(
            final_answer=final_answer,
            final_controller_state=controller_state,
            turn_traces=turn_traces,
            memory_record_counts=self._memory_store.counts(),
            signature="stage2_v2|" + "||".join(graph.source_topology_signatures),
            metadata={
                "elapsed_s": time.perf_counter() - start,
                "task_node_count": len(task_nodes),
                "source_graph_count": len(graph.source_topology_signatures),
                "turn_count": len(turn_traces),
                "finalizer_strategy": finalizer_strategy,
                "approx_token_cost": self._memory_store.total_token_estimate() * self.evaluator.config.token_cost_per_word,
                "stage2_version": "v2",
                "turn_token_estimates": list(turn_token_estimates),
                "turn_token_costs": list(turn_token_costs),
                "used_v2_features": [
                    feature
                    for feature, enabled in (
                        ("composer", self.v2_config.composer_enabled),
                        ("gnn", self.v2_config.gnn_enabled),
                        ("global_node", self.v2_config.global_node_enabled),
                        ("lmpo", self.v2_config.lmpo_enabled),
                    )
                    if enabled
                ],
                "policy_terms": len(self._pending_policy_log_probs),
            },
        )
        if replay_dir:
            self.save_replay_bundle(result, replay_dir, question_text=question_text, metadata=metadata)
        return result

    def learn_from_run(
        self,
        graph: UnionGraph,
        result: Stage2RunResult,
        *,
        dataset_profile: DatasetProfile,
        summary: Optional[EvalSummary] = None,
        reward_target: Optional[float] = None,
    ) -> Dict[str, float]:
        base_stats = super().learn_from_run(
            graph,
            result,
            dataset_profile=dataset_profile,
            summary=summary,
            reward_target=reward_target,
        )
        target = self._summary_target(summary, reward_target)
        policy_stats = self.lmpo_trainer.update_from_policy(
            self._pending_policy_log_probs,
            target,
            entropy_terms=self._pending_policy_entropies,
        )
        self._pending_policy_log_probs = []
        self._pending_policy_entropies = []
        merged = dict(base_stats)
        merged.update({f"lmpo_{key}": float(value) for key, value in policy_stats.items()})
        return merged
