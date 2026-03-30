"""Stage2 Runtime V2: 整合所有新组件的执行流程

支持模块化配置和消融实验
"""

from __future__ import annotations
import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .config_v2 import Stage2V2Config
from .composer import SimpleMemoryComposer
from .gnn import LightweightGNN
from .global_node import GlobalContextNode
from .lmpo import LMPOTrainer
from .types import MemoryRecord, ControllerState
from mas_treesearch.types import UnionGraph, UnionNode


@dataclass
class Stage2V2Result:
    """V2 执行结果"""
    final_answer: str
    latent_memories: Dict[str, torch.Tensor]
    trajectory: Dict
    used_v2_features: List[str]


class Stage2RuntimeV2:
    """二阶段 V2 运行时

    整合 Memory Composer + GNN + Global Node + LMPO
    """

    def __init__(
        self,
        config: Stage2V2Config,
        union_graph: UnionGraph,
        evaluator,
        embedder
    ):
        self.config = config
        self.union_graph = union_graph
        self.evaluator = evaluator
        self.embedder = embedder

        # 初始化组件
        self._init_components()

    def _init_components(self):
        """初始化所有组件"""
        # Memory Composer
        if self.config.composer_enabled:
            self.composer = SimpleMemoryComposer(
                self.config.get_composer_config()
            )
        else:
            self.composer = None

        # GNN
        if self.config.gnn_enabled:
            self.gnn = LightweightGNN(
                self.config.get_gnn_config()
            )
        else:
            self.gnn = None

        # Global Context Node
        if self.config.global_node_enabled:
            self.global_node = GlobalContextNode(
                self.config.get_global_node_config()
            )
        else:
            self.global_node = None

        # LMPO Trainer
        if self.config.lmpo_enabled and self.composer is not None:
            self.lmpo_trainer = LMPOTrainer(
                self.composer,
                self.config.get_lmpo_config()
            )
        else:
            self.lmpo_trainer = None

        # Private memory store (复用 V1)
        from .memory import PrivateEpisodeMemoryStore
        from .config import Stage2MemoryConfig
        self.memory_store = PrivateEpisodeMemoryStore(
            Stage2MemoryConfig()
        )

    def run(
        self,
        question_text: str,
        question_id: str,
        controller_state: ControllerState,
        learn: bool = False
    ) -> Stage2V2Result:
        """执行单个问题

        Args:
            question_text: 问题文本
            question_id: 问题 ID
            controller_state: Controller 状态
            learn: 是否更新学习模块
        """
        # 重置状态
        self.memory_store._records = {}
        if self.global_node:
            self.global_node.reset()

        # 多轮执行
        latent_memories = {}
        trajectory = {"turns": []}

        for turn in range(self.config.turn_count):
            turn_result = self._execute_turn(
                turn=turn,
                question_text=question_text,
                controller_state=controller_state,
                latent_memories=latent_memories
            )
            trajectory["turns"].append(turn_result)

        # 生成最终答案
        final_answer = self._finalize_answer(trajectory)

        # 学习更新
        if learn and self.lmpo_trainer:
            reward = self._compute_reward(final_answer, question_text)
            self.lmpo_trainer.update(
                trajectory,
                reward,
                list(latent_memories.values())
            )

        return Stage2V2Result(
            final_answer=final_answer,
            latent_memories=latent_memories,
            trajectory=trajectory,
            used_v2_features=self._get_used_features()
        )

    def _execute_turn(
        self,
        turn: int,
        question_text: str,
        controller_state: ControllerState,
        latent_memories: Dict[str, torch.Tensor]
    ) -> Dict:
        """执行单轮"""
        turn_trace = {"turn": turn, "nodes": []}

        # 获取 active nodes
        active_nodes = self._get_active_nodes(turn)

        for node in active_nodes:
            node_result = self._execute_node(
                node=node,
                turn=turn,
                question_text=question_text,
                controller_state=controller_state,
                latent_memories=latent_memories
            )
            turn_trace["nodes"].append(node_result)

        return turn_trace

    def _execute_node(
        self,
        node: UnionNode,
        turn: int,
        question_text: str,
        controller_state: ControllerState,
        latent_memories: Dict[str, torch.Tensor]
    ) -> Dict:
        """执行单个节点"""
        # 1. 记忆压缩阶段
        if self.composer:
            latent_mem = self._compress_memory(node, turn, question_text)
        else:
            # Fallback to V1: 文本摘要
            latent_mem = None

        # 2. 图传播阶段
        if self.gnn and latent_mem is not None:
            aggregated = self._aggregate_neighbors(
                node, latent_mem, latent_memories
            )
        else:
            aggregated = latent_mem

        # 3. 全局上下文整合
        if self.global_node and aggregated is not None:
            enhanced = self._integrate_global_context(aggregated)
        else:
            enhanced = aggregated

        # 存储
        if enhanced is not None:
            latent_memories[node.node_id] = enhanced

        # 4. 执行（这里简化，实际需要调用 LLM）
        output = self._execute_with_latent(
            node, enhanced, question_text, controller_state
        )

        return {
            "node_id": node.node_id,
            "output": output,
            "has_latent": enhanced is not None
        }

    def _compress_memory(
        self,
        node: UnionNode,
        turn: int,
        question_text: str
    ) -> Optional[torch.Tensor]:
        """压缩记忆"""
        # 获取原始记忆
        records = self.memory_store.get(node.node_id)
        if not records:
            return None

        # 选择记忆（简化版）
        selected = records[-self.config.max_selected_memories:]

        # Tokenize（简化：用随机 token ids）
        input_ids = torch.randint(0, 50000, (1, 100))
        attention_mask = torch.ones(1, 100, dtype=torch.bool)

        # 压缩
        with torch.no_grad():
            latent = self.composer(input_ids, attention_mask)

        return latent.squeeze(0)  # (L', D)

    def _aggregate_neighbors(
        self,
        node: UnionNode,
        self_latent: torch.Tensor,
        latent_memories: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """聚合邻居"""
        neighbor_latents = []
        edge_weights = []

        for edge in self.union_graph.task_edges:
            if edge.dst == node.node_id and edge.src in latent_memories:
                neighbor_latents.append(latent_memories[edge.src])
                edge_weights.append(0.5)  # 简化：固定权重

        if not neighbor_latents:
            return self_latent

        return self.gnn(self_latent, neighbor_latents, edge_weights)

    def _integrate_global_context(
        self,
        latent: torch.Tensor
    ) -> torch.Tensor:
        """整合全局上下文"""
        global_ctx = self.global_node.get_context(1)  # (1, D)
        # 简化：直接相加
        return latent + global_ctx.unsqueeze(0)  # (L', D)

    def _execute_with_latent(
        self,
        node: UnionNode,
        latent: Optional[torch.Tensor],
        question_text: str,
        controller_state: ControllerState
    ) -> str:
        """用 latent memory 执行节点（简化版）"""
        # 实际实现需要注入到 LLM hidden states
        # 这里返回占位输出
        return f"Output from {node.role} (with latent={latent is not None})"

    def _get_active_nodes(self, turn: int) -> List[UnionNode]:
        """获取活跃节点"""
        # 简化：返回所有 task nodes
        return [n for n in self.union_graph.nodes if n.node_type == "task"]

    def _finalize_answer(self, trajectory: Dict) -> str:
        """生成最终答案"""
        # 简化：返回最后一个输出
        if trajectory["turns"]:
            last_turn = trajectory["turns"][-1]
            if last_turn["nodes"]:
                return last_turn["nodes"][-1]["output"]
        return "No answer"

    def _compute_reward(self, answer: str, question: str) -> float:
        """计算 reward（简化版）"""
        # 实际需要调用 evaluator
        return 0.5

    def _get_used_features(self) -> List[str]:
        """获取使用的 V2 特性"""
        features = []
        if self.config.composer_enabled:
            features.append("composer")
        if self.config.gnn_enabled:
            features.append("gnn")
        if self.config.global_node_enabled:
            features.append("global_node")
        if self.config.lmpo_enabled:
            features.append("lmpo")
        return features
