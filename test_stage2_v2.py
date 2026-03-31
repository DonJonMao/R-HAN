"""测试 Stage2 V2 组件和最小闭环。"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mas_stage2.config_v2 import ABLATION_NO_GNN, FULL_V2_CONFIG
from mas_stage2.pipeline import _selection_metadata
from mas_stage2.runtime_v2 import Stage2RuntimeV2
from mas_stage2.composer import SimpleMemoryComposer
from mas_stage2.gnn import LightweightGNN
from mas_stage2.global_node import GlobalContextNode
from mas_treesearch.agents import AgentPool, AgentSpec
from mas_treesearch.profiles import DEFAULT_PROFILE
from mas_treesearch.types import EvalSummary, PromptSlots, UnionEdge, UnionGraph, UnionNode


def test_composer():
    """测试 Memory Composer。"""
    print("Testing Memory Composer...")
    config = FULL_V2_CONFIG.get_composer_config(embed_dim=256)
    composer = SimpleMemoryComposer(config, vocab_size=50000)
    input_ids = torch.randint(0, 50000, (2, 100))
    attention_mask = torch.ones(2, 100, dtype=torch.bool)
    latent = composer(input_ids, attention_mask)
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Output shape: {latent.shape}")
    assert latent.shape == (2, config.latent_length, config.hidden_dim)
    print("  ✓ Composer test passed\n")


def test_gnn():
    """测试 GNN。"""
    print("Testing GNN...")
    config = FULL_V2_CONFIG.get_gnn_config(embed_dim=256)
    gnn = LightweightGNN(config)
    l_val, d_val = config.hidden_dim // 32, config.hidden_dim
    self_latent = torch.randn(l_val, d_val)
    neighbor_latents = [torch.randn(l_val, d_val) for _ in range(3)]
    edge_weights = [0.5, 0.3, 0.2]
    aggregated = gnn(self_latent, neighbor_latents, edge_weights)
    print(f"  Output shape: {aggregated.shape}")
    assert aggregated.shape == (l_val, d_val)
    print("  ✓ GNN test passed\n")


def test_global_node():
    """测试 Global Context Node。"""
    print("Testing Global Context Node...")
    config = FULL_V2_CONFIG.get_global_node_config(embed_dim=256)
    global_node = GlobalContextNode(config)
    ctx = global_node.get_context(batch_size=2)
    assert ctx.shape == (2, config.hidden_dim)
    updates = [
        ("node1", torch.randn(config.hidden_dim)),
        ("node2", torch.randn(config.hidden_dim)),
    ]
    new_state = global_node.update(updates, ["out1", "out2"])
    assert new_state.shape == (1, config.hidden_dim)
    print("  ✓ Global Node test passed\n")


def test_ablation():
    """测试消融配置。"""
    print("Testing Ablation Configs...")
    config = ABLATION_NO_GNN
    assert not config.gnn_enabled
    assert config.composer_enabled
    print("  ✓ Ablation test passed\n")


@dataclass
class _FakeEmbeddingConfig:
    dim: int = 256


class _FakeEmbedder:
    def __init__(self, dim: int = 256):
        self.config = _FakeEmbeddingConfig(dim=dim)

    def embed(self, text: str):
        values = [0.0] * self.config.dim
        for index, byte in enumerate(text.encode("utf-8")):
            values[index % self.config.dim] += float((byte % 31) + 1)
        norm = sum(value * value for value in values) ** 0.5 or 1.0
        return [value / norm for value in values]


@dataclass
class _FakeEvalConfig:
    token_cost_per_word: float = 0.001


class _FakeEvaluator:
    def __init__(self):
        self.config = _FakeEvalConfig()

    def _task_context(self, *args, **kwargs):
        return ""

    def _resolve_runtime(self, *args, **kwargs):
        return None

    def _cached_chat(self, messages, runtime=None):
        system = messages[0]["content"].lower()
        user = messages[-1]["content"].lower()
        if "final answer synthesizer" in system:
            return "42"
        if "verifier" in system or "judge" in system:
            return "VERDICT: pass\nISSUES: none\nFIX: none"
        if "critic" in system:
            return "VERDICT: challenge\nISSUES: missing check\nFIX: verify arithmetic"
        if "output contract" in user:
            return "42"
        return "42"

    def _output_contract(self, *args, **kwargs):
        return ""

    def _sanitize_final_output(self, question_text, raw_output, **kwargs):
        del question_text, kwargs
        return raw_output.strip()

    def _strip_hidden_reasoning(self, raw_output):
        return raw_output.strip()


def _build_smoke_graph() -> tuple[UnionGraph, AgentPool]:
    pool = AgentPool(
        agents=[
            AgentSpec(
                agent_id="solver_agent",
                role="solver",
                profile="solve the problem",
                system_prompt="You solve the problem.",
            ),
            AgentSpec(
                agent_id="verifier_agent",
                role="verifier",
                profile="verify the candidate",
                system_prompt="You verify the answer.",
            ),
        ]
    )
    solver_node = UnionNode(
        node_id="solver_node",
        agent_id="solver_agent",
        role="solver",
        node_type="task",
        source_graph_ids=["g1"],
        support_count=1,
        avg_graph_score=0.8,
        topo_level_mean=0.0,
        metadata={"prompt_slots": {"reasoning_mode": PromptSlots().reasoning_mode}},
    )
    verifier_node = UnionNode(
        node_id="verifier_node",
        agent_id="verifier_agent",
        role="verifier",
        node_type="task",
        source_graph_ids=["g1"],
        support_count=1,
        avg_graph_score=0.8,
        topo_level_mean=1.0,
        metadata={"prompt_slots": {"reasoning_mode": PromptSlots().reasoning_mode}},
    )
    edge = UnionEdge(
        src="solver_node",
        dst="verifier_node",
        edge_type="task",
        source_graph_ids=["g1"],
        support_count=1,
        support_ratio=1.0,
        avg_parent_score=0.8,
        best_parent_score=0.8,
        initial_keep_logit=0.9,
        dynamic_keep_weight=0.0,
    )
    graph = UnionGraph(
        nodes={"solver_node": solver_node, "verifier_node": verifier_node},
        edges=[edge],
        source_topology_signatures=["g1"],
        root_node_ids=["solver_node"],
        sink_node_ids=["solver_node"],
    )
    return graph, pool


def test_runtime_smoke():
    """测试 V2 runtime 的最小闭环。"""
    print("Testing Runtime V2 smoke path...")
    graph, pool = _build_smoke_graph()
    config = FULL_V2_CONFIG
    config.graph.turn_count = 2
    config.memory.max_selected_records = 2
    runtime = Stage2RuntimeV2(
        config=config,
        evaluator=_FakeEvaluator(),
        agent_pool=pool,
        embedder=_FakeEmbedder(dim=256),
    )
    result = runtime.run(
        graph,
        question_text="What is 40 + 2?",
        dataset_profile=DEFAULT_PROFILE,
        metadata={"id": "smoke-1"},
        learn=True,
    )
    assert result.final_answer.strip()
    assert result.metadata.get("stage2_version") == "v2"
    assert len(result.turn_traces) == 2
    assert sum(result.memory_record_counts.values()) > 0
    print("  ✓ Runtime V2 smoke test passed\n")


def _fake_summary(*, task: float, success: float, latency: float, token_cost: float, safety: float = 0.0) -> EvalSummary:
    reward = 1.10 * task + 0.55 * success - 0.02 * latency - 0.015 * token_cost - 0.25 * safety
    return EvalSummary(
        tier="tier2",
        mean_reward=reward,
        reward_std=0.0,
        mean_task_score=task,
        mean_success=success,
        mean_latency=latency,
        mean_token_cost=token_cost,
        mean_safety_penalty=safety,
        evaluations=[],
    )


def test_selection_prefers_stage2_on_quality_tie():
    """质量完全相同时，不能仅因 stage2 更慢就回退到 stage1。"""
    print("Testing Stage2 selection policy...")
    stage1 = _fake_summary(task=1.0, success=1.0, latency=0.02, token_cost=0.0001)
    stage2 = _fake_summary(task=1.0, success=1.0, latency=1.20, token_cost=0.0020)
    meta = _selection_metadata(stage1_summary=stage1, stage2_summary=stage2, risk_std_penalty=0.0)
    assert meta["selection_decision"] == "use_stage2"
    assert meta["selection_reason"] == "quality_tie_prefer_stage2"
    assert meta["stage2_reward"] < meta["stage1_reward"]
    print("  ✓ Selection policy test passed\n")


if __name__ == "__main__":
    print("=" * 50)
    print("Stage2 V2 Tests")
    print("=" * 50 + "\n")
    test_composer()
    test_gnn()
    test_global_node()
    test_ablation()
    test_runtime_smoke()
    test_selection_prefers_stage2_on_quality_tie()
    print("=" * 50)
    print("All tests passed!")
    print("=" * 50)
