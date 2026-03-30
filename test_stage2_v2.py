"""测试 Stage2 V2 组件"""

import torch
from mas_stage2.config_v2 import FULL_V2_CONFIG, ABLATION_NO_GNN
from mas_stage2.composer import SimpleMemoryComposer
from mas_stage2.gnn import LightweightGNN
from mas_stage2.global_node import GlobalContextNode


def test_composer():
    """测试 Memory Composer"""
    print("Testing Memory Composer...")

    config = FULL_V2_CONFIG.get_composer_config()
    composer = SimpleMemoryComposer(config, vocab_size=50000)

    # 模拟输入
    input_ids = torch.randint(0, 50000, (2, 100))
    attention_mask = torch.ones(2, 100, dtype=torch.bool)

    # Forward
    latent = composer(input_ids, attention_mask)

    print(f"  Input shape: {input_ids.shape}")
    print(f"  Output shape: {latent.shape}")
    print(f"  Expected: (2, {config.latent_length}, {config.hidden_dim})")
    assert latent.shape == (2, config.latent_length, config.hidden_dim)
    print("  ✓ Composer test passed\n")


def test_gnn():
    """测试 GNN"""
    print("Testing GNN...")

    config = FULL_V2_CONFIG.get_gnn_config()
    gnn = LightweightGNN(config)

    # 模拟输入
    L, D = 8, 4096
    self_latent = torch.randn(L, D)
    neighbor_latents = [torch.randn(L, D) for _ in range(3)]
    edge_weights = [0.5, 0.3, 0.2]

    # Forward
    aggregated = gnn(self_latent, neighbor_latents, edge_weights)

    print(f"  Self shape: {self_latent.shape}")
    print(f"  Neighbors: {len(neighbor_latents)}")
    print(f"  Output shape: {aggregated.shape}")
    assert aggregated.shape == (L, D)
    print("  ✓ GNN test passed\n")


def test_global_node():
    """测试 Global Context Node"""
    print("Testing Global Context Node...")

    config = FULL_V2_CONFIG.get_global_node_config()
    global_node = GlobalContextNode(config)

    # 获取上下文
    ctx = global_node.get_context(batch_size=2)
    print(f"  Context shape: {ctx.shape}")
    assert ctx.shape == (2, config.hidden_dim)

    # 更新
    updates = [
        ("node1", torch.randn(config.hidden_dim)),
        ("node2", torch.randn(config.hidden_dim))
    ]
    new_state = global_node.update(updates, ["out1", "out2"])
    print(f"  Updated state shape: {new_state.shape}")
    assert new_state.shape == (1, config.hidden_dim)
    print("  ✓ Global Node test passed\n")


def test_ablation():
    """测试消融配置"""
    print("Testing Ablation Configs...")

    # 测试禁用 GNN
    config = ABLATION_NO_GNN
    print(f"  GNN enabled: {config.gnn_enabled}")
    print(f"  Composer enabled: {config.composer_enabled}")
    assert not config.gnn_enabled
    assert config.composer_enabled
    print("  ✓ Ablation test passed\n")


if __name__ == "__main__":
    print("=" * 50)
    print("Stage2 V2 Component Tests")
    print("=" * 50 + "\n")

    test_composer()
    test_gnn()
    test_global_node()
    test_ablation()

    print("=" * 50)
    print("All tests passed!")
    print("=" * 50)
