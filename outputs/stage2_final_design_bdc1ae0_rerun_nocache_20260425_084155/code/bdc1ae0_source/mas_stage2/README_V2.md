# Stage2 V2 实现说明

## 概述

基于 DESIGN_V2.md 的完整实现，引入真正的 latent memory 和可学习组件。

## 核心组件

### 1. Memory Composer (`composer.py`)
- 可学习的记忆压缩器
- 输入：role_profile + raw_memories (文本)
- 输出：latent_memory ∈ R^(L'×D)
- 训练后可冻结

### 2. Lightweight GNN (`gnn.py`)
- 1-2 层 GNN 用于邻居聚合
- 动态运行，不冻结
- 适应不同图结构

### 3. Global Context Node (`global_node.py`)
- 可学习的全局节点
- 所有智能体连接
- Attention pooling 更新

### 4. LMPO Trainer (`lmpo.py`)
- RL 训练 Memory Composer
- LLM 保持冻结
- Policy gradient 更新

### 5. Runtime V2 (`runtime_v2.py`)
- 整合所有组件
- 支持模块化配置
- 消融实验友好

## 配置系统

### 预定义配置 (`config_v2.py`)

```python
# 完整 V2
from mas_stage2.config_v2 import FULL_V2_CONFIG

# 消融实验
from mas_stage2.config_v2 import (
    ABLATION_NO_GNN,      # 禁用 GNN
    ABLATION_NO_GLOBAL,   # 禁用全局节点
    ABLATION_NO_LEARNING, # 禁用学习
    V1_COMPATIBLE         # 回退到 V1
)
```

### 自定义配置

```python
from mas_stage2.config_v2 import Stage2V2Config

config = Stage2V2Config(
    composer_enabled=True,
    composer_latent_length=16,  # 自定义长度
    gnn_enabled=True,
    gnn_num_layers=3,           # 自定义层数
    global_node_enabled=True,
    lmpo_enabled=True
)
```

## 使用方法

### 测试组件

```bash
python test_stage2_v2.py
```

### 训练

```python
from mas_stage2.runtime_v2 import Stage2RuntimeV2
from mas_stage2.config_v2 import FULL_V2_CONFIG

runtime = Stage2RuntimeV2(
    config=FULL_V2_CONFIG,
    union_graph=union_graph,
    evaluator=evaluator,
    embedder=embedder
)

result = runtime.run(
    question_text="...",
    question_id="...",
    controller_state=controller_state,
    learn=True  # 启用学习
)
```

## 消融实验

### 实验 1: 无 GNN
```python
from mas_stage2.config_v2 import ABLATION_NO_GNN
runtime = Stage2RuntimeV2(config=ABLATION_NO_GNN, ...)
```

### 实验 2: 无全局节点
```python
from mas_stage2.config_v2 import ABLATION_NO_GLOBAL
runtime = Stage2RuntimeV2(config=ABLATION_NO_GLOBAL, ...)
```

### 实验 3: 无学习（只推理）
```python
from mas_stage2.config_v2 import ABLATION_NO_LEARNING
runtime = Stage2RuntimeV2(config=ABLATION_NO_LEARNING, ...)
```

## 与 V1 的区别

| 特性 | V1 | V2 |
|------|----|----|
| 记忆表示 | 文本拼接 | Latent vector |
| 邻居聚合 | 文本拼接 | GNN |
| 全局协调 | 规则 Controller | 可学习 Global Node |
| 学习能力 | 3 个线性模型 | LMPO + Composer |
| 可配置性 | 低 | 高（模块化） |

## 下一步

1. 完善 LLM 集成（latent memory 注入）
2. 实现完整的 LMPO 训练循环
3. 在小数据集上验证
4. 完整数据集训练
5. 对比实验和消融研究

## 文件结构

```
mas_stage2/
├── composer.py       # Memory Composer
├── gnn.py           # Lightweight GNN
├── global_node.py   # Global Context Node
├── lmpo.py          # LMPO Trainer
├── runtime_v2.py    # Runtime V2
├── config_v2.py     # 统一配置
├── DESIGN_V2.md     # 详细设计文档
└── (V1 files...)    # V1 实现保留
```
