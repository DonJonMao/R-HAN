# MAS Stage2 详细实现方案 V2

## 0. 设计目标

在一阶段产出的 Union Graph 基础上，通过多轮协作执行优化任务表现。核心思想：

1. **固定结构，优化执行**：不再搜索图结构，而是学习如何在固定图上更好地协作
2. **Latent Memory**：用可学习的压缩器将原始记忆转为向量表示
3. **动态图传播**：用轻量 GNN 在图上聚合邻居信息
4. **端到端优化**：用 RL 训练 Memory Composer，LLM 保持冻结

---

## 1. 三层图架构

### 1.1 上层：Global Context Node

**定义**：
- 单个可学习的全局节点 `G ∈ R^D`
- 所有任务智能体都连接到它（全连接）
- 存储任务级的全局偏好和执行模式

**功能**：
- **读取**：每个智能体执行前读取全局上下文
- **更新**：每轮结束后，用 attention pooling 聚合所有智能体的更新

**更新公式**：
```
G_{t+1} = G_t + Σ_i α_i · Δ_i
其中 α_i = softmax(score(G_t, output_i))
```

### 1.2 中层：Union Graph

**定义**：
- 一阶段产出的联合图，包含：
  - Task nodes（任务智能体）
  - Task edges（智能体间的通信边）
  - 每条边的结构先验（support_ratio, avg_parent_score 等）

**功能**：
- 定义智能体间的通信拓扑
- 提供结构先验用于边权重计算

**动态边激活**：
- 每轮根据当前状态计算边权重
- 保留高权重边，剪枝低权重边

### 1.3 下层：Private Episodic Memory

**定义**：
- 每个智能体维护自己的私有记忆池
- 记忆类型：
  - `self_output`：自己的历史输出
  - `feedback`：收到的反馈（pass/challenge/reject 等）

**功能**：
- 存储智能体的局部经验
- 通过 Memory Composer 压缩后参与图传播

---

## 2. 核心组件设计

### 2.1 Memory Composer（可学习）

**输入**：
- `role_profile`：智能体角色描述（如 "math solver", "code verifier"）
- `raw_memories`：原始记忆文本列表（最多 K 条）

**输出**：
- `latent_memory ∈ R^(L'×D)`：固定长度的向量序列
  - L'：latent memory 序列长度（如 8）
  - D：LLM 隐藏维度（如 4096）

**架构**：
```
Encoder: role_profile + raw_memories → hidden states
Compressor: hidden states → latent_memory (fixed L')
```

**训练方式**：
- 用 RL（LMPO）训练
- Reward 来自最终任务表现
- LLM 保持冻结

**冻结策略**：
- 训练完成后冻结 Composer 参数
- 代表"这类任务如何理解和压缩记忆"
- 可迁移到新任务

### 2.2 Memory Selector（轻量学习）

**功能**：从私有记忆池中选择 K 条最相关的记忆

**方法**：Similarity-based retrieval
```python
query_vec = embed(question + role + controller_focus)
scores = [cosine(query_vec, embed(memory)) for memory in pool]
selected = top_K(scores)
```

**可选增强**：
- 加入 recency bias
- 加入 feedback type bias（优先选 challenge/pass）
- 用轻量 MLP 修正分数

### 2.3 Lightweight GNN（动态运行）

**功能**：聚合邻居的 latent memory

**输入**：
- `self_latent ∈ R^(L'×D)`：自己的 latent memory
- `neighbor_latents`：邻居的 latent memory 列表
- `edge_weights`：当前边权重

**输出**：
- `aggregated_latent ∈ R^(L'×D)`：聚合后的表示

**架构**（1-2 层 GNN）：
```
# Layer 1: 邻居聚合
neighbor_msg = Σ_j w_j · MLP(neighbor_latents[j])

# Layer 2: 自身更新
aggregated = self_latent + α · neighbor_msg
```

**边权重计算**：
```
w_j = softmax(
    structure_prior(edge_j) +          # 一阶段结构先验
    feedback_score(edge_j) +           # 上轮反馈
    controller_role_boost(edge_j) +    # 全局控制器加权
    learned_adjustment(edge_j)         # 轻量学习修正
)
```

**不冻结原因**：
- 需要适应不同图结构
- 需要根据当前状态动态调整
- 可以用简单方法（甚至启发式）

### 2.4 Global Context Integrator

**功能**：整合全局上下文到智能体

**方法**：
```python
# 读取全局上下文
global_vec = G_t  # 全局节点向量

# 拼接到 latent memory
enhanced_latent = concat([
    aggregated_latent,  # 来自 GNN
    global_vec.expand(L', D)  # 广播到序列长度
])
```

---

## 3. 单轮执行流程

### 3.1 准备阶段

```python
# 1. 计算边权重
edge_weights = compute_edge_weights(
    union_graph=union_graph,
    feedback_history=feedback_t,
    controller_state=controller_t
)

# 2. 边激活（剪枝）
active_edges = prune_edges(
    edge_weights=edge_weights,
    threshold=0.38,  # soft prune
    top_k=3,         # 每个目标节点最多保留 3 条入边
    hard_prune=(turn > 4)  # 后期硬剪枝
)

# 3. 节点激活
active_nodes = select_active_nodes(
    union_graph=union_graph,
    active_edges=active_edges,
    feedback_history=feedback_t,
    controller_state=controller_t
)
```

### 3.2 记忆压缩阶段

```python
for node in active_nodes:
    # 1. 选择记忆
    raw_memories = memory_selector.select(
        pool=private_memory[node],
        query=question + node.role + controller.focus,
        K=4
    )

    # 2. 压缩成 latent memory
    latent_memory[node] = memory_composer(
        role_profile=node.role,
        raw_memories=raw_memories
    )  # → R^(L'×D)
```

### 3.3 图传播阶段

```python
for node in active_nodes:
    # 1. 收集邻居 latent memory
    neighbor_latents = [
        latent_memory[src]
        for src in node.incoming_neighbors
        if (src, node) in active_edges
    ]

    # 2. GNN 聚合
    aggregated_latent[node] = gnn(
        self_latent=latent_memory[node],
        neighbor_latents=neighbor_latents,
        edge_weights=[edge_weights[(src, node)]
                      for src in node.incoming_neighbors]
    )

    # 3. 整合全局上下文
    enhanced_latent[node] = integrate_global_context(
        aggregated_latent=aggregated_latent[node],
        global_context=G_t
    )
```

### 3.4 智能体执行阶段

```python
for node in active_nodes:
    # 1. 构造 prompt
    prompt_text = build_prompt(
        question=question,
        role=node.role,
        task_context=task_context,
        controller_summary=controller.summary
    )

    # 2. 编码 prompt
    prompt_hidden = llm.encode(prompt_text)  # → R^(L×D)

    # 3. 拼接 latent memory
    input_hidden = concat([
        enhanced_latent[node],  # L'×D
        prompt_hidden           # L×D
    ])  # → R^((L'+L)×D)

    # 4. LLM 生成（frozen）
    output = llm.generate(input_hidden)

    # 5. 存储输出
    private_memory[node].add(
        record_type="self_output",
        text=output,
        turn=t
    )
```

### 3.5 反馈生成阶段

```python
# 只有 critic/verifier/judge 生成反馈
for node in [critic, verifier, judge]:
    if node not in active_nodes:
        continue

    # 1. 解析反馈类型
    feedback_type = parse_feedback(node.output)
    # → pass/challenge/reject/uncertain

    # 2. 沿入边回传反馈
    for src in node.incoming_neighbors:
        if (src, node) in active_edges:
            private_memory[src].add(
                record_type="feedback",
                feedback_type=feedback_type,
                text=node.output,
                source=node.node_id,
                turn=t
            )
```

### 3.6 全局节点更新阶段

```python
# 1. 收集所有智能体的更新
updates = []
for node in active_nodes:
    # 提取节点的"状态变化"
    delta = extract_update(
        node_output=node.output,
        latent_memory=enhanced_latent[node]
    )
    updates.append((node, delta))

# 2. Attention pooling 聚合
attention_scores = [
    score_function(G_t, delta)
    for node, delta in updates
]
alpha = softmax(attention_scores)

# 3. 更新全局节点
G_{t+1} = G_t + Σ_i alpha[i] · updates[i].delta
```

### 3.7 Controller 更新阶段

```python
# 1. 统计反馈
feedback_stats = count_feedback_types(feedback_t)

# 2. 更新模式
if remaining_turns <= 1:
    controller.mode = "finalize"
elif feedback_stats.challenge > feedback_stats.support:
    controller.mode = "tighten"
else:
    controller.mode = "refine"

# 3. 更新角色权重
controller.role_weights = update_role_weights(
    current_weights=controller.role_weights,
    feedback_stats=feedback_stats,
    mode=controller.mode
)

# 4. 更新 focus
controller.focus = generate_focus(
    mode=controller.mode,
    feedback_stats=feedback_stats
)
```

---

## 4. 多轮执行与终止

### 4.1 多轮循环

```python
for turn in range(5):  # 默认 5 轮
    # 执行单轮流程（见第 3 节）
    execute_turn(turn)

    # 记录轨迹
    trajectory.append({
        "turn": turn,
        "active_nodes": active_nodes,
        "active_edges": active_edges,
        "feedback": feedback_t,
        "controller_state": controller_t
    })
```

### 4.2 最终答案生成

```python
# 1. 收集所有 sink 节点的输出
sink_outputs = [
    node.output
    for node in union_graph.sink_nodes
]

# 2. 任务类型特定的 finalizer
if task_type == "code_generation":
    # 重新评估所有代码候选
    candidates = extract_code_candidates(trajectory)
    final_answer = select_best_code(
        candidates=candidates,
        evaluator=evaluator
    )

elif task_type in ["numeric", "math_expression"]:
    # 共识优先
    final_answer = consensus_finalizer(sink_outputs)

elif task_type in ["mcq", "boolean"]:
    # 多数投票
    final_answer = majority_vote(sink_outputs)

else:
    # 通用 LLM finalizer
    final_answer = llm_finalizer(
        question=question,
        sink_outputs=sink_outputs,
        controller_summary=controller.summary,
        global_context=G_final
    )
```

### 4.3 与 Stage1 Baseline 比较

```python
# 1. 评估 stage2 结果
stage2_summary = evaluator.evaluate(
    question=question,
    answer=final_answer,
    tier="tier2"
)

# 2. 计算风险调整分
stage2_reward = risk_adjusted_score(
    mean_reward=stage2_summary.mean_reward,
    reward_std=stage2_summary.reward_std,
    penalty=0.5
)

stage1_reward = risk_adjusted_score(
    mean_reward=stage1_summary.mean_reward,
    reward_std=stage1_summary.reward_std,
    penalty=0.5
)

# 3. 回退决策
if stage2_reward >= stage1_reward:
    return final_answer, "use_stage2"
else:
    return stage1_output, "fallback_stage1"
```

---

## 5. 训练策略

### 5.1 训练目标

**只训练 Memory Composer**，其他组件保持简单或冻结：
- LLM：冻结
- GNN：轻量（1-2 层），可以不训练或用简单规则
- Memory Selector：启发式或轻量 MLP
- Global Context Node：可学习，但用简单更新规则

### 5.2 LMPO 训练算法

**核心思想**：用 RL 优化 Memory Composer

```python
# 伪代码
for episode in training_data:
    # 1. Rollout：用当前 composer 执行完整流程
    trajectory, final_answer = execute_stage2(
        question=episode.question,
        composer=composer  # 当前参数
    )

    # 2. 计算 reward
    reward = evaluate(final_answer, episode.ground_truth)

    # 3. 计算 baseline（用于减少方差）
    baseline = moving_average_reward

    # 4. 计算 advantage
    advantage = reward - baseline

    # 5. Policy gradient 更新
    loss = -advantage * log_prob(trajectory | composer)
    composer.update(loss)
```

### 5.3 分阶段训练

**Phase 1: Warm-up（10% 数据）**
- 用简单任务训练 Composer
- 学习基本的记忆压缩能力

**Phase 2: Main Training（80% 数据）**
- 用完整任务训练
- 优化任务表现

**Phase 3: Fine-tuning（10% 数据）**
- 在困难样本上微调
- 提升鲁棒性

### 5.4 训练技巧

1. **Relative Reward**：
   - 在同一个 batch 内比较不同 latent memory 的效果
   - 减少绝对 reward 的方差

2. **Token-level Objectives**：
   - 不只看最终答案，也看中间步骤
   - 例如：verifier 的反馈是否准确

3. **Curriculum Learning**：
   - 从简单任务开始
   - 逐步增加难度

---

## 6. 关键设计决策

### 6.1 为什么冻结 Composer 而不是 GNN？

| 维度 | Memory Composer | GNN |
|------|----------------|-----|
| **任务相关性** | 任务无关（通用压缩能力） | 任务相关（图结构敏感） |
| **输入输出** | 固定（role + memories → vector） | 可变（图结构变化） |
| **泛化性** | 可迁移到新任务 | 难以泛化到新图 |
| **训练成本** | 高（需要 RL） | 低（可以用启发式） |

**结论**：Composer 值得训练和冻结，GNN 保持简单和动态

### 6.2 为什么用 Latent Memory 而不是文本？

| 维度 | Latent Memory | 文本摘要 |
|------|--------------|---------|
| **信息密度** | 高（L'×D 维向量） | 低（受 token 限制） |
| **语义保留** | 可学习优化 | 依赖手工规则 |
| **LLM 集成** | 直接拼接 hidden states | 需要重新编码 |
| **可训练性** | 端到端优化 | 难以优化 |

**结论**：Latent Memory 更适合学习和优化

### 6.3 为什么需要全局节点？

**作用**：
1. **跨智能体信息共享**：不通过图边也能传递全局信息
2. **任务级偏好存储**：学习"这类任务通常怎么做"
3. **协调机制**：避免智能体各自为政

**替代方案**：
- 不用全局节点，只用 Controller（当前实现）
- 问题：Controller 是规则驱动，不可学习

---

## 7. 实现优先级

### P0（核心功能）
1. Memory Composer 实现
2. Latent Memory 注入到 LLM
3. 轻量 GNN 邻居聚合
4. LMPO 训练框架

### P1（增强功能）
1. Global Context Node
2. 动态边权重学习
3. Token-level objectives

### P2（优化功能）
1. Curriculum learning
2. Multi-task training
3. 更复杂的 GNN 架构

---

## 8. 预期效果

### 8.1 相比当前实现的提升

| 维度 | 当前实现 | V2 设计 | 提升 |
|------|---------|---------|------|
| **记忆利用** | 文本拼接 | Latent vector | ✓✓✓ |
| **学习能力** | 3 个线性模型 | RL 训练 Composer | ✓✓✓ |
| **信息密度** | 受 token 限制 | 固定维度向量 | ✓✓ |
| **可解释性** | 高（文本可读） | 中（需要分析向量） | ✗ |
| **工程复杂度** | 低 | 高 | ✗✗ |

### 8.2 成功指标

1. **Stage2 采用率 > 50%**：
   - 当前：0%（全部 fallback）
   - 目标：>50% 样本 stage2 优于 stage1

2. **平均 reward 提升 > 5%**：
   - 相比 stage1 baseline

3. **Token 成本下降 > 20%**：
   - 通过更好的边剪枝和记忆压缩

---

## 9. 风险与缓解

### 风险 1：训练不稳定

**原因**：RL 训练方差大

**缓解**：
- 用 relative reward
- 增大 batch size
- 用 baseline 减少方差

### 风险 2：Latent Memory 信息损失

**原因**：压缩必然损失细节

**缓解**：
- 增大 L'（序列长度）
- Role-aware compression
- 保留原始记忆作为 fallback

### 风险 3：工程复杂度高

**原因**：需要实现 RL 训练、GNN、向量注入

**缓解**：
- 分阶段实现（P0 → P1 → P2）
- 先用简单版本验证可行性
- 保留当前实现作为 baseline

---

## 10. 下一步行动

### 10.1 验证阶段（1-2 周）

1. 实现 Memory Composer（不训练，用随机初始化）
2. 实现 Latent Memory 注入
3. 在小数据集上测试可行性

### 10.2 开发阶段（2-3 周）

1. 实现 LMPO 训练框架
2. 实现轻量 GNN
3. 在完整数据集上训练

### 10.3 优化阶段（1-2 周）

1. 调优超参数
2. 增加 Global Context Node
3. 对比实验和消融研究

---

## 参考文献

1. **Latent Memory 论文**：`latentmem.pdf`
   - Memory Composer 设计
   - LMPO 训练算法
   - Hidden state injection

2. **当前实现**：`mas_stage2/`
   - 三层架构
   - 反馈闭环
   - Stage1 baseline 比较

3. **一阶段设计**：`README.md`
   - Union Graph 生成
   - 结构先验
   - 评估体系
