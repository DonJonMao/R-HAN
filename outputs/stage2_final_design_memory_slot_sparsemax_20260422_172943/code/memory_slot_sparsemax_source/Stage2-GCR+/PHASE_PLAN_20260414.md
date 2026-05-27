# Stage2-GCR+ Code 路线分阶段计划

日期：2026-04-14

## 1. 文档目的

这份文档把当前 Stage2-GCR+ 的 Code 主线拆成三个阶段：

1. 先验证当前 graph-native code shell 是否成立。
2. 再把当前过于保守的 provenance-bound recovery 修成长期合法语义。
3. 最后再做更强但变量更多的执行升级和 selector learning。

核心原则是先证壳，再修入口语义，再上强机制。不要在主壳尚未站稳时同时改 recovery、rerun sparsity、selector learning。

## 2. 为什么当前版本叫“第一版”

当前版本之所以定义为 `gcr第一版`，不是因为它已经完整，而是因为它第一次把 Code 线的主壳闭合成了一个可训练、可分析、可继续叠代的最小完整系统。这个“第一版”已经明确具备以下骨架：

- 冻结 `UnionGraph`
- typed memory
- `GlobalNode`
- edge-only sparse gating
- per-turn pruning
- raw candidate bank
- class collapse
- typed verifier
- provenance-bound recovery
- reinsert
- hard final guard

但它仍然刻意保持了几个“简单可学习部分”：

- 稀疏 rerun 仍是固定 `top-k`，不是 learned support set，也不是 sparsemax support-set policy。
- recovery 仍是 prompt-level patch，不做 LoRA、adapter 或 base model finetune。
- execution depth 仍主要由 `Lean / Full` 和固定 rounds 控制，不是统一 recovery loop。
- slot 使用仍是固定模板，不做 slot mask / slot depth learning。
- selector learning 仍然是弱学习和局部统计，不负责 learned routing、global override 或 branch policy。
- anchor recovery 入口当前先以强 invariant 保守收口，而不是完整的“anchor 合法绑定”。

所以它是一个“主壳成立的第一版”，不是“机制已经打满的最终版”。

## 3. Phase 1：跑完当前版，验证主壳是否成立

### 3.1 目标

这一阶段不再改方法，只回答一个问题：

当前 `Stage2-GCR+` 的 graph-native code 主壳是否成立，值不值得继续往上加更强机制。

### 3.2 本阶段不改

- 不改 recovery operator
- 不改 Lean / Full
- 不改 fixed top-k
- 不改 slot 模板
- 不开新的 learned routing

### 3.3 只跑什么

只跑 Code 路线：

- MBPP
- HumanEval（MBPP 站稳后再上）

Reasoning / Graph 当前仍以 stabilization-first 和可运行 ablation 为主，不作为 gain route。

### 3.4 必须记录并分析

当前版必须强制记录并分析下列指标：

- `better / worse / same`
- `stage2_success_avg vs stage1_success_avg`
- `stage2_task_avg vs stage1_task_avg`
- `selection_reason` 分布
- `active_edge_ratio_by_turn`
- `active_node_ratio_by_turn`
- `candidate_provenance_coverage`
- `recovery_subgraph_size`
- `final_answer_source_type`
- `anchor_recoverable_but_blocked_no_provenance_count`

### 3.5 成功标准

- 多个 run 方向一致
- `better > worse`
- graph-faithfulness 指标有信号
- recovery invariant 不再爆炸

如果这一阶段站不住，后续 Phase 2 / 3 都没有继续叠的价值。

## 4. Phase 2：把保守修法升级成长期合法修法

### 4.1 要解决的问题

当前保守修法只允许带有 graph-backed seed provenance 的候选进入 code recovery target selection。它能阻止：

- 没 provenance 的 `stage1-anchor-only` 候选进入 recovery
- 最终在 reinsert / final selection 阶段触发 invariant 爆炸

但它也过于保守，因为存在一类 anchor：

- 本身是可修的
- 语义上确实来自 Stage1 图
- 只是没有被绑定成合法的 graph-backed seed provenance

这类 anchor 不应该被一刀切挡掉，而应该被“合法化绑定”后再进入 recovery。

### 4.2 长期合法修法：best-original-graph prior subgraph

当 `best_of(c*, anchor)` 选到了 anchor，且 anchor 缺少足够 provenance 时：

1. 从 Stage1 合图前的原始图集合里选择得分最高的图：

   `i* = argmax_i s_i`

2. 在 `UnionGraph` 中提取该原始图对应的节点/边，得到：

   `G_prior = G_union[V_i*]`

3. 用它构造 anchor 的 recovery subgraph：

   `G_rec-anchor = Induce(V(G_prior) ∪ checker_neighbors ∪ sink_guards ∪ recovery_ids)`

4. 如果与 sink 断开，则补一条最短连接路径。

5. 为 anchor 写回 graph-backed seed provenance 后，再允许其进入 recovery。

### 4.3 Phase 2 的边界

这一阶段只修 recovery 入口语义，不同时引入新的 recovery 深度和 rerun 稀疏策略。

仍然保留：

- prompt-level recovery
- Lean / Full
- fixed top-k
- 固定 slot 模板

### 4.4 成功标准

- `anchor_recoverable_but_blocked_no_provenance_count` 显著下降
- recovery 不再因 provenance 缺字段而爆炸
- performance 至少不差于 Phase 1，最好略有提升

## 5. Phase 3：增强版 Code 路线

这一阶段明确只服务于 Code，Reasoning / Graph 暂不共享 code recovery 形状。

### 5.1 Phase 3A：执行升级

#### 5.1.1 prompt-level 三步式 recovery

把当前单步 patch 升级为三步式：

- Diagnose：生成结构化 `patch_plan`
- Patch：只根据 `patch_plan` 生成 patch code，不允许全文自由重写
- Self-check：输出简短 patch rationale / risk note，仅作 tie-break 和日志

这一步仍完全保持 prompt-level。

#### 5.1.2 统一 recovery loop

不再显式维护 Lean / Full 两个 repair mode，而是统一成一个 loop：

- 当前 checkpoint 未 fully pass 且仍存在 recoverable failure 时继续
- 从 recovery subgraph 中取 proposal 节点作为 recovery sources
- 每个 source 最多生成 1 个 patch branch
- checker / verifier 统一评估
- 只保留 strict-improve 当前 checkpoint 的 branches
- 有改进则更新 checkpoint，继续下一轮
- 无改进则停止

把深度交给如下超参数控制：

- recovery rounds 上限
- 每轮 source 数量
- 每轮 branch 数量

#### 5.1.3 fixed top-k 改为 support-set sparsification

将当前固定保边规则：

- 第一轮保 3 条入边
- 后续轮保 2 条入边

替换为 sparsemax support set：

`a_{u→v}^t = sparsemax_{u ∈ N^-(v)}(γ_{u→v}^t)`

保留全部 `a_{u→v}^t > 0` 的边。

这一步必须挂 feature flag，保留：

- `top-k baseline`
- `support-set variant`

用于对照 graph-faithfulness 和 MBPP 效果是否同步改善。

### 5.2 Phase 3B：学习 slot mask + slot depth

这一步进入 selector learning 层，必须建立在 3A 稳定之后。

#### 5.2.1 slot mask

增加小 policy：

`α_τ^t = f_slot(h_v^t, g^t, π_v)`

决定每轮打开哪些 slot。

#### 5.2.2 slot depth

再预测每个 slot 取几条：

`r_τ^t ∈ {0, 1, 2, ...}`

这是从固定 slot 模板走向可学习 typed retrieval policy 的自然升级，但必须放在最后，因为当前 selector 仍只负责 retrieval，不负责 routing / ranking / override。

## 6. 当前实现约定

本次实现将 Phase 2 明确落成两部分：

- runtime 侧：对缺 provenance 的可修 stage1 anchor 做 `best_original_graph_prior` 合法绑定
- training/report 侧：强制记录并分析 Phase 1 / Phase 2 所需指标

因此，这一版可以直接作为下一轮训练和观察的起点。
