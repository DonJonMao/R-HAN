# Stage1 / Stage2 全流程算法说明书与 Stage2 当前性能瓶颈

更新时间：2026-04-03

本文档不是“功能概述”，而是按当前真实代码，把 `stage1` 和 `stage2 v3.1` 的算法设计逐步拆开，明确到：

1. 输入是什么。
2. 中间状态是什么。
3. 每一步先算什么、后算什么。
4. 每个 `score / reward / target` 是怎么定义的。
5. 当前 `stage2` 的退化发生在算法链路的哪一步。

本文坚持三条原则：

- 只写当前代码里真实存在并正在运行的机制。
- 只在“结论”部分加入基于日志和 checkpoint 的推断。
- 明确区分“结构搜索”和“固定结构上的执行重判”。

相关代码主入口：

- `stage1` 入口：`mas_treesearch/pipeline.py`
- `stage1` 搜索核心：`mas_treesearch/search.py`
- `stage1` 结构集合选择：`mas_treesearch/topology_set.py`
- `stage1` 合图：`mas_treesearch/union_runtime.py`
- `stage1 -> stage2` 产物：`mas_stage2/structure_io.py`
- `stage2` 执行骨架：`mas_stage2/runtime_v2.py`
- 记忆选择：`mas_stage2/memory.py`
- LMPO：`mas_stage2/lmpo.py`
- `stage2 v3` 候选聚合基类：`mas_stage2_v3/runtime.py`
- `stage2 v3.1` 选择与学习：`mas_stage2_v3_1/runtime.py`
- `stage2 v3.1` pipeline：`mas_stage2_v3_1/pipeline.py`

---

## 1. 总体框架

### 1.1 一句话定义

当前项目的主线可以严格分成两层：

- `stage1`：搜索“什么 MAS 结构更适合当前题目”。
- `stage2`：在 `stage1` 冻结好的 `UnionGraph` 上，做多轮执行、消息路由、候选聚合与最终重判。

也就是说：

- `stage1` 优化的是 `structure`
- `stage2` 优化的是 `execution dynamics + final selection`

### 1.2 两阶段的边界

`stage1` 结束后，不会把“搜索过程”带到 `stage2`，而是只把以下静态产物交给 `stage2`：

- 一张固定 `UnionGraph`
- `stage1` 自己的最终输出 `stage1_output`
- `stage1` 的最终签名 `stage1_signature`
- `stage1` 的评测摘要 `stage1_summary`
- 结构摘要 `structure_summary`

从此以后：

- `stage2` 不再改结构
- `stage2` 只在固定图上运行
- `stage2` 是否翻案，本质上是在“保留 `stage1 anchor`”和“用 `stage2` 候选覆盖它”之间做选择

### 1.3 本文使用的记号

为了便于后续逐步展开，先统一几个记号：

- 题目：`q`
- 参考答案：`y*`
- 数据集 profile：`p`
- `stage1` 搜索状态：`s`
- `stage1` 单个结构节点：`n`
- `UnionGraph`：`G = (V, E)`
- `stage2` 轮数：`T`
- 第 `t` 轮：`t = 0, 1, ..., T-1`
- `stage1` 输出候选：`a_stage1`
- `stage2` 候选集合：`C = {c_1, ..., c_k}`
- `stage1 anchor`：候选 bank 中对应 `a_stage1` 的那一项

---

## 2. Stage1 的目标与输入输出

### 2.1 Stage1 输入

`TreeSearchMASPipeline.search(...)` 的输入是：

- `question_text`
- 可选 `reference_answer`
- 可选 `metadata`
- `dataset_name`
- `learn`
- `pipeline_mode`

其中最关键的是数据集 profile `p`。它会决定：

- 允许哪些 root templates
- 偏好哪些 agent
- 搜索迭代次数
- `tier1 / tier2` 的保留比例
- 结构先验
- reward 权重

### 2.2 Stage1 输出

`stage1` 的最终输出不是单个答案，而是一个 `SearchResult`，其中关键内容有：

- `best_node`
- `top_nodes`
- `records`
- `nodes`
- `union_graph`
- `structure_summary`

后续真正交给 `stage2` 的是 `PreparedStage1Artifact`：

- `question_text`
- `union_graph`
- `stage1_signature`
- `stage1_output`
- `dataset_name`
- `stage1_summary`
- `structure_summary`

---

## 3. Stage1 详细算法设计

## 3.1 Stage1 的核心对象

`stage1` 搜索的不是网络参数，而是离散 MAS 结构状态 `ArchitectureState`。一个状态至少包含：

- `template`
- `role_to_agent`
- `role_to_prompt`
- `prompt_edit_count`
- `non_prompt_steps_since_edit`

也就是说，一个状态本质上定义了：

- 工作流模板是什么
- 每个 role 由哪个 agent 扮演
- 每个 role 的 prompt slot 如何设定

### 3.2 Step 0：解析数据集 profile

算法的第一步不是直接造图，而是先解析 `dataset_profile`：

1. 从 `dataset_name` 或 `metadata["mas_dataset_name"]` 解析出数据集。
2. 载入这个数据集对应的 `profile`。
3. 从 `profile` 中取出：
   - `allowed_templates`
   - `root_templates`
   - `role_agent_preferences`
   - `required_agent_ids`
   - `search_overrides`
   - `structure_prior`

这一层的作用是把后续搜索变成“数据集感知”的搜索，而不是所有任务一套固定模板。

### 3.3 Step 1：做 agent 候选选择

`TaskConditioner.select(...)` 会根据题目和 profile 先产出 `AgentSelection`。这个对象至少包含：

- `question_vector`
- `candidate_agent_ids`
- `core_agent_ids`
- `explore_agent_ids`
- `scores`

注意这里还没有开始搜索。它只是把“全体 agent 池”缩成了“当前题目可用的 agent 候选子集”。

### 3.4 Step 2：构造 root states

对每个 `root_template`，`RootTemplateBuilder.build(...)` 都会生成一个初始结构状态。

这个构造过程按 role 逐个进行：

1. 取出该模板需要的 roles。
2. 对每个 role，按以下优先级选 agent：
   - `profile.role_agent_preferences[role]`
   - role 对应的 capability 偏好
   - 剩余 candidate agent 中第一个可用的
3. 某些 role 要求 agent 唯一，不允许多个关键 role 复用同一 agent。
4. 如果 profile 指定了 `required_agent_ids`，则尝试把这些 agent 注入合适的 role。
5. 为每个 role 填默认 prompt slots。
6. 个别模板会有特化，例如 `solve_verify` 下 verifier 的 `verification_mode = strict`。

这一步的输出是根结构 `s_root`。

### 3.5 Step 3：把结构状态编译成图节点

每个状态都会经过 `compile_architecture(state)` 编译成可执行结构，并包装成 `SearchNode`。节点里会保存：

- `state`
- `compiled`
- `parent_signature`
- `action_from_parent`
- `unexpanded_actions`
- `stats`
- `proxy_score`
- `tier1`
- `tier2`

同时系统会为每个根节点：

1. 枚举全部可扩展 action。
2. 如果启用 learned edit prior，则先给 action 排序。
3. 计算 proxy score。
4. 放入根集合 `self._roots`。

### 3.6 Step 4：定义可扩展动作空间

`_enumerate_actions(state)` 会给一个状态列出所有离散编辑动作。

当前动作有四大类：

1. `change_template`
   - 把当前模板切换到另一个 workflow template
2. `swap_agent`
   - 把某个 role 的 agent 换成另一个候选 agent
3. `set_prompt_slot`
   - 修改某个 role 的 prompt slot
4. `stop`
   - 停止继续编辑该状态

其中 prompt edit 不是无限开的，而受两个约束：

- `prompt_edit_count < max_prompt_edits_per_state`
- `non_prompt_steps_since_edit >= prompt_edit_cooldown`

这意味着 prompt 修改是“稀疏编辑”，不是每一步都能随便改。

### 3.7 Step 5：如果启用 learned edit prior，则先给动作排序

`_rank_actions(...)` 的逻辑很简单：

1. 用 `FeatureBuilder.action_features(...)` 为每个 action 构造特征。
2. 用 `LearnableEditPrior.score(features)` 打分。
3. 按分数降序排列 action。

如果没有启用 learned edit prior，则保持原始动作顺序。

这里要强调：

- `stage1` 的 learned edit prior 是动作排序器
- 它不是最后答案判别器

### 3.8 Step 6：代理结构搜索主循环

`TreeSearchEngine.search(...)` 的主循环可以写成：

```text
for iter in 1..search_iterations:
    1. 选 parent
    2. 对 parent 做 progressive widening
    3. 按动作顺序展开 child
    4. 对 child 做 proxy
    5. 对代码任务中的一部分 child 做 precheck
    6. 对 top fraction child 做 tier1
    7. 对 tier1 top fraction child 做 tier2
    8. 用 tier2 reward 回传更新搜索树
    9. 如启用学习，则更新 value model 与 edit prior
```

下面逐步拆开。

### 3.9 Step 6.1：选 parent

`_select_parent()` 不是简单贪心，而是带随机重启和 PUCT 风格加权抽样。

#### 路径 A：root restart

如果随机数落在 `root_restart_prob` 内，则直接从根集合里随机选一个 root。

这一步的意义是：

- 防止搜索早早陷入局部结构
- 保留模板层面的重新探索能力

#### 路径 B：从高分节点中做加权抽样

否则系统会：

1. 对全部节点排序。
2. 只取前 `top_k_selection` 个候选 parent。
3. 对每个候选算一个抽样权重。

排序主键是：

1. `max(q_mean, observed_node_score)`
2. `observed_node_score`
3. `proxy_score`

其中 `observed_node_score(node)` 的定义是：

- 如果已有 `tier2`：用 `tier2` 的有效分
- 否则若有 `tier1`：用 `0.92 * tier1有效分`
- 否则若有 `precheck`：用 `0.85 * precheck有效分`
- 否则用 `proxy_score`

parent 的抽样权重大致是：

```text
prior   = proxy_score
exploit = max(q_mean, observed_node_score)
bonus   = puct_c * prior * sqrt(visits + 1) / (1 + child_count)
weight  = max(1e-6, exploit + bonus + 1.0)
```

然后按这些 `weight` 做一次随机抽样。

因此 `stage1` 的 parent 选择不是纯 best-first，而是：

- 偏向高分节点
- 但保留探索

### 3.10 Step 6.2：progressive widening

选到 parent 后，不是把它所有 action 一次性展开，而是只展开有限个：

```text
limit = progressive_widening_base + progressive_widening_alpha * sqrt(max(1, parent.visits))
```

含义是：

- 访问次数越多，允许展开的动作越多
- 刚开始只探索最前面的少量动作

### 3.11 Step 6.3：应用动作得到 child

`_apply_action(...)` 根据 action 类型修改结构状态：

#### `change_template`

1. 切换 workflow template。
2. 保留当前活跃 agent 作为优先候选池。
3. 用新模板重新做 role 分配。
4. prompt edit 计数重置。

#### `swap_agent`

1. 定位某个 role。
2. 替换为新 agent。
3. 如果该 role 要求唯一 agent，则先检查目标 agent 是否已被其他关键 role 占用。

#### `set_prompt_slot`

1. 定位 role。
2. 修改某个 slot。
3. `prompt_edit_count += 1`
4. `non_prompt_steps_since_edit = 0`

#### `stop`

直接返回当前状态，不再继续扩展。

### 3.12 Step 6.4：proxy evaluation

每个新 child 先过 `proxy_scorer`：

1. `StaticProxyScorer.score(compiled, question_vector, profile, metadata)` 产出：
   - `proxy.score`
   - `proxy.uncertainty`
2. 如果启用了 learned value model，则再混合一次 learned prediction：

```text
score       = (1 - alpha) * proxy_score       + alpha * learned_mean
uncertainty = (1 - alpha) * proxy_uncertainty + alpha * learned_uncertainty
```

最后记录到：

- `node.proxy_score`
- `node.proxy_uncertainty`
- `node.stats.proxy_mean`

这里的作用是：

- 先用廉价 proxy 估计结构是否值得继续深评
- learned value model 负责逐步纠正 proxy 偏差

### 3.13 Step 6.5：precheck

`precheck` 只在代码任务中启用，并且只对 proxy 排名前一部分 child 做快速语法 / 基本合法性检查。

筛选逻辑：

1. 对 expanded children 按 `proxy_score` 排序。
2. 取前 `ceil(len(nodes) * code_precheck_top_fraction)` 个。
3. 调 `fast_code_precheck(...)`。

`precheck` 的作用不是最终质量判断，而是：

- 快速剔除明显格式不合法的代码候选
- 避免把贵的 `tier1 / tier2` 浪费在明显坏样本上

### 3.14 Step 6.6：tier1 evaluation

`tier1` 是中等成本评测。

执行步骤：

1. 用 `_tier_filter(expanded, tier1_fraction)` 对 expanded children 先做排序。
2. 只保留前一部分进入 `tier1`。
3. 调 `evaluator.evaluate(..., tier="tier1")`。

排序依据是：

- 优先看 `tier1` 有效分
- 没有 `tier1` 时退回 `precheck`
- 再退回 `proxy`

### 3.15 Step 6.7：tier2 evaluation

`tier2` 是真正高成本、最可信的结构质量评测。

执行步骤：

1. 从 `tier1_nodes` 再做一次 `_tier_filter(..., tier2_fraction)`。
2. 对保留下来的少量节点做 `tier2`。
3. 记录：
   - `tier2_mean`
   - `tier2_std`
   - `tier2_feedback`

这里真正回传到搜索树的 reward 不是简单 `mean_reward`，而是：

```text
effective_summary_score = risk_adjusted_score(summary, std_penalty) + task_type_specific_feedback_bonus
```

其中：

- `risk_adjusted_score(summary, std_penalty) = summary.mean_reward - std_penalty * summary.reward_std`
- 对 `code_generation`、`graph_reasoning` 等任务还会叠加一部分 feedback signal

### 3.16 Step 6.8：搜索树回传

对拿到 `tier2` 的节点：

1. 计算 reward = `effective_summary_score(node.tier2, tier="tier2")`
2. 从当前节点一路回溯到 root
3. 沿途更新：
   - `visits`
   - `q_mean`
   - `q_max`

这是标准树搜索中的 value backpropagation。

### 3.17 Step 6.9：Stage1 在线学习

当 `learn=True` 时，`stage1` 会做两类在线更新。

#### 3.17.1 更新 LearnableValueModel

对所有拿到 `tier2` 的节点：

1. 构造 state features。
2. 以 `tier2.mean_reward` 为监督目标。
3. 更新 `value_model`。

其作用是让后续 proxy 更接近真实 `tier2`。

#### 3.17.2 更新 LearnableEditPrior

对每个 `child <- parent + action`：

1. 找到该 child 的 `tier2.mean_reward`
2. 取 parent 的 `proxy_score` 作为 baseline
3. 设

```text
target = 1, if child.tier2.mean_reward > parent_baseline + score_improvement_epsilon
target = 0, otherwise
```

4. 用 action features 更新 `edit_prior`

这意味着 edit prior 学的不是“动作有多好”，而是：

- 该动作相对 parent baseline 是否带来了足够改进

### 3.18 Step 7：finalists 兜底重评

搜索主循环结束后，系统不会直接拿“访问最高”或“proxy 最高”的节点结束，而是：

1. 把全部搜索节点按 `_node_rank_key` 再排一次序。
2. 取前 `final_top_k` 个 finalist。
3. 如果某个 finalist 之前没有 `tier2`，则补做一次 `tier2`。
4. 再排一次序。
5. 取第一名为 `best_node`。

因此最终 `best_node` 是“补齐 tier2 以后重新排序”的结果。

### 3.19 Step 8：从单个 best node 到一组互补拓扑

`stage1` 不直接把 `best_node` 交给 `stage2`，而是交给 `TopologySetScorer.select(...)` 选一组互补拓扑。

#### 3.19.1 单个拓扑如何打分

每个候选拓扑先算：

- `quality`
- `quality_score`
- `structure_score`
- `readiness`
- `affordability`
- `score`

其中：

```text
quality =
    tier2 risk_adjusted_score
    else tier1.mean_reward
    else proxy_score
```

```text
quality_score = squash_probe_score(quality)
              = clamp01(0.5 + 0.5 * tanh(quality))
```

`structure_score` 由两部分组成：

```text
structure_score = 0.60 * readiness + 0.40 * affordability
```

最终单图分数：

```text
score = quality_weight * quality_score + structure_weight * structure_score
```

这里的 `quality_weight / structure_weight` 来自数据集 profile 的 `structure_prior`。

#### 3.19.2 readiness 怎么算

`readiness` 主要奖励：

- 有 reviewer 类角色
- 有 router
- sink 数清晰
- 角色组覆盖完整
- 代码任务中有 coder / planner / verifier

它回答的是：

- 这张图是否具备“能执行、有校验、有收束”的基本组织能力

#### 3.19.3 affordability 怎么算

`affordability` 主要惩罚：

- 任务节点太多
- 边太多
- sink 数偏离期望

它回答的是：

- 这张图是否过大、过重、过贵

#### 3.19.4 集合级选择怎么做

在候选拓扑池上，系统用贪心方式选出一组：

1. 先拿单图分最高的那个。
2. 之后每次在剩余候选里，找一个与当前已选集合最“互补”的图。

互补性 `complementarity` 主要看：

- edge diversity
- agent diversity
- template bonus
- sink bonus

每一轮选择时的 blended score 是：

```text
blended =
    (1 - diversity_weight) * candidate.score
    + diversity_weight * diversity
    + topology_union_bonus * min(1.0, diversity)
```

所以这一步的目标不是再找一个最强图，而是找“质量高且和已有图不重复”的图。

### 3.20 Step 9：把多张拓扑合成 UnionGraph

`GraphMerger.merge(...)` 会把这组拓扑合成一张 `UnionGraph`。

`UnionGraph` 中每个 `UnionNode / UnionEdge` 都会保留统计量，例如：

- `source_graph_ids`
- `support_count`
- `support_ratio`
- `avg_graph_score`
- `avg_parent_score`
- `best_parent_score`
- `topo_level_mean`
- `topo_level_var`
- `initial_keep_logit`

这些量后面会直接被 `stage2` 用到，特别是：

- `initial_keep_logit`
- `support_ratio`
- `avg_parent_score`
- `best_parent_score`

它们是 `stage2` 图路由的结构先验。

这里要特别说明两点：

1. `UnionGraph` 是 `stage1` 多图汇总后的统计图，不是简单并集。
2. 这里即便出现了像 `global_controller` 这样的图结构节点名，也不代表当前 `stage2` 又回到了旧的 `controller_model / edge_model` 路线。当前执行层真正使用的是 `runtime_v2 + gnn + global node`。

### 3.21 Step 10：生成结构摘要 structure_summary

`TopologySetScorer.summarize(...)` 会对最终 `UnionGraph` 给一个结构层摘要。

它至少会计算：

- `coverage`
- `complementarity`
- `redundancy_quality`
- `structural_faithfulness`
- `runtime_affordability`
- `execution_probe`

然后合成为：

```text
topology_quality = coverage_mix * coverage + (1 - coverage_mix) * structural_faithfulness
diversity_quality = complementarity_mix * complementarity + (1 - complementarity_mix) * redundancy_quality
deployability = runtime_affordability

structure_reward =
    topology_quality_weight * topology_quality
    + diversity_quality_weight * diversity_quality
    + deployability_weight * deployability

total_reward =
    (1 - execution_probe_weight) * structure_reward
    + execution_probe_weight * execution_probe
```

这份摘要并不直接决定最终答案，但它记录了：

- 这张 UnionGraph 在结构上为什么被选中

### 3.22 Step 11：把 Stage1 结果打包给 Stage2

`prepared_stage1_from_search_result(...)` 会把 `SearchResult` 压成 `PreparedStage1Artifact`：

1. 检查 `result.union_graph` 已存在。
2. 取 `stage1_signature = result.final_signature or result.best_node.compiled.signature()`
3. 取 `stage1_output = resolve_result_output(result)`
4. 带上：
   - `stage1_summary = result.best_node.tier2`
   - `structure_summary = result.structure_summary`

从这一刻开始，`stage2` 的输入就不是搜索树，而是这份 artifact。

---

## 4. Stage2 的总体设计

### 4.1 Stage2 的核心任务

`stage2` 在当前版本里分两层：

- 底层：`Stage2RuntimeV2.run(...)`
- 顶层：`Stage2RuntimeV31` 对最终候选的聚合、比较和学习

所以 `v3.1` 的真实结构是：

```text
固定 UnionGraph
    -> runtime_v2 多轮执行
    -> v3/v3.1 candidate bank 聚合
    -> v3.1 用 candidate_model + pairwise_model + reviewer_model 做最终决策
```

### 4.2 当前版本保留了什么、删除了什么

当前真实版本保留的是：

- `stage1` 固定结构
- `runtime_v2` 执行骨架
- `GNN` 边赋值
- `global node`
- latent memory
- LMPO
- candidate bank
- `v3.1` 三个可学习头

当前真实版本已经不再使用旧路线中的：

- 旧 `active node` 控制逻辑
- 旧 `edge_model`
- 旧 `controller_model`

注意：

- `runtime_v2` 基类还保留 `_active_task_nodes_v2(...)` 的默认实现
- 但在 `Stage2RuntimeV3 / V31` 中，这个函数被覆盖为“返回全部 task nodes”

所以现在的真实逻辑是：

- 边会 pruning
- 但节点不会因为边被裁而停跑
- 所有 task nodes 每轮都会执行

### 4.3 Stage2 输入

`Stage2V31Pipeline.search_prepared(...)` 的输入是：

- `question_text`
- `prepared_structure`
- `reference_answer`
- `metadata`
- `dataset_name`
- `replay_dir`
- `learn`

它首先做两件事：

1. 用 `prepared_structure.union_graph` 作为固定图
2. 往 `metadata` 注入：
   - `stage1_anchor_output`
   - `stage1_anchor_signature`

这意味着 `stage2` 从一开始就知道 `stage1` 想给出的答案是什么。

---

## 5. Stage2 v2 执行层的详细算法

### 5.1 总体主循环

`Stage2RuntimeV2.run(...)` 的主循环可以写成：

```text
初始化私有记忆、global node、policy缓存
for turn in 0..T-1:
    1. 先根据上一轮 feedback 构造 base turn state
    2. 给所有 task nodes 预取本地记忆并编码本地 latent
    3. 用 GNN + global node 对边打分并做 pruning
    4. 再构造带 active_edge_ratio 的 turn state
    5. 选本轮要运行的 task nodes
       在 v3/v3.1 中，这一步恒等于“全部 task nodes”
    6. 逐节点执行
    7. 抽 reviewer feedback events
    8. 把 feedback 作为 memory 写回
    9. 更新 global node 与全局文本摘要
   10. 记录 TurnTrace
循环结束后：
    11. 用 v3.1 finalizer 选择最终答案
```

下面按步骤展开。

### 5.2 Step 0：初始化运行状态

运行开始时，会初始化：

- `PrivateEpisodeMemoryStore`
- `self._pending_policy_log_probs`
- `self._pending_policy_entropies`
- `self._current_learn`
- `self._global_text_summary`
- `global_node.reset()`

并定义：

- `task_nodes = graph 中 node_type == "task" 的节点`
- `previous_feedback = []`
- `previous_exports = {}`
- `turn_traces = []`

### 5.3 Step 1：先构造 turn state

每轮开始时都会先构造一个 `ControllerState`，里面包含：

- 当前轮号
- 总轮数
- support / challenge / uncertain 计数
- `uncertainty`
- `active_edge_ratio`
- `summary`

其中：

```text
support_count   = # {pass, preserve}
challenge_count = # {challenge, reject, conflict, revise}
uncertain_count = # {uncertain}
```

```text
uncertainty =
    0.35, if no previous_feedback
    min(1.0, (challenge_count + uncertain_count) / total_feedback), otherwise
```

`summary` 会把这些统计和当前 `global_text_summary` 拼成一段文本，供后续 selector 和 prompt 使用。

### 5.4 Step 2：每个节点先做本地记忆选择

`_prepare_turn_packages(...)` 会对每个 task node 先做本地准备。每个节点都执行：

1. 读取该节点自己的私有 memory records。
2. 构造 `records_by_id`。
3. 用 `RoleAwareMemorySelector.select(...)` 选当前轮要看的记录。
4. 把选中的记录编码成 `local_latent`。

这一步非常重要，因为后面边打分时，GNN 用的是这里得到的 `local_latent`。

### 5.5 Step 2.1：RoleAwareMemorySelector 的打分公式

对每条本地 `MemoryRecord`，selector 会先构造查询文本：

```text
query_text =
    question_text
    + role
    + turn
    + global_state
    + uncertainty
    + controller_state.summary
```

然后嵌入成 `query_vec`，并对每条 record 计算：

- `query_similarity = cosine(query_vec, record.embedding)`
- `recency = 1 / (1 + current_turn - record.turn_index)`
- `role_bonus = 0.08 if record.role == node.role else 0`
- `token_penalty = min(1.0, record.token_estimate / 120.0)`

heuristic 分数是：

```text
heuristic_score =
    0.62 * query_similarity
    + 0.14 * recency
    + 0.12 * feedback_bias(record)
    + role_bonus
    - 0.02 * token_penalty
```

其中 `feedback_bias(record)` 对不同 feedback type 有固定映射，例如：

- `pass`: 0.12
- `challenge`: 0.14
- `reject`: 0.16
- `uncertain`: 0.04
- `unresolved`: 0.02

如果启用 learned selector，则还会再加一个轻量线性模型修正：

```text
final_score = heuristic_score + learned_weight * (learned_score - 0.5)
```

### 5.6 Step 2.2：Selector 的强制保留规则

selector 不是纯分数排序，还会做几类强制保留：

1. `keep_latest_self_output`
2. `keep_latest_feedback`
3. `include_failure_memory`
4. `include_success_memory`

然后剩余槽位再按分数从高到低补齐。

所以 selector 的真实行为是：

- 先保留最近输出和最近反馈
- 再保留一个失败信号、一个成功信号
- 最后用分数补齐

### 5.7 Step 3：把本地记忆编码成 latent

`_compose_latent(...)` 不会把整段记忆直接塞到 prompt，而是先做 latent 编码。

输入文本列表是：

- `role=<node.role>`
- `question=<截断后的题目>`
- `turn=<当前轮>`
- `global_state=<截断后的全局状态>`
- 每条选中 record 的简短文本

这些文本先被 tokenized，再送进 `composer`，得到该节点当前轮的 `local_latent`。

这个 latent 是后续两件事的基础：

1. 边打分
2. latent-guided memory brief

### 5.8 Step 4：对边做 GNN 路由打分

`_activate_edges_v2(...)` 会对每条 task-to-task edge 打 gate score。

#### 5.8.1 每条边的显式特征

边的显式特征向量是：

```text
[
    initial_keep_logit,
    support_ratio,
    avg_parent_score,
    best_parent_score,
    normalized_level_delta,
    progress
]
```

其中：

- `progress = (turn_index + 1) / total_turns`

#### 5.8.2 GNN 如何打 gate

如果 `gnn` 存在，则：

```text
gate = gnn.edge_gate(src_local_latent, dst_local_latent, edge_features, global_state)
```

如果 `gnn` 不存在，则退化成手工 prior：

```text
prior = 0.45 * support_ratio + 0.35 * avg_parent_score + 0.20 * best_parent_score
gate  = clamp(prior, 0, 1)
```

当前你这版真实在走的是 `gnn.edge_gate(...)` 路线。

### 5.9 Step 5：逐轮 pruning 边

边打分后，不是全部保留，而是按目标节点 `dst` 分组做裁剪。

对每个 `dst`：

1. 按 gate score 降序排列所有入边。
2. 计算 `keep_k`：

```text
keep_k = _turn_keep_k(incoming_count, turn_index)
```

它会随着轮数推进，从较宽松逐步收紧到更少入边。

3. 计算动态阈值：

```text
threshold_scale  = 0.85 if current_turn < hard_prune_after_turn else 1.0
dynamic_threshold = soft_prune_threshold * threshold_scale
```

4. 从 top 边里选一组候选：
   - 训练时可以采样并记录 `log_prob / entropy`
   - 推理时直接 top-k
5. 某条边成为 active 的条件：
   - rank 小于 `min_incoming_edges`
   - 或分数高于 `dynamic_threshold`

因此当前图的真实行为是：

- 边会随轮数逐渐变少
- 但当前 `v3.1` 不会因为边减少而减少执行节点数

### 5.10 Step 6：节点级激活

在基类 `runtime_v2` 中，本来还有一步根据 active edges 再裁 active nodes。

但在 `Stage2RuntimeV3 / V31` 中，这个函数被覆盖为：

```text
return list(task_nodes)
```

所以当前版本的真实行为是：

- `turn_0 ~ turn_4` 的 active node 数恒等于 task node 数
- 边 pruning 只影响“消息从谁传给谁”
- 边 pruning 不再影响“这个节点这一轮能不能执行”

这就是你前面反复强调的那版真实逻辑。

### 5.11 Step 7：逐节点执行

对每个 task node，都执行 `_run_task_node(...)`。

这一步可以再拆成九个小步骤。

#### 5.11.1 读取本地选中记录

如果前面 `prepared_state` 已经准备好了，就直接取：

- `records_by_id`
- `selected_items`
- `local_latent`

否则现算一次。

#### 5.11.2 聚合邻居 latent

`_aggregate_v2_neighbors(...)` 会收集所有 active incoming edges 对应的上游 export latent：

1. 找到 `dst == 当前节点` 且 `active=True` 的边。
2. 取这些上游节点上一轮的 `ExportedMemoryMessage`。
3. 把 export 中携带的 latent sequence 解出来。
4. 用边分数归一化成权重。
5. 调 `gnn(self_latent, neighbor_latents, edge_weights)` 聚合。

如果当前没有可用邻居 export，则直接用本地 latent。

#### 5.11.3 注入 global node 上下文

`_integrate_global_context(latent)` 的逻辑很直接：

```text
enhanced_latent = latent + 0.25 * global_ctx
```

其中 `global_ctx = global_node.get_context(1)`。

所以当前全局节点的作用不是替代节点执行，而是：

- 给每个节点的 latent 追加一份全局进展向量

#### 5.11.4 构造 latent-guided memory brief

这一步是当前执行层里最关键的 prompt 构造步骤之一。

先用 `enhanced_latent` 生成一个查询向量 `query`，然后分别对两类候选做检索：

- 本地 memory candidates
- 邻居 export candidates

对每类候选：

1. 把候选 embedding 堆成矩阵。
2. 与 `query` 做相似度打分。
3. 若训练中则按 softmax 采样，并记录 `log_prob / entropy`
4. 若推理中则直接 top-k

最后生成文本 brief：

```text
[Global State]
...

[Latent-Selected Private Memory]
- ...

[Graph-Mediated Neighbour Signals]
- ...
```

这一步说明：

- prompt 里出现哪些历史记忆
- prompt 里出现哪些邻居消息

并不是固定规则，而是 latent policy 决定的。

#### 5.11.5 生成角色指令

不同 role 的 instruction 不同，例如：

- `solver`: 给出当前最强候选
- `critic`: 指出最可能缺陷
- `reviser`: 用 strongest feedback 修复
- `judge`: 判断哪个候选更可靠

如果是代码任务，还会进一步约束：

- reviewer 类角色不得输出代码
- candidate 类角色应输出满足函数签名的可执行 Python

#### 5.11.6 组装 prompt 并调用 LLM

prompt 由四部分组成：

1. `system_prompt`
2. `Question`
3. `Latent-guided memory brief`
4. `Current task`

必要时再加：

- `Task context`
- `Output contract`

然后用 `evaluator._cached_chat(...)` 调用 tier2 runtime，得到原始输出，再做 postprocess。

#### 5.11.7 写回私有 memory

每个节点本轮都会生成一条 `MemoryRecord(record_type="self_output")`，其中包含：

- `text = 当前输出`
- `embedding = embed(output)`
- `feedback_type = unresolved`
- `selected_record_ids`
- `neighbour_sources`

这条记录只属于当前 owner node。

#### 5.11.8 生成 ExportedMemoryMessage

`_build_v2_export(...)` 会把节点当前轮的输出压成一条对外广播消息：

1. 先取当前 latent 的查询向量。
2. 从本地 selected_items 中最多取前两条做 carry memory。
3. 构造 export summary：
   - `<role> update`
   - `Memory carry: ...`
   - `Current output: ...`
4. 写成 `ExportedMemoryMessage`

包含字段：

- `node_id`
- `turn_index`
- `summary`
- `latent_vector`
- `provenance_record_ids`
- `confidence`

因此 `ExportedMemoryMessage` 的含义不是：

- 把原始 memory 直接共享给别的 agent

而是：

- 把当前节点本轮输出和少量 carry memory 压缩成图上传播的消息

#### 5.11.9 在 v3/v3.1 中补 typed output 元数据

`Stage2RuntimeV3._run_task_node(...)` 会对输出再贴一个“类型标签”。

如果节点 role 属于 candidate roles，或节点本身是 sink，则标成候选：

- `candidate`
- 记录 `candidate_digest`
- 代码任务下记录 `parse_ok / entry_point_ok`

如果 role 属于 review roles，则标成 review：

- `review`
- 提取 `review_verdict`

否则标成 `analysis`。

### 5.12 Step 8：抽 reviewer feedback events

每轮节点执行完后，系统会从 reviewer 类输出中抽 `FeedbackEvent`。

这些 event 典型类型有：

- `pass`
- `challenge`
- `uncertain`
- `preserve`
- `reject`
- `conflict`

这些 feedback 后续有两条用途：

1. 写入 memory，供下一轮节点 prompt 使用
2. 进入 candidate bank，影响 reviewer 校准和最终排序

### 5.13 Step 9：把 feedback 写回 memory

抽出的 `FeedbackEvent` 会再转成 `MemoryRecord(record_type="feedback")`，写回对应节点的私有 memory。

因此下一轮 selector 看到的本地记忆，不只包括自己的过去输出，还包括别人对自己的评价。

### 5.14 Step 10：更新 global node 与全局文本摘要

`_update_global_state(...)` 会做两件事。

#### 5.14.1 更新 global node 向量状态

1. 对每个节点取当前 export latent。
2. 求均值或聚合后形成 update。
3. 用 `global_node.update(updates, outputs)` 更新全局隐状态。

#### 5.14.2 生成全局文本摘要

1. 用 global context 生成一个 query。
2. 对本轮各节点输出做相似度排序。
3. 取前 `global_summary_max_nodes` 条。
4. 拼成新的 `_global_text_summary`。

所以当前 global node 有两个面向：

- 向量态：供下轮 latent 融合和边打分
- 文本态：供下轮 selector 和 prompt brief 使用

### 5.15 Step 11：记录 TurnTrace

每轮会记录：

- `controller_state`
- `active_edges`
- `node_traces`
- `feedback_events`
- `sink_outputs`
- `active_node_ids`
- `skipped_node_ids`
- `global_summary`
- token 估计与 token cost

在当前 `v3.1` 中，正常现象应该是：

- `active_node_ids` 等于全部 task nodes
- `skipped_node_ids` 基本为空

---

## 6. Stage2 v3.1 的候选聚合与最终选择

### 6.1 v3.1 的设计边界

`v3.1` 并没有改 `runtime_v2` 的多轮执行壳。它只改最后一层：

- 候选怎么聚合
- reviewer 信号怎么解释
- challenger 与 `stage1 anchor` 怎么比较
- 在线学习监督信号怎么定义

所以 `v3.1` 的核心不是新执行图，而是新选择器。

### 6.2 candidate roles 和 review roles 是固定集合

当前代码中，这两个集合是提前写死的。

#### 候选角色 candidate roles

```text
{solver, solver_a, solver_b, generator, reviser, aggregator}
```

它们的输出会进入 candidate bank。

#### 评审角色 review roles

```text
{critic, verifier, judge}
```

它们的输出不会直接作为答案候选，而是会产出 reviewer 信号。

所以当前系统里：

- 不是所有 agent 都能当 reviewer
- 只有上述一组 role 的输出会被解释为 review signal

### 6.3 candidate bank 的整体流程

`_candidate_bank_bundle(...)` 可以写成：

```text
初始化空 bank
读取 turn_traces
对每个 turn_trace:
    对每个 node_trace:
        如果是 candidate role 或 sink 节点:
            1. 规范化输出文本
            2. 加入/找到对应 candidate entry
            3. 记录 occurrence_key = (turn, node)
            4. 聚合该 occurrence 收到的 reviewer feedback
            5. 更新 entry 的出现统计、来源统计、评审统计
把 stage1_anchor_output 也注入 bank
对 bank 中每个 entry:
    1. 做代码合法性检查
    2. 构造 candidate features
    3. candidate_model 打绝对质量分
排序 candidates
找到 anchor
返回 bundle
```

下面展开每个环节。

### 6.4 Step 1：规范化候选文本

每个候选不是直接用原始 LLM 输出，而是先经过：

```text
self._sanitize_candidate(question_text, raw_text, metadata)
```

这样做的目的有两个：

1. 把不同节点、不同轮次里语义相同但格式略有差异的答案尽量对齐
2. 让 candidate bank 面对的是“规范化后答案文本”，而不是原始杂乱输出

规范化后文本的哈希摘要：

```text
digest = sha1(text)[:12]
```

### 6.5 Step 2：按 occurrence 聚合

每个 `(turn_index, node_id)` 上的一次候选出现，称为一个 `occurrence`。

对于每个 occurrence，系统会记录：

- `digest`
- `role`
- `is_sink`
- `parse_ok`
- `entry_point_ok`

然后把以下统计累加到对应 candidate entry：

- `occurrence_count += 1`
- 如果该节点是 sink，则 `sink_support += 1`
- `source_node_ids.add(node_id)`
- `source_roles.add(role)`
- `turn_indices.add(turn_index)`

因此一个 candidate 最终代表的不是“一次输出”，而是：

- 多轮
- 多节点
- 多角色

上出现的“同一个规范化答案”。

### 6.6 Step 3：聚合 reviewer feedback

对每个 occurrence，系统会找到所有指向它的 `FeedbackEvent`，然后调用 `_aggregate_occurrence_feedback(...)`。

#### 6.6.1 reviewer features

每条 feedback event 会构造 reviewer 特征：

- `bias`
- `task::<task_type>`
- `reviewer::<source_kind>`
- `event::<event_type>`
- `target_role::<target_role>`
- `confidence`
- `target_is_sink`
- `detail_length`
- `parse_ok`
- `entry_point_ok`

#### 6.6.2 reviewer_model 的预测

`reviewer_model` 是 `OnlineLinearModel`，其预测形式是：

```text
mean = clamp01(bias + sum_i w_i * x_i)
uncertainty = max(init_uncertainty if steps==0 else residual_ema / sqrt(steps), 0.03)
```

当前 reviewer weight 的定义是：

```text
trust = reviewer_model.predict(features).mean
weight = confidence * (0.5 + trust)
```

因此 reviewer 不是简单一人一票，而是：

- 先给 reviewer event 一个 trust
- 再用 `confidence * (0.5 + trust)` 做校准

#### 6.6.3 entry 上累加的 reviewer 统计

每个 candidate entry 最终会累计：

- raw counts：
  - `feedback_pass`
  - `feedback_challenge`
  - `feedback_uncertain`
- calibrated weights：
  - `feedback_pass_calibrated`
  - `feedback_challenge_calibrated`
  - `feedback_uncertain_calibrated`
- reviewer 统计：
  - `reviewer_event_count`
  - `reviewer_trust_sum`
  - `reviewer_mean_trust`

### 6.7 Step 4：把 stage1 anchor 也放进 bank

如果 `metadata` 中存在 `stage1_anchor_output`，则系统会：

1. 规范化这段文本
2. 在 bank 中找到或新建对应 entry
3. 把 `stage1_anchor = True`

这意味着：

- `stage1` 的答案不是在 bank 外单独比较
- 而是 bank 内的一个特殊 candidate

### 6.8 Step 5：代码任务的合法性检查

如果当前任务是 `code_generation`，系统会对每个 candidate 做：

1. `ast.parse(text)` 检查是否可解析
2. 如果 metadata 中有 `entry_point`，再检查是否包含对应函数定义

得到两个布尔量：

- `parse_ok`
- `entry_point_ok`

这两个量会同时进入：

- candidate_model 特征
- pairwise_model 特征
- reviewer_model 特征

### 6.9 Step 6：构造 candidate_model 的绝对特征

在 `v3.1` 中，每个 candidate 的绝对特征包括：

- `bias`
- `task::<task_type>`
- `occurrence_log`
- `occurrence_saturation`
- `sink_support_log`
- `sink_ratio`
- `turn_ratio`
- `source_node_count`
- `source_role_count`
- `source_diversity`
- `feedback_pass_raw`
- `feedback_challenge_raw`
- `feedback_uncertain_raw`
- `feedback_pass_calibrated`
- `feedback_challenge_calibrated`
- `feedback_uncertain_calibrated`
- `review_margin`
- `review_volume`
- `review_consensus`
- `reviewer_event_count`
- `reviewer_mean_trust`
- `text_length`
- `line_count`
- `parse_ok`
- `entry_point_ok`
- 每个 `source_role::<role>` 的 one-hot

其中几个关键派生量：

```text
occurrence_saturation = 1 - exp(-occurrence_count / 3)
sink_ratio = sink_support / max(1, occurrence_count)
source_diversity = log(1 + #source_nodes) + 0.5 * log(1 + #source_roles)
review_margin = pass_calibrated - challenge_calibrated
review_consensus = review_margin / review_volume
```

当前 `v3.1` 相比 `v3` 的一个重要变化是：

- 不再把 `stage1_anchor` 身份直接作为绝对质量特征输入 `candidate_model`

### 6.10 Step 7：candidate_model 给绝对质量分

`candidate_model` 也是 `OnlineLinearModel`。

它输出：

- `candidate_model_score`
- `candidate_model_uncertainty`

然后系统把：

```text
support_score = candidate_model_score
```

也就是说，在 `v3.1` 中，最终的候选绝对排序主分已经是 learned 的，而不是手写多项加权。

### 6.11 Step 8：candidate 排序

在非代码任务中，排序主键是：

1. `quality_score`
2. `review_consensus`
3. `sink_ratio`
4. `source_diversity`
5. `review_advantage`
6. `digest`

在代码任务中，最前面还会先卡：

1. 是否 valid code
2. `quality_score`
3. `review_consensus`
4. `sink_ratio`
5. `source_diversity`
6. `review_advantage`

注意：

- 排序只是为了给出一个候选列表
- 最终是否翻掉 `stage1 anchor`，还要看 pairwise model

### 6.12 Step 9：构造 challenger vs anchor 的 pairwise 特征

`v3.1` 不再像旧 `v3` 那样只拿“absolute 第一名”去挑战 anchor，而是：

- 对 candidate bank 中每个 challenger 都和 anchor 做一次两两比较

pairwise 特征包括：

- `bias`
- `task::<task_type>`
- `challenger_quality`
- `anchor_quality`
- `quality_margin`
- `challenger_review_margin`
- `anchor_review_margin`
- `review_margin`
- `challenger_review_consensus`
- `anchor_review_consensus`
- `review_consensus_margin`
- `challenger_sink_ratio`
- `anchor_sink_ratio`
- `sink_ratio_margin`
- `challenger_source_diversity`
- `anchor_source_diversity`
- `source_diversity_margin`
- `challenger_occurrence_saturation`
- `anchor_occurrence_saturation`
- `occurrence_saturation_margin`
- `challenger_uncertainty`
- `anchor_uncertainty`
- `uncertainty_margin`
- `challenger_parse_ok`
- `anchor_parse_ok`
- `challenger_entry_point_ok`
- `anchor_entry_point_ok`

pairwise model 输出：

```text
P(challenger > anchor)
```

### 6.13 Step 10：最终答案选择规则

`_select_against_anchor(...)` 的逻辑按顺序是：

#### 情形 A：bank 为空

- 如果有 anchor，则直接用 anchor
- 否则返回空答案

#### 情形 B：没有 anchor

- 直接取 absolute 排序第一的 candidate

#### 情形 C：有 anchor

1. 对所有 `candidate != anchor` 计算：
   - `pairwise_probability`
   - `pairwise_uncertainty`
2. 按以下键排序 challenger：
   - `pairwise_probability`
   - `quality_score`
   - `review_consensus`
   - `sink_ratio`
   - `source_diversity`
3. 取第一名 challenger
4. 如果：

```text
pairwise_probability > 0.5
```

则 override，否则 preserve anchor。

#### 代码任务额外规则

代码任务下还有三个硬约束：

1. 如果 anchor 非法、但存在合法 challenger，则优先换成合法 challenger。
2. 如果没有合法 challenger，则保留 anchor。
3. 否则在“合法 challenger”集合里再用 `pairwise_probability > 0.5` 决定是否 override。

这说明当前 `v3.1` 的决策结构是：

```text
先 absolute 排序生成 challenger 集合
再 relative pairwise 决定要不要翻 anchor
```

---

## 7. Stage2 v3.1 的在线学习

### 7.1 Stage2 的最终 summary 和训练目标

`Stage2V31Pipeline.search_prepared(...)` 在 run 结束后，会先对最终答案重新做一次 `tier2` evaluator：

- `final_summary = evaluator.evaluate_output(stage2_result.final_answer, tier="tier2", ...)`

然后定义训练目标：

```text
learning_target = clamp01(0.55 * mean_success + 0.45 * mean_task_score)
```

如果最终保留的是 `stage1 anchor`，则还会减一个 `fallback_penalty`：

```text
if stage1_anchor_used:
    learning_target = max(0, learning_target - fallback_penalty)
```

这一步是训练目标，不是最终展示 reward。

### 7.2 真正记到日志里的 stage1/stage2 reward

日志里记录的 `stage1_reward / stage2_reward` 用的是：

```text
risk_adjusted_score(summary, search_config.risk_std_penalty)
```

即：

```text
mean_reward - std_penalty * reward_std
```

因此：

- `stage2_success` 只看是否做对
- `stage2_reward` 还会受 latency、token cost、标准差等影响

### 7.3 Step 1：先做 LMPO 更新

`Stage2RuntimeV2.learn_from_run(...)` 会先做一次 LMPO 的策略梯度更新。

LMPO 训练目标的核心是：

```text
loss = -advantage * sum(log_probs) - entropy_coef * sum(entropy_terms) + auxiliary_losses
```

其中：

```text
baseline  = baseline_momentum * baseline + (1 - baseline_momentum) * reward
advantage = reward - baseline
```

LMPO 更新的不是 LLM 本体，而是“latent memory 选择 / verbalization”这条离散策略。

当前它影响的主要采样点有：

- edge 选择中的 `_choose_indices`
- memory brief 中本地记忆选择
- memory brief 中邻居消息选择

### 7.4 Step 2：选一组学习候选

`v3.1` 不会拿 bank 中全部 candidate 去学习，而是：

1. 从排序后的 `ranked_candidates` 中去重
2. 最多保留前 `max_logged_candidates`
3. 如果 anchor 不在里面，再把 anchor 加入

记为 `learning_candidates`。

### 7.5 Step 3：对每个 candidate 重新做 evaluator 打分

`_evaluate_candidate_targets(...)` 会对每个 candidate 文本重新调用：

```text
evaluator.evaluate_output(question_text, candidate_text, tier="tier2", ...)
```

然后转成 candidate target：

```text
candidate_target = clamp01(0.55 * mean_success + 0.45 * mean_task_score)
```

这一步很关键，因为它说明：

- `candidate_model`
- `pairwise_model`
- `reviewer_model`

最终都不是用启发式 reward 学，而是用 evaluator 给出的候选质量学。

### 7.6 Step 4：更新 candidate_model

对每个 `learning_candidate`：

1. 取它的 `candidate_features`
2. 取它的 `candidate_target`
3. 用 `OnlineLinearModel.update(features, target)` 更新

更新公式是：

```text
pred  = clamp01(bias + w·x)
error = pred - target
bias  <- bias - lr * error
w_i   <- w_i - lr * error * x_i
```

因此 candidate_model 是在线线性回归/分类式更新，不是 MLP。

### 7.7 Step 5：更新 pairwise_model

对 `learning_candidates` 中任意两个不同候选 `left` 和 `right`：

1. 取 `left_target` 和 `right_target`
2. 构造 pairwise 监督：

```text
1.0 if left_target > right_target
0.0 if left_target < right_target
0.5 otherwise
```

3. 用 `_override_features(left, right)` 更新 `pairwise_model`

这意味着 `pairwise_model` 学的是真正的相对排序，而不是“只学 challenger vs anchor”的二元 gate。

### 7.8 Step 6：更新 reviewer_model

`v3.1` 的 reviewer 学习目标是“相对纠错价值”，不是“绝对好坏”。

定义：

```text
margin = candidate_target - reference_target
```

若 event 是：

- `pass / preserve`
  - `margin > 0` 时 target = 1
  - `margin < 0` 时 target = 0
  - `margin = 0` 时 target = 0.5
- `challenge / reject / conflict`
  - `margin < 0` 时 target = 1
  - `margin > 0` 时 target = 0
  - `margin = 0` 时 target = 0.5
- `uncertain`
  - target = `max(0, 1 - |margin|)`

这里的 `reference_target` 定义为：

- 如果当前事件指向的是 anchor，则 reference 取最好非 anchor 候选
- 如果当前事件指向的是非 anchor 候选，则 reference 取 anchor

这意味着 reviewer_model 学的是：

- 这个 reviewer 事件是否正确地区分了“当前候选”和“对照候选”

而不是：

- reviewer 说了 pass 就一定好
- reviewer 说了 challenge 就一定坏

---

## 8. Torch 具体用在什么地方

你前面问过“具体哪里用到了 torch”。当前真实代码里，`torch` 主要用在以下部分：

### 8.1 明显使用 torch 的部分

- `composer`
  - 本地记忆文本编码为 latent
- `gnn`
  - 边 gate 打分
  - 邻居 latent 聚合
- `global_node`
  - 全局上下文状态更新和读取
- `_candidate_scores / _choose_indices`
  - latent-guided 记忆 / 邻居选择
- `LMPOTrainer`
  - policy gradient 更新

### 8.2 不主要依赖 torch 的部分

- `candidate_model`
- `pairwise_model`
- `reviewer_model`

这三个当前都是 `OnlineLinearModel`，本质上是：

- 线性权重
- 在线梯度更新
- 不依赖深层 torch 网络结构

所以现在“自适应的三部分”不是 torch 的深网，而是三组在线线性判别器。

---

## 9. 当前 gsm8k 训练的直接证据

当前 `gsm8k` 的 checkpoint 反映的是 `175` 条训练样本上的已落盘状态。

关键数字如下：

- `stage1_success = 0.9714`
- `stage2_success = 0.9029`
- `stage1_reward = 1.5765`
- `stage2_reward = 0.4403`

更细地拆：

- success improved: `0`
- success worse: `12`
- reward improved: `0`
- reward worse: `175`
- preserve count: `161`
- override count: `14`
- anchor rate: `92.0%`

preserve 路径统计：

- `stage1_success = 0.9814`
- `stage2_success = 0.9814`
- `stage1_reward = 1.5954`
- `stage2_reward = 0.6154`
- `stage1_latency = 0.4165`
- `stage2_latency = 50.0492`

override 路径统计：

- `stage1_success = 0.8571`
- `stage2_success = 0.0`
- `stage1_reward = 1.3589`
- `stage2_reward = -1.5736`
- `override pairwise probability mean = 0.7416`

模型更新证据：

- `v3_1_candidate_model.steps = 601`
- `v3_1_pairwise_model.steps = 2098`
- `v3_1_reviewer_model.steps = 2233`

这说明：

- 训练不是“没在跑”
- 学习确实发生了
- 但学出来的决策方向当前是错的

---

## 10. 当前退化发生在算法链路的哪一步

这里不再泛泛而谈，而是按算法步骤定位问题。

### 10.1 问题 1：执行层没有把“更强 challenger”稳定做出来

对应算法位置：

- 第 5 节的多轮执行
- 第 6 节的 candidate bank 构造

当前直接证据是：

- `override count = 14`
- `success improved = 0`

这说明在当前已跑完的 `gsm8k` 样本里：

- 没有任何一次 `stage2` 真正把 `stage1` 错答案纠正成正确答案

所以问题的第一层不是“最终选择器太保守”，而是：

- 候选池里本来就几乎没有高质量、可稳定击败 anchor 的 challenger

### 10.2 问题 2：candidate bank 仍然有明显的“重复即高分”惯性

对应算法位置：

- 第 6.5 节到第 6.10 节

虽然 `v3.1` 已经去掉了 `stage1_anchor` 直接进入 absolute features，但 candidate 仍然会从这些统计项获益：

- `occurrence_log`
- `occurrence_saturation`
- `sink_support_log`
- `sink_ratio`
- `source_diversity`

如果一个答案本来就更像 `stage1` 路线，更容易被多个节点重复复述，那么它会自然获得：

- 更高 `occurrence`
- 更高 `sink_support`
- 更高 `source_diversity`

于是 candidate_model 依然可能学出：

- “像 stage1 的答案 = 高质量”

这不是因为显式喂了 `stage1_anchor` 标志，而是因为 bank 统计本身已经带有强烈的路径依赖。

### 10.3 问题 3：pairwise_model 在 override 路径上高置信但方向错误

对应算法位置：

- 第 6.12 节到第 6.13 节

当前最强的坏信号就是：

- override path 上 `pairwise_probability mean = 0.7416`
- 但 `stage2_success = 0.0`

这说明当前不是“pairwise 不敢 override”，而是：

- 一旦它 override，就经常在高置信下 override 错对象

也就是说当前真正塌的是：

- `P(challenger > anchor)` 的方向性

### 10.4 问题 4：reviewer 信号仍然更像“挑战放大器”，而不是“纠错证据”

对应算法位置：

- 第 6.6 节
- 第 7.8 节

你前面总结 reviewer 像“愤青”，这个判断和当前现象是一致的。

问题不在于 reviewer 完全没信号，而在于：

- `challenge` 类信号很多
- 但这些信号并不稳定对应“被挑战者真的更差”

于是它们被聚合进 bank 后，会带来两个副作用：

1. 把本来只是“不同表达”的候选也推成 challenger
2. 给 pairwise model 提供带噪声的 review margin / review consensus 特征

### 10.5 问题 5：preserve 路径 correctness 不掉，但 reward 全掉

对应算法位置：

- 第 5 节的多轮执行成本
- 第 7.2 节的最终 reward 计算

preserve 路径上：

- `stage1_success = stage2_success = 0.9814`
- 但 `stage2_latency = 50.0492`，远高于 `stage1_latency = 0.4165`

因此即便 `stage2` 没把答案做坏，只要它最终还是保留 `stage1 anchor`，它也会因为：

- 延迟暴涨
- token cost 增长

导致 `stage2_reward` 大幅下降。

所以当前在 `gsm8k` 上，`stage2` 处于一个非常不利的位置：

- preserve 不加正确率，但必然加成本
- override 目前又经常错

于是 reward 几乎必然全面变差。

### 10.6 问题 6：训练目标对 preserve 还有额外惩罚

对应算法位置：

- 第 7.1 节

当前训练时：

```text
if stage1_anchor_used:
    learning_target -= fallback_penalty
```

这意味着即便某个样本上 preserve 是正确且理性的，训练目标也会给它一个额外折扣。

它的原始设计意图是：

- 避免模型永远躺回 stage1

但在 `gsm8k` 这种 `stage1` 基线已经很高的数据集上，这个机制会进一步放大一个偏差：

- 客观上 preserve 常常是对的
- 训练上 preserve 又被额外扣分

这会让模型在“高基线数据集”上更难学到真正合理的保守策略。

### 10.7 问题 7：v3.1 的学习器并不复杂，但当前主矛盾也确实不在“网络太浅”

当前三头都是 `OnlineLinearModel`，这当然不复杂。

但从现有证据看，主要矛盾仍然不是：

- 模型不够深
- 没上 MLP

而是：

1. `stage2` 执行层没有稳定制造出高质量 challenger
2. 候选 bank 统计有路径依赖
3. reviewer 信号噪声高
4. pairwise 在少量 override 样本上学歪了方向
5. `gsm8k` 上 preserve 的客观收益本来就低，成本却很高

所以当前更准确的判断是：

- 现在的问题首先是“信号结构错位”，不是“网络容量不足”

---

## 11. 一二阶段的最终职责总结

### 11.1 Stage1 在做什么

`stage1` 的真实职责可以压成一句话：

- 搜索一批高质量 MAS 结构
- 从中挑出一组互补拓扑
- 合成一张带结构先验的 `UnionGraph`
- 同时给出一个高质量 `stage1 anchor answer`

它解决的是：

- 谁来协作
- 用什么 workflow
- 图怎么连
- 哪些结构值得进入固定执行阶段

### 11.2 Stage2 在做什么

`stage2 v3.1` 的真实职责可以压成一句话：

- 在固定图上做多轮协作执行
- 用私有记忆、邻居 export、global node 和 GNN 调节信息流
- 生成候选 bank
- 再用可学习的绝对质量、相对比较、reviewer 校准来决定是否翻掉 `stage1 anchor`

它解决的是：

- 固定结构如何真正运行
- 哪些中间结果该传播
- 最后是否应该信 `stage1`，还是信 `stage2`

### 11.3 当前 Stage2 的核心瓶颈

当前 `gsm8k` 上的核心矛盾也可以压成一句话：

- `stage1` 已经太强，`stage2` 若保留它就只会增加成本，若翻案又没有足够可靠的新证据，所以结果表现为“几乎全退化”

更具体地说，退化主要来自三点叠加：

1. 候选池里缺少真正优于 `stage1` 的 challenger
2. 一旦 override，pairwise 常常高置信选错
3. 即便 preserve 正确，reward 也会因为执行成本暴涨而下降

这就是当前 `stage2 v3.1` 在 `gsm8k` 上“训练生效但结果退化”的最准确算法级解释。
