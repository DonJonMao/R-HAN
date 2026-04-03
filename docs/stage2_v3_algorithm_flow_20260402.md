# Stage2 V3 算法流程详解

更新时间：2026-04-02

这份文档描述的是当前代码里真实存在的 `stage2 v3` 实现。当前版本已经修正为：

`stage1 固定结构 + runtime_v2 图执行壳 + GNN 边赋值 + global node 全局状态 + 每轮全量 task nodes 执行 + v3 三个 learned 选择头`

对应代码文件：

- `mas_stage2_v3/config.py`
- `mas_stage2_v3/runtime.py`
- `mas_stage2_v3/pipeline.py`
- `train_mas_stage2_v3_target_suite.py`
- `mas_stage2/runtime_v2.py`

## 1. 一句话总览

当前 `stage2 v3` 的核心逻辑是：

1. `stage1` 先为每道题生成固定 `UnionGraph`
2. `stage2 v3` 在这张固定图上运行 5 轮
3. 每轮先用 `GNN + global node` 给边打分并做 pruning
4. 但不再显式选择 `active node`，而是所有 task nodes 每轮都执行
5. 执行结束后构建 candidate bank
6. 再用三个 learned 模块决定最终答案：
   - `candidate_model`
   - `override_model`
   - `reviewer_model`

因此这版 `v3` 的重点不再是“谁能运行”，而是“边如何传信息、候选如何排序、何时覆盖 stage1”。

## 2. 它在整套系统中的位置

完整流程是：

1. `TreeSearchMASPipeline` 生成 stage1 搜索结构
2. 结构被保存为 `PreparedStage1Artifact`
3. `Stage2V3Pipeline.search_prepared(...)` 读取这张固定图
4. `Stage2RuntimeV3.run(...)` 执行多轮图通信
5. `Stage2RuntimeV3._finalize_answer(...)` 做最终候选选择
6. `Stage2RuntimeV3.learn_from_run(...)` 做在线学习

所以 `v3` 仍然和 `stage1` 相连，但 `stage2` 的训练对象已经清晰收敛到：

- 图上的边通信
- 最终候选排序
- stage1/stage2 覆盖决策
- reviewer 可靠性校准

## 3. 配置层

当前 `Stage2V3Config` 直接继承 `Stage2V2Config`。

这意味着 `v3` 复用了 `v2` 的这些基础组件配置：

- `composer_*`
- `gnn_*`
- `global_node_*`
- `lmpo_*`
- `latent_prompt_*`

同时 `v3` 自己新增并保留的配置主要是最终选择层：

- `numeric_stage1_anchor_prior`
- `numeric_override_margin`
- `numeric_min_support_to_override`
- `code_stage1_anchor_prior`
- `code_support_margin`
- `code_min_review_advantage`
- `code_require_entry_point`
- `generic_stage1_anchor_prior`
- `generic_override_margin`
- `max_logged_candidates`

其中需要特别说明：

- 这批旧字段现在主要承担兼容配置和日志接口的作用
- 当前主选择路径已经由 learned heads 主导
- 代码任务里真正仍然作为硬约束使用的是 `code_require_entry_point`

## 4. 运行时基底：复用 runtime_v2

当前 `Stage2RuntimeV3` 继承的是 `Stage2RuntimeV2`，不是旧的 `Stage2Runtime`。

这件事很重要，因为它决定了 `v3` 的执行语义：

- 复用 `latent memory composer`
- 复用 `GNN edge gate`
- 复用 `global node`
- 复用 `LMPO` 对离散选择项的更新
- 不再走旧版 `GlobalController + edge_model + controller_model` 的主路径

也就是说，当前 `v3` 的基础执行壳已经是：

`global node + edge-only graph gating + latent-guided prompt brief`

## 5. 每轮执行流程

`Stage2RuntimeV3.run(...)` 内部实际复用的是 `runtime_v2.run(...)` 的轮次流程。

每轮主要分成 8 步。

### 5.1 构建 turn state

每轮开始先构造一个轻量 `turn_state`。

这里虽然复用了 `ControllerState` 这个数据结构名字，但它已经不是旧意义上的 controller。

它只是一个轮次级摘要，包含：

- 当前是第几轮
- 上一轮 feedback 的支持/挑战统计
- 当前 active edge ratio
- global summary 文本摘要

它的作用更接近“轮次上下文容器”，而不是“显式指挥官”。

### 5.2 为所有节点预构建局部 latent

对每个 task node：

1. 从私有记忆池取本节点历史记录
2. 用 selector 选取少量局部 records
3. 用 composer 编成局部 latent

这一步是全图做的，不依赖 node activation。

### 5.3 用 GNN 对边打分

之后进入 `_activate_edges_v2(...)`。

每条边的 gate 由下面几类信息共同决定：

- `src local_latent`
- `dst local_latent`
- edge features
- `global_node` 当前状态

如果 `gnn_enabled=True`，则直接调用 `self.gnn.edge_gate(...)`。

边分数出来后，仍然会做逐轮 pruning：

- 前几轮更宽松
- 后几轮更严格
- 每个目标节点只保留有限条入边

所以图仍然是“逐轮收缩”的。

### 5.4 所有 task nodes 每轮都执行

这是这次修正后的关键点。

当前 `Stage2RuntimeV3` 显式覆盖了 `_active_task_nodes_v2(...)`，直接返回全部 task nodes。

也就是说：

- 不再根据 active edges 再做一次 node 级裁剪
- `skipped_node_ids` 现在应稳定为 0
- 每轮真正被执行的 node 数应等于 `task_node_count`

边 pruning 只影响：

- 哪些邻居消息会进入当前节点
- 这些邻居消息的强弱和排序

但不再直接决定“节点这一轮能不能运行”。

### 5.5 节点执行

每个节点执行时会做：

1. 读取本地 selected memory
2. 从活跃入边的上游 export 中聚合邻居 latent
3. 再与 `global node` 上下文融合
4. 把 latent 变成 prompt brief
5. 调用 tier2 LLM 生成本轮输出

因此当前 prompt 的信息来源是：

- 本节点私有记忆
- 经过 edge gate 筛选后的邻居消息
- global node 的全局摘要

### 5.6 记忆回写与 export

节点输出后会写回：

- 自己的 `MemoryRecord`
- 面向邻居的 `ExportedMemoryMessage`

这个 export 是图上跨节点传播的唯一显式载体。

### 5.7 reviewer feedback 抽取

每轮结束后系统会从 reviewer 角色输出里抽取 `FeedbackEvent`，例如：

- `pass`
- `challenge`
- `uncertain`

这些事件一方面会进入私有记忆，另一方面也会被 `v3` 用于 candidate bank 聚合和 reviewer calibration。

### 5.8 更新 global node

最后用本轮的 node outputs / exports 更新 `global node`，并生成新的全局摘要，供下一轮 edge gate 和 prompt brief 使用。

## 6. Candidate Bank

所有轮次结束后，`v3` 不直接从最后一轮某个 sink output 取答案，而是显式构建 candidate bank。

bank 的来源包括：

- 候选角色输出
- sink node 输出
- `stage1_anchor_output`

每个 candidate entry 会记录：

- 文本本身
- digest
- 出现次数
- 来自哪些 node / role
- 出现于哪些 turn
- sink support 次数
- reviewer feedback 原始计数
- reviewer feedback 校准后的权重和
- 平均 reviewer trust
- 是否是 stage1 anchor
- 对代码任务是否 `parse_ok`
- 对代码任务是否 `entry_point_ok`

这一步把分散在多轮、多节点中的候选输出汇总成一个可学习排序问题。

## 7. 三个 learned 头

### 7.1 candidate_model

`candidate_model` 接收每个 entry 的聚合特征，学习输出一个 candidate score。

特征包括：

- occurrence_count
- sink_support
- turn coverage
- calibrated pass / challenge / uncertain
- reviewer trust
- 文本长度与行数
- 是否 stage1 anchor
- 代码格式约束是否满足
- source roles

最终 `candidate_model_score` 会成为 candidate bank 排序的主信号。

### 7.2 reviewer_model

`reviewer_model` 不再把 reviewer 反馈当作“同权投票”。

它会根据 reviewer event 的特征学习一个 trust：

- reviewer 来源
- event 类型
- target role
- 是否 sink
- 文本 detail 长度
- 对代码候选的 `parse_ok / entry_point_ok`

这个 trust 会和原始 feedback confidence 相乘，得到 calibrated weight。

因此：

- pass 不再天然等于 pass
- challenge 也不再天然等于 challenge
- 不同 reviewer、不同目标对象的评论权重可以不同

### 7.3 override_model

`override_model` 负责比较：

- 当前最优 stage2 candidate
- stage1 anchor

它学习输出“是否应该覆盖 stage1”。

比较特征包括：

- challenger / anchor 的 candidate score
- 两者出现次数与 sink support
- 两者 review balance
- 两者 calibrated pass/challenge
- 对代码任务的 parse / entry-point 合法性
- 两者模型不确定性

## 8. 最终答案选择

最终选择逻辑分任务类型：

### 8.1 数值任务

如果最佳 stage2 candidate 和 stage1 anchor 相同，则直接保留 anchor。

否则调用 `override_model` 判断是否覆盖 stage1。

### 8.2 代码任务

代码任务比数值任务多两条硬约束：

- `parse_ok`
- `entry_point_ok`

如果 stage2 候选不合法，则不能覆盖合法的 stage1 anchor。

如果 stage1 非法而 stage2 合法，则允许 stage2 覆盖。

在两者都合法时，再交给 `override_model` 做 learned gate。

### 8.3 通用文本任务

逻辑与数值任务类似，但没有代码硬约束。

## 9. 学习流程

当前 `Stage2RuntimeV3.learn_from_run(...)` 有两层学习。

### 9.1 runtime_v2 的 LMPO 学习

先调用 `runtime_v2.learn_from_run(...)`。

这里更新的是：

- selector / brief 构建过程里产生的离散 policy 项
- edge keep 选择里的离散 sampling 项

也就是 `v2` 本身那套 LMPO 闭环仍然保留。

### 9.2 v3 自己的三类更新

在 LMPO 之后，`v3` 再做三类监督：

1. `candidate_updates`
2. `override_updates`
3. `reviewer_updates`

候选监督来自对 candidate 文本本身做 evaluator 打分。

override 监督来自：

- challenger target 是否优于 anchor target

reviewer 监督来自：

- 该 reviewer event 是否与 candidate 真实质量方向一致

因此当前 `v3` 的学习信号已经从“调一堆 heuristic weight”转成了：

- 图执行层：LMPO
- 最终选择层：candidate / override / reviewer 三头监督

## 10. 可观测性与日志

当前日志和结果里会记录：

- `v3_stage1_anchor_used`
- `v3_candidate_count`
- `v3_selection_reason`
- `v3_selected_candidate_source`
- `v3_selected_support_score`
- `v3_selected_model_uncertainty`
- `v3_override_probability`
- `v3_override_uncertainty`
- `v3_candidate_updates`
- `v3_override_updates`
- `v3_reviewer_updates`
- `v3_all_task_nodes_each_turn`

其中最重要的一条新增校验位是：

- `v3_all_task_nodes_each_turn = true`

它用来明确区分当前修正后的 `v3` 和之前那个错误继承旧壳的版本。

## 11. 当前版本相对旧误接版本的差异

修正后的 `v3` 和之前那个“误接到 base runtime”的版本相比，有三点本质变化：

1. 基底从 `Stage2Runtime` 改为 `Stage2RuntimeV2`
2. 不再走 `GlobalController + edge_model + controller_model` 主路径
3. 不再显式做 `active node` 裁剪，而是每轮全量 task nodes 执行

因此现在的 `v3` 更接近你已经确认的目标设计：

`每题固定 stage1 图 + private memory + global node + edge-only graph gating + per-turn pruning + learned final selection`

## 12. 当前限制

虽然这版已经比之前干净很多，但仍然有几个限制：

1. `Stage2V3Config` 仍然继承 `Stage2V2Config`
   - 所以配置打印中还会带出部分旧字段

2. runtime 类型名里仍然沿用 `ControllerState`
   - 但它现在只是 turn summary 容器，不是旧 controller

3. 边仍然是 “edge gating + pruning”
   - 还不是更强的端到端图策略网络

4. 三个最终选择头目前仍是 `OnlineLinearModel`
   - 稳定、便宜，但表达能力有限

## 13. 当前最准确的总结

当前 `stage2 v3` 的最准确定义是：

`在 stage1 固定图上，使用 runtime_v2 的 latent / GNN / global-node 图执行壳进行多轮全节点执行，用 edge gate 控制消息传播，再用 candidate_model、reviewer_model、override_model 完成最终答案选择。`
