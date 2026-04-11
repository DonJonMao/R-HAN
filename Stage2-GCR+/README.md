# Stage2-GCR+ (Revised)

更新时间：2026-04-11

## 1. 文档定位

这份文档定义新的 `Stage2-GCR+ (Revised)` 主线方案。

它的目的不是继续扩展一个越来越宽的 `stage2` 点子集合，而是把当前已经讨论清楚的设计收敛成一版**可实现、可验证、不会在实现过程中悄悄变题**的正式规格。

这版方案明确服务于下面这个两阶段主线：

- `Stage1`：`MCTS / Tree Search + Topology Set Selection + UnionGraph Merge`
- `Stage2`：在**冻结的每题专属 `UnionGraph`** 上，用 `typed memory + GlobalNode + edge-only sparse gating + per-turn pruning` 优化执行信息流，并只在 verifier 明确判定为可恢复时进入**图内局部恢复**

这版文档明确拒绝以下偏移：

- 把 `Stage2` 做成脱离 `UnionGraph` 的第二系统
- 把 repair 做成 free-form self-correction 主壳
- 用一套跨任务的硬编码加权总分同时决定 rerun、collapse、recovery 和 final selection
- 用多目标加权优化把所有模块耦成一个无法诊断的黑盒

因此，这份文档只保留以下几类思想，并且只保留其必要落点：

- `DAR`：只用于 `class collapse`
- `PAMAS`：只用于 `typed memory` 和分层可见性
- `CortexDebate`：只用于图上的稀疏激活
- `CalibraEval`：只用于最终 tie-case 的轻量去偏
- `MALT`：只用于 `Code / Full` 模式下的小规模浅层分支扩张

它们都不是被整篇照搬，而是被约束在一个统一的 graph-native 主壳内。

## 2. 一句话定义

`Stage2-GCR+ (Revised)` 是一个：

`graph-native rerun + typed memory + class collapse + verifier-triggered local recovery + hard final guard`

的二阶段系统。

其主流程只有一条：

```text
冻结 UnionGraph 上 rerun
-> typed memory 构造 local latent
-> GlobalNode 聚合轮次级全局状态
-> GNN 产生 edge-only sparse gate
-> per-turn pruning / active subgraph
-> 生成 raw candidate bank
-> route-specific collapse classes
-> 选当前 champion class
-> typed verifier 判断 recoverable / non-recoverable
-> 若可恢复，则在图内 recovery subgraph 上做局部恢复
-> recovery outputs 回流 candidate bank
-> re-collapse / re-select / re-verify
-> hard guard against stage1 anchor
```

这条链条里的每一段都必须可单独验证、可单独替换、可单独失败归因。

## 3. 四个顶层硬边界

这四条不是建议，而是方法定义本身的一部分。

### 3.1 `UnionGraph` 是唯一 substrate

`Stage2` 的一切 rerun、active subgraph、message passing、candidate provenance 和 recovery subgraph，都必须建立在 `Stage1` 产出的该题 `UnionGraph` 上。

允许做的事情：

- 基于现有边做 edge gating
- 基于现有边做 per-turn pruning
- 基于现有节点和现有边诱导出 local recovery subgraph
- 基于现有图的统计先验决定 rerun 行为

不允许做的事情：

- 在 `Stage2` 中生成新的 task nodes
- 在 `Stage2` 中创建新的自由 task edges
- 构造脱离 `UnionGraph` 的 repair graph 作为并行主壳
- 让 recovery 使用一个与原图无关的自由候选空间

`Stage1` 提供的图不是上下文装饰，而是 `Stage2` 的唯一合法运行骨架。

### 3.2 recovery outputs 必须绑定到原图节点的 `node-turn provenance`

任何 recovery 输出都必须带完整 provenance，并且 provenance 必须能回溯到原图中的具体执行位置。

每个 recovery candidate 至少必须记录：

- `origin_node_id`
- `origin_turn_index`
- `origin_role`
- `parent_candidate_digest`
- `recovery_subgraph_node_ids`
- `recovery_subgraph_edge_ids`
- `trigger_verifier_snapshot`
- `repair_operator_type`

没有 provenance 的 recovery output，一律视为非法 candidate。

这条边界的目的，是防止实现退化成：

`图给上下文 -> 外部 repair prompt 自由生成 -> 结果回灌`

一旦出现这种形态，`UnionGraph` 的方法地位就会再次下降为辅助上下文，而不是 substrate。

### 3.3 recovery outputs 必须 reinsert

任何 recovery output 都不能直接变成 final answer。

它必须重新进入：

```text
candidate bank
-> class collapse
-> champion re-selection
-> typed verifier
-> final guard
```

这条边界保证 recovery 的方法角色只是：

- 一个受 verifier 触发的 candidate generation enhancement

而不是：

- 一个能够绕开主壳的 privileged override channel

### 3.4 graph-faithfulness 必须显式评测

`Stage2-GCR+` 不能只看最终 reward / success。

它必须显式评测“图到底有没有被真正使用”，否则即使最后分数上升，也不能证明方法主线是成立的。

因此 `graph-faithfulness` 不是附加分析，而是主评测维度之一。

至少必须包含以下指标和对照实验：

- `active_edge_ratio_by_turn`
- `active_node_ratio_by_turn`
- `candidate_provenance_coverage`
- `sink_path_provenance_length`
- `recovery_subgraph_size`
- `recovery_subgraph_support_profile`
- `final_answer_source_type`
- `anchor_vs_graphless_ablation`
- `anchor_vs_random_pruned_graph_ablation`
- `anchor_vs_flat_star_graph_ablation`

如果去掉图、打乱图、或强烈弱化图之后，系统表现几乎不变，则说明当前实现没有真正利用 `UnionGraph`。

## 4. 两条实现级硬约束

在实现层面，再额外加两条代码级 invariant。

### 4.1 `reinsert` 是强 invariant，不是约定

任何 recovery output 如果未经过：

`reinsert -> collapse -> reselection -> verifier -> final_guard`

则在运行时直接判定为无效，禁止参与最终输出。

这条规则应在 final decision 入口处做硬检查，而不是靠调用者自觉遵守。

### 4.2 `graph-faithfulness` 结果必须进入每次运行的 metadata

所有正式运行都必须把 `graph-faithfulness` 相关指标写入结果 metadata 和实验日志。

不能只在“需要写论文时”临时补采。

如果一个实现版本没有稳定产出这些指标，则它不能被视为完整的 `Stage2-GCR+` 实现。

## 5. 输入、输出与核心对象

### 5.1 输入

单题输入为冻结的一阶段产物：

`A = (G_union, anchor, priors, structure_summary, stage1_signature)`

其中：

- `G_union`：该题专属的 `UnionGraph`
- `anchor`：`Stage1` 最终输出
- `priors`：节点和边上的结构先验统计
- `structure_summary`：一阶段的结构摘要
- `stage1_signature`：一阶段结构签名

默认工程流程是：

1. `Stage1` 为每题生成图
2. 保存为 `PreparedStage1Artifact`
3. `Stage2` 优先复用 prepared artifact
4. `Stage2` 训练时尽量不 live rerun `Stage1`

### 5.2 输出

最终输出只有一个：

- `final_answer`

它只能来自：

- `stage1 anchor`
- 或者通过最终 hard guard 的 `Stage2` class representative

`Stage2` 不允许因为“生成了看起来更好的新候选”就直接覆盖 `anchor`。

### 5.3 运行时核心状态

每轮维护：

`S_t = {h_v^t, M_v^t, y_v^t, e_{u->v}^t, g_t, C_t}`

其中：

- `h_v^t`：节点状态
- `M_v^t`：节点私有记忆
- `y_v^t`：节点本轮输出
- `e_{u->v}^t`：沿 active edge 传播的导出消息
- `g_t`：全局节点状态
- `C_t`：当前候选类集合

## 6. 节点运行时类型映射

`Stage2-GCR+` 不改 `UnionGraph` 拓扑，但必须给每个 task node 赋一个运行时类型：

- `proposal`
- `checker`
- `aggregator`
- `sink`

这一步不使用硬编码加权总分，而使用明确的层级规则。

### 6.1 输入信号

节点类型映射只允许使用以下四类输入：

- `role` 名称
- 是否为 `sink`
- 到最近 `sink` 的拓扑距离
- `Stage1 support` 统计

不引入额外 learned scorer。

### 6.2 映射规则

按以下顺序确定类型：

1. 若节点属于 `sink_node_ids`，则类型为 `sink`
2. 否则若 `role` 属于 checker 集合，则类型为 `checker`
3. 否则若节点满足以下全部条件，则类型为 `aggregator`
   - 不是 `sink`
   - 不是 `checker`
   - 到最近 `sink` 的拓扑距离处于靠近 sink 的最小壳层
   - 在同壳层非 checker 节点中具有较高 `Stage1 support`
4. 其余节点类型为 `proposal`

### 6.3 checker 集合

初版 checker 集合可由 role 名映射：

- `tester`
- `verifier`
- `critic`
- `judge`
- `checker`

### 6.4 aggregator 的定义方式

`aggregator` 不通过打分加权，而通过“靠近 sink 的拓扑壳层 + 高 support 排序”来确定。

更具体地说：

- 先在所有非 sink、非 checker 节点里计算到 sink 的最短距离
- 找到最靠近 sink 的一层或两层壳层
- 在该壳层中按 `Stage1 support` 排序
- 选出本题实际承担中间汇总作用的节点作为 `aggregator`

这保证 `aggregator` 的定义同时反映：

- 图结构位置
- 一阶段结构统计强度

而不是只看 prompt 角色名字。

## 7. 记忆系统：四个物理桶，typed views

### 7.1 四个物理 memory 桶

初版只保留四个物理桶：

- `self_output`
- `feedback`
- `class_summary`
- `repair_trace`

这是刻意收缩后的设计，不再把过多 typed state 直接做成物理存储桶。

### 7.2 `feedback` 的内部 typed view

`feedback` 桶内部再提供两种 view：

- `stable_view`
- `failure_view`

二者不是物理桶，而是 `feedback` 上的 typed filtering。

`stable_view` 包含：

- `pass`
- `preserve`
- `keep`
- 其它等价的稳定正信号

`failure_view` 包含：

- `challenge`
- `reject`
- `conflict`
- `revise`
- 其它等价的失败或不稳定信号

此外明确约定两个逻辑摘要视图：

- `checker verdict summary := view(feedback)`
- `recovery summary := view(repair_trace)`

它们只是从现有物理桶上取的聚合视图，不新增任何物理桶。

### 7.3 每轮写回规则

每轮结束后：

- 节点输出写入 `self_output`
- 审核类节点产生的 verdict 写入 `feedback`
- collapse 后的类代表与摘要写入 `class_summary`
- recovery 过程中的 patch / refine / block-fix provenance 写入 `repair_trace`

### 7.4 不允许的记忆设计

以下设计在这版中明确不采用：

- `query_similarity + recency + feedback_bias + role_bonus` 的 heuristic 加权总分
- 跨桶加权竞争
- 为 selector 设计统一多目标优化目标
- 让 memory selector 同时承担 retrieval、routing、ranking 和 final override 的职责

`memory` 只负责 retrieval 和局部状态支持，不负责一切。

## 8. Typed memory retrieval

### 8.1 slot 模板

不同节点类型读取不同 slot 模板。

`proposal` 读取：

- 1 条 `self_output`
- 1 条 `feedback`
- 1 条 `failure_view`
- 1 条 `stable_view`
- 最多 2 条 active neighbors 的 exported messages

`checker` 读取：

- `anchor class summary`
- `champion class summary`
- 1 条最近 `feedback`
- 最多 1 条 `failure_view`

`aggregator` / `sink` 读取：

- `class_summary`
- `checker verdict summary`
- `recovery summary`

这里再次强调：

- `checker verdict summary` 来自 `feedback` 的聚合视图
- `recovery summary` 来自 `repair_trace` 的聚合视图
- schema 仍然只有 4 个物理桶

### 8.2 slot 内选择函数

slot 内部不做硬编码加权，而只做兼容性检索：

`sel_tau(v,t) = TopR_{m in M_v^tau} cos(q_{v,tau}^t, k_tau(m))`

其中：

- `tau` 是 slot 类型
- `q_{v,tau}^t` 是该节点在该 slot 的查询向量
- `k_tau(m)` 是 record 的 key 表示

只按相似度检索 top-r，不引入跨目标的加权总分。

### 8.3 selector 的职责边界

selector 只决定：

- 哪些 typed records 进入当前 slot

selector 不决定：

- 最终 candidate 排序
- final answer
- override
- class collapse

## 9. Local latent 与 `memory brief`

### 9.1 typed latent 组合

slot 取出记录后，不直接拼成长文本，而是先形成 typed latent：

`z_v^t = GRU_{rho(v)}(z_v^{t-1}, [e_self, e_fb, e_stable, e_fail, e_class, g_t])`

其中：

- `e_self` 来自 `self_output`
- `e_fb` 来自 `feedback`
- `e_stable` 来自 `feedback.stable_view`
- `e_fail` 来自 `feedback.failure_view`
- `e_class` 来自 `class_summary`
- `g_t` 来自 `GlobalNode`

### 9.2 latent 到 LLM 的方式

这版仍然采用：

- `latent-guided prompt verbalization`

而不是：

- `direct latent injection into LLM hidden states`

也就是说，latent 先决定最该暴露给 prompt 的少量内容，再 verbalize 成短 `memory brief`。

### 9.3 `memory brief` 的职责

`memory brief` 只服务于：

- 当前轮 prompt
- exported message 生成
- edge gate 支持

它不直接作为 final selection 依据。

## 10. GlobalNode：只保留全局聚合，不保留 controller

这版只保留 `GlobalNode`。

明确移除以下 controller-heavy 设计：

- `GlobalController`
- `focus`
- `mode = explore/refine/finalize`
- `role_weights`
- controller linear model

`GlobalNode` 只承担三件事：

1. 聚合当前轮 active nodes 的全局状态
2. 为下一轮提供全局上下文
3. 作为 edge gating 的输入之一

它不是显式指挥官，不负责任何“高层文字控制”。

## 11. GNN：只做 edge-only sparse control

### 11.1 输入

每个节点的初始状态：

`h_v^0 = W_h[r_v, pi_v, z_v^0, phi_v^{t-1}]`

其中：

- `r_v`：节点类型或角色 embedding
- `pi_v`：来自 `Stage1` 的节点先验
- `z_v^0`：当前轮 local latent
- `phi_v^{t-1}`：上一轮 feedback 摘要

### 11.2 边 gate

每条边输出：

`gamma_{u->v}^t = phi([h_u^t, h_v^t, h_u^t ⊙ h_v^t, |h_u^t - h_v^t|, g_t, e_uv, pi_uv])`

该输出只服务于 edge gating，不做统一多目标总分。

### 11.3 pruning 规则

对每个目标节点 `v`，按 `gamma_{u->v}^t` 做 `per-destination top-k pruning`：

- 第 1 轮：最多保留 3 条入边
- 后续轮：最多保留 2 条入边
- `sink` / `protected nodes` 保留兜底边

这里使用离散预算，不使用连续加权总分。

### 11.4 active nodes

active set 由以下几类并集组成：

- active edge endpoints
- `sink` nodes
- `protected nodes`
- `recovery_ids`

其中两个实现时必须写死的集合定义为：

- `sink_guards = sink node ids ∪ nearest upstream aggregators`
- `protected_ids = sink node ids ∪ current champion provenance nodes ∪ checker nodes involved in current verifier state`

其中 `recovery_ids` 来自最近一轮被：

- `challenge`
- `reject`
- `conflict`
- `revise`

所指向的节点。

### 11.5 GNN 的职责边界

GNN 只负责：

- 稀疏边控制
- 活跃子图诱导
- 图上的局部信息传播

GNN 不负责：

- 最终答案裁决
- candidate 排序
- final override 判定

## 12. Graph-native rerun 主壳

每轮 rerun 的流程固定为：

1. 从私有记忆中按 typed slots 取记录
2. 形成 local latent
3. 聚合 `GlobalNode`
4. GNN 计算 edge gate
5. 做 per-turn pruning
6. 得到 active nodes
7. active nodes 读取 `memory brief + neighbour exports`
8. 节点 LLM 运行
9. 抽取 feedback / export / provenance 并回写 memory
10. 从规定入口收集 raw candidates

这一步是整个方法主轴。

`Stage2` 的收益首先应来自：

- 在固定图上优化信息流

而不是：

- 在强 anchor 周围做脱离图的自由修补

## 13. Candidate bank：更宽松但更稳的入口

raw candidate bank 只允许以下四类入口：

- `sink outputs`
- `checker-approved proposals`
- `checker-positive aggregators`
- `recovery outputs`

其中两个 checker 相关入口都使用离散谓词，不允许退化回阈值打分：

- `checker-approved proposals`：至少一个 checker 给出 `pass / preserve / keep / approve`，且没有 checker 给出 `reject / conflict`
- `checker-positive aggregators`：至少一个 checker 给出 `pass / preserve / keep / approve`，且没有 checker 给出 `reject / conflict`

### 13.1 为什么加入 `checker-positive aggregators`

如果只允许 `sink outputs`，candidate diversity 很容易饿死。

如果允许所有 `proposal outputs` 直接进 bank，bank 又会迅速噪声化。

`checker-positive aggregators` 提供了一条中间路线：

- 比普通 proposal 更干净
- 比只看 sink 更不容易丢失有效异议类

### 13.2 provenance 要求

所有进入 bank 的 candidate 都必须记录：

- `text` 或 `code`
- `node_id`
- `turn_index`
- `node_type`
- `role`
- `verifier_snapshot`
- `provenance`

没有 provenance 的 candidate，不允许进入 class collapse。

## 14. Class collapse：先收，再折类

### 14.1 collapse 的目的

collapse 的目标不是文本压缩，而是：

- 去掉重复多数的统计优势
- 给少数但结构独立的异议类独立生存空间
- 让后续 comparator 面向“独立类”而不是“重复次数”

### 14.2 collapse 的职责边界

`DAR` 在这版中的唯一主要落点是：

- `class collapse`

它不再扩展成跨阶段统一消息保留总机制。

### 14.3 route-specific class keys

#### Code

失败类：

`key_code_fail = (failure_kind, failing_tests_sig, patch_locus)`

通过类：

`key_code_pass = (entry_sig, normalized_code_sketch)`

#### Reasoning

`key_reason = (final_answer, claim_pattern, first_unsupported_step)`

#### Graph

`key_graph = (broken_blocks, violated_constraints, repair_locus)`

### 14.4 class representative

每类只保留一个 representative。

代表选择只用 route-specific 词典序 comparator，不使用 class size 参与主排序。

`class_size` 只保留在 metadata 中用于分析，不允许进入主 ranking。

## 15. Typed verifier

### 15.1 Code verifier

`V_code(c) = (syntax_ok, entry_ok, passed, total, failure_kind, failing_tests_sig, patch_locus)`

### 15.2 Reasoning verifier

`V_reason(c) = (final_answer, first_unsupported_step, constraint_violations, unit_mismatch, equation_conflict)`

### 15.3 Graph verifier

`V_graph(c) = (broken_blocks, violated_constraints, repair_locus)`

### 15.4 verifier 的职责

verifier 只提供 typed evidence。

它不输出统一“好坏总分”，不负责跨任务统一排序。

## 16. Champion selection：不用加权总分，只用词典序

### 16.1 Code champion

`kappa_code = (1[passed=total>0], syntax_ok, entry_ok, passed, -num_failed_examples)`

### 16.2 Graph champion

`kappa_graph = (num_satisfied_constraints, -num_broken_blocks, 1[local_consistency])`

### 16.3 Reasoning compare：`GCR-v1 stabilization policy`

这版 reasoning comparator 明确标注为：

- `GCR-v1 stabilization policy`

它是一个临时止损策略，不是 reasoning 路线的最终长期定义。

当前 comparator 只做保守比较，目标是：

- 不再负增益
- 不再高置信错翻

其保守词典序为：

`kappa_reason_v1 = (1[anchor_class], 1[no_fatal_contradiction], -num_major_contradictions, num_critical_supports)`

这意味着在当前阶段：

- reasoning 路线优先稳住 anchor
- 暂不把 recovery override 作为主力增益来源

## 17. 运行模式：`Bypass / Lean / Full`

模式只由离散谓词决定，不通过 learned mode classifier，不使用多目标加权总分。

### 17.1 `Bypass`

满足任一条件则进入：

- `anchor class` 仍是 champion
- champion 已 strict-improve over anchor
- 不存在 recoverable failure
- 当前预算紧
- 当前任务是 reasoning / graph 且无稳定 challenger class

### 17.2 `Lean`

满足：

- 存在明确 recoverable failure
- 只涉及单一局部 failure locus
- 当前任务是 code
- 一轮局部恢复有现实推进可能

### 17.3 `Full`

当前仅对 code 开启，并且满足：

- `Lean` 一轮后仍未 fully pass
- 仍存在至少一个 viable repair class
- 当前 failure 具有继续推进空间

## 18. Local recovery：图内、条件触发、结果回流

### 18.1 触发条件

只有当以下条件成立，才允许进入 recovery：

`RecoverTrigger = Recoverable(champion) OR Recoverable(anchor)`

这条规则保留两个入口：

- `champion` recoverable
- `anchor` recoverable

这样可以避免“当前 best seed 还不够强，所以明知 anchor 可修也被直接 preserve”的过保守问题。

进入 recovery 时，`best_of(c_star, anchor)` 的选择规则必须固定为：

- 若 `c_star` 可恢复，则优先修 `c_star`
- 否则若 `anchor` 可恢复，则修 `anchor`
- 否则不进入 recovery

### 18.2 recovery subgraph

recovery subgraph 只能由 `UnionGraph` 中的现有节点和现有边诱导：

`G_rec = Induce(champion_provenance ∪ recovery_ids ∪ checker_neighbors ∪ sink_guards)`

不允许在 recovery 中生成新 task node 或新 task edge。

### 18.3 当前仅对 Code 打开 recovery 主链

#### Lean recovery

- 1 个 repairer
- 1 个 checker
- 1 轮 patch 生成
- 最多 3 个 patch branches

#### Full recovery

- 2 个 repairers
- 1 个 checker
- 允许 2 层浅层扩张
- 可 restart 到下一个 verified class representative

### 18.4 Code recovery 输入约束

patch 输入只允许包含：

- 当前 champion 或 anchor 的代码
- typed verifier 输出
- failing tests summary
- patch locus
- preserve constraints

不允许：

- 全文自由重写
- 无约束生成与当前局部失败无关的大改版本

允许：

- local patch
- boundary fix
- signature-preserving modification

### 18.5 Reasoning / Graph recovery

当前阶段：

- `Reasoning` recovery 只留接口，不允许进入 final override 主链
- `Graph` recovery 只留接口，不允许进入 final override 主链

#### Reasoning recovery 接口

- 只修 `first_unsupported_step`
- 只生成少量替代子推导
- 当前仅用于日志和 ablation

#### Graph recovery 接口

- 只修当前 `repair_locus`
- 当前仅用于日志和 ablation

## 19. Recovery outputs 回流规则

所有 recovery outputs 必须：

```text
enter candidate bank
-> collapse classes
-> select champion again
-> verify again
-> final guard
```

这条链是方法定义的一部分，不能绕开。

## 20. Final guard：hard guard first, debias last

### 20.1 Code

最终只看 typed verifier 偏序：

`c > a0` 当且仅当：

- `syntax_c >= syntax_a0`
- `entry_c >= entry_a0`
- `passed_c >= passed_a0`
- 且至少一项严格更好

不使用连续加权总分。

### 20.2 Reasoning / Graph

先做 class-level conservative compare。

只有在：

- `anchor class`
- `challenger class`

仍然难分时，才触发一次轻量 debias compare：

1. 原顺序比较一次
2. 交换位置比较一次
3. 若结论一致，接受
4. 若结论冲突，视为 calibration failure，保 `anchor`

这一步只放在最后一小步，不参与主 rerun / recovery / collapse 逻辑。

## 21. 训练顺序：先局部弱标签，再自举，再更强轨迹

这版训练顺序的原则是：

- 先让信号可用
- 再让信号更强
- 不在系统尚未稳定时用其自身生成的不稳定强轨迹反过来监督自己

### Phase 0：冻结 `Stage1`

- 每题固定 prepared artifact
- `Stage2` 训练时不 live rerun `Stage1`

### Phase 1：local weak labels for memory selector

目标：

- 先让 typed slot retrieval 脱离 heuristic score

正样本：

- 被读取后 downstream 出现稳定正反馈
- 或被读取后触发 useful recovery

负样本：

- 被读取后 downstream 仍然 conflict / reject
- 或与当前节点当前问题明显无关

损失：

- per-slot contrastive retrieval loss

### Phase 2：bootstrap edge labels for edge gate

目标：

- 让 edge gate 先学会“哪些边值得保留”

正样本边：

- 位于最终 champion provenance
- 或位于 successful recovery path

负样本边：

- 被激活但未被消费
- 或只传播噪声

损失：

- per-destination BCE / ranking loss

### Phase 3：code recovery operator

当前阶段不训练新的 recovery LLM。

只做：

- prompt-level code recovery

等成功 recovery traces 足够多后，再考虑更强监督。

### Phase 4：stronger trajectory labels as later-stage refinement

只有在前面阶段稳定后，才考虑引入更强的 trajectory labels。

此阶段只允许作为后续增强阶段，不作为第一版训练入口。

### 训练设计明确不做的事情

- 不做大一统联合多目标优化
- 不做统一加权 reward 把 selector / gate / recovery / finalizer 一起绑死
- 不直接引入 RL 主线
- 不把 LLM 主体拖入 joint training

## 22. Graph-faithfulness 评测细则

这一节是主评测的一部分，不是附录。

### 22.1 运行时统计

每次运行必须记录：

- `turn_count`
- `active_edge_count_by_turn`
- `active_node_count_by_turn`
- `pruned_edge_count_by_turn`
- `candidate_count_by_source`
- `candidate_class_count`
- `candidate_provenance_coverage`
- `selected_candidate_source_type`
- `selected_candidate_provenance_length`
- `selected_candidate_recovery_bound`

### 22.2 恢复统计

如果进入 recovery，还必须记录：

- `recovery_trigger_source`
- `recovery_mode`
- `recovery_subgraph_node_count`
- `recovery_subgraph_edge_count`
- `recovery_branch_count`
- `recovery_reinserted_count`
- `recovery_selected_count`

### 22.3 结构消融

至少做以下三种对照：

- `graphless / anchor-only`
- `random pruned graph`
- `flat star graph`

预期是：

- 若 `Stage2-GCR+` 真正利用图，则这些消融应明显削弱效果或 graph-faithfulness 指标

若没有显著差异，则说明实现偏离了 graph-native 主线。

## 23. 明确删除或弱化的内容

这版方案建议从主线中移除或弱化：

- `GlobalController`
- `focus / mode / role_weights`
- controller linear model
- heuristic memory score mixing
- “GNN 只是 surviving-neighbor aggregator，而 pruning 靠外部 heuristics 主导”的过渡态
- 把 repair 当成脱离图的第二系统

## 24. 仓库中的实现落点

### `mas_stage2/memory.py`

保留：

- 私有记忆存储
- local composer
- export message builder
- verbalizer

修改方向：

- 物理桶收缩为 4 个
- `feedback` 提供 `stable/failure` typed view
- 用 typed slot retrieval 替换 heuristic 加权选择

### `mas_stage2/gnn.py`

保留：

- `LightweightGNN` 骨架

修改方向：

- 真正输出 edge-only sparse gate
- `per-destination top-k pruning` 进入 runtime 主路径
- 不再依赖 controller 逻辑决定 focus / mode

### `mas_stage2/runtime.py` 或后续 `Stage2-GCR+` 专用 runtime

至少需要承载以下职责：

- node type mapping
- typed memory read / write
- graph-native rerun
- candidate bank construction
- route-specific class collapse
- champion selection
- recovery subgraph construction
- code recovery round
- hard final guard
- graph-faithfulness logging

## 25. 最终伪代码

```text
Input: PreparedStage1Artifact A = (UnionGraph, anchor, priors), sample x

route <- AffordanceRoute(x)    # Code / Reasoning / Graph

for turn t = 1..T:
    for each node v:
        node_type_v <- ResolveNodeType(role, sink_flag, sink_distance, stage1_support)
        selected_slots_v <- TypedMemorySelect(v, M_v, current_state)
        z_v <- ComposeLocalLatent(selected_slots_v, g_{t-1})

    g_t <- GlobalNodePool({z_v})
    edge_logits <- EdgeGate(UnionGraph, {z_v}, g_t, priors)
    active_edges <- PerDestinationTopK(edge_logits)
    active_nodes <- BuildActiveSet(active_edges, sinks, protected_ids, recovery_ids)

    for v in active_nodes:
        brief_v <- Verbalize(z_v, neighbour_exports(v), g_t)
        y_v <- RunNode(v, brief_v)
        WriteBack(self_output, feedback, class_summary, repair_trace, exports)

    collect raw candidates from:
        sink outputs
        checker-approved proposals
        checker-positive aggregators
        recovery outputs

C <- CollapseClasses_route(C_raw U {anchor})
c_star <- SelectChampion_route(C)
v_c <- Verify_route(c_star)
v_a <- Verify_route(anchor)

mode <- SelectMode(route, c_star, anchor, v_c, v_a)

if route == Code and mode in {Lean, Full} and (Recoverable(v_c) or Recoverable(v_a)):
    G_rec <- BuildRecoverySubgraph(UnionGraph, c_star, anchor, recovery_ids)
    B <- RunCodeRecovery(G_rec, target = best_of(c_star, anchor), mode = mode)
    assert every b in B has node-turn provenance
    C <- CollapseClasses_code(C U B)
    c_star <- SelectChampion_code(C)
    v_c <- Verify_code(c_star)

final_answer <- FinalGuard(route, anchor, c_star)
LogGraphFaithfulnessMetrics()
return final_answer
```

## 26. 一句话总结

`Stage2-GCR+ (Revised)` 的核心不是把 `Stage2` 变成更复杂的 repair 系统，而是把它收敛成一套不会偏离方法主线的 graph-native 执行系统：

- `Stage1` 提供每题专属 `UnionGraph`
- `Stage2` 在该图上用 typed memory 和 sparse edge gating rerun
- candidate 先 collapse classes，再选 champion
- 只有 verifier 明确判定可恢复时才进入图内 local recovery
- recovery outputs 必须绑定原图 provenance，必须 reinsert，必须再经过 class collapse / verifier / final guard
- 全过程必须显式评测 graph-faithfulness

这版设计的目标不是一次性追求最强结果，而是先保证：

- 一二阶段不割裂
- 方法边界不漂
- 实现路径不震荡
- 失败归因可诊断
