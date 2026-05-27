根据你 **2026-04-07/08** 的实验总结文档，以及你当前分支里 `mas_stage2` 的目标说明，我建议把二阶段统一收敛成一个**图内主壳 + 条件恢复**的版本。你仓库当前已经把目标写得很清楚：二阶段应收敛成“**每题一张 stage1 `UnionGraph` + prepared artifact 复用 + 单题私有记忆 + 仅保留 `GlobalNode` + edge-only gating + 每轮强制 pruning + latent-guided prompt brief**”，而不是继续保留 controller 驱动的混合系统。与此同时，现有训练记录说明：`v3.1` 在 GSM8K 上会高置信错翻，而 `v4.3` 在 MBPP 上已经出现了真实有效的 code repair overturn，但覆盖率仍低、整体 reward 仍为负。([GitHub][1])  

我下面给你的，是一版**从头到尾、可以单独阅读**的 `Stage2-GCR+` 详细技术方案。它不依赖多目标加权总分，不引入一堆硬编码权重，不试图把所有顶会点子拼成大杂烩，而是只吸收五个最必要的思想：

* **DAR**：只用于候选折类
* **CortexDebate**：只用于图上的稀疏激活
* **PAMAS**：只用于 typed memory 和分层可见性
* **CalibraEval**：只用于最后一小步去偏
* **MALT**：只用于 code 的 `Full` 模式下小规模分支扩张
     

---

# Stage2-GCR+：总体定义

## 一句话定义

`Stage2-GCR+` 是一个 **graph-native rerun + class collapse + verifier-triggered local recovery + final guard** 的二阶段系统。

它的主线只有这一条：

```text
冻结 UnionGraph 上 rerun
→ GNN + typed memory 控制信息流
→ 生成 candidate bank
→ 按任务类型折类
→ 选当前 champion
→ 用 typed verifier 判断
→ 若存在 recoverable failure，则进入图内 local recovery
→ recovery 结果重新回流到 classes
→ final adjudication / hard guard against anchor
```

这条主线有三个最重要的边界：

第一，**图不变**。
二阶段不重新造自由图，只能在 `Stage1` 输出的 `UnionGraph` 上做 edge gating、pruning、active subgraph 和 recovery subgraph。这样一二阶段不会割裂。([GitHub][1])

第二，**图是唯一合法信息通道**。
原始记忆不能跨 agent 直读，跨 agent 信息只允许通过图上的 exported messages 传播。这个边界和你仓库当前目标是一致的。([GitHub][1])

第三，**repair 不是主壳，而是图内局部恢复模式**。
这点非常重要。MBPP 现有结果说明 code repair 有真实正信号，但 GSM8K 的旧 override 路径会高置信错翻，所以 repair 应被保留，但只能作为 verifier 触发的局部恢复回路，不再做成脱离图的第二系统。 

---

# 1. 输入、状态与输出

## 输入

输入仍然是 `PreparedStage1Artifact`：

[
A = (G_{\text{union}}, a_0, \Pi, S)
]

其中：

* (G_{\text{union}})：冻结的 `UnionGraph`
* (a_0)：`Stage1 anchor`
* (\Pi)：Stage1 的节点/边先验统计
* (S)：Stage1 的结构摘要与签名

你仓库当前明确建议：二阶段训练尽量复用 prepared artifact，而不是训练时 live 重跑 Stage1，这一点应该保持不变。([GitHub][1])

## 运行时状态

在第 (t) 轮，维护：

[
\mathcal{S}*t = {h_v^t,\ M_v^t,\ y_v^t,\ e*{u\to v}^t,\ g_t,\ C_t}
]

含义分别是：

* (h_v^t)：节点 (v) 的图状态
* (M_v^t)：节点 (v) 的私有 typed memory
* (y_v^t)：节点本轮输出
* (e_{u\to v}^t)：边上传播的导出消息
* (g_t)：全局节点状态
* (C_t)：当前候选等价类集合

---

# 2. 图的角色划分：不改结构，只改运行时类型

这一步不重新造图，只给现有节点赋运行时类型。

每个 task node 在 Stage2 中被标成以下四类之一：

* `proposal`
* `checker`
* `aggregator`
* `sink`

这是为了后面决定：

* 它该读什么记忆
* 它该接收什么消息
* 它是否能进入 recovery subgraph

这一步不需要再学新模型，初版完全可以根据已有 role 名和 sink 标识做静态映射：

* role 名包含 `tester / verifier / critic / judge / checker` → `checker`
* sink nodes → `sink`
* 其余默认 `proposal`
* sink 上游、且 Stage1 support 高的节点可标为 `aggregator`

这样做的意义是：
你不再需要 `GlobalController` 去显式调 `role_weights`，角色差异会通过**节点类型 + memory 可见性 + edge gating**自然体现。仓库目标文档也明确建议：不要再单独学 node importance / role weight，只学 edge gate。([GitHub][1])

---

# 3. 记忆系统：typed memory，而不是一锅 heuristic score soup

你仓库当前的 memory 骨架其实很好，`PrivateEpisodeMemoryStore`、`LocalMemoryComposer`、`ExportMessageBuilder`、`MemoryBriefVerbalizer` 都应该保留。真正该改的是 `RoleAwareMemorySelector` 里那种把 `query_similarity / recency / feedback_bias / role_bonus` 线性加权的 heuristic 排序思路。仓库目标文档也明确建议“简化记忆选择目标，只保留 relevance retrieval 与 outcome-aware bias，避免继续堆叠额外小模型”。([GitHub][1])

## 记忆类型

每个节点只维护六种 typed record：

* `self_output`
* `feedback`
* `stable_signal`
* `failure_signal`
* `class_summary`
* `repair_trace`

其中：

* `self_output`：最近几轮本节点自己的输出
* `feedback`：针对本节点或其输出的反馈事件
* `stable_signal`：`pass / preserve / keep` 一类正信号
* `failure_signal`：`challenge / reject / conflict / revise` 一类负信号
* `class_summary`：当前 class collapse 后的摘要
* `repair_trace`：只给 recovery 用，存 patch/step/block 修改的 provenance

## 记忆读取规则

这里不用任何多目标总分，只用**按角色固定 slot 模板 + slot 内相似度检索**。

### proposal 节点读取

固定读取：

* 1 条 `self_output`
* 1 条 `feedback`
* 1 条 `failure_signal`
* 1 条 `stable_signal`
* 最多 2 条来自 active neighbors 的 exported messages

### checker 节点读取

固定读取：

* `anchor class summary`
* `champion class summary`
* 1 条最近 feedback
* 最多 1 条 failure memory

### aggregator / sink 节点读取

主要读取：

* class summaries
* checker verdict summaries
* recovery summaries

这一步直接吸收 **PAMAS** 的核心，而不是它的整个框架：
不同层的节点看到的信息范围不同，底层不看全文，中层做局部聚合，顶层才整合。PAMAS 强调的正是 full-context overexposure 和 information drowning 的问题。 

## 记忆选择函数

每个 slot 内只按兼容性检索：

[
\text{sel}*\tau(v,t)=\operatorname{TopR}*{m\in M_v^\tau}\cos(q_{v,\tau}^t,\ k_\tau(m))
]

这里：

* (\tau) 是 slot 类型
* (q_{v,\tau}^t) 是节点在当前轮对该 slot 的查询向量
* (k_\tau(m)) 是 record 的 key

没有任何线性权重加总。
如果 slot 为空，只补最近一条同类 record 作为兜底。

## 记忆聚合

slot 取出来以后，不直接拼长文本，而是先形成一个 typed latent state：

[
z_v^t=\mathrm{GRU}_{\rho(v)}\big(z_v^{t-1},[e^{self},e^{fb},e^{stable},e^{fail},e^{class},g_t]\big)
]

然后只把它 verbalize 成短 `memory brief` 进 prompt。
这和你仓库当前“latent-guided prompt verbalization，而不是 direct latent injection into hidden states”的目标完全一致。([GitHub][1])

---

# 4. GNN：保留 LightweightGNN，但让它真正进入主路径

你仓库当前已经明确说了：最终目标是 **edge-only graph gating + per-turn pruning**，GNN 应该进入 edge gating / message passing 主路径，而不是只是 surviving-neighbor aggregator。([GitHub][1])

## 选择建议

这版不要换成重型 Graph Transformer。
继续用你现有的 `LightweightGNN` 思路，只做两个修改：

### 第一，输入更干净

每个节点初始状态：

[
h_v^0 = W_h[r_v,\pi_v,z_v^0,\phi_v^{t-1}]
]

其中：

* (r_v)：节点类型 embedding
* (\pi_v)：Stage1 先验
* (z_v^0)：当前 local latent summary
* (\phi_v^{t-1})：上一轮反馈摘要

### 第二，edge gate 主导 pruning

每条边输出一个 logit：

[
\gamma_{u\to v}^t = \phi([h_u^t,h_v^t,g_t,e_{uv},\pi_{uv}])
]

然后**对每个目标节点 (v)** 做 top-k incoming 边保留：

* 前两轮：`k = 3`
* 中后轮：`k = 2`
* sink / protected nodes：保留一条兜底边

这样做的好处是：

* 不需要 softmax/sparsemax 之类新算子
* 仍然能得到很强的稀疏执行效果
* 实现上也和你当前 runtime 更接近

这一步吸收的是 **CortexDebate** 的核心，不是其全部框架：
不是每个节点都值得听，不是全图所有节点每轮都要发声。([aclanthology.org](https://aclanthology.org/2025.findings-acl.495/?utm_source=chatgpt.com))

## active node 选择

继续沿用你 runtime 里的好逻辑：

* `incident_ids`
* `preserved_ids`
* `recovery_ids`
* `protected_ids`

合并成 active set。
尤其是 `recovery_ids` 已经很有价值，因为当前 runtime 已经把 `challenge / reject / conflict / revise` 这类事件映射成 recovery 触发信号。

---

# 5. rerun：图内主壳，不是为了直接给 final answer

Stage2 的每轮 rerun 执行如下：

1. 从节点私有 memory 里按 slot 读记录
2. 形成 local latent summary
3. GNN 计算 edge gates
4. 依据 top-k gate 做 pruning
5. 得到本轮 active nodes
6. 每个 active node 读 `memory brief + neighbour exports`
7. 运行节点 LLM
8. 抽 feedback 并写回 memory
9. sink / checker-approved proposal 输出 candidate

这一步仍然是主轴。
因为你想保住的一二阶段耦合，本质上就在这里：
不是 Stage1 给个 anchor、Stage2 另起炉灶，而是 Stage2 在冻结图上**重新优化信息流**。这正是你仓库当前明确的目标方向。([GitHub][1])

---

# 6. candidate bank：先收，再折类

## 收集规则

每轮只把三类东西放进原始 `candidate bank`：

* sink 输出
* checker 明确 approve 的 proposal 输出
* recovery subgraph 产生的 recovery branches

每个 candidate 都带完整 provenance：

[
c=(text/code,\ node_id,\ turn,\ role,\ verifier_snapshot,\ provenance)
]

## 为什么一定要折类

你现有总结文档已经很清楚：
`v3.1` 在 GSM8K 上之所以会高置信错翻，关键原因之一就是 `candidate bank` 里重复答案天然占优，模型学成了“像多数路线的 challenger 更可信”，而不是真能纠正 anchor 的 challenger 更可信。

所以 `candidate bank` 后必须先做 `CollapseClasses`。
这一步只吸收 **DAR** 的核心思想：去掉重复多数的统计优势，给少数异议类独立生存空间。DAR 原文强调的就是保留 informative disagreement，而不是重复多数。

---

# 7. route-specific class collapse：三类任务三种折法

## code

这条线最重要，因为当前实证最强。

### 失败候选

[
key_{\text{code-fail}}=(failure_kind,\ failing_tests_sig,\ patch_locus)
]

### 通过测试的候选

[
key_{\text{code-pass}}=(entry_sig,\ normalized_code_sketch)
]

也就是说：

* 修同一个 failure locus 的 patch 归为一类
* 只是文本改写但 verifier 行为等价的代码也归为一类

## reasoning

这版先采取最实用的保守折法：

[
key_{\text{reason}}=(final_answer,\ reasoning_signature)
]

其中 `reasoning_signature` 先别做太复杂，只保留：

* 首个关键 claim
* 简化后的中间步骤骨架
* 若有 verifier，再加 `first_unsupported_step`

## graph

[
key_{\text{graph}}=(broken_blocks,\ violated_constraints,\ repair_locus)
]

---

# 8. champion 选择：不用总分，只用词典序

## code champion

[
\kappa_{\text{code}}=(1[p=n>0],\ syntax,\ entry,\ passed,\ -|E_{fail}|)
]

也就是：

* fully pass 优先
* 再看 syntax
* 再看 entry
* 再看 passed 数
* 最后才看失败数

## reasoning champion

这版要实用，所以先走保守 comparator：

[
\kappa_{\text{reason}}=
(1[\text{anchor class}],\ 1[\text{no fatal contradiction}],\ -#major_contradictions)
]

也就是 reasoning 路径这版**anchor 优先**。
这是为了先把 GSM8K 从负增益拉回不增不减，再考虑激进翻案。

## graph champion

[
\kappa_{\text{graph}}=(#satisfied_constraints,\ -#broken_blocks)
]

没有线性加权。
只用词典序。

---

# 9. verifier：typed 输出，不再用统一模糊好坏分数

这是这版非常关键的一处。

## code verifier

[
V_{\text{code}}(c)=
(\text{syntax_ok},\text{entry_ok},\text{passed},\text{total},\text{failure_kind},\text{failing_tests_sig},\text{patch_locus})
]

这和你当前 `v4.3` 的成功经验一致：
真正起作用的是 verifier-backed patch improvement，而不是语义 override。MBPP 文档也明确说明，成功 override 主要来自 repair branch，而不是纯 verified seed。 

## reasoning verifier

[
V_{\text{reason}}(c)=
(\text{final_answer},\text{first_unsupported_step},\text{constraint_violations},\text{unit_mismatch})
]

但这版只把它用于：

* collapse signature
* conflict logging
* final conservative compare

**不用于激进 recovery 主链。**

## graph verifier

[
V_{\text{graph}}(c)=
(\text{broken_blocks},\text{violated_constraints},\text{repair_locus})
]

---

# 10. mode：Bypass / Lean / Full，但只做离散判断

你当前 `v4.4` 文档里对 `Bypass / Lean / Full` 的语义已经写得很清楚，而且它们本来就是离散预算模式，而不是连续总分驱动的。`Lean` 是一次轻量闭环，`Full` 允许多轮扩张、检查点和重启。

这版不要学新的 mode classifier，先用离散谓词即可。

## `Bypass`

满足任一条件就进入：

* champion 已 strict-improve over anchor
* anchor 仍是 champion，且无 recoverable failure
* reasoning 任务中无可靠 challenger class
* 预算紧

## `Lean`

满足：

* 存在 recoverable failure
* 只涉及单个 failure locus
* 当前任务是 code
* 一轮恢复足够可能有用

## `Full`

这版只对 code 开启，且满足：

* champion 或 anchor 有明确局部 failure
* 一轮 `Lean` 后仍未 fully pass
* 仍存在至少一个可继续推进的 recovery class

也就是说：
**`Full` 这版只服务 code。**
数学和图结构暂不开 `Full`，这是基于你现有训练记录的务实选择。

---

# 11. local recovery：图内恢复，而不是图外系统

这是整个方案的关键。

## recovery 触发条件

[
\text{RecoverTrigger}=
(m\in{\text{Lean},\text{Full}})\land(\text{Recoverable}(c^\star)\lor \text{Recoverable}(a_0))
]

这里已经包含了一个很重要的修正：

* 不是只修 champion
* 如果 champion 不可推进，但 anchor 自己有明确可修失败，也允许从 anchor 进入 recovery

这能修复“明明 anchor 自己没过 verifier，却因为没更强 seed 就直接 preserve”的过保守问题。

## recovery subgraph

恢复子图不造新图，只从 `UnionGraph` 中抽：

[
G_{\text{rec}} = \operatorname{Induce}\big(
\text{champion provenance}
\cup \text{recovery ids}
\cup \text{checker neighbors}
\cup \text{sink guards}
\big)
]

这保证 repair 仍然是图内局部模式。

## code recovery：这版真正打开的唯一恢复主链

### Lean

* 1 个 repairer
* 1 个 checker
* 1 轮 patch
* 最多 3 个 patch branches

### Full

* 2 个 repairers
* 1 个 checker
* 允许 2 轮 shallow expansion
* 中间允许 restart 到下一个 verified class rep

这一步只吸收 **MALT** 最实用的一部分：
**小规模 search-tree expansion**，而不是整套多模型训练。

patch 输入只包括：

* 当前 champion 或 anchor 的代码
* typed verifier 输出
* failing tests summary
* patch locus
* preserve constraints

不允许全文随意重写，只允许：

* local patch
* boundary fix
* signature-preserving modification

## reasoning recovery：先留接口，默认关闭

这一版 reasoning recovery 只做日志和 ablation 接口：

* 记录 `first_unsupported_step`
* 可尝试生成局部替代子推导
* 但默认不允许进入最终 override 主链

原因很简单：
GSM8K 当前历史结果不支持现在就把 recovery 当主力打开。
先止损，再求进。

## graph recovery：接口保留，但初版不开

---

# 12. recovery 结果必须回流，不允许直接强行替换

任何 recovery 输出都必须重新进入：

```text
candidate bank
→ collapse classes
→ select champion again
→ verifier again
→ final guard
```

这意味着 repair 不是一次性 override。
它只是 candidate generation 的一个强化模式。

这样你就不会再出现“一边说图是主壳，一边 repair 结果直接绕过图把答案改掉”的割裂。

---

# 13. final guard：最后一步才做裁决

## code

只看 typed verifier 偏序：

[
c \succ a_0 \iff
syntax_c \ge syntax_{a_0},;
entry_c \ge entry_{a_0},;
passed_c \ge passed_{a_0}
]

且至少一项严格更好。

## reasoning / graph

先做 class-level 保守 compare。
若仍然难分，再做一个极轻量的 **CalibraEval spirit** 去偏：

1. `anchor = A, challenger = B` 比一次
2. 交换位置再比一次
3. 若结论一致，接受
4. 若结论冲突，不 override

这一步只放在最后一小步，不参与前面的 rerun、折类、recovery 主链。CalibraEval 原文强调的也是 inference-time、label-free 的最后去偏，而不是前面的大脑。

---

# 14. 训练方案：分四阶段，不做大一统联合目标

你仓库当前的推荐收敛顺序非常合理：
固定结构输入、删 controller、把图学习收敛为 edge-only、让 pruning 真正进入主路径、简化记忆选择目标。([GitHub][1])

这版训练也只做四阶段。

## Phase 0：冻结 Stage1

* 每题固定 prepared artifact
* 训练时不 live rerun Stage1

## Phase 1：memory selector

目标：学会**按 slot 检索对的记录**

* 正样本：被读取后 downstream 产生 stable signal，或成功触发 useful recovery
* 负样本：被读取但 downstream 仍 reject/conflict，或与当前问题无关

损失：**per-slot contrastive retrieval**

## Phase 2：edge gate

目标：学会**哪条边值得保留**

* 正样本：位于最终 champion provenance 或 useful recovery path 上
* 负样本：active 但未被消费，或只传播噪声

损失：**per-destination edge ranking / BCE**

## Phase 3：code recovery operator

先不训大模型，只做 prompt-level recovery。
等有足够成功轨迹，再考虑 very small patch-SFT。

## Phase 4：轻量 joint refinement

只 joint tune：

* memory selector
* edge gate

不 joint tune主 LLM。

---

# 15. 当前版本的任务优先级

## 优先级 1：MBPP / HumanEval

这是你现在最应该打穿的一条线。
现有日志已经证明：

* repair 有真实正信号
* 但覆盖率不够，reward 和效率还没成立。 

这条线应该完整落地：

* typed memory
* edge-only sparse gating
* class collapse
* code local recovery
* reinsert
* hard guard

## 优先级 2：GSM8K / MATH

目标先别定成“立刻提分”，先定成：

* 不再负增益
* 不再高置信错翻

所以这版对 reasoning 的策略是：

* rerun
* collapse classes
* sparse activation
* anchor-first conservative compare
* optional final swap-check

**先稳，再开 recovery。**

---

# 16. 你现在应该删掉什么

这一版建议明确弱化或删除：

* `GlobalController`
* `focus / mode / role_weights`
* `controller linear model`
* `RoleAwareMemorySelector` 里的 heuristic 加权排序
* “GNN 只做 surviving-neighbor aggregation，pruning 靠外部 heuristics” 这套过渡态
  ([GitHub][1])

---

# 17. 最终伪代码

```text
Input: PreparedStage1Artifact A = (UnionGraph, anchor, priors), sample x

# 1. route protocol family
p <- AffordanceRoute(x)        # Code / Reasoning / Graph

# 2. graph-native rerun
for turn t = 1..T:
    for each node v:
        retrieve typed slots from private memory M_v
        compose local latent z_v^t
    g_t <- GlobalNode({z_v^t})

    edge_logits <- LightweightGNN(UnionGraph, {z_v^t}, g_t, priors)
    active_edges <- TopKPerDestination(edge_logits)
    active_nodes <- Endpoints(active_edges) ∪ protected_ids ∪ recovery_ids

    for v in active_nodes:
        brief_v <- Verbalize(z_v^t, neighbor_exports(v), g_t)
        y_v^t <- RunNode(v, brief_v)
        write feedback and exports back to memory

    collect raw candidates from sinks / checker-approved proposals

# 3. collapse classes
C <- CollapseClasses_p(C_raw ∪ {anchor})

# 4. choose champion
c* <- argmax_{c in C} kappa_p(c)

# 5. typed verifier
v_c <- Verify_p(c*)
v_a <- Verify_p(anchor)

# 6. choose mode
m <- ModeByPredicates(c*, anchor, v_c, v_a)

# 7. local recovery
if m in {Lean, Full} and (Recoverable(v_c) or Recoverable(v_a)):
    G_rec <- BuildRecoverySubgraph(UnionGraph, c*, anchor, recovery_ids)
    B <- RunRecovery_p(G_rec, target = best of {c*, anchor}, mode = m)
    C <- CollapseClasses_p(C ∪ B)
    c* <- argmax_{c in C} kappa_p(c)

# 8. final adjudication
y <- FinalGuard_p(anchor, c*)
return y
```

---

# 最后一段总结

如果把这版 `Stage2-GCR+` 压成最短总结，就是：

> **它把你仓库当前已经明确的 graph-native 目标真正落地：Stage1 提供每题专属 `UnionGraph`，Stage2 在这张图上用 typed memory 和 sparse edge gating rerun 产生候选；再用 DAR 式 class collapse 去重；随后只在 verifier 判定为可恢复时进入图内 local recovery；最终通过 conservative guard 与 optional debias 决定是否相对 anchor 覆盖。**

这版最重要的优点有三个：

* 一二阶段不割裂
* code 上保住了已有正信号
* GSM8K 不会因为过早强行 opening recovery 再次被打崩

如果你下一步真要开工，我建议第一批只做三件事：
**先去掉 heuristic memory score、把 edge gate 变成真正的主 pruning 路径、把 code recovery 改成 recovery-subgraph + reinsert。**

[1]: https://raw.githubusercontent.com/DonJonMao/R-HAN/feat/tree-multigraph-upgrade-base/mas_stage2/STAGE2_CURRENT_VS_TARGET.md "raw.githubusercontent.com"


According to a document from 2026-04-08, the current MBPP line already shows real **code-repair** gains but still has negative average reward; according to the current `feat/tree-multigraph-upgrade-base` branch, the intended Stage2 direction is a **graph-native** runtime built around a per-question `UnionGraph`, private memory, a `GlobalNode`, and edge-only gating rather than a controller-heavy second system. Based on that, the cleanest next design is the following **Stage2-GCR+**.   ([GitHub][1])

# Stage2-GCR+：从头到尾的完整设计

## 1. 设计目标

Stage2-GCR+ 只做一件事：**在冻结的 `UnionGraph` 上，把一阶段已经很强的 `anchor` 再优化一轮，但不把 Stage2 变成一套脱离图的第二系统。**

它的基本判断来自两个现实约束。第一，当前 `v4.3` 的正信号主要来自 **code repair**，说明“围绕局部失败点做 verifier-guided 修复”对代码题是有效的；第二，历史上的 GSM8K 负增益说明“在强 `anchor` 条件下，光靠 rerun 后的 pairwise/reviewer 线路很容易高置信错翻”。所以，Stage2 既不能只做旧式 dense rerun，也不能继续把 repair 做成图外主线。  

因此，Stage2-GCR+ 的总原则是：

[
\text{Frozen UnionGraph}
\rightarrow
\text{Graph-native rerun}
\rightarrow
\text{Class collapse}
\rightarrow
\text{Champion selection}
\rightarrow
\text{Verifier-triggered local recovery}
\rightarrow
\text{Final guard against anchor}
]

这里统一的是**控制流程**，不统一的是**恢复算子**。代码题允许 patch recovery；数学题预留 step-refine 接口但默认不启用；图结构题预留 block-fix 接口。这样做的依据不是数据集名字，而是任务的**可验证结构**，这比“按 benchmark 名字切 if-else”更稳，也更符合当前 modular / hierarchical 的主流方向。([GitHub][2])

---

## 2. 输入、输出与核心对象

### 2.1 输入

Stage2-GCR+ 的输入是每题一份冻结的一阶段产物：

[
A = (G_{\text{union}}, a_0, \Pi, S)
]

其中：

* (G_{\text{union}})：该题的冻结 `UnionGraph`
* (a_0)：Stage1 的最终输出，也就是 `anchor`
* (\Pi)：Stage1 产生的图先验
* (S)：结构摘要 / signature

这和你当前分支的目标一致：**每题一张 stage1 union graph，支持缓存复用，Stage2 不脱离 Stage1 独立存在。** ([GitHub][2])

### 2.2 输出

最终输出仍然是：

[
y^\star
]

它要么是 `anchor`，要么是一个通过最终 guard 的 Stage2 候选。
Stage2-GCR+ 不追求“只要生成了新答案就覆盖”，而是要求：**只有在 typed verifier 下，新的候选对 `anchor` 形成严格改进时，才允许 override。** 这点和你当前 code line 的成功逻辑一致。

### 2.3 运行时核心状态

每轮维护：

[
\mathcal{S}*t={h_v^t, M_v^t, y_v^t, e*{u\to v}^t, g_t, C_t}
]

其中：

* (h_v^t)：节点状态
* (M_v^t)：节点私有记忆
* (y_v^t)：节点本轮输出
* (e_{u\to v}^t)：沿 active edge 传播的导出消息
* (g_t)：全局节点状态
* (C_t)：当前候选等价类集合

---

## 3. 整体运行流程

### 3.1 主流程

Stage2-GCR+ 的一次完整运行，分成六段。

第一段，在冻结 `UnionGraph` 上 rerun。
第二段，收集原始 candidate bank。
第三段，对 candidate bank 做 route-specific class collapse。
第四段，选出当前 `champion class`。
第五段，如果 verifier 认为它存在**可局部恢复的失败**，就在图内 recovery subgraph 上做 local recovery。
第六段，对 recovery 结果重新折类、重新选 champion，最后做 hard guard against anchor。

用伪代码写就是：

```text
Input: PreparedStage1Artifact A=(UnionGraph, anchor, priors), sample x

p <- ProtocolFromAffordance(x)

for t = 1..T:
    run graph-native rerun on UnionGraph
    collect raw candidates C_raw

C <- CollapseClasses_p(C_raw ∪ {anchor})
c* <- SelectChampion_p(C)

v_c <- Verify_p(c*)
v_a <- Verify_p(anchor)

if Recoverable(v_c) or Recoverable(v_a):
    B <- LocalRecovery_p(UnionGraph, c*, anchor, verifier_state)
    C <- CollapseClasses_p(C ∪ B)
    c* <- SelectChampion_p(C)

y <- FinalGuard_p(anchor, c*)
return y
```

这个流程看上去简单，但它把图、memory、折类、恢复和 guard 放到了同一条链上，不再像旧 `v4.3/v4.4` 那样在“图 rerun”和“图外 repair”之间摇摆。 ([GitHub][2])

### 3.2 任务协议的划分

为了实用和提分优先，协议族先只保留三类：

* `Code`
* `Reasoning`
* `Graph`

但当前版本的启用策略要更保守：

* **Code**：完整启用 local recovery
* **Reasoning**：先只启用 rerun + collapse + guard，不启用 recovery override
* **Graph**：先只启用 rerun + collapse + guard，block-fix 接口先预留

这么做不是保守过度，而是直接响应你的训练记录：**当前硬正信号只在 code line 上出现，reasoning line 暂时更需要“止损和稳住”而不是激进恢复。**  

---

## 4. 节点类型与图内角色

Stage2-GCR+ 不改 `UnionGraph` 拓扑，但给每个节点赋一个运行时类型：

* `proposal`
* `checker`
* `aggregator`
* `sink`

这个类型不是新学一个复杂模块，而是从已有 role 名和图位置映射出来：

* `tester / verifier / critic / judge / checker` 这类角色 → `checker`
* sink 节点 → `sink`
* 其余普通任务节点 → `proposal`
* 靠近 sink、且在 Stage1 里先验较强的中间节点 → `aggregator`

这样做的目的是：**不同类型节点看不同记忆、承担不同职责。** 这正好吸收了 PAMAS 的核心：底层只看局部视角，中层做聚合，顶层才最终判断，而不是所有节点都看全文、做同样的事。 

---

## 5. 记忆系统：typed memory，而不是 heuristic score soup

你当前分支里的 `memory.py` 骨架是对的：已经有私有记忆、局部 latent、导出消息、memory brief 这些好东西；真正需要改的是**选择机制**。现在的 heuristic `query_similarity + recency + feedback_bias + role_bonus` 不够稳，也太容易变成工程化调分。([github.com](https://github.com/DonJonMao/R-HAN/blob/feat/tree-multigraph-upgrade-base/mas_stage2/runtime.py)) ([GitHub][3])

### 5.1 记忆类型

每个节点只维护 6 类 record：

* `self_output`
* `feedback`
* `stable_signal`
* `failure_signal`
* `class_summary`
* `repair_trace`

不是所有节点都用到全部 6 类，但 schema 统一。

### 5.2 记忆写回

每轮结束后：

* 节点自身输出写成 `self_output`
* checker/judge/critic 的反馈写成 `feedback`
* 若反馈是 `pass / preserve`，额外写入 `stable_signal`
* 若反馈是 `challenge / reject / conflict / revise`，额外写入 `failure_signal`
* collapse 之后的 class 代表和 class 摘要写入 `class_summary`
* recovery 模式中产生的 patch / refined step / block-fix 记录写入 `repair_trace`

这一步和你当前 runtime 里的 feedback 提取逻辑是兼容的。现有实现已经会把 `critic / verifier / judge` 产出的文本归成 `pass / challenge / uncertain / reject / conflict` 等事件，并回写为 feedback record。([github.com](https://github.com/DonJonMao/R-HAN/blob/feat/tree-multigraph-upgrade-base/mas_stage2/runtime.py)) ([GitHub][3])

### 5.3 记忆读取：slot retrieval

对不同节点类型，预先指定固定 slot 模板。

`proposal` 节点读取：

* 1 条 `self_output`
* 1 条 `feedback`
* 1 条 `failure_signal`
* 1 条 `stable_signal`
* 最多 2 条邻居 export

`checker` 节点读取：

* `anchor class summary`
* `champion class summary`
* 1 条最近 `feedback`
* 最多 1 条 `failure_signal`

`aggregator/sink` 节点读取：

* class summaries
* checker verdict summaries
* recovery summaries

slot 内部的选择函数不做线性加权，而只做**兼容性检索**：

[
\text{sel}*\tau(v,t)=\operatorname{TopR}*{m\in M_v^\tau}\cos(q_{v,\tau}^t,\ k_\tau(m))
]

也就是每个 slot 一个 query，每条 record 一个 key，只按相似度取 top-r。
这样你避免了多目标总分，也让 selector 变成一个清晰的“检索头”，不是手工搅和器。这个设计和 PAMAS 的“不同层看不同信息范围”是同向的。 

### 5.4 局部 latent

保留你当前的 local latent / memory brief 思路，但改成 typed 聚合：

[
z_v^t=\mathrm{GRU}_{\rho(v)}\big(z_v^{t-1},[e^{self},e^{fb},e^{stable},e^{fail},e^{class},g_t]\big)
]

这里：

* 每个 slot 先单独编码
* 再拼接进 role-specific GRU
* 输出 `local latent`
* 再 verbalize 成简短 `memory brief`

注意：这个 latent **不直接用于最终选答案**，它只服务于：

* 当前轮 prompt brief
* edge gate
* exported message

这样 memory 还是图内主资产，但不再演变成一个 opaque 的“谁赢了都怪它”的黑盒。

---

## 6. GNN：只做 edge-only sparse control

你的分支目标文档已经很明确：要把二阶段收敛到 `GlobalNode + edge-only gating + per-turn pruning`，去掉 controller-heavy 的旧混合体。([raw.githubusercontent.com](https://raw.githubusercontent.com/DonJonMao/R-HAN/feat/tree-multigraph-upgrade-base/mas_stage2/STAGE2_CURRENT_VS_TARGET.md)) 这就是 GNN 在 Stage2-GCR+ 里的定位。

### 6.1 输入

每个节点的初始状态是：

[
h_v^0 = W_h[r_v,\pi_v,z_v^0,\phi_v^{t-1}]
]

其中：

* (r_v)：角色 embedding
* (\pi_v)：Stage1 的节点先验
* (z_v^0)：local latent summary
* (\phi_v^{t-1})：上一轮反馈摘要

全局节点 (g_t) 由所有 active node 状态池化而成。

### 6.2 边 gate

保留你现在 `LightweightGNN` 的轻量消息聚合骨架，不换大模型。([github.com](https://github.com/DonJonMao/R-HAN/tree/feat/tree-multigraph-upgrade-base/mas_stage2)) ([GitHub][4])

每条边输出一个 logit：

[
\gamma_{u\to v}^t = \phi([h_u^t,h_v^t,h_u^t\odot h_v^t,|h_u^t-h_v^t|,g_t,e_{uv}])
]

然后不是全量 softmax，而是 **per-destination top-k pruning**：

* 第 1 轮：每个节点最多保留 3 条入边
* 后续轮：最多 2 条入边
* sink / protected nodes 保留兜底边

这一步直接吸收 **CortexDebate** 的核心思想：稀疏、按帮助关系传播，而不是全连接互聊。([aclanthology.org](https://aclanthology.org/2025.findings-acl.495/)) ([ACL Anthology][5])

### 6.3 active nodes

active set 由四部分并起来：

* active edge endpoints
* sink nodes
* protected nodes
* recovery ids

这里 `recovery ids` 就沿用你 runtime 现有的思路：凡是最近一轮被 `challenge / reject / conflict / revise` 指到的节点，自动进入下一轮 recovery 候选集合。([github.com](https://github.com/DonJonMao/R-HAN/blob/feat/tree-multigraph-upgrade-base/mas_stage2/runtime.py)) ([GitHub][3])

---

## 7. candidate bank 与 class collapse

这是 Stage2-GCR+ 的第二个核心。

### 7.1 raw candidate bank

每轮只收三类候选：

* sink 输出
* checker 明确 approve 的 proposal 输出
* recovery subgraph 里产生的 recovery branches

每个 candidate 附上完整 provenance：

[
c=(\text{text/code},\ \text{node_id},\ \text{turn},\ \text{role},\ \text{verifier_snapshot},\ \text{provenance})
]

### 7.2 为什么必须先折类

因为你过去的 GSM8K 失败已经说明：
**重复答案在 bank 里会天然占优，pairwise 学到的是“像多数路线的 challenger 更可信”，而不是真能纠正 `anchor` 的 challenger 更可信。** 

所以，collapse 的目标不是“压缩文本”，而是：

* 去掉重复多数的统计优势
* 给少数但结构独立的异议类独立生存空间

这就是 DAR 在这个系统里的唯一落点：**class collapse，而不是显式轮间消息过滤器。** DAR 原论文强调的正是 informative disagreement 比重复多数更重要。

### 7.3 route-specific class keys

`Code` 路径：

* 失败候选
  [
  key_{\text{code-fail}}=(failure_kind,\ failing_tests_sig,\ patch_locus)
  ]
* 通过测试候选
  [
  key_{\text{code-pass}}=(entry_sig,\ ast_sketch)
  ]

`Reasoning` 路径：

[
key_{\text{adv}}=(final_answer,\ claim_pattern,\ first_unsupported_step)
]

`Graph` 路径：

[
key_{\text{graph}}=(broken_blocks,\ violated_constraints,\ repair_locus)
]

### 7.4 representative

每类只保留一个 representative。
选择时不用加权总分，只用词典序。

`Code`：

[
\kappa_{\text{code}}=(1[p=n>0],\ syntax,\ entry,\ passed,\ -|E_{fail}|)
]

`Reasoning`：

[
\kappa_{\text{adv}}=(1[\text{no fatal contradiction}],\ -#major_contradictions,\ #critical_supports)
]

`Graph`：

[
\kappa_{\text{graph}}=(#satisfied_constraints,\ -#broken_blocks,\ 1[\text{local consistency}])
]

这一步很重要：**class size 可以保留到 metadata 里，但不能参与主排序。**

---

## 8. typed verifier 与 champion

### 8.1 typed verifier

Stage2-GCR+ 不允许一个统一“总分 judge”通吃所有任务。
verifier 必须 typed。

`Code` verifier：

[
V_{\text{code}}(c)=
(\text{syntax_ok},\text{entry_ok},\text{passed},\text{total},\text{failure_kind},\text{failing_tests_sig},\text{patch_locus})
]

`Reasoning` verifier：

[
V_{\text{math}}(c)=
(\text{final_answer},\text{first_unsupported_step},\text{constraint_violations},\text{unit_mismatch},\text{equation_conflict})
]

`Graph` verifier：

[
V_{\text{graph}}(c)=
(\text{broken_blocks},\text{violated_constraints},\text{repair_locus})
]

### 8.2 champion

当前 champion class 只是**最值得继续推进的类**，不是最终答案：

[
c^\star = \arg\max_{c\in C_t}\kappa_p(c)
]

### 8.3 运行模式

为了实用和止损，当前版本的 mode 不学分类器，只用离散谓词。

`Bypass`：

* `anchor class` 仍是 champion
* 或 champion 已经 strict-improve over anchor
* 或不存在 recoverable failure
* 或预算紧

`Lean`：

* recoverable failure 明确
* 只涉及单一局部 failure locus
* 且当前冲突规模小

`Full`：

* 仅对 `Code` 开
* `Lean` 一轮后仍未 fully pass
* 或存在多个 viable repair classes

这意味着：

* **Code**：允许 `Bypass / Lean / Full`
* **Reasoning / Graph**：当前版本先只允许 `Bypass`，最多做 very light adjudication，不启用 recovery override

这不是偷懒，而是严格顺着你现有训练记录走，优先保住已经能提分的 code line。

---

## 9. local recovery：图内、条件触发、结果回流

这是整个方案真正把 `v4.3` 正信号收编进 graph-native 主壳的地方。

### 9.1 触发条件

只有当存在 recoverable failure 时才进入恢复：

[
\text{RecoverTrigger}=
(\text{Recoverable}(c^\star)\lor \text{Recoverable}(a_0))
]

这里保留两个入口：

* champion recoverable
* anchor recoverable

这样即使当前 best seed 还不能支配 anchor，只要 anchor 自己有明确局部 failure，也不会被过早 preserve。

### 9.2 recovery subgraph

恢复子图不是新图，而是 `UnionGraph` 的一个局部诱导子图：

[
G_{\text{rec}} = \operatorname{Induce}\big(
\text{champion provenance}
\cup \text{recovery ids}
\cup \text{checker neighbors}
\cup \text{sink guards}
\big)
]

这保证一二阶段不割裂。

### 9.3 当前版本：只对 Code 打开 recovery

#### Lean recovery

* 1 个 repairer
* 1 个 checker
* 1 轮 patch 生成
* 最多 3 个 patch branches

#### Full recovery

* 2 个 repairers
* 1 个 checker
* 允许 2 层浅树扩张
* 中间允许 restart 到下一个 verified class rep

repair 输入只包含：

* 当前 champion code
* verifier 输出
* failing tests summary
* patch locus
* preserve constraints

不允许自由全文重写，只允许：

* 局部 patch
* boundary fix
* signature-preserving change

这里吸收的是 MALT 的**小规模 search-tree expansion**，但只用于 code `Full`，不引入它整套多角色训练。MALT 真正可借的就是“当静态 bank 不够强时，要允许受 verifier 约束的浅层扩张”，而不是把整篇论文全部搬过来。

### 9.4 Reasoning / Graph recovery：先留接口，不启用 override

`Reasoning` 的恢复算子定义成 `step-refine`：

* 只修 `first_unsupported_step`
* 只生成 2–3 个替代子推导
* 不改全文
* 当前版本只记录到日志和 candidate bank，不参与 final override

`Graph` 的恢复算子定义成 `block-fix`：

* 只修当前 break block
* 当前版本也只留接口，不进入 final override 主链

这一步的意义是：**接口先统一，主链先稳住。**

### 9.5 recovery 结果必须回流

repair / refine / block-fix 结果不能直接当 final output。
必须：

```text
recovery outputs
→ candidate bank
→ class collapse
→ champion re-selection
→ verifier
→ final guard
```

这样 recovery 就不是图外临终抢救，而是图内恢复回路。

---

## 10. final decision：hard guard first，debias last

### 10.1 Code

最终比较只看 typed verifier 偏序：

[
c \succ a_0 \iff
syntax_c \ge syntax_{a_0},;
entry_c \ge entry_{a_0},;
passed_c \ge passed_{a_0}
]

且至少一项严格更好。

### 10.2 Reasoning / Graph

先用 class-level comparator 比较；
只有当 `anchor class` 和 `challenger class` 仍然难分时，才做一次轻量 debias compare：

1. 原顺序比一次
2. 交换位置再比一次
3. 若结论一致，接受
4. 若结论不一致，视为 calibration failure，保 `anchor`

这一步吸收的是 CalibraEval 的核心：**pairwise judge 在位置和 token 上有偏差，最后一票必须去偏。** 但当前版本只取它的轻量 spirit，不完整实现 NOA。

---

## 11. 训练方案：只训图内控制，不开 RL 主线

当前版本最重要的是把 graph-native 主壳训稳，不要同时训一堆东西。

### Phase 0：冻结 Stage1

* 每题固定 prepared artifact
* Stage2 训练时不 live rerun Stage1

### Phase 1：训练 memory selector

目标不是答案质量，而是**slot retrieval 是否选对**。

正样本：

* 被读取后 downstream 进入 stable signal
* 或被读取后触发了 useful code recovery

负样本：

* 被读取但 downstream 仍 conflict/reject
* 或未被读取且与当前节点无关

损失：**per-slot contrastive retrieval loss**

### Phase 2：训练 edge gate

目标是**这条边对 downstream 是否真的有帮助**。

正样本边：

* 位于最终 champion provenance
* 或位于 successful recovery path

负样本边：

* 被激活但未被消费
* 或只传播噪声

损失：**per-destination BCE / ranking loss**

### Phase 3：code recovery operator

当前版本不训新的 LLM repair model，先只做 prompt-level recovery。
等成功 recovery traces 累积够多，再单独考虑 code-patch SFT。

### Phase 4：轻量 joint refinement

只 joint tune：

* memory selector
* edge gate

不 joint tune LLM 主体。
这和你仓库目前“先 memory，再 edge pruning，再轻量联合”的方向一致。([github.com](https://github.com/DonJonMao/R-HAN/tree/feat/tree-multigraph-upgrade-base)) ([GitHub][1])

---

## 12. 实施顺序：先提分，再扩展

### 第一优先级：MBPP / HumanEval

这里当前已有真实正信号。
先完整落地：

* typed memory
* sparse edge gate
* class collapse
* code local recovery
* reinsert
* hard guard

目标不是一下子大幅提升，而是：

* 提高 repair 覆盖率
* 降低无效 rerun 候选
* 把 reward 拉回到不那么负

### 第二优先级：GSM8K / MATH

目标先定成：

* **不再负增益**
* **不再高置信错翻**

当前版本这里只启用：

* rerun
* class collapse
* sparse activation
* final guard
* optional debias compare

不启用 recovery override。

### 第三优先级：Graph 类任务

等 local-constraint verifier 做扎实以后，再启用 block-fix recovery。

---

## 13. 仓库里的具体落点

### `mas_stage2/memory.py`

保留：

* 私有记忆存储
* local composer
* export message builder
* verbalizer

替换：

* heuristic linear score
* 固定 bias 混合

新增：

* typed slot query/key heads
* `class_summary` / `repair_trace` 两类 record

### `mas_stage2/gnn.py`

保留：

* `LightweightGNN` 骨架
* 轻量消息聚合

改成：

* per-destination top-k edge pruning
* active edge mask 真正进入主路径
* 不再依赖 controller 决定 focus/mode

### `mas_stage2/runtime.py`

新增五个函数：

* `collapse_code_classes`
* `collapse_reasoning_classes`
* `collapse_graph_classes`
* `select_champion_class`
* `build_recovery_subgraph`
* `run_code_recovery_round`
* `final_debias_compare`

现有的 feedback extraction、memory write-back、candidate-preserving finalizer 可以继续复用。([github.com](https://github.com/DonJonMao/R-HAN/blob/feat/tree-multigraph-upgrade-base/mas_stage2/runtime.py)) ([GitHub][3])

---

## 14. 最终伪代码

```text
Input: PreparedStage1Artifact A=(UnionGraph, anchor, priors), sample x

p <- AffordanceRoute(x)    # Code / Reasoning / Graph

for t = 1..T:
    for each node v:
        selected_slots <- TypedMemorySelect(v, M_v, current_state)
        z_v <- ComposeLocalLatent(selected_slots, g_{t-1})
    g_t <- GlobalNodePool({z_v})
    active_edges <- EdgeGate(UnionGraph, {z_v}, g_t)
    active_edges <- PerDestinationTopK(active_edges)
    active_nodes <- BuildActiveSet(active_edges, sinks, protected_ids, recovery_ids)

    for v in active_nodes:
        brief_v <- Verbalize(z_v, neighbour_exports(v), g_t)
        y_v <- RunNode(v, brief_v)
        write self_output / export / feedback

    collect C_raw from sinks + approved proposals + recovery outputs

C <- CollapseClasses_p(C_raw ∪ {anchor})
c* <- SelectChampion_p(C)

v_c <- Verify_p(c*)
v_a <- Verify_p(anchor)

if p == Code and Recoverable(v_c, v_a):
    G_rec <- BuildRecoverySubgraph(UnionGraph, c*, recovery_ids)
    B <- RunCodeRecovery(G_rec, target=best of {c*, anchor}, mode=Lean/Full)
    C <- CollapseClasses_code(C ∪ B)
    c* <- SelectChampion_code(C)

y <- FinalGuard_p(anchor, c*)
return y
```

---

## 15. 一句话总结

**Stage2-GCR+ 的本质，不是再造一套图外 self-correction 系统，而是把你当前仓库已经明确的 graph-native Stage2 真正收敛：图负责 rerun、memory 负责局部状态、GNN 负责稀疏控制、DAR 只做折类、PAMAS 只做分层可见性、CortexDebate 只做稀疏交互、CalibraEval 只做最后去偏，而 code recovery 作为 verifier-triggered 的图内局部恢复模式被统一收编进主壳。**

如果你下一步要真正开始实现，我建议先只做两件事：
**先把 `memory.py` 的 heuristic selector 换成 typed slot retrieval，再把 `gnn.py` 的 active edge pruning 变成 runtime 主路径。**

[1]: https://github.com/DonJonMao/R-HAN/tree/feat/tree-multigraph-upgrade-base "GitHub - DonJonMao/R-HAN at feat/tree-multigraph-upgrade-base · GitHub"
[2]: https://raw.githubusercontent.com/DonJonMao/R-HAN/feat/tree-multigraph-upgrade-base/mas_stage2/STAGE2_CURRENT_VS_TARGET.md "raw.githubusercontent.com"
[3]: https://raw.githubusercontent.com/DonJonMao/R-HAN/feat/tree-multigraph-upgrade-base/mas_stage2/runtime.py "raw.githubusercontent.com"
[4]: https://github.com/DonJonMao/R-HAN/tree/feat/tree-multigraph-upgrade-base/mas_stage2 "R-HAN/mas_stage2 at feat/tree-multigraph-upgrade-base · DonJonMao/R-HAN · GitHub"
[5]: https://aclanthology.org/2025.findings-acl.495/?utm_source=chatgpt.com "Debating Sparsely and Equally for Multi-Agent Debate"
