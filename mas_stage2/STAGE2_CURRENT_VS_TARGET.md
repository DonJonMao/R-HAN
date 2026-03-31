# Stage2 当前实现与目标设计对比

## 1. 文档目的

这份文档只做一件事：

- 说明 `mas_stage2/` 当前真实实现到底是什么
- 说明我们当前已经澄清后的目标设计是什么
- 说明两者之间的差异、取舍和后续收敛方向

本文档优先反映当前讨论后的共识，不以旧版 `README_V2.md` 或 `DESIGN_V2.md` 的理想化描述为准。

---

## 2. 当前实现的真实设计

当前 `stage2` 不是完全重写的一套新系统，而是：

- 以 `Stage2Runtime` 为执行骨架
- 在其上叠加 `Stage2RuntimeV2` 的 latent / GNN / global node 增强

因此它更接近：

`V1 可运行执行壳 + V2 局部增强`

而不是：

`完全按最终三层图理念重构后的纯净系统`

### 2.1 输入结构

当前 `stage2` 的输入单位是“每题一个 stage1 结构产物”：

- 默认流程：对当前题先跑一次 stage1，生成该题自己的 `UnionGraph`
- 可选流程：把该题的 stage1 结果保存成 `PreparedStage1Artifact`，后续重复复用

因此：

- 不是整个数据集共用一张固定结构图
- 也不是 stage2 完全脱离 stage1 独立存在
- 实际上是“每题一张 stage1 union graph，支持缓存复用”

### 2.2 三层语义结构

当前实现已经具备三层语义，但不是严格意义上的“三层异构图算法”：

1. 上层：全局控制层
   - 目前实际上同时存在两套东西：
   - `GlobalController`：规则驱动，维护 `mode / focus / role_weights / uncertainty`
   - `GlobalContextNode`：可学习的全局向量状态

2. 中层：`UnionGraph`
   - 复用 stage1 为当前题生成的图结构
   - 边上保留结构先验，例如 `support_ratio / avg_parent_score / initial_keep_logit`

3. 下层：每个 agent 的题内私有记忆
   - 当前是 memory store，不是显式 memory-node graph
   - 每个 agent 只能直接读取自己的原始记忆
   - 跨 agent 传播只能通过图上的 exported messages 完成

### 2.3 当前单轮运行流程

当前 `stage2` 的一轮大致如下：

1. 读取当前题的 `UnionGraph`
2. controller 生成本轮高层控制状态
3. 计算每条边的激活分数并执行 pruning
4. 决定本轮哪些 task nodes 实际参与
5. 每个参与节点从自己的私有记忆中选择少量记录
6. 用 `MemoryComposer` 或本地 composer 压缩这些记忆
7. 聚合邻居导出的压缩消息
8. 拼出给该 agent 的 `memory brief`
9. 调用 LLM 完成本轮节点执行
10. 抽取 feedback events 并写回私有记忆
11. 更新全局状态，进入下一轮

### 2.4 当前实际在学习什么

当前实现里，学习信号并不干净，存在两套机制并行：

1. V1 风格轻量在线模型
   - `selector linear model`
   - `edge linear model`
   - `controller linear model`

2. V2 风格增强模块
   - `MemoryComposer`
   - `latent_to_embed bridge`
   - `LMPO / REINFORCE`
   - `LightweightGNN`
   - `GlobalContextNode`

这导致当前 `stage2` 不是“单一图学习系统”，而是“线性小模型 + latent 组件”的混合体。

### 2.5 当前 latent memory 如何作用到 LLM

这是当前实现里最容易被误解的一点。

当前不是：

- latent memory 直接作为 hidden-state prefix 输入 LLM

当前实际是：

- 先把私有记忆压成 latent
- 再用 latent 去选择最该暴露给 prompt 的本地/邻居片段
- 最后把这些片段 verbalize 成短文本 `memory brief`
- 再把 `memory brief` 作为 prompt 的一部分送入 LLM

因此当前更准确的说法是：

`latent-guided prompt verbalization`

而不是：

`direct latent injection into hidden states`

### 2.6 当前 pruning 的真实情况

当前 pruning 已经存在，但仍然偏工程化：

- 主要基于边先验、feedback、controller 信息和线性 edge model
- GNN 主要负责聚合 surviving neighbors 的信息
- pruning 的主导逻辑并不在 GNN 内部

换句话说，当前是：

- 先由外部逻辑决定哪些边保留
- 再由 GNN 在保留下来的子图上做信息聚合

而不是：

- 由统一的图模型直接给出 edge gate，并用它同时完成消息传播与 pruning

---

## 3. 当前已确认的目标设计

经过最近一轮讨论，我们对二阶段的目标设计已经进一步收缩，原则是：

- 只保留真正必要的模块
- 不追求形式上复杂
- 优先保证系统语义统一、训练目标清晰

### 3.1 总目标

目标中的 `stage2` 应该是：

`固定 stage1 per-question union graph + 单题私有记忆 + global node + edge-only graph gating + per-turn pruning`

其中：

- stage1 负责生成当前题的结构骨架
- stage2 负责在固定骨架上学习通信强度与收缩路径
- 两阶段训练逻辑应尽量解耦

### 3.2 输入结构

目标设计里仍然保留“每题一张 stage1 图”的设定：

- 每道题对应自己的 `UnionGraph`
- stage2 运行在该题的图上
- 为了让二阶段训练稳定，优先复用 prepared artifact

因此未来的推荐工程流程是：

1. stage1 为每题生成图
2. 保存为 prepared artifact
3. stage2 尽量复用缓存，不在训练时反复 live 重跑 stage1

### 3.3 上层：只保留 Global Node

未来设计里不再保留独立 `controller`。

也就是说，不再需要：

- `focus`
- `mode = explore / refine / finalize`
- `role_weights`
- 规则驱动的高层文字指导

上层只保留一个 `GlobalNode`，职责缩减为：

1. 聚合本轮所有活跃节点的全局信息
2. 为下一轮提供全局上下文
3. 作为 edge gating 的输入之一

它不再承担“显式指挥官”的职责。

### 3.4 中层：只学习 Edge Weight，不学习 Node Weight

这是当前最明确的设计决定之一。

未来设计中：

- 不单独设计 node importance / node weight
- 不单独给角色做显式重要性偏置
- 只学习 edge gate / edge weight

节点是否参与，不通过独立 node model 决定，而由边自动诱导：

- 如果一个普通节点在当前轮没有足够强的 incident edges，则跳过
- `root / sink` 可保留少量规则保护

这比“同时学 node 和 edge”更符合当前项目主线，因为：

- stage1 已经完成拓扑生成
- stage2 重点应是优化通信路径，而不是重新决定节点价值

### 3.5 每轮强制 pruning

未来设计中，收缩趋势不再由 controller 的 `focus / mode` 承担，而直接通过图算法完成：

- 每一轮都强制执行一次 pruning
- pruning 依据当前 edge gate 和轮次推进
- 前几轮以 soft prune 为主
- 后几轮再做 hard prune

因此，“轮次推进导致图逐步收缩”将成为系统内生行为，而不是外部策略文本。

### 3.6 下层：保留私有记忆，但不强求形式化异构图

未来设计里，下层仍然是每个 agent 的私有记忆池，但不强求一定做成形式上严格的 memory-node graph。

只要满足以下约束即可：

1. 每个 agent 只直接读取自己的原始记忆
2. 原始记忆不能跨 agent 直接共享
3. 跨 agent 信息必须通过图上的导出消息传播
4. global node 只处理全局信息，不替代原始局部记忆

因此：

- 可以保留“语义上三层图”
- 不必为了形式完整性强行引入一套复杂异构图框架

### 3.7 latent memory 的未来定位

未来设计中，latent memory 仍然保留，但当前更偏向现实可落地路线：

- 不强制要求 hidden-state prefix / soft prompt adapter
- 优先保留 `latent -> verbalized brief -> prompt` 这条稳定路线

原因是：

- 当前推理栈以 API / vLLM 调用为主
- 直接做 hidden-state 注入会显著改变 serving 方式和工程复杂度

因此当前的主路线不是“必须做连续隐状态注入”，而是：

- 先让 latent 真正用于记忆压缩、邻居聚合和 prompt 选择
- 后续如确有必要，再评估是否上更重的 hidden-state 方案

### 3.8 记忆选择的未来定位

未来设计中，记忆选择不需要引入复杂的 DP/PSO。

更合适的方向是：

- 保留 embedding relevance
- 加入 success / failure / revise 等 outcome bias
- 让每轮既看到“值得延续的片段”，也看到“应避免重复的错误片段”

也就是说，未来更倾向：

`relevance + outcome-aware bias`

而不是：

`复杂搜索式记忆选择器`

---

## 4. 当前实现与目标设计的核心对比

| 维度 | 当前实现 | 目标设计 |
|------|---------|---------|
| stage1 输入方式 | 每题 live stage1，支持 artifact 缓存 | 每题一张 stage1 图，训练时尽量固定复用 prepared artifact |
| 上层模块 | `GlobalController + GlobalContextNode` 并存 | 只保留 `GlobalNode` |
| 高层信号 | `focus / mode / role_weights` | 不再显式存在 |
| 节点权重 | 间接存在节点活跃控制 | 不单独学习 node weight |
| 角色偏置 | controller 会调整 role weight | 不显式建 role importance |
| 边权机制 | heuristic + edge linear model 主导 | edge-only graph gate 主导 |
| GNN 作用 | 聚合 surviving neighbors，非 pruning 核心 | 统一进入 edge gating / message passing 主路径 |
| pruning | 外部工程逻辑主导 | 每轮强制 pruning，逐轮收缩 |
| 下层记忆 | 私有 memory store | 仍保留私有记忆池，不强求形式化异构图 |
| 跨 agent 记忆传播 | 通过 exported message | 保持不变 |
| latent 到 LLM | latent-guided verbalization | 仍优先走 verbalized brief 路线 |
| 训练目标 | V1 小模型 + V2 模块混合 | 尽量收敛到 edge gate + memory/latent 的清晰闭环 |

---

## 5. 哪些当前设计应保留

以下部分已经符合目标方向，后续不需要推翻：

1. 每题由 stage1 生成专属 `UnionGraph`
2. stage2 支持 prepared artifact 复用
3. 每个 agent 只访问自己的题内私有记忆
4. 原始记忆不能跨 agent 直接共享
5. 邻居间通过 exported messages 沿图传播
6. latent memory 已经进入系统主路径
7. 逐轮运行、逐轮反馈、逐轮记忆回写这一闭环是正确的

---

## 6. 哪些当前设计应逐步移除或弱化

以下部分与当前确认后的目标设计不完全一致，应逐步收缩：

1. `GlobalController`
   - 不再作为长期核心模块
   - 后续应由 `GlobalNode + edge gating + turn schedule` 替代

2. `focus / mode / role_weights`
   - 这些高层文本控制信号不再视为目标设计的一部分

3. `controller linear model`
   - 与目标设计不符，后续应移除

4. 显式角色重要性偏置
   - 不再单独建模
   - 让角色差异通过记忆状态、节点表示和边 gating 自然体现

5. “GNN 只是聚合器，pruning 主要靠外部 heuristics”
   - 这是过渡状态，不应视为最终方案

---

## 7. 推荐的未来收敛方向

后续如果继续重构 `stage2`，推荐按下面顺序收敛：

### 7.1 第一步：固定结构输入

- 默认优先复用 prepared artifact
- 把 stage1 结构波动从 stage2 训练中隔离出去

### 7.2 第二步：删掉 controller 依赖

- 去掉 `focus / mode / role_weights`
- 仅保留 `GlobalNode`
- global 信息通过向量状态和摘要进行传播，不再通过“控制文本”驱动

### 7.3 第三步：把图学习收敛为 edge-only

- 只学习 edge gate
- 不学习 node weight
- 不单独学习 role weight
- 节点活跃性由边自动诱导

### 7.4 第四步：让 pruning 真正进入图主路径

- 每轮强制 pruning 一次
- 早期 soft prune
- 后期 hard prune
- pruning 结果直接影响下一轮活跃子图

### 7.5 第五步：简化记忆选择目标

- 保留 relevance retrieval
- 增加 outcome-aware bias
- 避免继续堆叠额外小模型导致训练目标混乱

---

## 8. 当前最终共识

截至本轮讨论，二阶段的未来目标可以概括为：

`每题 stage1 union graph + prepared artifact 复用 + 单题私有记忆 + 仅保留 global node + edge-only gating + 每轮强制 pruning + latent-guided prompt brief`

换句话说：

- 二阶段不再追求“控制器驱动的多模块混合系统”
- 而是收敛为“由图边权和全局状态主导的信息流优化系统”

这也是当前后续实现、重构和训练设计的推荐基线。
