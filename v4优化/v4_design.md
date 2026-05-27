# R-HAN Stage2 V4 设计文档

更新时间：2026-04-06

本文档统一描述 `stage2 v4` 的递进式路线，而不再把后续所有工作都压缩进一个过宽的 `v4.2`。

当前建议的版本分组为：

- `v4.1`：先消除有害 override，建立安全的 challenger 通道
- `v4.2`：放宽 challenger 供给，并把最终仲裁改成结构化假设检验
- `v4.3`：从语义仲裁转向 verifier 驱动的局部搜索（GEF-Loop）
- `v4.4`：从一刀切协议转向任务感知的协议路由与预算感知模式选择
- `v4.5`：用可验证进展重构训练信号，做事件级 credit assignment 与 preserve 中立化
- `v4.6`：在协议与信用都稳定后，再学习 coordinator policy 与 RL / MARL

设计原则保持不变，但顺序更明确：

1. 不再引入新的多参数加权目标函数，不靠堆阈值和工程 trick 纠偏。
2. 所有改动都要写成清晰的算法机制，而不是“经验上多减一点、多加一点”。
3. 当前问题首先是信号结构错位，而不是模型容量不够，因此先修正 challenger 的产生、保留、仲裁与执行动力学，再讨论更重的学习控制器。
4. 后续“放宽”不是简单放低 override 门槛，而是让更多有资格的 challenger 进入竞争，并让仲裁过程更结构化。
5. 对 code / graph / structured 任务，challenger 的定义必须由 verifier affordance 决定，而不能继续沿用自然语言推理任务中的“语义上离 anchor 更远”。

---

## 1. 经验诊断：为什么需要 `v4`

### 1.1 `v3.1` 的直接问题

基于当前 `v3.1` 的 `gsm8k` 训练快照：

- 训练样本数：`264`
- `stage1_success = 0.9697`
- `stage2 final success = 0.8902`
- `selection_decision = use_stage2_pairwise` 的样本数：`25`
- 这 `25` 次 override 全部失败
- 其中 `21` 次是“`stage1` 原本正确，但 `stage2` 覆盖后变错”的负回归
- `stage1 anchor` 仍被保留 `239/264` 次，但真正发生 override 时几乎全是有害 override

这说明当时的主要问题不是“系统太保守”，而是：

1. `stage2` 很少产生真正高质量的 challenger。
2. candidate bank 中存在明显的回声放大，重复答案会以统计优势挤压异见答案。
3. override 资格过宽，导致一旦出现 challenger，就可能在证据不足时错误推翻强 `stage1 anchor`。

因此，`v4` 的第一步必须先把“谁有资格挑战 anchor”重新算法化。

### 1.2 `v4.1` 的当前新现象

截至当前 `gsm8k` 已完成的 `v4.1` 训练结果，`v4.1` 已经表现出与 `v3.1` 不同的失效形态：

- 训练样本数：`264`
- `avg_success = 0.9697`
- `override_count = 0`
- `explicit_positive = 3`
- 其中 `261` 次原因为 `v4_1_preserve_no_explicit_challenger`
- 其余 preserve 原因也主要来自 inspector / pairwise 拒绝

这说明 `v4.1` 已经基本压住了“错误 override”这一旧问题，但新的主要瓶颈也非常明确：

1. challenger 供给过少，几乎没有样本真的进入“可挑战 anchor”的状态。
2. 当前的 override 资格制过于依赖“显式 challenger 已经先出现”这一前提，因此系统很容易退化为纯 preserve。
3. 问题的重点已经从“如何防止错误翻案”转移为“如何让真正值得挑战的假设有机会被提出、被保留、被仲裁”。

因此，后续版本的正确推进顺序不应是立刻跳到 RL 或分层 coordinator，而应先解决 challenger 的供给与仲裁。

---

## 2. 总体路线

### 2.1 `v4.1` 的目标

`v4.1` 只做三件事：

1. 显式 challenger
2. DAR 式消息保留
3. 更严格的 override 资格

这三个改动都直接对应 `v3.1` 中最明确的负效应来源，并且可以在不重写整个 `stage2` 框架的前提下上线训练。

### 2.2 `v4.1` 之后的推进顺序

当前更合理的路线不是“`v4.1` 之后直接进入一个大而全的 `v4.2`”，而是拆成四个逐层放开的阶段：

1. `v4.2` 先解决 challenger 供给与结构化仲裁。
2. `v4.3` 再把 Stage2 重构成 verifier 驱动的局部搜索，优先解决 code / graph 任务中的 challenger 空转。
3. `v4.4` 再把不同任务族拆成不同协议，并让路由器决定走哪条协议、走多深。
4. `v4.5` 以后才重构训练目标与 credit assignment，并把信用下沉到事件级。
5. `v4.6` 最后才讨论 coordinator policy、RL / MARL 与分层控制。

这个顺序的依据很直接：

- 当前观察到的主要问题是“challenger 根本进不来”，不是“policy learner 还不够强”。
- 如果连 challenger 的供给和仲裁都没有打通，提前做 coordinator / RL 只会学习到更保守的 preserve 行为。
- 先把推理时的供给侧与决策协议改对，再把这些运行信号转化为更稳定的学习信号，失效归因才清晰。

---

## 3. `v4.1` 算法设计

`v4.1` 保持如下部分不变：

- 固定 `stage1 -> UnionGraph -> stage2` 的两阶段边界
- `runtime_v2` 的多轮执行骨架
- 现有 latent memory / GNN / global node / candidate bank / online learning 主体

`v4.1` 只修改“候选形成与最终翻案”这条链路。

### 3.1 模块 A：显式 Challenger

#### 动机

当前 `v3.1` 的 challenger 主要来自普通 task nodes 的自然输出。这条路径有两个问题：

1. task nodes 的默认目标不是“反驳 anchor”，因此很难主动生成高质量异见。
2. 现有 candidate bank 更容易聚合“被重复说过很多次”的答案，而不是“专门为推翻 anchor 而构造”的答案。

#### 机制

在常规 candidate bank 首次构建完成后，`v4.1` 不直接进入最终选择，而是额外执行一个显式 challenger 步骤：

1. 从 bank 中识别 `stage1 anchor`
2. 收集当前已有的非 anchor 候选，作为 challenger 的证据上下文
3. 只调度预定义 challenger agent 集合，例如：
   - `skeptic`
   - `debater_b`
4. 每个 challenger agent 必须完成如下任务：
   - 判断 anchor 是否应被推翻
   - 给出最关键 flaw
   - 给出自己的替代答案
5. 只有当 challenger 显式输出：
   - `VERDICT: challenge`
   - 且 `ANSWER != anchor`
   时，才把该答案并入 bank，并标记为 `explicit_challenger = True`

#### 形式化

设 anchor 为 `a`，已有非 anchor 候选集合为 `C_nonanchor`，challenger agent 集合为 `H`。

对每个 `h ∈ H`，执行：

`(v_h, flaw_h, c_h) = Challenger(h, q, a, C_nonanchor)`

只有满足：

- `v_h = challenge`
- `c_h != a`

时，`c_h` 才进入候选集合，并被标记为显式 challenger。

#### 设计含义

这一步不靠调低 anchor prior 来“逼出 challenger”，而是单独开辟一个“以推翻 anchor 为目标”的生成通道。

---

### 3.2 模块 B：DAR 式消息保留

#### 动机

当前负效应的一部分来自消息回声：

- 相似答案在多轮传播中不断重复
- 传播带宽被同质信息占满
- 真正的少数异见更难被后续节点看到

DAR 的启发是：每轮不必保留所有邻居消息，而应保留一个“共识代表 + 差异代表”的子集。

#### 机制

对某个节点在当前轮接收到的邻居导出消息集合 `M = {m_1, ..., m_n}`，若 `n > K`，则执行多样性保留：

1. 对每条消息的 latent vector 做归一化
2. 计算集合质心 `g`
3. 先选择最接近质心的消息，作为“共识代表”
4. 再迭代选择与当前已选集合最不相似的消息，直到达到 `K`

其中 `K` 直接复用已有 `memory.max_neighbour_exports`，不新增调参维度。

#### 形式化

给定消息向量 `z_i`，定义：

- `g = normalize(mean_i z_i)`
- 第一条保留消息：
  `m* = argmax_i cos(z_i, g)`
- 后续每一步选择：
  `argmax_j min_{s in S}(1 - cos(z_j, z_s))`

其中 `S` 为当前已保留集合。

#### 设计含义

这一步不是给“多样性”加一个额外 reward 权重，而是直接改变通信子图中哪些消息有资格继续传播。它属于结构性约束，不是目标函数上的软偏好。

---

### 3.3 模块 C：更严格的 Override 资格

#### 动机

`v3.1` 的问题不是 override 太少，而是 override 一旦发生，经常发生在不该发生的时候。当前需要的不是“更激进”，而是“更有资格约束的激进”。

#### 机制

`v4.1` 的 override 规则分成两道门：

1. challenger 必须是显式 challenger
2. inspector 必须同意 challenger 胜出

具体流程如下：

1. 对所有 challenger 与 anchor 做 pairwise 比较，得到：
   `P(c > a)`
2. 只保留其中被标记为 `explicit_challenger = True` 的候选
3. 取最强显式 challenger `c*`
4. 调用 inspector 比较 `a` 与 `c*`
5. 只有当以下条件同时满足时才 override：
   - `P(c* > a) > 0.5`
   - `Inspector(a, c*) = challenger`

否则一律保留 anchor。

#### 形式化

设显式 challenger 集合为 `C_explicit`。

若 `C_explicit = ∅`，则：

`y = a`

否则：

`c* = argmax_{c ∈ C_explicit} P(c > a)`

再令：

`d = Inspector(q, a, c*) ∈ {anchor, challenger, uncertain}`

最终决策：

`y = c*` 当且仅当 `P(c* > a) > 0.5` 且 `d = challenger`

否则：

`y = a`

#### 设计含义

这一步把“是否翻案”从单一 pairwise score 改为“显式反驳 + 二次审查”的资格制。它的目标不是提高 override 率，而是降低有害 override 率。

---

### 3.4 `v4.1` 的在线学习策略

`v4.1` 暂时不重写已有 online learning 主干。保留：

- candidate 绝对质量学习
- pairwise 相对比较学习
- reviewer 相对校准学习

原因有两点：

1. 当前最紧迫的问题是推理阶段的决策链，而不是学习器容量。
2. 如果短期同时改生成、通信、决策和学习目标，会让失效归因变得困难。

因此 `v4.1` 的训练目标保持最小变动，只在推理图和 override 资格上动刀。

---

### 3.5 `v4.1` 的预期收益与风险

#### 预期收益

1. 显式 challenger 提高“真正反驳 anchor”的候选产出率。
2. DAR 式保留降低回声放大，给少数异见更多生存空间。
3. inspector-gated override 降低“错误 challenger 抢走最终答案”的概率。

#### 主要风险

1. 显式 challenger 可能增加 token 开销。
2. 如果 challenger 质量仍然不足，override 次数会进一步下降。
3. inspector 与 challenger 若共享同类偏差，仍可能出现系统性误判。

当前日志已经表明，第 2 点风险正在真实发生，因此 `v4.2` 的首要任务不再是继续收紧，而是以算法方式放宽 challenger 的供给与晋升。

---

## 4. `v4.1` 当前实现映射

当前代码落点如下：

- `mas_stage2_v4_1/config.py`
- `mas_stage2_v4_1/runtime.py`
- `mas_stage2_v4_1/pipeline.py`
- `train_mas_stage2_v4_1_target_suite.py`
- `run_stage2_v4_1_selected_serial_20260404.sh`

其中：

- `runtime.py` 实现显式 challenger、DAR 消息保留、inspector-gated override
- `pipeline.py` 与训练入口对齐 `v4.1` 元数据
- 训练脚本移除了 `v3` 的旧 override 阈值参数入口
- 当前训练启动范围先锁定 `gsm8k`

先跑 `gsm8k` 的原因很直接：

- 现有负效应证据最明确地出现在 `gsm8k`
- `v4.1` 的目标首先是修复“错误 override”而不是一口气扩展到所有任务

---

## 4.1 `v4.2` 当前实现映射

当前 `v4.2` 已按上述递进思路落到独立代码路径中：

- `mas_stage2_v4_2/config.py`
- `mas_stage2_v4_2/runtime.py`
- `mas_stage2_v4_2/pipeline.py`
- `train_mas_stage2_v4_2_target_suite.py`
- `run_stage2_v4_2_selected_serial_20260404.sh`
- `test_stage2_v4_2.py`

其中：

- `runtime.py` 保留 `v4.1` 的显式 challenger 与 DAR 保留，但把最终决策链扩展为：
  1. `explicit challenger`
  2. `auditor` 支持的 `provisional challenger` 软晋升
  3. ACH 风格的结构化 adjudication
  4. 置换式 calibration 后才允许 override
- `pipeline.py` 与训练入口改写为 `v4_2_*` 元数据和统计口径
- `test_stage2_v4_2.py` 直接覆盖当前最关键的五个算法点：
  - DAR 保留
  - 软性 challenger 晋升
  - 无 provisional challenger 时 preserve
  - calibration 放行后的 override
  - calibration 拒绝后的 preserve
- 当前 runner 仍先锁定 `gsm8k`，因为 `v4.2` 的直接目标是修复强 anchor 数学推理任务中的 preserve-only 退化

这意味着 `v4.2` 已不只是设计草案，而是可以独立训练和记录诊断信号的一条新实验分支。

---

## 5. `v4.2` 到 `v4.6` 的递进研究设计

### 5.1 `v4.2`：放宽 challenger 供给 + 结构化仲裁

#### 核心判断

`v4.1` 当前不是“又回到了错误 override”，而是“几乎没人拿到挑战资格”。因此 `v4.2` 的目标不是放松安全门，而是改变 challenger 的供给与晋升协议，让更多有根据的非 anchor 假设进入最终竞争。

#### 模块 A：两阶段 challenger 供给

`v4.2` 将 challenger 从“固定 agent 直接给出最终 challenger”改为两阶段协议：

1. `Challenge Proposal`
2. `Challenge Promotion`

第一阶段中，challenger agent 不再只输出一个替代答案，而必须输出：

- 目标假设 `h`
- 对 anchor 的关键反驳 `r`
- 支撑该反驳的证据片段 `e`

第二阶段中，系统不再要求 challenger 必须来自“固定显式 agent 的最终答案”，而是允许以下两类候选被晋升为 `provisional challenger`：

1. 显式 challenger 产出的新假设
2. candidate bank 中已有、但能被至少一条结构化反驳支持的非 anchor 独特答案

这意味着“挑战资格”从单纯依赖 agent 身份，改为依赖“假设是否被结构化反驳协议支持”。

除此之外，`v4.2` 还可以加入一条软性晋升通道，用来专门处理 `v4.1` 中“显式 challenger 为零时系统直接僵住”的问题。

借鉴 PAMAS 的 perspective aggregation 思路，不再只依赖 `VERDICT: challenge` 这一硬标签，而是增加 `Auditor` 层，对 candidate bank 中的独特非 anchor 假设做异常线索审计：

1. 先枚举所有独特非 anchor 假设 `h`
2. 由多个 auditor 分别只查看局部证据子集，而不是全量长上下文
3. 每个 auditor 输出：
   - `anomaly_support(h | a)`
   - 触发该支持的异常证据片段
   - 对 `h` 的局部一致性判断
4. 若某个 `h` 未被显式标记为 challenge，但满足：
   - 与 `anchor` 存在非平凡语义分歧
   - 至少获得一组 auditor 的异常支持
   - 局部 verifier 未明确判其自相矛盾
   则将其晋升为 `provisional challenger`

这里可以把语义熵、judge 置信度波动、与 anchor 的语义距离作为 auditor 的观测量，但不应把它们直接写成单一阈值判决。它们只是异常线索的输入，不是最终晋升规则本身。

这样做的意义是：

- 即使显式 challenger agent 没有产出 `VERDICT: challenge`，系统仍然能从 candidate bank 中挖出“少数但可能有价值”的候选
- 让 challenger 供给不再被单一 prompt 触发条件完全卡死
- 保持晋升过程仍然以结构化证据为核心，而不是单纯奖励“离 anchor 更远”的答案

#### 模块 B：ACH 式结构化仲裁

`v4.2` 不再只做 `anchor` 对单个 `c*` 的一次 pairwise，而是把最终决策改成竞争假设分析：

1. 将 `anchor` 与所有独特非 anchor 答案列为假设集合 `H = {H_0, H_1, ..., H_m}`
2. 从 feedback events、review traces、challenger 反驳中抽取证据集合 `E = {E_1, ..., E_n}`
3. 构造一致性矩阵 `M(E_i, H_j)`
4. 对当前领先假设执行一次压力测试：
   - 若该假设是错的，最强反证是什么？
5. inspector / finalizer 基于矩阵与压力测试给出最终仲裁

这样做的核心作用是：

- 不再让单次 pairwise 直接定义 override
- 把“少数但强的反证”显式保留下来
- 避免 `anchor` 只因为先验位置优势就天然占优

#### 模块 C：judge 去偏与校准

借鉴 CalibraEval，`v4.2` 的最终仲裁不应依赖单一顺序下的一次 judge 输出，而应在推理时做有限次置换：

- 交换 `anchor` / `challenger` 的呈现顺序
- 打乱候选假设顺序
- 对同一组假设做多次仲裁采样

更具体地，`v4.2` 可以把 finalizer 的 override 判决改成一个非参数实时校准协议：

1. 对 `a` 与当前最强 `c*` 至少执行 `K >= 3` 次位置置换与 ID 置换
2. 收集每次置换下的原始判别概率 `P_ob^{(k)}`
3. 用置换样本构造经验无偏分布，并由非参数映射 `g(·)` 得到去偏后的 `P_de`
4. 只有当：
   - `P_de(c* > a) > 0.5`
   - 且跨置换结果没有表现出高不稳定性
   时，最终才允许 override

若置换后概率明显回落，或不同置换之间分歧很大，则 finalizer 应直接降级为 `uncertain` 并 preserve。

这样做的目的不是多模型堆叠，而是削弱位置偏见与令牌偏见，并防止系统重新回到 `v3.1` 式“高置信但方向错”的 override。

#### 为什么这是下一步

`v4.2` 直接对应当前最新日志暴露的问题：

1. 让 challenger 不再只能靠固定 agent 的一次命中出现。
2. 让 candidate bank 中已有的少数派答案也有机会被提升为正式竞争假设。
3. 让 finalizer 的工作从“二选一打分”升级为“结构化证据仲裁”。

如果这一步还没有打通，就不应该提前推进更重的控制器学习。

---

### 5.2 `v4.3`：从语义仲裁转向 Verifier-Triggered Local Search（GEF-Loop）

#### 核心判断

当前 `v4.2` 的主要改动落在最终选择链：`provisional challenger -> adjudication -> calibration -> override`。这条链在 `gsm8k` 上只让系统“略微更活”，但没有形成稳定收益；在 `humaneval` 上更明显，当前连 `provisional challenger` 都几乎起不来。这说明问题不再主要是“最后谁判赢”，而是执行过程中根本没有形成可验证、可晋升的 challenger。

因此，`v4.3` 不再继续围绕 finalizer 做文章，而是把 `stage2` 改造成 verifier 驱动的局部搜索过程。对于 code / graph / structured 任务，challenger 的定义不再是“另一段语义上不同的答案”，而是“能够修复 anchor 已暴露失败点的局部候选”。

#### 统一抽象

`v4.3` 将每个样本的 `stage2` 状态表示为：

- 当前 anchor `a_t`
- 局部分支池 `B_t`
- 已验证检查点集合 `C_t`
- 当前失败事件集合 `F_t`
- 全局摘要 `G_t`

统一循环写成：

```text
1. Detect: 对当前 anchor 或活跃分支执行 verifier / executor，得到失败事件 F_t
2. Localize: 将失败事件定位到局部修复单元 u_t
3. Expand: 围绕 u_t 生成一个小预算的局部分支集 B_t
4. Execute + Verify: 对 B_t 中的分支执行并验证
5. Select:
   - 若存在分支在 verifier 关系下严格优于当前 anchor，则升级 anchor
   - 若没有严格改进，但出现不可调和冲突，则从最近检查点 restart
   - 否则 preserve 并停止
```

其中最重要的不是“挑出最像正确答案的候选”，而是定义一个任务相关的 verifier 支配关系 `b ≻ a`：

1. `b` 至少保留了 `a` 已满足的关键约束。
2. `b` 修复了 `a` 的至少一个已知失败点。
3. `b` 没有引入更高优先级的新违例。

这一定义是偏序关系，不是多指标线性加权打分。若两个分支不可比，则允许并存进入下一轮，而不是强行通过单次 judge 打出一个赢家。

#### 模块 A：Task-Aware Expansion

##### A1. `v4.3-code-repair`：面向 `mbpp / humaneval`

这是 `v4.3` 的第一优先级，因为当前最明确的失效就发生在 code task 上。

其核心变化有三点：

1. Challenger 改角色。
   不再让 challenger 直接输出“替代全文代码”，而是先扮演 `Test Designer / Counterexample Proposer`，围绕当前 anchor 生成一个小预算的反例集合 `T_c`，并指出最可疑的失败约束。
2. Verifier 改输出。
   verifier 不再只输出 `VERDICT: pass|challenge|uncertain`，而要显式给出失败事件：
   - `FAIL_EVENT`
   - `LOCATION`
   - `REPAIR_FOCUS`
   - `CONSTRAINT`
3. 分支粒度改成 patch。
   默认只生成局部补丁分支，而不是全文重写。只有当失败点无法局部化，或连续若干轮局部补丁都没有严格改进时，才允许升级到 function-body 级 rewrite。

对应的 GEF-Loop 为：

```text
Anchor code
-> 执行可见测试 / 显式约束检查
-> 若失败，抽取 failure event
-> Challenger 生成 counterexample set 与 repair focus
-> Solver / Reviser 只围绕 repair focus 生成 patch branches
-> 对 patch branches 执行并验证
-> 若某个 branch 严格改善 verified pass set，则晋升为新 anchor
```

这里的 “严格改善” 不是靠语义 judge，而是靠 verifier 支配关系。最简单的实现可以把验证摘要写成有序元组，例如：

- `syntax_ok`
- `entry_point_ok`
- `visible_constraints_passed`
- `known_failure_fixed`

比较时优先按支配关系或字典序，而不是手工写一个总分。

需要特别强调：

1. `mbpp` 上自动测试生成更可信，因此适合作为 `v4.3-code-repair` 的首个主战场。
2. `humaneval` 上自动生成测试只能作为 branch trigger 与 repair guidance，而不能当最终真理；隐藏测试的泛化仍要靠最终评估验证。
3. 因此 `v4.3-code-repair` 的第一目标不是“立刻在所有 code benchmark 上涨分”，而是先让 challenger 以 verifier-backed 的方式真正出现。

##### A2. `v4.3-graph-local`：面向 `nlgraph`

对 `nlgraph`，`v4.3` 不再把 challenger 定义为“另一整张图”，而是引入 `GoV` 风格的局部结构验证图。

具体做法：

1. 将当前图答案拆成 `node blocks / edge blocks / local constraints`。
2. verifier 先检查局部块合法性，而不是直接做整图 yes/no。
3. 若发现某个 block 断裂，只在该 block 邻域做局部分支扩张。
4. 若局部修复成功，则把该 block 记入已验证检查点。

这使得 `nlgraph` 的 challenger 变成“针对某个断裂点的局部修复提案”，而不是另起一条整图候选。由于 `nlgraph` 本身就更接近有可靠 checker 的任务族，这条线理论上比 code task 更容易先跑出正收益。

##### A3. `v4.3-reasoning-local`：面向 `gsm8k / math`

对数值推理与数学任务，`v4.3` 不再把 `v4.2` 完全推翻，而是保留一条更轻的局部搜索变体：

1. 由 verifier 提取最可疑的子结论或局部等式变换。
2. 只对该子结论附近做局部重推与短分支扩张。
3. 若局部修复不能带来新的 verifier 证据，则回退到 `v4.2` 的轻量仲裁。

这条线不是 `v4.3` 的第一优先级，因为当前最紧迫的问题不是 `gsm8k` 上的少量有害 override，而是 code task 上 challenger 根本起不来。

#### 模块 B：Verifiable Restart

`restart` 在 `v4.3` 中不是“重试一次”这种工程化补救，而是一个明确的算法动作。

我准备把 restart 定义为：

1. 维护一个已验证检查点集合 `C_t`。
2. 每当 verifier 确认某个局部状态合法，就把它写入 `C_t`。
3. 若当前路径出现不可调和冲突，或连续若干轮没有任何严格改进，则回滚到最近检查点。

这里的“最近检查点”不是抽象的 global latent，而是 verifier 真正认可过的局部合法状态。这样 restart 的语义就是：

`restart = rollback to the latest verifier-approved local state`

它解决的不是“多探索几次”，而是回声室问题：防止 agent 在同一条错误路径上通过多轮对话不断自我强化错误。

#### 模块 C：稀疏动态执行

在 `v4.2` 中，实际运行仍非常接近“每轮所有 task nodes 都执行一次”。这对 verifier 驱动的局部搜索是不合适的。

因此，`v4.3` 的第三个公共机制是 task-aware 的稀疏执行：

1. 不是所有节点每轮都执行。
2. 不是所有边每轮都传播。
3. 当前 failure event 没涉及到的子图，默认不激活。
4. 只有围绕 `repair focus / failure block / subclaim` 的局部子图被激活。

这一步的意义不是单纯省 token，而是让 Stage2 的轨迹分布真正从“固定图上的重复传播”转成“围绕失败事件的局部搜索”。

#### 第一波实现顺序

为了让明天开始改代码时路线清晰，`v4.3` 的实现顺序应明确写死为：

1. 先做 `v4.3-code-repair`
   - 主跑 `mbpp`
   - 再跑 `humaneval`
2. 再做 `v4.3-graph-local`
   - 主跑 `nlgraph`
3. `v4.3-reasoning-local` 作为后续增强
   - 面向 `gsm8k / math`

不建议一开始把三条线一起实现，否则很容易又回到“大而全但没有一条线真正打通”的状态。

#### 预计代码落点

`v4.3` 不建议继续简单继承 `v4.2` 的 finalizer 链，而应单独起一个 runtime 家族。建议的代码落点为：

- `mas_stage2_v4_3/config.py`
  - 新增 `protocol_family`、`branch_budget`、`checkpoint_budget` 等协议级配置
- `mas_stage2_v4_3/runtime.py`
  - 实现 `Detect -> Localize -> Expand -> Execute -> Select -> Restart`
- `mas_stage2_v4_3/protocols/base.py`
  - 抽象各任务协议的接口
- `mas_stage2_v4_3/protocols/code_repair.py`
  - `mbpp / humaneval` 的 patch-loop
- `mas_stage2_v4_3/protocols/graph_local.py`
  - `nlgraph` 的 block-level repair
- `mas_stage2_v4_3/pipeline.py`
  - 对齐新元数据统计
- `test_stage2_v4_3.py`
  - 覆盖 failure event、strict improvement、checkpoint restart 等核心机制

`v4.2` 中的 `explicit challenger / DAR / pairwise public view / calibration` 不必删除，但对 code / graph 协议来说，它们不再是主决策链，只作为轻量 fallback 或日志参考。

---

### 5.3 `v4.4`：任务感知的协议路由 + 预算感知模式选择

#### 核心判断

一旦 `v4.3` 让 code、graph、reasoning 任务拥有了不同的 verifier affordance，“统一协议”本身就不再成立。`v4.4` 的核心不是单纯节省 token，而是正式放弃“一刀切”的 Stage2 协议，把不同任务送入不同的 Stage2 机制。

更准确地说，`v4.4` 解决两个问题：

1. 当前样本应该走哪一类协议。
2. 在该协议内部，应走 `Bypass / Lean / Full` 中的哪一种深度。

#### 先分协议，再分模式

`v4.4` 的路由输出不再是单一模式，而是一个二元决策：

`route(x) = (protocol_family, execution_mode)`

其中 `execution_mode` 保持三档：

1. `Bypass`
   - 直接保留 `stage1 anchor`
2. `Lean`
   - 只运行轻量协议，不做局部扩张或只做一次轻量修复
3. `Full`
   - 运行完整的局部分支、checkpoint 与 restart

#### 首批三条主协议

首批正式纳入 `v4.4` 的三条协议如下：

| 协议 | 任务 affordance | 代表数据集 | challenger 定义 | 主判据 |
|---|---|---|---|---|
| `v4.4-Adversarial` | 没有强执行 oracle，但可做结构化反证 | `gsm8k`, `math`，以及 `mmlu` 的轻量版本 | 竞争性假设 | ACH + calibration |
| `v4.4-Code-Repair` | 有可执行产物，可由失败反馈局部修复 | `mbpp`, `humaneval` | 修复已知失败点的 patch | verifier strict dominance |
| `v4.4-Graph-Constrained` | 有局部结构约束，可做 block-level 检查 | `nlgraph` | 修复断裂 block 的局部候选 | 局部约束增益 |

需要强调两点：

1. 这些协议不是按 benchmark 名字手工定制，而是按任务 affordance 划分。
2. `mmlu` 不应该被强行送入重型 `Full Stage2`，它更适合 `Adversarial` 协议下的 `Bypass / Lean`。

#### 对 `knowledge_crosswords` 的处理

`knowledge_crosswords` 不适合直接并入 `Graph-Constrained`。它更接近 `slot-wise repair`，因此在 `v4.4` 中应被明确标记为“后续保留协议”：

- 首批不实现单独的 `Structured-Repair`
- 现阶段默认只允许它走 `Lean` 路线
- 不强行把它塞进 `Full Graph-Constrained`

这样做可以避免为了追求统一协议而在 structured task 上再次引入系统性失配。

#### 路由器的输入

`v4.4` 的路由器应使用少量、语义明确的输入，而不是一长串阈值：

- `DatasetProfile.task_type`
- `answer_format`
- `stage1` 不确定性
- 当前 verifier affordance 是否强
- candidate disagreement
- 最近几轮 failure event 密度
- 剩余 token / latency budget

这里 `DatasetProfile` 只提供任务先验，不直接决定最终动作；最终仍由路由器根据当前状态决定走 `Bypass / Lean / Full`。

#### 与 `v4.3` 的关系

`v4.4` 不是替代 `v4.3`，而是把 `v4.2` 与 `v4.3` 重新组织成“按任务调用”的体系：

1. 对 reasoning 任务，保留 `v4.2` 资产，形成 `Adversarial` 协议。
2. 对 code / graph 任务，把 `v4.3` 作为主协议。
3. 对高成本或低收益任务，优先 `Bypass` 或 `Lean`。

换句话说，`v4.2` 不应被彻底废弃；它更适合作为 reasoning 任务的轻量分支，而不再被当成所有任务的统一 Stage2。

---

### 5.4 `v4.5`：基于可验证进展的训练信号重构

#### 核心判断

在 `v4.3` 和 `v4.4` 之前，训练器学到的大概率只会是“preserve 更安全”。因此 `v4.5` 的前提是：协议本身已经能稳定地产生 challenger、局部修复、checkpoint 与 restart。

`v4.5` 的目标不是发明更多 reward，而是把 verifier 已经提供的结构化事件转化为更干净的学习信号。

#### 方向 A：preserve 路径中立化

首先要明确一条原则：

- `preserve anchor` 是合法动作，不应额外惩罚。

否则系统会再次学出“为了学 override 而学会错误 override”的偏差。`v4.5` 的学习目标必须保持：

1. preserve 中立
2. 正确修复受奖
3. 无效分支与错误重启受罚

这和“通过惩罚 preserve 来逼模型多动”是完全不同的逻辑。

#### 方向 B：事件级 credit assignment

`v4.5` 不再只用整条 trajectory 的最终成败做唯一回报，而是把 credit 下沉到 verifier 事件：

- 触发了有效 failure localization
- 生成了严格改进的局部分支
- 正确执行了 restart
- 消除了某个已知失败点

也就是说，信用单元不再是“整条轨迹看起来像不像正确”，而是“这一步是否让 verifier 认可的状态真正前进了”。

#### 方向 C：Ordered Verification Potential

为了避免重新掉回多目标加权函数，`v4.5` 不建议把很多局部指标做线性加权求和，而建议引入“有序验证层级”或“有序验证势函数”。

具体做法是：

1. 对每类任务定义一个有序的验证层级 `L0 < L1 < ... < LK`
2. 训练时只奖励层级上升，惩罚层级下降
3. 若需要多维状态，则用字典序或偏序，而不是人工写总分

例如：

- code task：
  - `L0`: 语法无效
  - `L1`: 语法有效但入口不合法
  - `L2`: 入口合法但已知失败未修复
  - `L3`: 修复了至少一个已知失败点
  - `L4`: 通过全部可见测试
- graph task：
  - `L0`: 输出格式不合法
  - `L1`: 格式合法但局部约束明显冲突
  - `L2`: 关键局部块合法
  - `L3`: 已修复主要断裂点
  - `L4`: 整体结构合法
- reasoning task：
  - `L0`: 输出合同不合法
  - `L1`: 主结论存在明显矛盾
  - `L2`: 关键子结论被 verifier 支持
  - `L3`: 主结论与已验证子结论一致
  - `L4`: 最终答案正确

这样定义后，最自然的 shaping 形式就是：

`r_t^shape = level(s_{t+1}) - level(s_t)`

它的优点是：

1. 直接依赖 verifier 可观察事件。
2. 不需要人为堆很多权重。
3. 能和 `v4.3` 的局部修复动作天然对齐。

#### 方向 D：agent-wise normalization

在信用已经事件化之后，再做角色内归一化才有意义。此时 `agent-wise normalization` 的作用是：

1. 避免 solver、verifier、skeptic 这些角色因为天然 reward 分布不同而互相污染。
2. 让“修复型角色”和“质疑型角色”的梯度规模可比。

这里的重点是先有干净的事件信用，再做角色内标准化，而不是在脏信号上直接归一化。

#### 方向 E：明确不做什么

`v4.5` 应明确排除以下做法：

1. 不把 `sim(y_failed, y_correct)` 当成跨任务统一主 reward。
2. 不把 parse、tests、distance、uncertainty、consistency 等全部揉成一个手工总分。
3. 不通过给 preserve 加额外负奖励来“逼出 challenger”。

语义相似度最多只能做辅助排序，不能成为主训练目标。

---

### 5.5 `v4.6`：协议级 coordinator 与 RL / MARL

#### 核心判断

`v4.6` 的学习对象不应再是“直接学答案”，而应是“学如何调用已经定义清楚的协议动作”。只有当 `v4.3 -> v4.5` 已经把协议、状态和信用都写清楚以后，coordinator policy 才值得进入。

#### 推荐形态

更合理的长期形态是三层：

- 顶层：protocol coordinator
- 中层：branch / restart / route manager
- 底层：worker agents

此时 coordinator 的动作空间应尽量小且语义明确：

- `bypass`
- `lean`
- `full`
- `spawn_local_branch`
- `restart_from_checkpoint`
- `stop_and_preserve`

这意味着 `v4.6` 学习的是“协议控制”，不是“再造一个更大的 judge”。

#### 状态输入

只有以下状态已经稳定可观测时，`v4.6` 才有意义：

- 任务协议族
- `stage1` 不确定性
- candidate disagreement
- 最近 failure event 序列
- 当前 checkpoint 数与最近 restart 结果
- 当前预算消耗
- 当前 verifier 层级

否则 coordinator 只会学到最稳定但最无趣的动作：始终 preserve。

#### 为什么必须放到最后

如果在协议还没定义清楚时就把 RL / MARL 拉进来，policy 只会放大当前偏差，而不是修复协议本身的问题。当前最需要的不是“更会学”，而是“先给它一个值得学的动作空间”。

---

## 6. 版本之间的衔接关系

### 6.1 `v4.1 -> v4.2`

`v4.1` 先解决“错误 override 太危险”，`v4.2` 再解决“为什么 challenger 进不来”。二者仍然是先后关系，而不是替代关系。

`v4.1` 留下的关键资产包括：

1. 显式 challenger 的出现率
2. inspector 对 challenger 的拒绝 / uncertain / 放行统计
3. DAR 保留后异见候选的覆盖情况

`v4.2` 则进一步把 challenger 的资格从“固定 agent 身份”扩展到“有 auditor 支持的竞争假设”。

### 6.2 `v4.2 -> v4.3`

当前经验已经给出一个很明确的推进条件：

1. `gsm8k` 上 `v4.2` 虽然比 `v4.1` 更“活”，但有效 override 仍几乎没有形成。
2. `humaneval` 上更明显，当前连 `provisional challenger` 都很难出现。
3. 若 `mbpp` 结果也延续这一趋势，那么就说明 `v4.2` 的主要问题不是“judge 还不够准”，而是 code task 的 challenger 定义本身错了。

在这种条件下，下一步就不应继续调 `v4.2` 的末端仲裁，而应直接进入 `v4.3-code-repair`。

### 6.3 `v4.3 -> v4.4`

只有当至少两条不同协议真的存在时，`v4.4` 的 protocol routing 才不是空壳。因此实际顺序应是：

1. 先打通 `v4.3-code-repair`
2. 再补 `v4.3-graph-local`
3. 然后再做 `v4.4` 的协议路由器

### 6.4 `v4.4 -> v4.5`

只有在 route 与 mode 都已成形后，训练器才可能学到“什么时候该保守、什么时候该扩张、什么时候该 restart”。否则事件级 credit assignment 只会面对单一、贫瘠、不可比的轨迹。

### 6.5 `v4.5 -> v4.6`

当 verifier 层级、事件信用和动作空间都稳定后，coordinator policy 才有足够干净的状态与回报去学习。此时 RL / MARL 学到的才是 protocol policy，而不是放大现有偏差的 preserve-only 策略。

---

## 7. 实验建议

### 7.1 `v4.1` 当前继续跟踪的指标

建议继续保留 `v4.1` 的安全基线指标：

1. 最终 `success`
2. `beneficial override` 数量
3. `harmful override` 数量
4. `explicit challenger` 触发率
5. `inspector` 的 `anchor / challenger / uncertain` 分布
6. 平均 token cost

`v4.1` 的价值主要是“安全基线”，不是长期主线。

### 7.2 `v4.2` 的核心验证指标与止损条件

`v4.2` 仍应继续记录：

1. `provisional challenger` 覆盖率
2. 独特假设数 `|H|`
3. adjudication 路径占比
4. calibration 放行率
5. override precision

但对 `v4.2` 的评价标准应更严格：

1. 若 `gsm8k` 只是让 adjudication 路径稍微增多，却没有形成有效收益，则说明它只修了表层活性。
2. 若 `humaneval` 与 `mbpp` 上长期接近 `0` 个 `provisional challenger`，则说明它不是 code task 友好的统一协议。
3. 一旦出现上述模式，就不应继续把 `v4.2` 当成普适主线，而应把它收缩为 reasoning 任务的轻量协议。

### 7.3 `v4.3` 的核心验证指标

`v4.3` 的指标需要分成公共指标与协议专属指标。

公共指标：

1. failure event 触发率
2. strict improvement 分支率
3. checkpoint 使用率
4. restart 触发率与 restart 后恢复率
5. 激活子图比例
6. success 与 token / latency 的 Pareto 关系

`v4.3-code-repair` 专属指标：

1. counterexample 触发后的有效 repair rate
2. patch 分支中能修复已知 failure 的比例
3. function-body rewrite 被触发的比例
4. 可见测试通过数的净提升

`v4.3-graph-local` 专属指标：

1. block-level 失败定位率
2. 局部修复成功率
3. repaired block 被后续破坏的比例
4. 全局图合法性的净提升

### 7.4 `v4.4` 的核心验证指标

`v4.4` 的重点不再只是 `Bypass / Lean / Full` 的占比，而是“协议是否真的按任务族分化”：

1. `protocol_family` 分布
2. 各协议下 `Bypass / Lean / Full` 的样本占比
3. 各协议的成功率
4. 各协议的平均 token / latency
5. route ablation 后的退化幅度
6. success-cost Pareto 是否改善

对 `mmlu` 与 `knowledge_crosswords`，还应特别观察：

1. 是否被错误送入重型协议
2. `Bypass / Lean` 是否已经足够

### 7.5 `v4.5` 与 `v4.6` 的核心验证指标

当进入学习层升级后，应重点观察：

1. verifier 层级上升事件与最终成功的相关性
2. event-level credit 是否比 trajectory-only reward 更稳定
3. agent-wise advantage 方差是否更稳定
4. 不同角色的梯度规模是否更均衡
5. coordinator policy 的动作分布是否真的依赖状态
6. coordinator 是否避免退化为 preserve-only

---

## 8. 当前结论

当前结论应更新为：

1. `v4.1` 的价值已经明确，它是必要的安全基线。
2. `v4.2` 的价值也已经明确，它适合作为 reasoning 任务的轻量 challenger / adjudication 分支，但不是 code task 友好的统一协议。
3. 当前 `gsm8k` 与 `humaneval` 的证据共同说明：下一步最值得做的，不是继续围绕 `v4.2` 末端仲裁打补丁，而是进入 `v4.3`，把 Stage2 改成 verifier 驱动的局部搜索。
4. 若 `mbpp` 明天的结果也没有出现有效修复或有效翻案，则应立即启动 `v4.3-code-repair` 的代码实现。
5. `v4.4` 的作用不是“再加一个路由器”，而是正式承认不同任务族需要不同协议。
6. `v4.5` 与 `v4.6` 都必须建立在协议先被写清楚、先能跑通的前提上。

换句话说：

- `v4.1` 解决“错误地翻案”
- `v4.2` 解决“为什么没人能合法挑战”
- `v4.3` 解决“为什么系统搜不出 verifier-backed challenger”
- `v4.4` 解决“不同任务应该走哪条协议、走多深”
- `v4.5` 解决“如何把 verifier 认可的进展学进去”
- `v4.6` 才解决“如何让 coordinator 学会调协议、调分支、调 restart”
