# Stage2-Rollback 整体设计

更新时间：2026-04-17

## 0. 绑定结论

本文档是 `stage2-rollback/` 的正式设计约束，以下结论全部视为硬约束，不作为可选建议：

1. 新实现目录固定落在 repo 根目录 `stage2-rollback/`，与 `mas_stage2_v4_4/` 同级。
2. 概念基线固定为 `2026-04-17` 的提交 `362aff3`。
3. 代码壳参考固定为 `Stage2-UCC/stage2_phase3a_unified@362aff3`。
4. 对象语义、verifier 语义与 safe override 语义参考固定为：
   - `Stage2-UCC/UCC整体设计.md@362aff3`
   - `Stage2-UCC/stage2_phase1_semantic_safe_override/`
5. 实现约束固定为：不 import、不继承、不 vendor 上述目录中的 stage2 私有实现。
6. 不从 `Stage2-GCR+` 的 `runtime_v44` 直接 fork，不继承 `route_family`、`execution_mode`、`code-only recovery 主链`、`hard guard` 逻辑。
7. `runtime_v2.py`、`runtime_v41.py`、`runtime_v44.py` 只作为参考对象，不能作为代码继承链。
8. 老代码保持原样不动；`stage2-rollback/` 只新增一套完全隔离的 `pure rollback / local rerun` 实现。
9. `stage2-rollback/` 后续必须落一版独立代码；旧版 stage2 实现只能阅读和借鉴，不能共享实现文件。

一句话总结：

`stage2-rollback` 保留原始 stage2 的物理世界观，即 frozen `UnionGraph` + Global Controller Node + Per-Agent Episodic Memory；但在新目录里独立重写一套 `pure rollback / local rerun` 二阶段算子，不改写老实现。

## 1. 基线如何拆分

这里的“基线”必须拆成两部分理解，不能只写成“参考 `phase3a_unified`”。

### 1.1 代码壳参考

代码壳参考固定为 `Stage2-UCC/stage2_phase3a_unified@362aff3`，原因是：

- 它已经是当前 UCC 的独立 home。
- 它具备完整 runtime、pipeline、candidate bank、replay logging、训练入口。
- `UCC整体设计.md` 已经明确把它降级为 `P0: Bootstrap Teacher Scaffold`。
- 这意味着它适合作为“统一接口、统一数据流、teacher 轨迹”的外壳参考。

### 1.2 语义参考

方法语义不能只看 `phase3a_unified`，因为当前仓库里正式的语义阶段已经右移：

- `Stage2-UCC/UCC整体设计.md@362aff3`
  - 明确了 `P0 -> 阶段一 -> 阶段二 -> 阶段三` 的新阶段定义。
- `Stage2-UCC/stage2_phase1_semantic_safe_override/`
  - 对应正式的“阶段一：Unified Semantic Verification & Safe Override”。
  - 负责 `ArtifactIR++`、answer/evidence-aware verifier、pairwise overturn、safe override。

仓库事实是：`Phase1SemanticSafeOverrideRuntime` 直接继承 `Phase3aUnifiedRuntime`。因此，`stage2-rollback` 的正式表述必须是：

- 代码壳参考 `phase3a_unified`
- 对象语义、verifier 语义与 safe override 语义参考 `UCC整体设计.md + stage2_phase1_semantic_safe_override`
- 实现上不 import、不继承、不 vendor 这些目录里的 stage2 私有实现

## 2. 为什么不直接从 `Stage2-GCR+` 或 `mas_stage2_v4_4` 起 fork

`Stage2-GCR+ runtime_v44` 虽然最接近 graph-native rerun，但它不适合作为本方案基座，原因如下：

- 它已经强绑定 `route-specific collapse`。
- 它带有明显的 `code-main-chain` 倾向。
- 它依赖 `protected ids / sink guards / hard final guard`。
- 它的 `route_family / execution_mode` 是任务特异离散 policy。

这些都和当前目标冲突：当前目标要求统一一套 pure rollback 算子，并显式禁止任务特异离散 policy。

因此，正式约束如下：

- 代码壳参考：`Stage2-UCC/stage2_phase3a_unified@362aff3`
- 语义参考：
  - `Stage2-UCC/UCC整体设计.md@362aff3`
  - `Stage2-UCC/stage2_phase1_semantic_safe_override/`
- 借鉴对象：
  - `Stage2-GCR+/stage2_gcr_plus/runtime_v2.py` 的 rerun 外壳
  - `Stage2-GCR+/stage2_gcr_plus/runtime_v41.py` 的 candidate-bank / provenance 绑定
  - `Stage2-GCR+/stage2_gcr_plus/runtime_v44.py` 的 recovery provenance 字段设计
- 明确不继承：
  - `route_family`
  - `execution_mode`
  - `code-only recovery 主链`
  - `hard guard`
  - 所有 route-specific collapse

## 3. 代码独立性约束

这是本设计最重要的工程边界。

### 3.1 独立 home

`stage2-rollback/` 是新的独立 home，不是 `Stage2-UCC/` 的子目录，也不是 `Stage2-GCR+/` 的变体目录。

### 3.2 允许参考，不允许共用

允许：

- 阅读 `Stage2-UCC/stage2_phase3a_unified/`
- 阅读 `Stage2-UCC/stage2_phase1_semantic_safe_override/`
- 阅读 `Stage2-UCC/UCC整体设计.md`
- 阅读 `Stage2-GCR+/stage2_gcr_plus/runtime_v2.py`
- 阅读 `Stage2-GCR+/stage2_gcr_plus/runtime_v41.py`
- 阅读 `Stage2-GCR+/stage2_gcr_plus/runtime_v44.py`

不允许：

- 直接 import `Stage2-UCC/stage2_phase3a_unified.*`
- 直接 import `Stage2-UCC/stage2_phase1_semantic_safe_override.*`
- 直接 import `Stage2-GCR+/stage2_gcr_plus/runtime_v2.py`
- 直接 import `Stage2-GCR+/stage2_gcr_plus/runtime_v41.py`
- 直接 import `Stage2-GCR+/stage2_gcr_plus/runtime_v44.py`
- vendor 上述目录里的 stage2 私有实现
- 复制 correction 链并局部改名继续用
- 复用 `route_family / execution_mode / hard guard` 逻辑

### 3.3 共享依赖边界

第一版允许继续依赖 repo 级通用基础设施：

- `mas_treesearch` 的数据加载、dataset profile、evaluator、prompt slots、`UnionGraph` 类型
- 通用 agent pool / evaluator / embedding 接口
- 通用训练数据读写工具
- repo 级共享桥接类型 `PreparedStage1Artifact`

但 `stage2-rollback` 自己的以下模块必须独立实现：

- runtime
- pipeline
- artifacts
- trace IR
- verifier
- rollback boundary
- absorbing diffusion rerun
- rerun candidate emitter
- pairwise/listwise selector
- trainer

## 4. 新目录布局

后续代码固定落位如下：

```text
stage2-rollback/
  README.md
  ROLLBACK整体设计.md
  stage2_rollback/
    __init__.py
    artifacts.py
    trace_ir.py
    verifier.py
    boundary.py
    diffusion.py
    emitter.py
    selector.py
    runtime.py
    pipeline.py
    train_bank.py
    train_verifier.py
    train_boundary.py
    train_diffusion.py
    train_selector.py
```

其中：

- `artifacts.py`：在新目录内独立重写 `ArtifactIR++`
- `trace_ir.py`：负责 stage1 压缩 trace 与 replay brief
- `verifier.py`：rollback verifier，不服务 local edit
- `boundary.py`：rollback 边界打分
- `diffusion.py`：带吸收态的局部 rerun 子图算子
- `emitter.py`：rerun candidate 发射与 request build
- `selector.py`：pairwise/listwise safe selector
- `runtime.py`：pure rollback 主 runtime
- `pipeline.py`：沿用 frozen `UnionGraph` 的 stage2 入口，但不复用旧 UCC pipeline 文件
- `train_bank.py`：确定性的 teacher bank 构建
- `train_verifier.py / train_boundary.py / train_diffusion.py / train_selector.py`：分阶段训练

## 5. 保留什么，新代码不纳入什么

### 5.1 保留的世界观

需要保留的是物理壳，而不是旧算子：

- frozen `UnionGraph`
- Global Controller Node
- Per-Agent Episodic Memory
- candidate bank
- replay / trajectory logging
- repo 级共享桥 `PreparedStage1Artifact`

### 5.2 新代码明确不纳入的旧二阶段链路

以下链路属于旧的 local self-correction 主链。老代码保持不动；但在 `stage2-rollback/` 这条新实现里，不复用、不继承、不改造后沿用：

- `localize_units`
- `deterministic_critique`
- `deterministic_proposal`
- `apply_correction_artifact`
- `predict_delta`
- `preserve_heatmap`
- 整条 `localize -> preserve -> propose -> apply -> delta -> value`
- 所有 `replace / insert_before / insert_after / delete / reorder` 局部文本编辑路径

### 5.3 新的 operator 链

新方案统一写成：

```text
shared prepared bridge
  -> local rollback adapter
  -> trace compress / replay brief
  -> verifier latent maps
  -> rollback boundary
  -> absorbing diffusion rerun
  -> rerun emitter
  -> safe selector
```

这不是 correction 的变体，而是另一条 operator 主链。

## 6. 总体目标

本方案的目标不是继续做“更聪明的 self-correction”，而是：

1. 把错误 unit 投影回 stage1 的错误后缀。
2. 在 frozen union graph 上重新执行局部 rerun。
3. 从 rerun 子图重新发射候选，而不是对旧答案做文本手术。
4. 用 learned safe override 决定是否推翻 anchor。

一句话：

它修的是“搜索后缀”，不是“答案表面”。

## 7. 输入边界：共享桥不动，runtime 壳归 stage2

这一节必须严格区分：

- repo 级共享桥
- `stage2-rollback` 本地 adapter
- stage2 自己的 runtime 壳

### 7.1 共享桥保持不动

repo 级共享桥仍然是 `mas_stage2/structure_io.py` 中定义的：

```text
PreparedStage1Artifact(
    question_text,
    union_graph,
    stage1_signature,
    stage1_output,
    dataset_name,
    stage1_summary,
    structure_summary,
    metadata,
)
```

这个 dataclass 是全 repo 共用桥接类型，`stage2-rollback` 不修改它，不扩字段，不回写定义。

### 7.2 stage1 提供什么

对 `stage2-rollback` 而言，stage1 直接提供的共享输入只有：

- 问题 `q`
- `PreparedStage1Artifact base_prepared`
- 其中的 `union_graph`
- 其中的 `stage1_output / stage1_signature / metadata`

### 7.3 stage2 自己持有什么

下面这些是 stage2 的 runtime 壳，不是 stage1 直接吐给本方案的对象：

- frozen `UnionGraph G = base_prepared.union_graph`
- Global Controller Node 状态 `g`
- 每个图节点的 Per-Agent Episodic Memory `M_v`
- anchor warm-start pass 生成的 replay cache

也就是说，stage2 依赖 stage1 产出的 union graph，但不把 stage2 runtime 壳误写成“stage1 输出”。

### 7.4 压缩 trace 必须可执行

新增一个压缩 trace：

```text
T = {tau_j}_{j=1..L}
tau_j = (s_j, a_j, o_j, Pi_j, nu_j, b_j)
```

各字段含义：

- `s_j`：stage1 第 `j` 步内部状态摘要
- `a_j`：该步扩展动作或控制决策摘要
- `o_j`：该步局部输出摘要
- `Pi_j ⊆ V × E`：该步触碰到的 provenance 子图
- `nu_j`：该步局部得分、不确定度、frontier 信号
- `b_j`：可重放的 prompt brief 或 controller brief

这里：

- `a_j` 保证 trace 不只是 post-hoc diagnosis，而是能支持“从哪里重新长”。
- `b_j` 保证第一版在不改 executor 的前提下，仍能构造可执行 rerun request，而不是把裸张量直接喂给 executor。

没有压缩 trace，就无法做“回到安全边界”的 rollback，因此 `trace_ir.py` 是第一优先级前置模块。

## 8. 本地 adapter：不修改共享桥

`stage2-rollback` 不扩展 repo 级 `PreparedStage1Artifact`，而是在本地新增一层 adapter：

```text
RollbackPreparedStage1Artifact(
    base_prepared=PreparedStage1Artifact(...),
    anchor_artifact=...,
    rollback_trace=...,
    anchor_replay_cache=...,
)
```

各字段含义：

- `base_prepared`
  - repo 级共享桥，保持原样不动
- `anchor_artifact`
  - `base_prepared.stage1_output` 经过本地 `artifacts.py` 解析后的 `ArtifactIR++`
- `rollback_trace`
  - 压缩后的 anchor 主路径、可执行动作摘要、brief、近邻分支摘要
- `anchor_replay_cache`
  - 不是 stage1 输出
  - 由 `stage2-rollback` 在 anchor warm-start pass 中生成
  - 包含：
    - `node_states`
    - `global_state`
    - `local_memory_cache`
    - `prompt_briefs / controller_briefs`

因此，`RollbackPreparedStage1Artifact` 是新目录的本地桥，不改 repo 共享桥。

## 9. 统一 `ArtifactIR++` 表示

候选统一拆成 unit 集合 `U = {u_i}_{i=1..N}`。

每个 unit 不再带任务特异标签，而是统一做四个 view 编码：

```text
z_i^(m) = Enc_m(u_i)
m in {surface, step, struct, exec}
```

全局 view 可靠性：

```text
rho = sparsemax(W_rho [zbar(surface); zbar(step); zbar(struct); zbar(exec)])
```

unit 表示：

```text
h_i = sum_m rho_m W_m z_i^(m)
```

soft role：

```text
r_i = softmax(W_r h_i) in Delta^3
```

三维分别对应：

- `answer`
- `evidence`
- `mixed`

这里保留 `mixed` 不是装饰项。后续 boundary、diffusion、selector 都必须显式使用 `mixed` 通道，不能让它成为死通道。

## 10. trace-unit-graph 三重对齐

pure rollback 的关键不是只看答案哪里错，而是把错误 unit 投影回：

- stage1 的哪段后缀
- union graph 的哪些节点

### 10.1 trace 编码

trace step 表示：

```text
x_j = [e_s(s_j), e_a(a_j), e_o(o_j), e_Pi(Pi_j), nu_j, e_b(b_j), rho_j^ctr]
g_j = BiGRU(x_1:L)_j
```

额外定义一个 no-rerun 虚拟边界状态：

```text
g_0 = g_empty
```

### 10.2 unit -> trace 对齐

为同时支持真实 trace step 与 no-rerun latent slot，定义边界索引域为：

```text
j in {0, 1, ..., L}
```

其中：

- `j = 0` 表示 no-rerun latent slot
- `j >= 1` 对应真实 trace step

于是 unit 到 trace 的软对齐定义为：

```text
B_ij = softmax_{j in {0, ..., L}}((h_i)^T W_b g_j)
```

解释：

- `B_i0` 表示 unit `u_i` 被直接解释为“无需回滚”的质量
- `B_ij`（`j >= 1`）表示 unit `u_i` 来自真实 trace 第 `j` 步的质量

### 10.3 unit -> graph node 对齐

设图节点编码为 `s_v`，定义：

```text
R_iv = softmax_v((h_i)^T W_n s_v)
```

`R_iv` 表示 unit `u_i` 与图节点 `v` 的软绑定程度。

### 10.4 对齐一致性

设 `pi_{Pi_j}` 是 provenance 子图 `Pi_j` 对应节点集合上的均匀分布。
训练时加入结构一致性项：

```text
L_align = sum_i sum_{j=1}^L 1[|Pi_j| > 0] B_ij KL(pi_{Pi_j} || R_i)
```

这里显式跳过：

- `j = 0` 的 no-rerun latent slot
- 空 provenance 的 step

这一步强制：

- 如果某个 unit 更像来自第 `j` 步
- 那么它在图节点上的对齐也应更像落在 `Pi_j` 所覆盖的 provenance 区域

## 11. rollback verifier

新 verifier 不服务 local edit，而服务 rollback。

对每个 unit 输出三个连续量：

### 11.1 correctness belief

```text
c_i = sigma(f_c(h_i))
```

### 11.2 support belief

```text
s_i = sigma(f_s([h_i, sum_v R_iv s_v]))
```

### 11.3 keep belief

这个量不表示“别改它”的硬规则，而表示它更可能属于安全前缀：

```text
k_i = sigma(f_k([h_i, sum_j B_ij g_j]))
```

### 11.4 三张连续热图

```text
epsilon_i = 1 - c_i
mu_i = 1 - s_i
kappa_i = k_i
```

含义：

- `epsilon_i`：错误热度
- `mu_i`：缺证据热度
- `kappa_i`：安全前缀强度

### 11.5 role-aware 风险通道

后续模块统一使用四个 role-aware 风险通道，而不是过早压成一个标量：

```text
w_i^ans     = r_i,ans epsilon_i
w_i^evd     = r_i,evd mu_i
w_i^mix,eps = r_i,mix epsilon_i
w_i^mix,mu  = r_i,mix mu_i
```

这样 `mixed` 会在 boundary、diffusion、selector 中持续生效，不会变成“模型把难定义 unit 全塞给 mixed，然后后续完全不用”。

### 11.6 typed support 与 signature displacement

为让 rollback 在 `MMLU-Pro`、`MATH` 这类 typed-answer 任务上更稳，阶段二不只维护 generic `answer/evidence consistency`，还要显式维护候选级 typed support。对任一 candidate `A`，设其 typed answer signature 为 `sig_A`，则定义：

```text
H_A^ans = {h_i | u_i in A^ans}
H_A^evd = {h_i | u_i in A^evd}
```

```text
s_A^type = sigma(
  f_type(
    [Pool(H_A^ans), Pool(H_A^evd), e(sig_A), Pool({r_i}_{u_i in A})]
  )
)
```

其中：

- `s_A^type` 表示“当前 evidence units 是否真的支持当前 typed answer object”
- 这不是 lexical consistency，而是 typed-support consistency
- 对 `MMLU-Pro`，它对应“当前证据是否真的支持该 option 标签”
- 对 `MATH / GSM8K`，它对应“当前证据链是否真的支持该 numeric object”

定义 anchor-relative typed support 增量：

```text
Delta_type(c, a) = s_c^type - s_a^type
```

同时定义候选与 anchor 的 signature displacement：

```text
d_sig(c, a) = 1 - cos(e(sig_c), e(sig_a))
```

解释：

- `d_sig` 大，表示候选与 anchor 的答案对象差异更大
- `Delta_type` 大，表示候选对自己答案对象的 typed support 更强

### 11.7 step-level contract residual

为让 rollback 更早发现 `MATH` 类任务里的 plan drift、substitution drift、constraint drift，trace 侧必须显式维护 contract residual。对每个 trace step `j`，定义 step object：

```text
Y_j^step = ParseContract(o_j; dataset_name, answer_format, task_subtype)
```

再定义 step-level contract residual vector：

```text
rho_j^ctr = [r_j^parse, r_j^comp, r_j^const, r_j^exec]_j
```

其中：

- `r_j^parse`：该 step 输出对象是否能被 typed contract 解析
- `r_j^comp`：typed object 所需关键字段是否缺失
- `r_j^const`：对象级约束是否被破坏
- `r_j^exec`：可执行 / 可计算检查是否失败

这样边界网络就不会只盯“哪一段像错了”，还能看见“哪一段开始违背 typed contract”。

## 12. Rollback 边界选择

边界集合必须显式允许 no-rerun：

```text
j in {0, 1, ..., L}
```

其中：

- `j = 0` 表示 no-rerun
- `g_0 = g_empty` 是 no-rerun 的虚拟边界状态

为了避免把 no-rerun latent slot 混进 prefix 质量，定义：

```text
C_i0 = 0
C_ij = sum_{t=1}^j B_it,  j >= 1
```

也就是说：

- `B_i0` 只表示“无需回滚”的后验质量
- `C_ij` 只累计真实 trace step `1..j` 的 prefix 质量

推理期强约束：

- 若 `j* = 0`
- 则 runtime 直接 short-circuit 返回 anchor：
  - 不进入 diffusion
  - 不构造 rerun request
  - 不调用 emitter
  - candidate bank 仅保留 anchor

### 12.1 prefix / suffix mass

```text
M_j^pre = sum_i C_ij
M_j^suf = sum_i (1 - C_ij)
```

### 12.2 prefix safe mean

```text
Sbar_j = (1 / (M_j^pre + eps)) sum_i C_ij kappa_i
```

### 12.3 suffix risk means

```text
Rbar_j^ans     = (1 / (M_j^suf + eps)) sum_i (1 - C_ij) w_i^ans
Rbar_j^evd     = (1 / (M_j^suf + eps)) sum_i (1 - C_ij) w_i^evd
Rbar_j^mix,eps = (1 / (M_j^suf + eps)) sum_i (1 - C_ij) w_i^mix,eps
Rbar_j^mix,mu  = (1 / (M_j^suf + eps)) sum_i (1 - C_ij) w_i^mix,mu
```

这里显式拆开：

- `mass`
- `mean`
- `answer / evidence / mixed`

目的是避免：

- 因为 prefix 更长，`S_j` 总量天然更大
- 因为 suffix 更长，`R_j` 总量天然更大

也就是避免“范围大小”伪装成“质量高低”。

### 12.4 step-level micro feature encoder

单尺度 `step-level` 边界网络不再直接产出最终 `pi_j^rb`，而只负责生成供多尺度边界使用的 step-level feature：

```text
psi_j^micro = f_rb^feat(
  [g_j, Sbar_j, M_j^pre, Rbar_j^ans, Rbar_j^evd, Rbar_j^mix,eps, Rbar_j^mix,mu, M_j^suf, j / L]
)
```

也就是说：

- §12.4 不再定义最终 rollback posterior
- 最终边界后验只由 §12.5 的 macro / micro 两级结构给出
- `train_boundary.py` 与 runtime 都只消费同一套最终多尺度边界分布

### 12.5 多尺度 rollback 边界

单一 step-level rollback boundary 对代码和图任务已经够强，但对 `MATH` 往往太晚。为此，rollback 改成两级边界：

- 宏观 block 边界：先判断“哪一个推导块开始漂了”
- 微观 step 边界：再判断“块内从哪一步开始坏”

#### 12.5.1 macro block 构造

trace 不按任务标签切块，而按 replay brief / controller brief 的连续同类片段切块。设得到 `K` 个连续 macro blocks：

```text
B_1, ..., B_K
```

并保留 no-rerun slot：

```text
B_0 = {0}
```

对 `k >= 1`，定义 block summary：

```text
gbar_k = AttnPool({g_j}_{j in B_k})
Sbar_blk_k = (1 / |B_k|) sum_{j in B_k} Sbar_j
Rbar_blk_k^sup = (1 / |B_k|) sum_{j in B_k} (Rbar_j^evd + Rbar_j^mix,mu)
Rbar_blk_k^ctr = (1 / |B_k|) sum_{j in B_k} Pool(rho_j^ctr)
```

#### 12.5.2 macro posterior

```text
pi_tilde_k^macro = softmax_k(
  f_M([gbar_k, Sbar_blk_k, Rbar_blk_k^sup, Rbar_blk_k^ctr, |B_k| / L])
)
```

解释：

- 它先决定“应不应该 rollback”
- 如果应该 rollback，再决定“哪一个大块最值得回到”

#### 12.5.3 micro posterior

对每个真实 block `k >= 1`，在块内定义 step posterior：

```text
pi_tilde_{j|k}^micro = softmax_{j in B_k}(
  f_m([psi_j^micro, rho_j^ctr, gbar_k])
)
```

#### 12.5.4 最终边界后验

```text
pi_tilde_0^rb = pi_tilde_0^macro
pi_tilde_j^rb = sum_{k >= 1: j in B_k} pi_tilde_k^macro pi_tilde_{j|k}^micro
```

后续所有依赖边界后验的连续特征统一使用：

```text
pi_bar_j =
  pi_tilde_j^rb,  train
  1[j = j*],      infer
```

训练时，`L_rb` 只对最终的多尺度稠密后验 `pi_tilde^rb` 求损失。
推理时，仍取：

```text
j* = argmax_j pi_tilde_j^rb
```

这样：

- `MMLU-Pro` 往往会把 mass 压到 `j = 0` 或 very shallow evidence block
- `MATH` 则更容易把 mass 压到“最早开始偏离正确计划”的推导块，而不是最后一个错误算式

## 13. 局部 rerun 子图：吸收扩散，不用 hard guard

### 13.1 role-aware 节点风险特征

将 suffix 风险投影到图节点时，统一使用 `pi_bar`，且只对真实边界 `j = 1..L` 求和：

```text
q_v^ans     = sum_i R_iv w_i^ans     sum_{j=1}^L pi_bar_j (1 - C_ij)
q_v^evd     = sum_i R_iv w_i^evd     sum_{j=1}^L pi_bar_j (1 - C_ij)
q_v^mix,eps = sum_i R_iv w_i^mix,eps sum_{j=1}^L pi_bar_j (1 - C_ij)
q_v^mix,mu  = sum_i R_iv w_i^mix,mu  sum_{j=1}^L pi_bar_j (1 - C_ij)
```

### 13.2 节点前缀安全质量

```text
p_v^safe = sum_i R_iv kappa_i sum_{j=1}^L pi_bar_j C_ij
```

### 13.3 吸收门

```text
a_v = sigma(f_a([s_v, q_v^ans, q_v^evd, q_v^mix,eps, q_v^mix,mu, p_v^safe]))
```

### 13.4 扩散 seed 与零风险回退

先把 role-aware 风险通道压到一个非负 seed score：

```text
q_v = softplus(f_seed([q_v^ans, q_v^evd, q_v^mix,eps, q_v^mix,mu]))
```

seed 初值定义为：

```text
p^(0) =
  q / ||q||_1,   if ||q||_1 > 0
  1 / |V|,       if ||q||_1 = 0
```

### 13.5 边转移

```text
phi_uv = f_e([s_u, s_v, e_uv, q_u^ans, q_u^evd, q_u^mix,eps, q_u^mix,mu, q_v^ans, q_v^evd, q_v^mix,eps, q_v^mix,mu, p_u^safe, p_v^safe])
P(v | u) = softmax_{v in N+(u)}(phi_uv)
```

### 13.6 吸收扩散算子

令 `D(a)` 为以 `a_v` 为对角元的对角矩阵：

```text
P_tilde = D(a) + (I - D(a)) P
```

扩散步数与 rerun rollout 步数必须分离。定义扩散步数为 `T_diff`，则：

```text
p^(t+1) = P_tilde^T p^(t),   t = 0, ..., T_diff - 1
```

扩散终态后，训练时使用数值稳定的稠密节点后验：

```text
alpha_tilde_v = (p_v^(T_diff) + eps) / sum_w (p_w^(T_diff) + eps)
```

推理时再投影成稀疏 rerun 子图支持：

```text
alpha = sparsemax(p^(T_diff))
```

统一记：

```text
alpha_bar =
  alpha_tilde,  train
  alpha,        infer
```

真正用于 local rerun 的双侧 gating 转移定义为：

```text
P_rr(v | u) = alpha_bar_v P_tilde(v | u) / (sum_w alpha_bar_w P_tilde(w | u) + eps)
```

这里：

- sender 活跃性由 `alpha_bar_u` 控制
- receiver 是否属于 rerun 子图由 `alpha_bar_v` 控制

#### 13.6.1 `Diff / DiffTilde` 的统一 contract

为避免 `alpha^sup`、`alpha_tilde^ans`、`alpha_bar^ans` 在实现层漂移，扩散算子在文档层固定成两种 API：

```text
DiffTilde(q, P_tilde; T_diff) -> alpha_tilde
Diff(q, P_tilde; T_diff) -> alpha
```

其中：

- `DiffTilde`
  - 输入：初始 seed `q`、吸收转移 `P_tilde`、扩散步数 `T_diff`
  - 输出：训练期使用的稠密节点后验 `alpha_tilde`
- `Diff`
  - 输入：同上
  - 输出：推理期使用的稀疏节点支持 `alpha`

两者的关系固定为：

```text
alpha_tilde = DiffTilde(q, P_tilde; T_diff)
alpha = sparsemax(alpha_tilde)
```

因此后续所有阶段都遵守同一个桥接规则：

```text
alpha_bar =
  alpha_tilde,  train
  alpha,        infer
```

也就是说：

- `DiffTilde` 只负责产生训练期的 dense posterior
- `Diff` 只负责产生推理期的 sparse support
- `alpha_bar` 是唯一允许下游 rollout / emitter / selector 继续消费的统一接口

### 13.7 support-first 双通道 rerun

为同时适配 `MMLU-Pro` 与 `MATH`，local rerun 不再是一条单通道扩散，而改成：

1. 先做 `support rerun`：先修“为什么应该这么答”
2. 再做 `answer rerun`：再修“最后到底答什么”

#### 13.7.1 evidence-first seed

定义 evidence-oriented 节点种子：

```text
q_v^sup = sum_i R_iv (w_i^evd + w_i^mix,mu) sum_{j=1}^L pi_bar_j (1 - C_ij)
```

support-stage 吸收门：

```text
a_v^sup = sigma(f_a^sup([s_v, q_v^sup, p_v^safe]))
P_tilde^sup = D(a^sup) + (I - D(a^sup)) P
alpha_tilde^sup = DiffTilde(q^sup; P_tilde^sup)
alpha^sup = sparsemax(alpha_tilde^sup)
alpha_bar^sup =
  alpha_tilde^sup,  train
  alpha^sup,        infer
P_rr^sup(v | u) = alpha_bar_v^sup P_tilde^sup(v | u) / (sum_w alpha_bar_w^sup P_tilde^sup(w | u) + eps)
```

其中 `DiffTilde` 表示 support-stage 扩散在训练期对应的稠密节点后验版本。

#### 13.7.2 support-stage memory refresh

support rerun 不直接发射答案，只先刷新 evidence / memory / local context：

```text
m_v^sup = f_supmem([alpha_bar_v^sup, M_v, anchor_replay_cache, trace_suffix])
```

#### 13.7.3 answer-stage seed conditioned on support

再定义 answer-oriented 节点种子：

```text
q_v^ans = sum_i R_iv (w_i^ans + w_i^mix,eps) sum_{j=1}^L pi_bar_j (1 - C_ij)
```

并用 support-stage 输出进行调制：

```text
q_v^{ans|sup} = softplus(
  f_couple([q_v^ans, alpha_bar_v^sup, m_v^sup, p_v^safe])
)
```

answer-stage 吸收门：

```text
a_v^ans = sigma(f_a^ans([s_v, q_v^{ans|sup}, p_v^safe, alpha_bar_v^sup]))
P_tilde^ans = D(a^ans) + (I - D(a^ans)) P
alpha_tilde^ans = DiffTilde(q^{ans|sup}; P_tilde^ans)
alpha^ans = sparsemax(alpha_tilde^ans)
alpha_bar^ans =
  alpha_tilde^ans,  train
  alpha^ans,        infer
P_rr^ans(v | u) = alpha_bar_v^ans P_tilde^ans(v | u) / (sum_w alpha_bar_w^ans P_tilde^ans(w | u) + eps)
```

其中 `DiffTilde` 表示 answer-stage 扩散在训练期对应的稠密节点后验版本。

#### 13.7.4 与现有 rerun 动力学的衔接

从这一节开始，§14 与 §15 中真正用于 rerun rollout 和 candidate emission 的子图支持，统一替换为：

```text
a <- a^ans
alpha <- alpha^ans
alpha_bar <- alpha_bar^ans
P_rr <- P_rr^ans
```

也就是说：

- support-stage 只负责把 evidence / constraint 拉回正轨
- answer-stage 才负责真正的后缀重算与 candidate 发射

这个改动对 `MMLU-Pro` 的作用是：
如果 support-stage 没有形成足够强的证据净增，answer-stage 会自然收缩，最后更容易走 no-rerun / null emission。

对 `MATH` 的作用是：
先修约束、定义域、关键中间量，再重算最终 numeric object，而不是直接对最后答案做“补丁式重跑”。

## 14. 局部 rerun 动力学

以下 rerun 动力学独立运行 `T_run` 步。
其中：

- `T_diff`：只负责诱导 rerun 子图支持
- `T_run`：只负责在该子图内做局部重算

两者不共享步数，也不复用同一个时间符号。

### 14.1 初始化

设 `anchor_replay_cache` 中包含：

- `h_v^anchor`：anchor warm-start pass 的节点状态
- `g^anchor`：anchor warm-start pass 的全局状态
- `m_vk^anchor`：本地 memory brief / replay cache

并设 `h_hat_v^0` 为 fresh init。则：

```text
h_v^(0) = a_v^ans h_v^anchor + (1 - a_v^ans) h_hat_v^0
g^(0) = g^anchor
```

memory read 保持原定义：

```text
w_vk^(t) = softmax_k((W_q h_v^(t-1))^T (W_k m_vk))
c_v^mem,t = sum_k w_vk^(t) W_v m_vk
```

### 14.2 graph message

真正的 local rerun message passing 定义为：

```text
m_v^(t) = sum_u alpha_bar_u^ans P_rr^ans(v | u) W_m h_u^(t-1)
```

这一步中 sender 与 receiver 两端都被 rerun 子图约束，因此它是名副其实的 local rerun，而不是未受限的 global spread。

### 14.3 node update

```text
h_v^(t) = GRU(h_v^(t-1), [m_v^(t), c_v^mem,t, g^(t-1)])
```

### 14.4 global node update

```text
g^(t) = AttnPool({alpha_bar_v h_v^(t)}_{v in V})
```

其中这里的 `alpha_bar` 延续 §13.7.4 的约定，统一指向：

```text
alpha_bar <- alpha_bar^ans
```

这里完整保留原始 stage2 的三层结构：

- memory layer：`M_v`
- graph layer：`V`
- global node：`g`

而且没有任何任务分支。

## 15.0 null emission slot

为了让 rollback 在 `MMLU-Pro` 这类 closed-set 任务上真正稳定，emitter 必须显式允许“不发射任何新 candidate”。因此，在所有 rerun 节点发射 logit 之外，新增一个空槽：

```text
eta_null = f_null([g^(T_run), s_a^type, Pool(kappa), Pool(alpha_bar^sup), 1[j* = 0]])
```

训练期发射分布改成：

```text
gamma_tilde_plus = softmax([eta_1, ..., eta_|V|, eta_null])
```

推理期改成：

```text
gamma_plus = sparsemax([eta_1, ..., eta_|V|, eta_null])
```

运行规则：

- 若 `argmax gamma_plus = null`，则本轮不发射任何 rerun candidate
- 若 `j* = 0`，runtime 已在边界阶段 short-circuit 返回 anchor
- 若 `j* > 0` 但 `argmax gamma_plus = null`，则表示“虽然探测到后缀风险，但当前 support-stage 仍不足以支撑新的答案发射”

训练标签也同步扩成带空槽的分布：

```text
phat_emit_plus(v) = sum_{A emitted by v} phat_B(A)
phat_emit_plus(null) = phat_B(A^(0))
```

这样 no-rerun / null-emission 就不是“失败兜底”，而是 rollback 正常的一等公民输出。

## 15. candidate 发射：request-based rerun，不做 edit

pure rollback 的关键约束保持不变：

- 新候选来自 rerun 子图的重新发射
- 新候选不来自对旧答案的文本手术
- executor 第一版保持冻结，不接裸张量，只接 request 对象

对每个节点先计算发射 logit：

```text
eta_v = f_emit([h_v^(T_run), g^(T_run), alpha_bar_v])
```

再与 §15.0 中的空槽 `eta_null` 共同组成发射后验。

训练时使用带空槽的稠密发射后验：

```text
gamma_tilde_plus = softmax([eta_1, ..., eta_|V|, eta_null])
```

推理时再投影成带空槽的稀疏发射支持：

```text
gamma_plus = sparsemax([eta_1, ..., eta_|V|, eta_null])
```

其中：

- 训练阶段所有 emitter 监督都作用在 `gamma_tilde_plus` 上
- 推理阶段真正触发 candidate emission 的是 `gamma_plus` 中除 `null` 外的节点支持集

定义 rerun 子图：

```text
G_rr = supp(alpha^ans)
```

对所有 `gamma_plus(v) > 0` 且 `v != null` 的节点，先构造 request：

```text
X_v^rr = BuildRerunRequest(
  q,
  j*,
  v,
  G_rr,
  trace_suffix={tau_t}_{t > j*},
  memory_briefs={b_t}_{t > j*},
  anchor_replay_cache,
)
```

再调用冻结的统一 executor：

```text
A_v^rr = E(X_v^rr)
```

这样第一版的执行边界是清楚的：

- 不改 executor
- 不在这一版动 generator
- 先把“回滚到哪、局部重算哪里、从哪里重新发射”学对

每个 rerun 候选都必须带完整 provenance：

```text
prov(A_v^rr) = (
  origin_node_id,
  j*,
  rerun_subgraph_node_ids,
  rerun_subgraph_edge_ids,
  trigger_verifier_snapshot,
  request_brief_signature
)
```

这里可以借 `runtime_v44` 的 recovery provenance 字段设计思想，但字段实现仍须独立重写。

## 16. 最终选择：learned safe override

不回到手工线性 utility。

给定候选 `c` 与 anchor `a`，先构造候选级摘要 `z_c`、`z_a`，再先做跨候选 unit 对齐：

```text
Omega_ij = softmax_j((h_i^a)^T W_Omega h_j^c)
```

这个对齐矩阵定义了：

- anchor 的第 `i` 个 unit
- 与 candidate 的哪些 unit 最可比

只有在有了 `Omega_ij` 之后，差分特征才在同一个定义域上。

### 16.1 错误修复增量

```text
Delta_err(c, a) = sum_{i,j} Omega_ij r_i,ans^a (epsilon_i^a - epsilon_j^c)
```

### 16.2 证据补全增量

```text
Delta_sup(c, a) = sum_{i,j} Omega_ij r_i,evd^a (mu_i^a - mu_j^c)
```

### 16.3 mixed 通道增量

```text
Delta_mix_err(c, a) = sum_{i,j} Omega_ij r_i,mix^a (epsilon_i^a - epsilon_j^c)
Delta_mix_sup(c, a) = sum_{i,j} Omega_ij r_i,mix^a (mu_i^a - mu_j^c)
```

### 16.4 前缀保留匹配强度

仅有跨候选对齐矩阵 `Omega_ij` 还不够，因为：

```text
sum_j Omega_ij = 1
```

会导致旧版 `M_keep(c, a)` 退化成常数。
因此必须额外引入 candidate-dependent 的保留匹配强度：

```text
m_keep_ij = sigma(f_keep([h_i^a, h_j^c, ell_ij, delta_pos_ij]))
```

其中：

- `ell_ij`：词面 / 语义 overlap 特征
- `delta_pos_ij`：相对位置差特征

### 16.5 前缀保留质量

```text
M_keep(c, a) = sum_{i,j} Omega_ij kappa_i^a m_keep_ij
```

这样它才真正表示：

- anchor 中高-keep 的 unit
- 在 candidate 中有没有被保住

### 16.6 pairwise selector

给定候选 `c` 与 anchor `a`，先做跨候选 unit 对齐：

```text
Omega_ij = softmax_j((h_i^a)^T W_Omega h_j^c)
```

然后定义四类差分特征：

```text
Delta_err(c, a)     = sum_{i,j} Omega_ij r_i,ans^a (epsilon_i^a - epsilon_j^c)
Delta_sup(c, a)     = sum_{i,j} Omega_ij r_i,evd^a (mu_i^a - mu_j^c)
Delta_mix_err(c, a) = sum_{i,j} Omega_ij r_i,mix^a (epsilon_i^a - epsilon_j^c)
Delta_mix_sup(c, a) = sum_{i,j} Omega_ij r_i,mix^a (mu_i^a - mu_j^c)
```

于是 pairwise selector 写成：

```text
s(c, a) = f_sel(
  [z_c, z_a, z_c - z_a, Delta_err(c, a), Delta_sup(c, a), Delta_mix_err(c, a), Delta_mix_sup(c, a), M_keep(c, a)]
)
```

### 16.7 bank 上的 listwise 选择

对于当前 candidate bank：

```text
B = {a} union C
```

统一定义 anchor-relative 的标量选择分数：

```text
u_theta(a) = 0
u_theta(c) = s(c, a),   c in C
```

然后得到 listwise 选择分布：

```text
p_theta^sel(A | B) = exp(u_theta(A)) / sum_{A' in B} exp(u_theta(A'))
```

这意味着 selector 的训练与推理都基于同一个 bank-level 分布，而不是“pairwise 定义一套、listwise 训练再另起一套”。

selector 的判断标准只有四件事：

- 有没有修掉 answer 错误
- 有没有补上 evidence 缺口
- mixed 区域有没有整体更稳
- 有没有保住安全前缀

### 16.8 typed-support-preserving safe override

为让 `MMLU` 的翻案更稳、`MATH` 的数值修正更可控，selector 不能只看 generic gain，还必须显式看“答案对象位移”与“typed support 净增”之间的关系。

#### 16.8.1 typed support gain 与 contract gain

定义：

```text
Delta_type(c, a) = s_c^type - s_a^type
```

候选级 contract residual 定义为：

```text
r_A^ctr = ContractResidual(answer_object_A, sig_A, Pool(H_A^evd))
```

于是有：

```text
Delta_ctr(c, a) = r_a^ctr - r_c^ctr
```

其中：

- `Delta_ctr > 0` 表示 candidate 的 contract residual 更低
- 对 `MATH`，它反映对象级约束与可计算一致性是否改善
- 对 `MMLU-Pro`，它通常退化成 option-typed object 是否更自洽

#### 16.8.2 flip barrier

定义答案对象位移 barrier：

```text
b_flip(c, a) = softplus(d_sig(c, a) - Delta_type(c, a))
```

解释：

- 若候选与 anchor 的 signature 差异很大，但 typed support 没有同步上涨，则 `b_flip` 大
- 若 candidate 确实带来了强 typed support 净增，则 barrier 自动下降

这是一条统一规则：

- 在 `MMLU-Pro` 上，它等价于“换选项必须拿出更强 typed support”
- 在 `MATH` 上，它等价于“改 numeric object 可以，但要同时带来更强的 typed support 与 contract 改善”

#### 16.8.3 base score 与最终 safe score

将 §16 现有差分特征扩展为：

```text
u_base(c, a) = f_sel^base(
  [
    z_c,
    z_a,
    z_c - z_a,
    Delta_err(c, a),
    Delta_sup(c, a),
    Delta_mix_err(c, a),
    Delta_mix_sup(c, a),
    M_keep(c, a),
    Delta_type(c, a),
    Delta_ctr(c, a)
  ]
)
```

最终 safe override 分数定义为：

```text
u_safe(c, a) = u_base(c, a) - b_flip(c, a)
```

bank-level 选择分布改成：

```text
u_theta(a) = 0
u_theta(c) = u_safe(c, a)
```

```text
p_theta^sel(A | B) = exp(u_theta(A)) / sum_{A' in B} exp(u_theta(A'))
```

这样 selector 学到的就是：

- answer 错误有没有修掉
- evidence 缺口有没有补上
- contract residual 有没有下降
- typed object 改动时，typed support 是否真的净增
- 安全前缀有没有保住

## 17. 训练方案

第一版训练原则非常明确：

- `stage1` 冻结
- `executor` 冻结
- `stage2-rollback` 在 frozen graph + frozen trace 上单独训练
- 不伪装成端到端
- 不写一锅大杂烩的大总损失

### 17.1 阶段 A：确定性 teacher bank 构造

这一阶段不依赖 learned rerun。teacher bank 必须先由 deterministic builder 生成。

对每个训练样本：

1. 跑冻结 stage1，拿到 `A^(0), G, T`
2. 枚举边界 `j in {0, 1, ..., L}`
3. 对每个 `j > 0`，取 trace suffix 的 provenance 闭包：

```text
V_prov^{>j} = union_{t > j} V(Pi_t)
```

4. 记 anchor replay cache 中可访问到的节点集合为 `V_anchor`。定义 deterministic seed：

```text
q_v^teach(j) =
  1 / |V_prov^{>j}|,   if v in V_prov^{>j} and |V_prov^{>j}| > 0
  1 / |V_anchor|,      if |V_prov^{>j}| = 0 and v in V_anchor and |V_anchor| > 0
  1 / |V|,             if |V_prov^{>j}| = 0 and |V_anchor| = 0
  0,                   otherwise
```

这样即使 suffix provenance 闭包为空，也不会出现未定义 seed。

5. 用固定图扩散或 PPR 构造 teacher subgraph：

```text
p_teach^(j) = PPR(q_teach^(j); A)
alpha_teach^(j) = sparsemax(p_teach^(j))
```

6. 基于 `alpha_teach^(j)` 支持集构造 verbalized rerun request，交给冻结 executor 生成候选
7. `j = 0` 直接对应 no-rerun anchor 候选 `A^(0)`
8. 得到：

```text
B = {A^(0), A_1^rr, ..., A_M^rr}
```

9. 用外部 evaluator 对每个候选打分 `m(A)`
10. 对每个候选 `A in B`，显式构造 teacher-side 语义包：

```text
Y_A^teach        = ParseContract(answer_A; dataset_name, answer_format, task_subtype)
sig_A^teach      = AnswerSignature(Y_A^teach)
r_A^{ctr,teach}  = ContractResidualTeach(Y_A^teach, sig_A^teach)
s_A^{type,teach} = TypedSupportTeach(A)
d_sig^teach(A, A^(0))
M_keep^teach(A, A^(0))
```

其中：

- `ContractResidualTeach`
  - 来自 `ParseContract` 与 evaluator 可见的 deterministic contract checks
- `TypedSupportTeach`
  - 来自 deterministic answer/evidence overlap、typed object check、evaluator 可见 typed checks
- `M_keep^teach`
  - 来自 anchor 安全前缀 units 与 candidate 的 deterministic overlap

11. 在 bank `B` 上使用 §17.7.1 的 lexicographic teacher order `≻` 进行排序：

```text
A* = argmax_{A in B} under ≻
```

12. candidate bank 的 soft label 也由 teacher order 构造，而不是由当前模型输出反推：

```text
ord_B(A) = LexRank_≻(A; B)
phat_B(A) = exp(-ord_B(A)) / sum_{A' in B} exp(-ord_B(A'))
```

注意：

- 8 个数据集的 evaluator 可以不同
- 但 `stage2-rollback` 不吃 `task id`
- `stage2-rollback` 只吃同题候选间的相对优劣
- `A*`、`j*`、`phat_B` 都必须由 teacher-side deterministic quantities 给出
- 这些 teacher quantities 不能依赖：
  - `f_type`
  - selector 对齐网络
  - keep matching 网络
  - 任何当前正在训练的 `stage2-rollback` 模块

### 17.2 阶段 B：训练 verifier + unit latent maps

训练对象：

- 四视角 encoder
- soft role head
- `c_i`
- `s_i`
- `k_i`
- 候选级 verifier score head

候选级分数：

```text
S_ver(A) = f_ver(Pool({h_i}))
```

损失使用 bank 内 listwise 排序：

```text
L_ver = -log exp(S_ver(A*)) / sum_{A in B} exp(S_ver(A))
```

其中 `A*` 为当前 bank 中按 §17.7.1 词典序 teacher order 选出的最优候选。

同时加入结构对齐一致性：

```text
L_align = sum_i sum_{j=1}^L 1[|Pi_j| > 0] B_ij KL(pi_{Pi_j} || R_i)
```

这里不要强行构造 noisy 的 unit hard labels，让 `c_i / s_i / k_i` 先作为潜变量。

### 17.3 阶段 C：训练 rollback boundary scorer

对每个样本，离线枚举边界 `j in {0, 1, ..., L}` 并产生对应候选，定义 teacher 边界：

```text
j* =
  0,                                    if A^(0) is a ≻-maximal element in {A^(0)} union {A_j^rr}_{j >= 1}
  lexargmax_{j >= 1}(A_j^rr under ≻, j), otherwise
```

含义：

- 如果 anchor 在词典序 teacher order 下已经是极大元，则 teacher 直接选择 `j = 0`
- 否则按 §17.7.1 的词典序 teacher order 选择最好候选对应的边界
- 若并列，选更深的边界，也就是更局部的 rollback

边界网络训练时不直接对 sparsemax 后验取对数，而是对稠密训练后验优化：

```text
L_rb = -log(pi_tilde_{j*}^rb)
```

这一阶段只训练：

- trace encoder
- step-level feature encoder `f_rb^feat`
- macro scorer `f_M`
- micro scorer `f_m`

verifier 暂时冻结。

### 17.4 阶段 D：训练 absorbing diffusion rerun operator

训练对象：

- shared edge transfer：
  - `f_e`
- support-stage：
  - `f_a^sup`
  - support-stage diffusion
  - `f_supmem`
- answer-stage：
  - `f_seed`
  - `f_couple`
  - `f_a^ans`
  - answer-stage diffusion
  - rerun 状态更新网络

teacher 子图来自 deterministic builder 或 successful rerun candidate 的 provenance 闭包，并按 support-stage / answer-stage 分别构造 teacher 分布。

support-stage 节点 / 边监督：

```text
L_node^sup = KL(q_sup^* || alpha_tilde^sup)
L_edge^sup = -sum_u sum_{v in N+(u)} P_sup^*(v | u) log P_rr^sup(v | u)
```

answer-stage 节点 / 边监督：

```text
L_node^ans = KL(q_ans^* || alpha_tilde^ans)
L_edge^ans = -sum_u sum_{v in N+(u)} P_ans^*(v | u) log P_rr^ans(v | u)
```

其中：

- `alpha_tilde^sup / alpha_tilde^ans` 分别是两条 diffusion 在训练期的稠密节点后验
- `P_rr^sup / P_rr^ans` 分别由对应阶段的训练期节点后验构造
- 第一版这里仍然不加入 `L_rr-rank`

### 17.5 阶段 E：训练 emitter + final selector

如果第一版采用“一发射节点 -> 一候选”的实现，则 emitter 直接对 evaluator soft label 学习，并显式带上 `null emission` 空槽：

```text
phat_emit_plus(v) = sum_{A emitted by v} phat_B(A)
phat_emit_plus(null) = phat_B(A^(0))
L_emit = KL(phat_emit_plus || gamma_tilde_plus)
```

也就是说，训练期 emitter 使用的是带空槽的稠密发射后验 `gamma_tilde_plus`，而不是推理期稀疏支持 `gamma_plus`。

final selector 直接使用 evaluator soft label：

```text
L_sel = KL(phat_B || p_theta^sel(. | B))
```

其中 `phat_B` 是 bank 上由 evaluator 排序得到的 soft label。

### 17.6 阶段 F：轻量联合微调

只有在前几阶段稳定后，才做一轮轻量 joint finetune：

- verifier 解冻
- boundary 解冻
- diffusion 解冻
- selector 解冻
- emitter 解冻
- executor 仍冻结

联合策略不写一堆线性加权总损失，而是按 curriculum 分轮：

1. 一轮 `L_ver + L_align + L_rb`
2. 一轮 `L_node^sup + L_edge^sup + L_node^ans + L_edge^ans`
3. 一轮 `L_emit + L_sel`

不要一开始就全放一起。

### 17.7 MMLU / MATH 稳定化 teacher order 与 curriculum

为避免 training signal 只追 evaluator、却忽略 `typed support / contract / preserve`，teacher bank 上的候选排序改成词典序，而不是手工线性加权。

#### 17.7.1 lexicographic teacher order

teacher order 中使用的量，必须全部是 teacher 版本，而不是当前模型输出。统一定义：

- `sig_A^teach`：由 `ParseContract + answer_signature` 得到的 teacher signature
- `d_sig^teach(c, a)`：由 teacher signature 计算的 signature displacement
- `s_A^{type,teach}`：由 `ParseContract`、evaluator 可见 typed checks、deterministic answer/evidence overlap 构造的 teacher typed support
- `Delta_type^teach(c, a) = s_c^{type,teach} - s_a^{type,teach}`
- `Delta_ctr^teach(c, a)`：由 deterministic contract residual 得到的 teacher contract gain
- `M_keep^teach(c, a)`：由 anchor 安全前缀 units 与 candidate 的 deterministic overlap 得到的 teacher preserve 质量

这些量都不依赖：

- `f_type`
- selector 对齐网络
- keep matching 网络

也就是说，teacher bank、`A*`、`j*` 与 `phat_B` 都不能被当前模型参数反向影响。

给定同题 bank 中两个候选 `c1, c2`，定义：

```text
c1 ≻ c2
iff
  (m(c1) > m(c2))
  or (m(c1) = m(c2) and Delta_type^teach(c1, a) > Delta_type^teach(c2, a))
  or (m(c1) = m(c2) and Delta_type^teach(c1, a) = Delta_type^teach(c2, a) and Delta_ctr^teach(c1, a) > Delta_ctr^teach(c2, a))
  or (m(c1) = m(c2) and Delta_type^teach(c1, a) = Delta_type^teach(c2, a) and Delta_ctr^teach(c1, a) = Delta_ctr^teach(c2, a) and M_keep^teach(c1, a) > M_keep^teach(c2, a))
  or (m(c1) = m(c2) and Delta_type^teach(c1, a) = Delta_type^teach(c2, a) and Delta_ctr^teach(c1, a) = Delta_ctr^teach(c2, a) and M_keep^teach(c1, a) = M_keep^teach(c2, a) and d_sig^teach(c1, a) < d_sig^teach(c2, a))
```

解释：

1. 先看 evaluator 真正结果
2. 打平时，看 typed support 谁更强
3. 再打平时，看 contract residual 谁更低
4. 再打平时，看谁更保前缀
5. 最后才偏向“离 anchor 更近”

这对 `MMLU-Pro` 会自然偏向“少乱翻”；
对 `MATH` 则仍保持“真正算对优先”。

#### 17.7.2 teacher boundary 的重新定义

原 §17.3 的 teacher 边界在稳定化阶段替换为：

```text
j* = 0
iff
  A^(0) is a ≻-maximal element in {A^(0)} union {A_j^rr}_{j >= 1}
```

否则：

```text
j* = lexargmax_{j >= 1}(A_j^rr under ≻, j)
```

其中 tie-break 仍偏向更深边界，也就是更局部的 rollback。

#### 17.7.3 support-first curriculum

训练顺序也改成 support-first：

`Phase S1: typed support / contract calibration`

- 训练 `f_type`
- 训练 step-level `contract residual` 头
- 训练 verifier 内部的 `answer / evidence` 与 `keep` latent maps
- answer diffusion / emitter / selector 暂不训练

`Phase S2: macro / micro boundary`

- 训练 `f_M` 与 `f_m`
- 用上面的 lexicographic teacher boundary
- answer diffusion 仍冻结

`Phase S3: support diffusion`

- 只训练 evidence-first 通道：
  - `f_a^sup`
  - support-stage diffusion
  - `f_supmem`
- 不发射答案

`Phase S4: answer diffusion + emitter + selector`

- 在 support channel 冻结或半冻结的前提下，训练：
  - `f_couple`
  - `f_a^ans`
  - answer-stage diffusion
  - emitter
  - typed-support-preserving selector

`Phase S5: 轻量联合微调`

- joint finetune：
  - verifier
  - boundary
  - support diffusion
  - answer diffusion
  - emitter
  - selector
- `executor` 仍冻结

这个 curriculum 的作用是：

- `MMLU-Pro` 先把“谁有资格翻案”训稳
- `MATH` 先把“证据 / 约束 / 中间量”拉回来，再训练最终答案重算

## 18. 第一版工程起手顺序

第一版工程顺序改成下面这版，而不是把 `train_bank.py` 放到最后：

### 第一批：桥接层与 teacher bank

`artifacts.py`

- 独立重写 `ArtifactIR++`
- 完成 `stage1_output -> anchor_artifact` 解析

`trace_ir.py`

- 先补 stage1 压缩 trace
- 补 `a_j / b_j`
- 形成可执行 replay brief

`pipeline.py`

- 新增 `RollbackPreparedStage1Artifact`
- 保持 repo 级 `PreparedStage1Artifact` 原样不动

`train_bank.py`

- 先实现 deterministic teacher bank builder
- 先把 `A^(0), G, T -> B, phat_B` 这条链打通

### 第二批：诊断层

`verifier.py`

- 先做 verifier latent maps
- 先做 role-aware 风险通道

`boundary.py`

- 先做 rollback boundary scorer
- 先打通 `j = 0` 的 no-rerun 分支

### 第三批：局部 rerun 层

`diffusion.py`

- 做 absorbing diffusion rerun
- 做双侧 gating 的局部 message passing
- 区分 `T_diff` 与 `T_run`

### 第四批：运行面与选择层

`runtime.py`

- 先实现 no-rerun short-circuit
- 再接 rerun path

`emitter.py`

- 做 request-based emission
- 接冻结 executor

`selector.py`

- 做 pairwise selector
- 再做 bank-level listwise selection

第一版明确不碰：

- executor
- 生成器
- fancy controller
- local correction 实现
- repo 级 `PreparedStage1Artifact` 定义

## 19. 与现有 UCC 路线的关系

这条线不是 UCC 正式阶段二的 `local correction block`。

UCC 当前正式阶段二定义仍然是：

```text
localize -> preserve -> propose -> apply -> delta -> value
```

`stage2-rollback` 这条新实现不沿用这条链，但旧 UCC 路线保持原样不动。

它与现有 UCC 的关系是：

- 代码壳参考 `phase3a_unified`
- 对象语义、verifier 语义与 safe override 语义参考 `UCC整体设计.md + phase1_semantic_safe_override`
- 实现上另起独立目录
- 不在 `stage2-rollback/` 中 import、复制或改造 `correction.py`
- 不在 `Stage2-UCC/stage2_phase3a_unified/` 内继续叠补丁
- 不把 rollback 伪装成 correction 的一种特殊 case

它是另一条独立的 stage2 二阶段算子路线：

```text
shared bridge -> local rollback adapter -> trace compress -> boundary -> absorbing diffusion rerun -> emit -> safe select
```

## 20. 最终结论

本方案的最终判断固定如下：

1. 代码壳参考从哪里来：
   - `Stage2-UCC/stage2_phase3a_unified@362aff3`
2. 对象语义、verifier 与 safe override 语义从哪里来：
   - `Stage2-UCC/UCC整体设计.md@362aff3`
   - `Stage2-UCC/stage2_phase1_semantic_safe_override/`
3. 哪些只借思路：
   - `runtime_v2 / runtime_v41 / runtime_v44`
4. 哪些在新目录中明确不纳入：
   - 整条 local self-correction 链对应的旧实现路径
5. 新方案的核心：
   - 用统一 verifier 把错误 unit 投影回 trace 后缀
   - 再用带吸收态、双侧 gating 的局部扩散算子在 frozen union graph 上重跑
   - 最后从 rerun 子图重新发射候选，用 learned safe override 选 winner
6. 最重要的工程边界：
   - 后续 `stage2-rollback/` 必须落一版独立代码
   - 不共用旧版 stage2 专有代码
   - 旧实现只允许参考，不允许共享
   - repo 级 `PreparedStage1Artifact` 保持不动，只在新目录里新增本地 adapter

这条路线的本质不是“去改老 correction”，而是在老代码完全不动的前提下，把新目录的 stage2 独立重写成 `pure rollback / local rerun`。
