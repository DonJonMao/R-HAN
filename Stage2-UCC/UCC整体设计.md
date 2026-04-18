# UCC整体设计

更新时间：2026-04-17

本文档正式采用新的阶段定义。

最重要的重定位只有一句话：当前 `phase3a_unified` 代码不再计入正式三阶段，而被定义为 `P0: Bootstrap Teacher Scaffold`。它负责统一接口、统一数据流、产 teacher 轨迹、暴露真实失败模式；新的正式阶段一从 `answer/evidence` 对象语义和 `overturn` 校准开始。

## 1. 为什么要重定义阶段

当前系统已经暴露出一个非常明确的现实瓶颈：

- 在 `MMLU-Pro` 这类 finite-answer / closed-set 任务上，主要损失不是候选不够多，而是错误 override 会把本来正确的 `stage1 anchor` 翻坏。
- 在 `MBPP/HumanEval` 上，系统已经能产生少量有效修复，但现有 hard guard 和粗粒度 correction 让这些收益很难稳定穿透 final selection。
- 在 `GSM8K/MATH/NLGraph/Knowledge Crosswords` 上，当前统一对象还不够语义化，verifier 能看到的是“一团 units”，而不是“最终答案单元”和“支持证据单元”的区别。

因此，旧定义里的“阶段一=当前代码”已经不再准确。当前代码更像一个 bootstrap 平台，而不是正式的方法阶段。

## 2. 新的总览

新的统一路线图如下：

- `P0: Bootstrap Teacher Scaffold`
- `阶段一: Unified Semantic Verification & Safe Override`
- `阶段二: Unified Local Correction`
- `阶段三: Fully Continuous Unified Controller`

实现优先级严格按下面顺序推进：

`1 -> 2 -> 5 -> 4 -> 3 -> 6`

其中：

- `1`：`answer/evidence` 对象语义
- `2`：`pairwise overturn verifier`
- `5`：风险感知 final utility / safe override
- `4`：`localize/preserve` 与 correction value
- `3`：`CorrectionArtifact` proposal / apply / delta
- `6`：训练连续化与 on-policy self-correction

叙事上，阶段二里的 `4/3` 不是两块相互独立的系统，而是同一个 `local correction block`：

`localize -> preserve -> propose -> apply -> delta -> value`

只是工程实现顺序上，先做 `localize/preserve`，再接 `artifact proposal` 更稳。

## 3. P0：Bootstrap Teacher Scaffold

### 3.1 定位

`P0` 对应当前 `phase3a_unified` 代码。

它的职责只有三个：

- 提供统一数据流：`candidate -> artifact -> verifier -> correction trace -> reinsert`
- 提供稳定但保守的 teacher 轨迹
- 暴露真实失败模式，例如：
  - `MMLU-Pro` 的错误 overturn
  - `MBPP/HumanEval` 的“内部修好但最终放不出来”
  - `GSM8K/MATH` 的“解释更完整但答案更错”

### 3.2 P0 保留什么

P0 继续保留当前主壳：

- 冻结 `UnionGraph`
- graph-native rerun
- candidate bank
- 统一化的 artifact / verifier / correction 接口
- teacher-style hard guard
- replay / trajectory logging

### 3.3 P0 不承担什么

P0 不承担以下正式方法责任：

- 不承担真正的 answer/evidence 语义建模
- 不承担显式的 pairwise overturn 校准
- 不承担正式的 risk-aware safe override
- 不承担 learnable local correction block
- 不承担 fully continuous controller

一句话：`P0` 是训练与分析平台，不是最终算法阶段。

## 4. 阶段一：Unified Semantic Verification & Safe Override

阶段一的目标不是做 correction，而是先把“谁有资格翻案”这件事做对。

核心原则：先修 `semantic verification` 和 `safe override`，再让 correction 进入主舞台。

### 4.1 阶段一的三项核心任务

阶段一只做三件事：

1. 把 candidate 统一成 `answer/evidence-aware ArtifactIR++`
2. 把 verifier 统一成 `single-candidate + pairwise overturn` 的单接口系统
3. 把 final selection 改成风险感知的 `safe override`

### 4.2 ArtifactIR++

阶段一的统一对象定义为：

```text
A_i = (
  U_i,
  E_i,
  V_i,
  y_i,
  p_i,
  q_i,
  R_i,
  S_i
)
```

其中：

- `U_i`：全部 units
- `E_i`：unit graph
- `V_i`：多视图表示
- `y_i`：rendered answer
- `p_i`：provenance
- `q_i`：各视图置信度
- `R_i`：unit roles
- `S_i`：schema features / answer signature

### 4.3 unit role 语义

每个 unit 不只存硬标签，而是存一个软角色分布：

```text
r_u = softmax(W_r h_u) in R^3
```

三维对应：

- `answer`
- `evidence`
- `mixed`

同时，在 metadata 中缓存一个 hard role：

```text
unit_role(u) = argmax r_u
```

### 4.4 顶层 answer/evidence 索引

阶段一要求在顶层显式缓存：

- `answer_unit_ids`
- `evidence_unit_ids`
- `schema_features`
- `answer_signature`
- `answer_object`

定义：

```text
A_i^ans = { u in U_i | r_u^ans > tau_a }
A_i^evd = { u in U_i | r_u^evd > tau_e }
```

这一步的意义非常直接：系统终于能明确区分“真正决定最终输出的 answer units”和“只是支持它的 evidence units”。

### 4.5 Typed Answer Contract

阶段一必须新增统一的 typed answer contract：

```text
Y_i = ParseContract(
  y_i;
  dataset_name,
  answer_format,
  task_subtype
)
```

其中：

- `Y_i` 就是 `answer_object`
- 它至少包含 `kind / value / valid / fields / signature`
- `schema_features S_i` 必须扩成：

```text
S_i = (
  schema_features,
  answer_signature,
  answer_object,
  schema_valid,
  task_subtype
)
```

实现约束：

- `phase1 canonicalizer` 与 `evaluator` 必须共用同一套 `ParseContract`
- 不能再出现 `profile` 声明了 `answer_format`，但 `phase1` 只按 `task_type` 粗分的实现
- `MMLU-Pro / NLGraph / MBPP` 都必须落到 typed object，而不是统一退化成普通文本 surface

### 4.6 answer signature

`answer_signature` 必须统一定义，而不是按任务硬 route。

统一规则如下：

- `MMLU-Pro` / 多选题：`option::<int>`
- 数学 / GSM：`numeric::<normalized value>`
- `NLGraph`：`graph_json::<subtask>::<field-or-canonical-form>`
- 代码：`code::<entry_point>::<canonical ast hash>`
- 结构任务：`structured::<canonical form>` 或更细的 typed signature
- 其他文本：`text::<normalized surface>`

这不是任务分流，而是统一答案表征。

最小实现例子：

- `MMLU-Pro`：`option::8`
- `NLGraph/connectivity`：`graph_bool::answer::no`
- `NLGraph/flow`：`graph_scalar::max_flow::31`
- `MBPP`：`code::solve::<ast_hash>`

### 4.7 多视图统一编码

阶段一保留四视图，但不按任务硬切换，而是软加权：

- `surface_view`
- `step_view`
- `struct_view`
- `exec_view`

视图置信度：

```text
q_i^(m) = sigma(MLP_view^(m)(f_i^(m)))
q~_i = sparsemax(q_i)
```

统一 unit 表示：

```text
h_{i,u} = sum_m q~_i^(m) W_m z_{i,u}^(m)
```

解释：

- 在 `MBPP/HumanEval` 上，`exec_view` 会自然更强
- 在 `NLGraph/Knowledge Crosswords` 上，`struct_view` 会自然更强
- 在 `MMLU/GSM8K/MATH` 上，`step_view` 和 `answer/evidence` 结构更重要

权重来自视图可靠性，不来自任务标签。

### 4.8 阶段一统一 verifier

阶段一的 verifier 是一套单接口系统，但输出三类结果：

- `single-candidate verification`
- `answer/evidence consistency`
- `pairwise overturn evaluation`

统一表示为：

```text
V_i = (
  r_i,
  epsilon_i,
  s_i,
  c_i,
  p_i,
  omega_i
)
```

其中：

- `r_i`：统一残差向量
- `epsilon_i`：unit-level error map
- `s_i`：unit-level support map
- `c_i`：candidate confidence
- `p_i`：progress score
- `omega_i`：overturn bundle

### 4.9 统一残差向量

统一残差定义为：

```text
r_i = [
  r_parse,
  r_cons,
  r_comp,
  r_exec,
  r_const,
  r_support,
  r_preserve
]_i
```

所有任务都投影到这一残差空间里，不再分别维护 reasoning/code/graph verifier。

语义约束必须写死：

- `r_parse`：`answer_object` 解析失败或 schema 解析失败
- `r_comp`：typed contract 要求的关键字段缺失
- `r_const`：对象级约束不满足
- `r_exec`：可执行 / 可计算检查失败
- `r_support`：当前 answer 缺少 evidence 支撑
- `r_preserve`：anchor 关键单元流失风险

实现上不允许把 `r_exec / r_const` 继续退化成 `parse_confidence` 的代理量。

### 4.10 meta verifier

meta verifier 负责检查：

- 完整性
- 一致性
- unsupported units
- 格式风险
- preserve 风险

形式上：

```text
r_i^meta = MLP_meta([Pool(H_i), e_X, S_i, p_i])
```

### 4.11 证据通道融合

证据通道保持统一定义：

- `parser`
- `executor`
- `constraint`
- `search`
- `symbolic`

对每个通道 `k`：

```text
alpha_ik = sparsemax_k(MLP_rel([g_ik, q~_i, tool_health_k, S_i]))
r_i = r_i^meta + sum_k alpha_ik T_k(o_ik)
```

解释：这是 evidence-channel weighting，不是 task routing。

### 4.12 unit-level error / support

错误热度：

```text
epsilon_{i,u} = sigma(MLP_err([h_{i,u}, r_i]))
```

支持度：

```text
s_{i,u} = sigma(MLP_sup([h_{i,u}, e_X, evidence_{i,u}]))
```

这两个输出在阶段一就要训练好，因为阶段二的 `localize/preserve` 会直接吃它们。

### 4.13 answer/evidence consistency

阶段一新增显式的答案一致性分数：

```text
a_i^cons = sigma(MLP_ans([
  Pool(H_i^ans),
  Pool(H_i^evd),
  S_i,
  r_i
]))
```

它回答的是：

“当前 evidence units 是否真的支持当前 answer units？”

这是当前 `MMLU-Pro` 最缺的一层。

冷启动阶段必须先有 typed bootstrap：

- `parse invalid` -> 低
- `schema missing` -> 低
- `graph/code executable contradiction` -> 低
- `answer unit 缺 evidence support` -> 低

然后再让 `MLP_ans` 在此基础上学习校准。

### 4.14 overturn bundle

定义：

```text
omega_i = (
  Delta a_i,
  Delta c_i,
  Delta r_i,
  pi_i^pres,
  rho_{i > a0}
)
```

其中：

- `Delta a_i`：candidate 与 anchor 在 answer units 上的差异
- `Delta c_i`：confidence 差异
- `Delta r_i`：residual 差异
- `pi_i^pres`：candidate preserve risk
- `rho_{i > a0}`：candidate 推翻 anchor 的风险

具体定义：

```text
Delta a_i = g_ans(A_i^ans, A_a0^ans, S_i)

rho_{i > a0} = sigma(MLP_ovr([
  Pool(H_i), r_i,
  Pool(H_a0), r_a0,
  Delta a_i,
  pi_i^pres,
  1 - a_i^cons
]))
```

其中 `g_ans` 必须是 answer-type-aware typed distance，不能退化成 generic lexical similarity。

最小实现约定：

- categorical / bool：exact mismatch
- scalar：typed numeric discrepancy
- sequence / path / order：constraint-set discrepancy
- code：API / entry-point / AST discrepancy

这一步的关键不在于“candidate 好不好”，而在于“candidate 是否真的有资格推翻当前 anchor”。

### 4.15 风险感知 utility

阶段一先定义基础 utility：

```text
u_i = MLP_u([
  Pool(H_i),
  r_i,
  c_i,
  p_i,
  a_i^cons,
  prov_i
])
```

相似度核：

```text
K_ij = (2 * cos(Pool(H_i), Pool(H_j)) + cos(r_i, r_j)) / 3
```

软去重后的 utility：

```text
u~_i = u_i - gamma * sum_j pi_j K_ij
pi_i = softmax(u_i / tau)
```

最终安全效用：

```text
u_i^safe = u~_i
           - eta1 * rho_{i > a0}
           - eta2 * pi_i^pres
           + eta3 * a_i^cons
```

### 4.16 safe override 规则

阶段一的 final selection 由 `risk-aware safe utility` 主导，同时保留一层很薄的 fail-safe：

```text
override(i)
iff
u_i^safe > u_a0^safe
and rho_{i > a0} < tau_ovr
and not catastrophic_answer_rewrite
```

其中：

```text
catastrophic_answer_rewrite
iff
|Delta a_i| > tau_a
and a_i^cons < tau_c
```

这层 fail-safe 不是旧的 hard anchor guard，而是防止 closed-set 任务继续大规模错翻的极薄保护层。

### 4.17 阶段一训练目标

阶段一的损失定义为：

```text
L^(1) = L_verify + L_ans + L_ovr + L_rank + L_safe
```

各项含义：

- `L_verify`：统一 verifier 校准
- `L_ans`：answer/evidence consistency 校准
- `L_ovr`：pairwise overturn 校准
- `L_rank`：utility 排序损失
- `L_safe`：safe override 二分类损失

阶段一成功的判据非常明确：

- `MMLU-Pro` 的错误 overturn 显著下降
- `MBPP/HumanEval` 不因为过强保守而明显掉分
- utility / overturn / safe override 三个头都能产出稳定可解释的轨迹

## 5. 阶段二：Unified Local Correction

阶段二才引入 correction 主体，而且必须建立在阶段一已经训好的语义 verifier 之上。

### 5.1 阶段二的唯一主线

阶段二的方法主线只有一条：

```text
localize -> preserve -> propose -> apply -> delta -> value
```

这就是统一的 `local correction block`。

### 5.2 阶段二的目标

阶段二要同时做到两件事：

- 放宽：让真正有价值的 correction 能穿透 final selection
- 收紧：让 correction 本身更局部、更可验证、更可解释

### 5.3 answer-first / evidence-first localize

阶段二引入 gate：

```text
g_i^ans = sigma(MLP_af([r_i, omega_i, a_i^cons, u~_i]))
```

解释：

- `g_i^ans` 接近 1：优先修 answer units
- `g_i^ans` 接近 0：优先修 evidence units

localize：

```text
l_{i,u} = MLP_loc([h_{i,u}, epsilon_{i,u}, s_{i,u}, m_{i,u}^{mem}, Delta_{i,u}^{hist}])
```

按角色混合得到：

```text
lambda_{i,u} = g_i^ans * 1[u in A_i^ans] * l_{i,u}
             + (1 - g_i^ans) * 1[u in A_i^evd] * l_{i,u}

lambda^_i = sparsemax_u(lambda_{i,u})
```

### 5.4 preserve heatmap

```text
rho_{i,u}^{keep} = sigma(MLP_keep([
  h_{i,u},
  s_{i,u},
  a_{i,u}^{anchor},
  m_{i,u}^{stable}
]))
```

这一步统一学习“哪些 unit 最不该动”，而不是再写任务型 preserve rule。

### 5.5 critique summary

```text
c_i^{loc} = sum_u lambda^_{i,u} h_{i,u}
k_i^{pres} = sum_u rho_{i,u}^{keep} h_{i,u}
d_i = MLP_crit([Pool(H_i), c_i^{loc}, k_i^{pres}, r_i, omega_i, g_t])
```

### 5.6 CorrectionArtifact

阶段二的 correction 只能是局部 artifact，不允许自由全文改写：

```json
{
  "target_units": [...],
  "operation": "replace | insert_before | insert_after | delete | reorder",
  "new_units": [...],
  "preserve_units": [...],
  "expected_delta": {
    "delta_answer": "...",
    "delta_parse": 0.0,
    "delta_constraint": 0.0,
    "delta_exec": 0.0,
    "delta_preserve": 0.0
  },
  "rationale": "..."
}
```

其中 `expected_delta.delta_answer` 必须复用阶段一的 typed `g_ans`，不能重新退化成“全文文本更顺”的自由表述。

### 5.7 apply + delta predictor

```text
A'_i = Apply(A_i, a_i)

Delta^_i = MLP_Delta([
  Pool(H_i),
  Pool(H'_i),
  r_i,
  omega_i,
  Emb(a_i)
])
```

输出：

- `Delta r_i`
- `Delta c_i`
- `Delta p_i`
- `risk_i`

### 5.8 correction value

阶段二先判断“这次 correction 值不值得做”：

```text
v_i^corr = MLP_cv([
  -Delta ||r_i||_1,
  Delta c_i,
  -risk_i,
  -rho_{i > a0}
])
```

这一步是 selective refinement 的统一形式。

### 5.9 阶段二 final selection

```text
u_i^final = u_i^safe + eta4 * v_i^corr
```

通过规则：

```text
override(i)
iff
u_i^final > u_a0^final
and rho_{i > a0} < tau_ovr
and risk_i < tau_risk
```

### 5.10 阶段二训练

阶段二损失定义：

```text
L^(2) = L^(1) + L_loc + L_keep + L_art + L_Delta + L_cv
```

推荐训练顺序：

1. `supervised calibration`
2. 小规模 `on-policy` 更新 `overturn + correction value + final utility`
3. 最后再把 aggression / halting / controller 放进来

## 6. 阶段三：Fully Continuous Unified Controller

阶段三不再改 verifier 或 correction operator，而是把 controller 本身连续化。

显式约束：

- controller 只控制参与强度、编辑 aggression、halting
- controller 不能定义或改写 `ParseContract / answer_signature / g_ans`
- phase3 只能调度 phase1 / phase2 的 typed semantics，不能回写覆盖它们

### 6.1 连续 controller 的五个部分

- 节点参与 `alpha_t`
- 边支持 `beta_{u->v}^t`
- memory-view attention `mu_{v,t}`
- correction aggression `g_i^edit`
- halting `zeta_t`

### 6.2 关键变量

节点参与：

```text
alpha_t = sparsemax_v(MLP_node([h_v^{t-1}, z_v^t, g_{t-1}, r_F^bar]))
```

边支持：

```text
beta_{u->v}^t = alpha_u^t alpha_v^t sparsemax(MLP_edge(...))
```

correction aggression：

```text
g_i^edit = sigma(MLP_aggr([r_i, omega_i, v_i^corr, u_i^final]))
```

halting：

```text
zeta_t = sigma(MLP_halt([
  g_t,
  sum_i w_i p_i,
  sum_i w_i ||r_i||_1,
  max_i Delta u_i^final,
  cost_t
]))
```

### 6.3 阶段三训练

阶段三采用 on-policy multi-turn self-correction：

```text
R_t = Delta Acc
    + eta1 * (-Delta ||r||_1)
    + eta2 * Delta u_final
    - eta3 * preserve_violation
    - eta4 * cost
```

控制器损失：

```text
L_ctrl = -E[R_t log pi_theta(alpha, beta, mu, g_edit, zeta)]
```

总损失：

```text
L^(3) = L^(2) + lambda9 * L_ctrl + lambda10 * L_halt
```

实现时 `Delta Acc` 的优先级应采用字典序：

1. 先看 `success`
2. 再看 `task_score`
3. 最后才看 `cost`

不能重新退回一个不透明的大权重和，掩盖 phase1 typed semantics 是否已经做对。

## 7. 新旧阶段映射

旧定义与新定义的映射如下：

- 旧“阶段一” `Bootstrap Unified Scaffold` -> 新 `P0: Bootstrap Teacher Scaffold`
- 旧“阶段二” `Unified Verifier + Unified Local Correction` -> 拆成：
  - 新 `阶段一: Unified Semantic Verification & Safe Override`
  - 新 `阶段二: Unified Local Correction`
- 旧“阶段三” `Fully Continuous Unified Controller` -> 保持为新 `阶段三`

## 8. 代码落位约定

从这版设计开始，目录职责明确如下：

- `Stage2-UCC/stage2_phase3a_unified/`
  - 对应 `P0`
  - 保持 teacher scaffold 身份，不再被叙述成正式阶段一
- `Stage2-UCC/stage2_phase1_semantic_safe_override/`
  - 对应新正式阶段一
  - 负责 `ArtifactIR++`、语义 verifier、pairwise overturn、safe override
- 阶段二与阶段三应当继续采用新的独立目录，不回写覆盖 P0 或阶段一实现

## 9. 当前实现优先级

后续工程推进顺序固定为：

1. `ArtifactIR++`：先把 answer/evidence 语义对象做对
2. `pairwise overturn verifier`：先把翻案资格做对
3. `risk-aware safe utility`：先把 final selection 做对
4. `localize/preserve`：先把 correction block 的前半段做对
5. `CorrectionArtifact + apply + delta + value`：再把 correction 后半段接上
6. `continuous controller + on-policy self-improvement`：最后再连续化

## 10. 最终结论

新的 UCC 主线可以压缩成一句话：

当前 `phase3a` 不应该继续被训练成“更会保 anchor 的正式阶段一系统”；它应该被稳定地当作 `P0 teacher scaffold`。正式方法的第一步必须先把 `answer/evidence` 语义和 `safe override` 做对，再让 `local correction` 成为主角，最后才把 controller 连续化。
