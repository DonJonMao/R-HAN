# Code Route 与 Deductive Route 设计对照

本文档记录 `stage2_final_design` 当前两条 route-specific final-stage self-refinement 路线：

- `code_repair`: 面向 MBPP / HumanEval 的代码修复路线。
- `deductive_reasoning`: 面向 GSM8K / MATH 的数学推理路线。

对应当前实现基线：

```text
commit: 632b5ae64d727de76eb885957b2f8d862cf130f8
main entry: stage2_final_design/train_mas_stage2_v4_4_target_suite.py
runtime: stage2_final_design/Stage2-GCR+/stage2_gcr_plus/runtime_v44.py
```

两条路线都不改 Stage2-GCR+ 的共享外壳：

```text
runtime_v2 graph loop
learned memory selector
sparsemax edge support-set
soft slot mask
candidate bank / class collapse / finalizer
```

差异集中在最后一跳的 route-specific 算子：如何把候选答案解析成可验证 artifact、如何定义 residual、如何触发局部修复或探针、以及 selector 什么时候允许覆盖 stage1 anchor。

## 1. 共同外壳

当前主链路是：

```text
train_mas_stage2_v4_4_target_suite.py
  -> mas_stage2_v4_4.Stage2V44Pipeline
  -> Stage2-GCR+/stage2_gcr_plus/runtime_v44.py
  -> runtime_v2 graph loop + learned memory selector
```

`runtime_v44.py` 通过 `_route_family()` 选择 final-stage route：

```text
GSM8K / MATH dataset name -> deductive_reasoning
code_generation task type -> code_repair
graph_reasoning / knowledge_crosswords -> graph_constrained
else -> adversarial
```

两条 route 的共性是：

```text
stage1 anchor
  + stage2 candidate bank
  -> parse artifact
  -> deterministic verifier residual
  -> class collapse / ranking
  -> local repair or verification probe
  -> verifier-dominance / certificate selection
```

核心设计原则是一样的：不让模型泛泛反思，而是把错误转成可定位 residual，再做局部修复或受约束翻案。

## 2. Code Route 详细设计

### 2.1 Artifact

代码路线的 artifact 是从候选文本中抽取出的 Python code：

```text
candidate text -> MultiFidelityEvaluator._extract_python_code -> candidate_code
```

实现位置：

```text
stage2_final_design/Stage2-GCR+/stage2_gcr_plus/code_repair.py
```

`CodeRepairEval` 记录：

```text
code_text
syntax_ok
entry_point_ok
passed
total
failure_kind
failing_examples
stdout / stderr / exec_error
```

### 2.2 Verifier Residual

代码 verifier 是外部可执行测试：

```text
1. Python compile
2. entry point check
3. MBPP visible tests or HumanEval check()
4. timeout / execution error classification
```

主要 failure kind：

```text
empty_code
syntax_error
entry_point_missing
visible_test_failure
dataset_test_failure
timeout
execution_error
no_dataset_tests
```

这些 residual 的语义比较强，因为它们来自执行环境而不是模型自评。

### 2.3 Class Collapse 与 Ranking

代码 route 的基本排序信号是：

```text
fully_passed
syntax_ok
entry_point_ok
passed
failing_examples count
```

`CodeRepairEval.dominates()` 的偏序是：

```python
dominance_tuple = (syntax_ok, entry_point_ok, passed)
new dominates old
  <=> every dimension is not worse
      and at least one dimension is better
```

这和 code route 的安全性直接相关：如果一个候选通过了更多测试，同时没有丢掉语法和入口点，它就有清楚的可执行证据支撑。

### 2.4 Repair

代码 repair 的路径是：

```text
failed candidate / anchor
  -> failure summary
  -> recovery subgraph context
  -> code_repair_patch branch
  -> re-run compile / tests
  -> only accept if CodeRepairEval.dominates()
```

`runtime_v44.py` 中 `_select_code_repair_against_anchor_v44()` 会：

```text
1. verify full candidate pool
2. collapse code classes
3. choose challenger class only if it dominates anchor
4. build recovery target for current champion
5. generate repair branches
6. reinsert approved repairs
7. recollapse classes
8. final anchor guard: if selected no longer dominates anchor, preserve anchor
```

### 2.5 Code Route 的优势

Code route 的 verifier 是强 oracle：

```text
syntax_ok / entry_point_ok / passed tests
```

因此 code route 可以更敢于跨答案覆盖。不同代码文本之间只要执行测试维度不退化，并且至少一项变好，就具备相对清晰的选择依据。

此前 code route 实验记录中，MBPP 从 `163/195` 到 `169/195`，HumanEval 从 `29/33` 到 `31/33`，且 `0 regression`。这说明当前共享框架适合承载 route-specific 最后一跳修复。

## 3. Deductive Route 详细设计

### 3.1 Dataset Routing

数学 route 不按“对象形态自动识别”，当前按 dataset name 保守路由：

```text
gsm8k -> deductive_reasoning
math  -> deductive_reasoning
```

实现位置：

```text
stage2_final_design/Stage2-GCR+/stage2_gcr_plus/deductive_reasoning.py
stage2_final_design/Stage2-GCR+/stage2_gcr_plus/runtime_v44.py
```

### 3.2 Artifact Contract

Deductive route 要求内部候选尽量输出：

```text
SOLUTION:
1. ...
2. ...
3. ...

FINAL: <number_or_expression>
```

`DeductiveArtifact` 记录：

```text
raw_text
dataset_name
final_answer
normalized_final_answer
steps
derivation_signature
stable_prefix_signature
parser_confidence
final_source
step_source
contract_ok
```

关键字段语义：

```text
final_source:
  final_line | gsm_hash | answer_line | answer_only | boxed | last_line | none

step_source:
  solution_numbered | numbered | fallback_lines | none

contract_ok:
  has SOLUTION header
  + explicit final
  + numbered steps
```

### 3.3 Final Answer Extraction

GSM8K 做了明确止血：

```text
允许:
  FINAL:
  ####
  Answer:
  answer-only 数字

禁止:
  长文本 fallback 到最后一个数字
```

原因是 GSM8K 长推导里经常出现中间数值、比例和单位数字。把最后一个数字当 final answer 会制造伪 final，导致 stage2 错推导覆盖 stage1 正确 anchor。

MATH 目前允许：

```text
\boxed{...}
FINAL:
last_line fallback
```

其中 `last_line` 只给低置信度，不当作强 process evidence。

### 3.4 Residual

`DeductiveResidual` 记录：

```text
fatal
local
support
first_bad_step
verified_prefix_len
residual_kind
repair_locus
final_consistent
residual_signature
checked_equation_count
```

当前 deterministic fatal family：

```text
arithmetic_mismatch
final_inconsistent_with_derivation
final_inconsistent_with_last_expression
parseable_algebra_mismatch
algebra_non_equivalence
missing_final_answer
missing_boxed_answer
empty_math_expression
```

当前 unverified anchor family：

```text
answer_only_no_derivation
no_checked_equations
semantic_unverified_step
```

这两类 residual 必须分开：

```text
deterministic fatal:
  verifier 有确定证据说明当前 artifact 有错，可以修或覆盖。

unverified anchor:
  verifier 不能证明 anchor 错，只能说明 anchor 缺少可验证过程。
  它允许触发 probe，但不能直接当作 fatal 让 challenger 覆盖。
```

`checked_equation_count` 比 `verified_prefix_len` 更关键。它只统计真正被算术或代数 checker 检查过的 equation，避免把“长推导没有被抓错”误当作“强过程证据”。

### 3.5 GSM8K Verifier

GSM8K verifier 当前只做确定性算术 residual：

```text
1. parse equation
2. safe arithmetic eval
3. left/right mismatch -> arithmetic_mismatch
4. last verified value vs final -> final_inconsistent_with_derivation
5. answer-only anchor -> answer_only_no_derivation local
6. no checked equations -> no_checked_equations / semantic_unverified_step local
```

它不尝试证明完整语义绑定，例如：

```text
"5 more" 是同一天多 5 片，还是第二天 5 片
"1/16 weekday minutes each weekend day" 是否应乘以两个 weekend days
变量是否绑定了正确对象
```

所以 GSM8K 的 selector 必须保守。局部算式自洽不等于题意正确。

### 3.6 MATH Verifier

MATH verifier 当前覆盖：

```text
boxed / final presence
numeric equation check
simple sympy equivalence check
final vs last parseable expression consistency
semantic_unverified local
```

MATH 的 deterministic coverage 仍然有限。它对简单算式、幂表达式、分式和部分代数等价有帮助，但还不能可靠判断完整证明或复杂符号语义。

### 3.7 Dominance

Deductive route 的 dominance 比 code route 更保守：

```text
same answer:
  可以用更干净、更可验证的推导 enrich anchor。
  这不会改变 final answer，因此不会制造 final regression。

cross answer:
  如果 old 没有 deterministic fatal，不允许仅凭更长推导覆盖。
  如果 old 有 deterministic fatal，new 也必须有 explicit final 且 fatal 更少。
```

这条规则是上一轮 GSM8K regression 的直接修复：不能再让 `verified_prefix_len` 跨答案支配 answer-only anchor。

### 3.8 Verification Probe

当前版本新增了 answer-only / unverified anchor 的低风险 probe：

```text
answer-only or unverified anchor
  -> deductive_anchor_verification_probe
  -> equation-chain candidate
  -> deductive verifier
  -> only accept clean certificate
```

probe prompt 明确要求：

```text
Do not assume the proposed answer is correct.
Re-solve the problem from the question quantities only.
Every step must contain a computable equation.
Do not add prose-only reasoning steps.
```

如果 candidate pool 已经有 cross-answer clean witness，则先走 pairwise challenger probe：

```text
clean challenger
  + independent pairwise probe returns same challenger answer
  + both are clean witnesses
  -> allow override
```

如果没有 challenger，则 independent anchor probe 只有在以下条件下可覆盖：

```text
probe answer != anchor answer
probe is clean witness
anchor answer class has no clean witness
```

### 3.9 Clean Witness

`_deductive_entry_is_clean_witness()` 当前要求：

```text
GSM8K:
  final_source in {final_line, gsm_hash}

MATH:
  final_source in {boxed, final_line}

common:
  contract_ok
  fatal_count == 0
  final_consistent
  local_kinds only allow unconsumed_question_quantity
  checked_equation_count >= 2 for GSM8K
  checked_equation_count >= 1 for MATH
  normalized final answer exists
```

这使 deductive route 的跨答案覆盖从“长链条优先”变成“证书优先”。

## 4. 两条 Route 的同构关系

两条路线现在已经基本同构：

```text
code route:
  code artifact
  -> execution/test residual
  -> failure kind / patch locus
  -> local repair
  -> verified selection

deductive route:
  derivation artifact
  -> arithmetic/algebra/final-consistency residual
  -> first bad step / stable prefix / unverified anchor
  -> suffix repair or verification probe
  -> residual-dominance / certificate selection
```

共同点：

```text
1. 都保留 stage1 anchor 作为默认安全点。
2. 都先把候选转成 typed artifact。
3. 都用 verifier residual 而不是自由文本 critique 做最终选择。
4. 都只接受 verifier 维度不退化的候选。
5. 都把 repair 结果重新评估后再进 selector。
6. 都把 route-specific metadata 写回 candidate entry，供日志和后续学习使用。
```

## 5. 关键差距

### 5.1 Verifier 强度差距

Code route 的 verifier 是执行测试，语义强：

```text
compile + entry point + dataset tests
```

Deductive route 的 verifier 是确定性局部 checker，语义弱：

```text
arithmetic equation
simple algebra equivalence
final consistency
boxed/final extraction
```

这意味着 code route 的 `passed` 可以近似当成真实任务证据，而 deductive route 的 `checked_equation_count` 只能表示“有多少局部式子被检查过”，不能表示完整题意正确。

### 5.2 Residual 密度差距

Code candidate 失败时通常有明确 residual：

```text
syntax_error
entry_point_missing
visible_test_failure
dataset_test_failure
```

GSM8K/MATH 的 stage1 anchor 经常只是最终答案：

```text
answer_only
```

这种 anchor 对 deterministic verifier 来说不可证伪。当前 probe 正是为了解决这个残差信号稀疏问题：先把 answer-only 转成 equation-chain artifact，再决定是否翻案。

### 5.3 Cross-Answer Override 风险差距

Code route 可以较自然地跨代码覆盖，因为测试通过数提供了强排序。

Deductive route 跨答案覆盖风险高。GSM8K 的错误候选可能每个局部算式都自洽，但题意绑定错。MATH 的表达式可能看似等价，但符号域、题目约束或最终格式不匹配。

所以 deductive route 采用：

```text
safe anchor preserve
same-answer enrichment
deterministic fatal repair
clean witness
pairwise / independent probe certificate
```

而不是直接用推导长度或 checked prefix 做跨答案支配。

### 5.4 Class Collapse 差距

Code route 的 class 更接近可执行行为类：

```text
same code behavior / same failure surface / same tests
```

Deductive route 的 class 依赖：

```text
normalized_final_answer
residual_kind
first_bad_step
final_source
contract_ok
```

MATH normalization 还不够强。比如 `26^3` 与 `17576`、`5^5` 与 `3125` 在数学上等价，但字符串归一化和最终 evaluator 的接受格式可能不同。这次 MATH 的 +2 就来自 probe 把表达式式答案转成 evaluator 接受的数值答案。

### 5.5 Repair 收益差距

Code route 已经证明能通过测试 residual 拿到稳定收益。

Deductive route 当前收益更保守：

```text
GSM8K:
  regression 已压住，但 wrong -> correct 仍为 0。

MATH:
  通过 anchor probe / final normalization 拿到 +2，且 0 regression。
```

这说明 deductive route 的方向是对的，但 GSM8K 的 probe generation 和语义 quantity binding 仍不足。

### 5.6 成本差距

Code repair 的额外计算通常集中在失败样本或未完全通过样本。

Deductive probe 会在 answer-only / unverified anchor 上触发。GSM8K 本次 300 条中有 256 条触发 probe，但没有 accepted override，因此当前 GSM8K 的 probe 成本收益比不理想。

## 6. 本次 GSM8K / MATH 实验结果

实验目录：

```text
outputs/stage2_final_design_deductive_probe_qwen_gsm8k300_math_20260426_223015
```

实验代码版本：

```text
632b5ae64d727de76eb885957b2f8d862cf130f8
```

运行配置：

```text
datasets: gsm8k, math
workers: 3
mode: 3x
embedding: /mnt/nvme/Qwen3-Embedding-8B
train_script: train_mas_stage2_v4_4_target_suite.py
gsm8k max_train: 100 per worker, total 300
math max_train: 41 per worker, total 123
```

结果汇总：

| dataset | count | stage1 | stage2 | delta | wrong -> correct | correct -> wrong | same |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GSM8K | 300 | 285/300 | 285/300 | 0 | 0 | 0 | 300 |
| MATH | 123 | 82/123 | 84/123 | +2 | 2 | 0 | 121 |

选择原因：

| dataset | preserve anchor | same-answer enrichment | anchor probe override | pairwise probe override |
| --- | ---: | ---: | ---: | ---: |
| GSM8K | 256 | 44 | 0 | 0 |
| MATH | 117 | 4 | 2 | 0 |

Probe 触发与接受：

| dataset | probe rows | accepted override | repair improvement count |
| --- | ---: | ---: | ---: |
| GSM8K | 256 | 0 | 0 |
| MATH | 119 | 2 | 2 |

MATH 两个 accepted override：

```text
math_aflow:train/prealgebra/1369.json
  stage1: 26^3
  stage2: 17576

math_aflow:train/prealgebra/1740.json
  stage1: 5^5
  stage2: 3125
```

结论：

```text
GSM8K:
  这版没有提升，但也没有再出现二阶段改坏一阶段的 regression。
  probe 通道已经打开，但当前 probe 没产生足够 clean 的 cross-answer certificate。

MATH:
  这版有小幅正向，82/123 -> 84/123。
  两个收益都来自 anchor probe override，且 correct -> wrong = 0。
```

## 7. 后续改进方向

GSM8K：

```text
1. 先记录更完整的 probe diagnostic 到 row whitelist。
2. 区分“anchor answer class 有 clean witness”和“anchor 只有 answer-only”。
3. 改进 equation-chain probe 的题意 quantity binding。
4. 对 probe 触发加预算门控，避免 256/300 触发但 0 accepted 的成本浪费。
5. 继续禁止长文本最后数字兜底和 verified_prefix_len 跨答案覆盖。
```

MATH：

```text
1. 强化 canonicalize_math_answer，例如幂表达式、分式、sqrt、简单多项式。
2. 保留 boxed/final_line 优先，不把 last_line 当 clean witness。
3. 扩展 sympy equivalence，但只用于 parseable expression。
4. 跨答案 override 继续要求 clean witness + probe certificate。
```

两条 route 共同：

```text
1. 不动共享 graph / memory slot / tree search 主结构。
2. 继续把 route-specific evidence 写入 candidate metadata。
3. selector 继续使用硬 guard，而不是自由加权 score。
4. 后续提升应优先提高 verifier residual 的可靠性，而不是放宽覆盖规则。
```
