# Stage2 V3 在 GSM8K 上的当前问题总结

更新时间：2026-04-02 21:21 CST

本文档只基于当前正在运行的 `stage2 v3` 在 `gsm8k` 上的表现做问题归纳，不提前替 `mbpp` 和 `humaneval` 下结论。

当前分析目标只有两个：

1. 记录 `gsm8k` 上已经明确成立的问题
2. 区分“已经可以确认的机制问题”和“需要等 code 数据集再综合判断的问题”

相关文件：

- `logs/stage2_v3_gsm8k_20260402.log`
- `outputs/mas_stage2_v3_selected_runs_20260402/gsm8k/gsm8k/checkpoint.json`
- `mas_stage2_v3/runtime.py`
- `mas_stage2_v3/pipeline.py`
- `mas_stage2/runtime_v2.py`
- `mas_stage2/learning.py`

## 1. 当前运行快照

统计口径：

- 基于 `logs/stage2_v3_gsm8k_20260402.log` 中当前已经完成的训练样本
- 快照时间：`2026-04-02 21:21 CST`
- 当前进度：`166 / 264`

### 1.1 现象统计

| 指标 | 当前值 | 说明 |
|---|---:|---|
| 已完成训练样本数 | 166 | 当前 `gsm8k` 已跑到 `166/264` |
| `task_success` | 161 | 说明 stage1 本身已经非常强 |
| `task_fail` | 5 | 真正失败样本很少 |
| `decision=use_stage1_anchor` | 166 | 最终决策 100% 保留 stage1 anchor |
| `decision=use_stage2_override` | 0 | 二阶段一次都没有真正覆盖 stage1 |
| `reason=v3_numeric_keep_stage1_anchor` | 150 | 多数情况下 top candidate 本身就是 anchor |
| `reason=v3_numeric_preserve_stage1_gate` | 16 | 少数情况下 challenger 出现，但 gate 仍拒绝覆盖 |
| `sel_src=stage1+stage2` | 165 | 大多数最终答案既是 stage1，也是 stage2 重复生成的答案 |
| `sel_src=stage2` | 0 | 没有任何一次最终答案来自纯 stage2 新候选 |
| 平均 `ovr_p` | 0.0050 | override 概率几乎塌到 0 |
| 最大 `ovr_p` | 0.5000 | 出现在最早期初始化阶段，不代表学出了 override |
| 负 reward 样本数 | 7 | 有负回报样本，但仍未触发 override |
| 平均 reward | 0.5488 | 训练流程在跑，但行为上仍明显保守 |

### 1.2 训练是否“活着”

从参数更新角度看，当前训练不是死的。

最近一次落盘 checkpoint：

- 保存时间：`2026-04-02T12:59:56Z`
- `v3_candidate_steps = 544`
- `v3_override_steps = 394`
- `v3_reviewer_steps = 1923`
- `LMPO num_updates = 100`

这说明：

- `candidate / override / reviewer` 三个头在持续更新
- `runtime_v2` 的 `LMPO` 也在更新
- 当前问题不是“训练没发生”，而是“训练发生了，但学到的行为退化成了保守策略”

## 2. 当前已经确认的问题

### 2.1 最终策略塌成了“stage1 anchor 保守器”

这是当前最核心的问题。

在 `gsm8k` 上，到目前为止：

- 最终决策 `166/166` 都是 `use_stage1_anchor`
- `use_stage2_override = 0`

这意味着当前 `v3` 虽然名义上是“二阶段重排与覆盖”，但实际行为已经退化为：

- 如果 stage2 复现了 stage1 答案，就继续选它
- 如果 stage2 给出不同答案，也几乎不会覆盖 stage1

换句话说，当前 `v3` 的主要功能不是“纠正 stage1”，而是“保护 stage1”。

对应代码位置：

- `Stage2V3Pipeline._metadata_with_stage1_anchor(...)`
- `Stage2RuntimeV3._candidate_bank_bundle(...)`
- `Stage2RuntimeV3._select_numeric_candidate(...)`

## 2.2 Override 监督严重失衡，几乎没有正样本

当前 `gsm8k` 的 stage1 精度已经非常高。前 `166` 个样本里有 `161` 个 `task_success`。

这直接带来一个训练层面的后果：

- 真正需要 stage2 覆盖 stage1 的样本非常少
- `override_model` 在大多数样本上看到的监督都是“不要覆盖”

当前 `override` 的监督逻辑是：

- 只有当 `challenger_target > anchor_target` 时，才给 `override_target = 1`
- 否则给 `0`

对应实现：

- `Stage2RuntimeV3.learn_from_run(...)`
- `override_target = 1.0 if challenger_target > anchor_target else 0.0`

因此在 `gsm8k` 这种 stage1 ceiling 很高的数据集上，`override_model` 天然会快速学成“永远别动 stage1”。

这不是参数没更新，而是训练标签分布本身就把它推向保守解。

## 2.3 “反保守”的惩罚没有真正传到最终决策头

当前 `v3` 管线里虽然有一条对保留 anchor 的惩罚：

- 如果最终用了 `stage1 anchor`
- 就会对 `learning_target` 扣一次 `fallback_penalty`

对应代码：

- `mas_stage2_v3/pipeline.py`

但这条惩罚主要通过 `reward_target` 传给了 `super().learn_from_run(...)`，也就是：

- 更偏向影响 `LMPO`
- 并没有直接改变 `candidate / override / reviewer` 三个头自己的监督标签

而 `v3` 三个头的监督是在 `Stage2RuntimeV3.learn_from_run(...)` 里单独重算的：

- `candidate` 监督来自 candidate 自己的 evaluator 分数
- `override` 监督来自 challenger 是否优于 anchor
- `reviewer` 监督来自 reviewer 事件是否与 candidate 真实质量方向一致

所以当前系统出现了一个重要的不一致：

- 管线级目标在说“不要总保 stage1”
- 但最终做决策的三个头并没有被直接训练成“敢于在必要时推翻 stage1”

这会让整体系统更容易退化成：

- 图层在学
- 选择层在更新
- 但最终仍持续保守

## 2.4 Candidate Bank 的聚合方式会放大 anchor 类答案

当前 candidate entry 的特征里，显式包含了这些强放大项：

- `occurrence_count`
- `sink_support`
- `turn_coverage`
- `stage1_anchor`

对应代码：

- `Stage2RuntimeV3._candidate_entry_features(...)`

这在当前 `v3` 设计下会产生一个非常强的偏置：

1. `stage1 anchor` 被直接注入 candidate bank
2. 所有 task nodes 每轮都执行
3. 如果若干 stage2 节点复现了 stage1 的答案，这个答案会在多轮、多节点中被重复累计
4. `occurrence_count` 和 `sink_support` 会持续变大
5. 排序时这个“重复过很多次的旧答案”就更容易压住真正新的 challenger

当前日志里：

- `sel_src=stage1+stage2` 有 `165` 次
- `sel_src=stage2` 为 `0`

这说明现在的主导模式不是“stage2 提出新答案”，而是“stage2 反复复制 stage1 答案并强化其统计优势”。

这会让二阶段逐渐丧失纠错能力。

## 2.5 所有节点每轮都执行，放大了重复答案，而不是放大差异化探索

当前修正后的 `v3` 明确取消了显式 `active node` 裁剪。

对应实现：

- `Stage2RuntimeV3._active_task_nodes_v2(...)` 直接返回全部 `task_nodes`

这意味着：

- 当前不是“边收缩后，活跃节点也随之变少”
- 而是“所有节点每轮都执行，只是边决定谁能收到谁的信息”

这个设计本身不一定错，但在 `gsm8k` 上它有一个明显副作用：

- 当 stage1 已经很强时，节点之间更容易收敛到相同答案
- 结果不是产生更多 challenger
- 而是更高频地重复 anchor 类答案

于是 candidate bank 中最强的统计信号，变成了“谁最像 stage1，谁出现得最多”，而不是“谁真正修正了 stage1”。

## 2.6 Edge pruning 在当前配置下没有形成足够强的去噪作用

从设计目标看，当前希望是：

- 随着轮数推进，边逐渐收缩
- 无效通信减少
- 图逐步聚焦

但就当前实现和已有 smoke 观察来看，这条机制还没有表现出明显的“逐轮去噪”效果。

原因主要有两层：

1. `runtime_v2` 的 `keep_k` 机制在小图上本来就偏宽松
2. 当前 gate 分数普遍高于阈值时，边会被持续保留

对应实现：

- `Stage2RuntimeV2._turn_keep_k(...)`
- `Stage2RuntimeV2._activate_edges_v2(...)`

在之前的 smoke replay 中，5 条边从 `turn_0` 到 `turn_4` 一直都是活跃的。这说明当前边门控至少在一些小图上没有形成明显稀疏化。

对 `gsm8k` 而言，这会进一步放大重复答案传播，而不是帮助系统逐轮聚焦到“真正有效的 challenger 路径”。

## 2.7 Reviewer 学到的是“支持强候选的共识”，不是“主动发现 anchor 错误”

当前 reviewer 的 target 设计是：

- 如果某 candidate 最终质量高，则支持它的 `pass/preserve` 被奖励
- 挑战它的 `challenge/reject/conflict` 被惩罚

对应实现：

- `Stage2RuntimeV3._reviewer_target(...)`

这在 stage1 强势的数据集上会产生一个次生效应：

- reviewer 更容易学会“相信当前共识答案”
- 而不是“在必要时稳定地指出共识答案有错”

当共识本身大多与 stage1 一致时，这条 reviewer 信号会继续反向强化 anchor。

于是形成一个闭环：

1. anchor 更常胜
2. reviewer 更信 anchor
3. candidate 聚合里 reviewer 权重更偏向 anchor
4. override 更不敢触发

## 2.8 不确定性估计过早塌到下限，失去区分作用

三个 `v3` 头目前使用的是 `OnlineLinearModel`。

它的 uncertainty 由：

- `residual_ema / sqrt(steps)`

决定，并且下限是 `0.03`。

对应实现：

- `mas_stage2/learning.py`

实际日志里几乎一直看到：

- `sel_unc=0.03`

这说明当前不确定性估计很快就掉到了下限，后面几乎不再提供有效区分信息。

这会带来两个问题：

1. `override_model` 中关于 `challenger_model_uncertainty / anchor_model_uncertainty` 的特征意义变弱
2. 日志上看似“模型非常自信”，但这种自信大概率只是估计器塌了，不代表模型真的学稳了

## 3. 当前可以明确下的结论

基于 `gsm8k`，当前已经可以确认以下几点。

### 3.1 当前问题不是“训练没生效”

可以确认：

- 参数在更新
- `LMPO` 在更新
- 三个 `v3` 头也在更新

所以问题不在于训练停了或者梯度没走到，而在于：

- 训练目标
- 选择机制
- 候选聚合方式

共同把系统推向了一个保守局部最优。

### 3.2 当前问题是“行为没有体现出二阶段价值”

在 `gsm8k` 上，二阶段当前没有表现出以下任何一个关键特征：

- 能稳定产出纯 stage2 新答案
- 能在失败样本上触发 override
- 能把 reviewer 变成对抗性纠错器
- 能通过边稀疏化减少重复答案传播

所以从行为上看，当前 `v3` 还没有形成强意义上的二阶段纠错能力。

### 3.3 GSM8K 本身对二阶段很不友好

这也是需要记录的重要背景。

由于当前一阶段在 `gsm8k` 上已经接近 `97%`：

- 它天生就不是最能放大二阶段收益的数据集
- 也不是最适合用来判断“override 能否真正学起来”的数据集

因此：

- 当前 `gsm8k` 结果足以暴露保守退化问题
- 但不足以单独决定最终算法重构方案

## 4. 暂时不应只靠 GSM8K 下结论的部分

以下问题需要等 `mbpp` 和 `humaneval` 结果出来后再统一判断：

1. `override` 的塌缩是否是所有任务上的共性问题  
   说明：`gsm8k` 上正样本太少，code 任务上可能更容易观察到真正的覆盖需求。

2. candidate bank 的重复放大是否只在 numeric 任务上特别严重  
   说明：代码任务里候选差异通常更大，重复答案未必像 numeric 任务这样容易形成绝对主导。

3. reviewer 信号是否在 code 任务上更有价值  
   说明：对于 `humaneval/mbpp`，`parse_ok`、`entry_point_ok`、结构性 critique 可能更能体现 reviewer 的作用。

4. edge pruning 不明显是否只是当前小图和当前阈值的问题  
   说明：不同数据集的图结构复杂度不一样，code 任务可能更容易体现边门控价值。

## 5. 三个可学习头的逐项问题归因

### 5.1 `candidate_model`

当前 `candidate_model` 真正在学的不是“谁更可能纠正 stage1”，而更接近“谁更像当前候选共识”。

原因是它当前吃进去的特征里，强相关项主要是：

- `occurrence_count`
- `sink_support`
- `turn_coverage`
- `reviewer_mean_trust`
- `stage1_anchor`

对应实现：

- `Stage2RuntimeV3._candidate_entry_features(...)`

这些特征在 `gsm8k` 上会天然偏向下面这类答案：

- 被更多节点重复输出的答案
- 被更多轮次重复保留的答案
- 已经和 stage1 anchor 对齐的答案

因此 `candidate_model` 目前更像是在学：

- “谁更稳定地重复出现”

而不是：

- “谁对 stage1 构成了真正改进”

这也是为什么当前日志中大量样本显示：

- `sel_src=stage1+stage2`
- 但 `sel_src=stage2` 为 `0`

### 5.2 `override_model`

当前 `override_model` 的问题最直接。

它的训练目标定义为：

- `challenger_target > anchor_target` 才给 `1`
- 否则给 `0`

对应实现：

- `Stage2RuntimeV3.learn_from_run(...)`

在 `gsm8k` 上，这个目标会天然塌成极端不平衡监督：

- stage1 anchor 大多本来就是对的
- 真正比 anchor 更好的 challenger 极少
- 所以 `override_model` 会看到大量负样本、极少正样本

于是这个头被训练成：

- “不要 override”

而不是：

- “在关键失败样本上果断 override”

当前行为证据也完全一致：

- `override=0`
- `avg_ovr_p≈0.005`

### 5.3 `reviewer_model`

当前 `reviewer_model` 的问题不是它完全没用，而是它很容易形成“赢家偏置”。

当前 reviewer 的训练方式是：

- 先给每条 review event 赋 trust
- 再把这个 trust 聚合回 candidate 特征
- 最后根据 candidate 的真实质量，反过来训练 reviewer

对应实现：

- reviewer 赋权：`_reviewer_weight(...)`
- reviewer 聚合：`_aggregate_occurrence_feedback(...)`
- reviewer 监督：`_reviewer_target(...)`

在 `gsm8k` 上，这会形成一个稳定闭环：

1. stage1 anchor 本来就更常是对的
2. 支持 anchor 的 reviewer 更容易被学成“可信”
3. 这些 reviewer trust 又进一步抬高 anchor 类候选
4. 挑战 anchor 的 reviewer 更难被保留下来

所以 `reviewer_model` 当前更像是在学：

- “谁更符合当前赢家”

而不是：

- “谁更擅长指出当前赢家何时错了”

### 5.4 三者之间的耦合关系

这三个头不是独立出问题，而是串起来形成了一个保守闭环：

1. `candidate_model` 放大重复出现的 anchor 类答案
2. `override_model` 在极度不平衡监督下学成几乎永不触发
3. `reviewer_model` 继续提升支持 anchor 的反馈权重

于是系统逐渐收敛到：

- 候选排序保守
- 覆盖决策保守
- reviewer 反馈也保守

这就是当前 `gsm8k` 上“训练在继续，但二阶段价值没有体现”的核心机制。

## 6. 基于当前 GSM8K 的预估修改方向

以下修改建议只是在 `gsm8k` 当前证据基础上的预估，不等同于最终定案。最终是否全部实施，还需要结合 `mbpp` 与 `humaneval` 结果。

### 6.1 对 `candidate_model` 的预估修改

建议方向：

- 弱化 `occurrence_count / sink_support / turn_coverage / stage1_anchor` 对排序的主导作用
- 显式区分：
  - `anchor_support`
  - `stage2_unique_support`
  - `stage2_unique_roles`
  - `stage2_unique_turns`
- 对“和 anchor 相同的重复答案”做 capped 计数，而不是持续累加

预期效果：

- 降低 stage2 对 stage1 答案的重复放大
- 让“少数但更优”的 challenger 有机会进入 top-1

### 6.2 对 `override_model` 的预估修改

建议方向：

- 不再只用二值目标 `challenger > anchor`
- 改为相对增益学习，例如让目标与 `challenger_target - anchor_target` 正相关
- 对这类样本提高权重：
  - `anchor fail, challenger success`
  - `anchor low score, challenger clear improvement`
- 对 numeric 任务加入更强的 rescue 训练信号：
  - 如果 anchor 错且 challenger 对，应强推 override

预期效果：

- 让 `override_model` 学“什么时候值得推翻 stage1”
- 而不是学“绝大多数时候别动 stage1”

### 6.3 对 `reviewer_model` 的预估修改

建议方向：

- 先削弱 reviewer trust 对 candidate 排序的回流权重
- 或在训练前期冻结 reviewer 为统一权重
- 把 reviewer 的重点训练样本限制在：
  - `anchor` 与 `challenger` 真正分歧的样本
  - 失败样本
  - 高价值 challenge 样本

预期效果：

- 避免 reviewer 过早学成“共识放大器”
- 提高 reviewer 对错误 anchor 的挑战价值

### 6.4 对图执行层的预估修改

建议方向：

- 进一步增强 per-turn edge sparsification
- 不一定恢复旧 `active node` 机制，但至少应减少“全图持续重复传播同一答案”
- 如果继续保留全节点执行，可以考虑：
  - 对重复输出 anchor 的节点降低后续轮次贡献
  - 对产生新候选的节点提高探索权重

预期效果：

- 降低重复答案在 5 轮内被不断刷大的问题
- 让图通信更接近“筛选有效 challenger”，而不是“强化现有共识”

### 6.5 优先级预估

如果后续要按最小改动、最大收益来改，我当前建议的优先级是：

1. 先改 `candidate bank / candidate_model`
2. 再改 `override_model` 的目标定义
3. 再改 `reviewer_model` 的闭环权重
4. 最后视 `mbpp / humaneval` 结果再决定图层是否继续大改

原因是：

- 当前最强的退化首先来自 candidate bank 对 anchor 的统计放大
- 如果 top candidate 本身就很难脱离 anchor，那么后面的 override 和 reviewer 再好也很难救回来

## 7. 当前文档的最终结论

只基于当前 `gsm8k`，对 `stage2 v3` 的最准确定义是：

`训练在进行，参数在更新，但当前行为已经退化为“stage1 anchor 保守器”；其根本原因不是训练失效，而是 override 监督极度失衡、candidate bank 放大 anchor 重复、reviewer 反馈自证循环、以及当前边/节点执行策略未能形成足够强的去噪与差异化探索。`

因此，`gsm8k` 已经足以支持下面这个判断：

- 当前 `v3` 在 numeric 任务上没有学出强二阶段价值

但 `gsm8k` 还不足以支持下面这个更强判断：

- 当前 `v3` 在所有任务上都必然无效

这个更强判断需要等 `mbpp` 与 `humaneval` 跑完后，再做综合分析。


## 8. 2026-04-03 补充证据：训练阶段已结束后的硬结论

下面这部分不是基于训练中途日志，而是基于 `post_train checkpoint` 中保存的完整 `264` 条训练样本统计，以及截至 `2026-04-03` 当前日志里已经落盘的 `439/1055` 条测试样本。

### 8.1 训练阶段已经可以下一个更强结论

从 `checkpoint.json` 可以直接确认：

- `train_index_completed = 264`
- `train_phase_complete = true`
- `phase = post_train`

也就是说，训练阶段本身已经完整结束，不再是“只看到前半段训练现象”。

### 8.2 训练集上，二阶段对最终任务结果的净增益为零

训练集 `264` 条样本上的结果非常明确：

- `stage2_task_score > stage1_task_score` 的样本数：`0`
- `stage2_task_score = stage1_task_score` 的样本数：`264`
- `stage2_task_score < stage1_task_score` 的样本数：`0`

也就是说，在训练集上，`stage2 v3` 从来没有一次把最终任务正确性从 `stage1` 往上推。

进一步看 reward：

- `stage2_reward > stage1_reward` 的样本数：`0`
- `stage2_reward < stage1_reward` 的样本数：`264`

原因也很直接：

- `stage1` 与 `stage2` 的任务成功数完全相同，都是 `256/264`
- `stage2` 额外付出了通信、推理轮次、token 与延迟成本

所以训练完成后，`gsm8k` 上的 `stage2 v3` 不是“增益很小”，而是：

- 任务正确率增益为 `0`
- 总 reward 净增益为负

### 8.3 最终选择实际上 100% 等于 stage1 anchor

训练集上还有一个比“override 很少触发”更强的事实：

- `fallback_applied = 264`
- `v3_stage1_anchor_used = 264`
- `v3_selected_candidate_digest == v3_stage1_anchor_digest` 的样本数：`264/264`

这说明最终被选中的答案并不只是“偏向 stage1”，而是：

- 每一条训练样本，最终答案都和 `stage1 anchor` 完全同一个候选

因此，当前 `v3` 在 `gsm8k` 上已经不是“保守但偶尔能改进”，而是：

- 二阶段生成了很多候选
- 但最终决策层把所有样本都收缩回了 `stage1 anchor`

### 8.4 Candidate bank 对 anchor 的重复放大，已经达到数量级差异

训练集统计显示，`stage1 anchor` 在 candidate bank 中的统计优势非常夸张：

- `anchor` 平均 `occurrence_count = 19.54`
- 最强 `non-anchor` 候选平均 `occurrence_count = 2.88`
- `anchor` 平均 `sink_support = 8.37`
- 最强 `non-anchor` 候选平均 `sink_support = 1.96`

并且在存在 `non-anchor` 候选的 `223` 条样本里：

- 有 `209` 条样本中，`anchor` 的出现次数不低于所有 `non-anchor`

这说明 candidate bank 当前不是在“保留多样化候选”，而是在：

- 反复累计 anchor 的重复出现次数
- 让 anchor 在统计量上获得近乎不可逆的领先

### 8.5 `candidate_model` 已经把 anchor 学成压倒性 top-1

训练集上，`candidate_model` 对 anchor 与 challenger 的打分差距已经非常大：

- `anchor` 平均 `candidate_model_score = 0.883`
- 最强 `non-anchor` 平均 `candidate_model_score = 0.119`

看训练最后 `50` 条样本，这个差距仍然存在，甚至更稳：

- `anchor` 平均分 `0.902`
- 最强 `non-anchor` 平均分 `0.175`
- 在最后 `50` 条里，有 `43/45` 条是 `anchor score >= best non-anchor score`

这说明 `candidate_model` 不是“还没学出来”，而是已经明确学成：

- `anchor` 应该排在最前面

值得注意的是，`non-anchor` 的 `reviewer_mean_trust` 平均值其实高于 `anchor`，但它仍然进不了 top-1。说明真正主导排序的，仍是：

- `stage1_anchor`
- `occurrence_count`
- `sink_support`

而不是 reviewer 对 challenger 的支持。

### 8.6 `override_model` 在训练前几步就塌缩到了接近永不触发

训练集上的 `override probability` 衰减速度极快：

- 第 `1` 条样本：`ovr_p = 0.5`
- 到第 `2` 条样本时，已经 `ovr_p <= 0.01`
- `264` 条训练样本里，有 `259` 条样本 `ovr_p = 0.0`

同时，所选候选的不确定性也很快塌到地板：

- 到第 `7` 条样本时，`selected_model_uncertainty <= 0.03`

这说明当前问题不是“后期过拟合”，而是：

- 训练前期几步，override 头和 uncertainty 就已经进入坏的吸引子
- 后续训练基本只是在强化这种保守状态

### 8.7 截至当前测试日志，测试分布与训练结论一致

截至 `2026-04-03` 当前落盘的测试日志，`test` 只进行到 `439/1055`，尚未完整结束；但已经落盘的这 `439` 条测试样本与训练结论完全一致：

- `decision=use_stage1_anchor`：`439/439`
- `sel_src=stage2`：`0`
- 平均 `ovr_p = 0.00023`
- 平均 `sel_unc = 0.03`

这说明至少到当前为止，测试集上也没有出现“训练虽然保守，但测试能学出 override”的迹象。

### 8.8 这一批补充证据带来的结论更新

基于训练阶段完整结果，现在可以把原文档里的判断进一步收紧为：

- `gsm8k` 上当前 `stage2 v3` 已经不是“价值有限的二阶段”，而是“严格退化为 stage1 复制器”
- 问题主因不是 runtime 没跑通，也不是参数没更新
- 问题主因是三件事叠加：
  - `candidate bank` 对 anchor 的重复统计放大
  - `candidate_model` 把 anchor 学成压倒性 top-1
  - `override_model` 在极早期就塌为近乎永不触发

因此，对 `gsm8k` 来说，后续修改不应再以“继续调权重看看”为主，而应直接改：

1. `candidate bank / candidate_model` 的统计与特征定义
2. `override_model` 的目标定义与样本加权
3. reviewer 信号回流到 candidate 排序的方式
