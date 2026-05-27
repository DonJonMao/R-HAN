# Stage2 V3.1 算法设计方案

更新时间：2026-04-03

本文档描述新的 `stage2 v3.1` 方案。设计目标不是继续给 `v3` 补更多阈值和人工权重，而是把当前最关键的三段选择逻辑改成更明确的、实例自适应的可学习判别流程。

相关代码入口计划为：

- `mas_stage2_v3_1/config.py`
- `mas_stage2_v3_1/runtime.py`
- `mas_stage2_v3_1/pipeline.py`
- `train_mas_stage2_v3_1_target_suite.py`
- `run_stage2_v3_1_selected_serial_20260403.sh`

## 1. 总体目标

`v3.1` 的核心目标只有三条：

1. 让 `stage2` 不再先被 `candidate_model` 的绝对分数卡死，再由 `override_model` 被动兜底。
2. 让 reviewer 信号不再围绕“当前候选自身是否还行”打转，而是围绕“它相对 anchor 到底有没有纠错价值”来校准。
3. 让 `gsm8k` 上的保守、`mbpp/humaneval` 上的激进，都由同一套按样本自适应的学习规则处理，而不是数据集特判。

因此，`v3.1` 不再把“最终决策”写成若干显式加权项之后再手工门控，而是重构为：

`候选质量估计 + 相对优势比较 + reviewer 相对校准`

其中：

- `候选质量估计` 回答“这个 candidate 本身好不好”
- `相对优势比较` 回答“这个 challenger 是否真的优于 stage1 anchor”
- `reviewer 相对校准` 回答“这个 review 事件是否真的在帮助区分 anchor 与 challenger”

## 2. 背景问题：为什么需要从 v3 走向 v3.1

当前 `v3` 已经暴露出三个一阶问题：

1. `candidate_model` 会把“更像 stage1、被重复得更多”的答案打高分，导致 challenger 还没进入最终比较，就先被绝对排序压住。
2. `override_model` 的训练标签极不平衡，在 `gsm8k` 上快速塌成“默认不翻案”。
3. `reviewer_model` 当前更像高召回、低精度的挑战器，经常反对 anchor，但并不能稳定指出 anchor 何时真的错了。

这三个问题串联起来以后，会形成一个闭环：

1. `stage1 anchor` 被注入 candidate bank。
2. 多轮多节点反复复现同一答案后，anchor 类候选的统计量被继续放大。
3. `candidate_model` 优先把 anchor 类候选排在前面。
4. `override_model` 只看这个已经被筛过一遍的 challenger。
5. reviewer 对 challenger/anchor 的噪声信号又会反过来加剧前两步偏置。

所以 `v3.1` 的重点不是“把三个线性头换成更深的网络”，而是先把监督结构和决策结构改对。

## 3. 总体流程

`v3.1` 保留 `v3` 的图执行骨架，不动下面这些已经存在的部分：

- `stage1` 固定结构
- `runtime_v2` 的多轮执行壳
- `GNN` 边赋值
- `global node`
- 每轮全量 task nodes 执行
- latent memory / LMPO / export / private memory

`v3.1` 只重构最后的候选选择与在线学习部分。

完整流程如下：

1. `stage1` 生成固定 `UnionGraph`
2. `stage2 v3.1` 在固定图上运行多轮
3. 所有 task nodes 每轮执行，边由 `GNN + global node` 决定信息流
4. 运行结束后构建 candidate bank
5. 对每个 candidate 估计一个绝对质量分数 `Q(c)`
6. 对每个 challenger 与 `stage1 anchor` 估计一个相对胜率 `P(c > anchor)`
7. reviewer 事件不直接投票，而是先经过“相对纠错价值”校准
8. 最终答案按“相对优势优先”的原则选出
9. 训练时同时做：
   - candidate 绝对质量学习
   - pairwise 相对排序学习
   - reviewer 相对可靠性学习

## 4. 模块设计

### 4.1 图执行层：沿用 v3，不再改执行骨架

这一层与 `v3` 保持一致：

- 仍然继承 `runtime_v2`
- 仍然使用 `GNN` 对边做 gate
- 仍然使用 `global node` 聚合全局状态
- 仍然不恢复旧的 `active node`
- 仍然让所有 task nodes 每轮执行

`v3.1` 不在这里引入新的硬编码控制器，避免再次回到旧版 `controller/edge_model/active_node` 那一套。

### 4.2 Candidate Bank 2.0：保留聚合，但降低“重复回声”优势

`v3.1` 仍然保留 candidate bank，因为它天然适合把多轮多节点输出转成统一的学习对象。

每个 `candidate entry` 仍然维护：

- `text`
- `digest`
- `occurrence_count`
- `sink_support`
- `source_node_ids`
- `source_roles`
- `turn_indices`
- reviewer 的 `pass/challenge/uncertain`
- `stage1_anchor`
- 代码任务的 `parse_ok / entry_point_ok`

但在 `v3.1` 中，candidate 的“绝对质量头”不再直接吃“stage1_anchor 这一身份”。

新增并强调的特征是：

- `source_node_count`
- `source_role_count`
- `turn_count`
- `sink_ratio = sink_support / occurrence_count`
- `review_pass_ratio`
- `review_challenge_ratio`
- `review_margin`
- `review_consensus`
- `occurrence_saturation`

设计动机是：

- `occurrence_count` 仍然有用，但只能表示“被重复看到”，不能再等价于“更正确”
- `source_role_count / source_node_count / sink_ratio` 更接近“多路支持”
- `review_margin / review_consensus` 更接近“经过校准后的外部证据”

这一步的核心是：把“重复”从强主导证据降回弱证据。

### 4.3 Candidate Quality Head：学习 candidate 本身的质量

对每个 candidate 构造绝对特征 `phi_abs(c)`，学习一个质量头：

`Q(c) = candidate_model(phi_abs(c))`

这里的 `Q(c)` 表示：

- 如果只看 candidate 本身，不跟 anchor 比，它作为最终答案的质量有多高

`Q(c)` 的训练目标仍然来自 evaluator 对该 candidate 的直接评估：

- `mean_success`
- `mean_task_score`

合成为同一个 `[0,1]` 目标值。

这一层保留的意义是：

- 当没有 `stage1 anchor` 时，系统仍然能独立选答案
- 当存在多个非 anchor 候选时，可以给 pairwise 模块提供基础质量参考

但 `v3.1` 不再让这层单独决定“是否翻案 stage1”。

### 4.4 Pairwise Advantage Head：用相对比较替代 v3 的被动 override

这是 `v3.1` 的主变化。

对任意 challenger `c` 与 anchor `a`，构造差分特征：

`phi_pair(c, a)`

主要包括：

- `Q(c), Q(a), Q(c)-Q(a)`
- `sink_ratio` 的绝对值与差值
- `source_diversity` 的绝对值与差值
- `review_margin` 的绝对值与差值
- `review_consensus` 的绝对值与差值
- `candidate_model_uncertainty` 的绝对值与差值
- `occurrence_saturation` 的绝对值与差值
- 对代码任务的 `parse_ok / entry_point_ok`

再学习一个 pairwise 头：

`P(c > a) = pairwise_model(phi_pair(c, a))`

这意味着 `v3.1` 的最终决策不再是：

1. 先用绝对分数排出一个第一名
2. 再让 override 去判断要不要翻 anchor

而是：

1. 对所有 challenger 都分别和 anchor 做一次相对比较
2. 选出 `P(c > a)` 最大的 challenger
3. 如果其胜率大于 `0.5`，则翻案；否则保留 anchor

这一步解决的是 `v3` 中最关键的结构偏差：

- 以前 challenger 会先被绝对排序错杀
- 现在只要某个 challenger 相对 anchor 真有优势，它就有机会直接进入最终决策

### 4.5 Reviewer Calibration 2.0：把 reviewer 从“愤青”改成“相对纠错证据”

`v3` 中 reviewer 的 target 更接近：

- 支持高质量 candidate 就是好 reviewer
- 反对高质量 candidate 就是坏 reviewer

这个定义太“绝对”，会导致 reviewer 更容易追随当前多数共识，而不是帮助识别 anchor 错误。

`v3.1` 改成相对校准：

#### 对非 anchor candidate 的 reviewer 事件

如果 reviewer 在评论 challenger `c`，则参考对象是 anchor `a`。

- 若 `target(c) > target(a)`，则说明 challenger 确有翻案价值
- 此时 `pass/preserve` 属于高质量 reviewer 信号
- `challenge/reject/conflict` 属于低质量 reviewer 信号

#### 对 anchor 的 reviewer 事件

如果 reviewer 在评论 anchor，本次参考对象变成当前最强非 anchor challenger `c*`。

- 若 `target(anchor) < target(c*)`，说明 reviewer 对 anchor 的质疑是有价值的
- 若 `target(anchor) >= target(c*)`，说明 reviewer 对 anchor 的攻击更可能是噪声

这样 reviewer 学到的就不再是：

- “谁看起来像共识答案，我就支持谁”

而变成：

- “当 anchor 真该被推翻时，谁的评论更可信”

### 4.6 最终答案选择

`v3.1` 的最终选择规则如下。

#### 情况 A：没有 stage1 anchor

直接选择 `Q(c)` 最高的 candidate。

#### 情况 B：有 stage1 anchor

1. 枚举所有非 anchor challenger
2. 对每个 challenger 计算 `P(c > anchor)`
3. 选出 `P(c > anchor)` 最大的 challenger `c*`
4. 若 `P(c* > anchor) > 0.5`，则输出 `c*`
5. 否则保留 anchor

对代码任务仍保留必要的硬约束：

- challenger 若无法通过基本解析或入口函数检查，则不能直接翻案

这里保留硬约束是合理的，因为这是任务合法性约束，不是人工经验权重。

## 5. 训练设计

### 5.1 Candidate 绝对质量学习

对本题保留下来的 top candidates 与 anchor 做 evaluator 打分，得到每个 `candidate_target(c)`。

然后更新：

- `candidate_model(phi_abs(c)) -> candidate_target(c)`

这一层仍是点式监督。

### 5.2 Pairwise 排序学习

`v3.1` 不再只训练“best challenger 对 anchor”的单条 override 标签，而是对一个样本内的多个 candidate 对构造 pairwise 监督。

做法是：

1. 取本题的 top candidates 与 anchor
2. 对任意有序对 `(c_i, c_j)` 构造 `phi_pair(c_i, c_j)`
3. 若 `target(c_i) > target(c_j)`，则 label 为 `1`
4. 若 `target(c_i) < target(c_j)`，则 label 为 `0`
5. 若两者相同，则 label 为 `0.5`

这一步的意义是：

- 单个样本内部就能产生更丰富的监督
- 不再把 pairwise 监督全部压在“anchor 是否需要翻案”这一个极不平衡的二分类问题上
- 在 `gsm8k` 上，虽然 anchor 往往更强，但非 anchor 之间的相对好坏也仍然可以训练 pairwise 头

### 5.3 Reviewer 相对可靠性学习

对每个 reviewer event：

1. 找到它评论的 candidate
2. 根据 candidate 是 anchor 还是 challenger，确定参考对象
3. 用 `candidate_target - reference_target` 构造相对真值
4. 再根据事件类型生成 reviewer target

大体原则是：

- `pass/preserve` 应支持相对更好的答案
- `challenge/reject/conflict` 应反对相对更差的答案
- `uncertain` 只在两者接近时更合理

### 5.4 数据量是否需要增加

`v3.1` 的第一版不要求立刻增加训练样本。

原因是：

- pairwise 监督会把每个样本内部的监督信号显著放大
- reviewer 的相对校准也比 `v3` 更高效

但如果后续还想继续提升，优先增加的不是“更多普通样本”，而是：

- `stage1` 出错但存在有效 challenger 的样本
- reviewer 对 anchor 存在明显分歧的样本
- `mbpp/humaneval` 中 stage2 曾产生正确新代码但未被选中的样本

也就是更偏“纠错型样本”，而不是简单扩量。

## 6. 与 v3 的核心区别

### 6.1 总结版

`v3` 的核心路径是：

`绝对 candidate 排序 -> 只拿第一名 challenger 去和 anchor 做 override -> reviewer 以绝对好坏做校准`

`v3.1` 的核心路径是：

`绝对质量估计 -> 全 challenger 与 anchor 的 pairwise 比较 -> reviewer 以相对纠错价值做校准`

### 6.2 逐项对比

| 模块 | v3 | v3.1 |
|---|---|---|
| candidate 绝对评分 | 是最终主排序入口，容易先把 challenger 压掉 | 只负责估计 candidate 本身质量，不单独决定翻案 |
| override | 只看绝对排序第一名 challenger | 对所有 challenger 与 anchor 做 pairwise 比较 |
| reviewer target | 围绕 candidate 绝对质量定义 | 围绕 candidate 相对 anchor 的纠错价值定义 |
| anchor 信息使用方式 | 既进入 candidate bank，又可能直接被绝对排序特征强化 | 仍保留为基线候选，但从绝对质量头中移除身份先验 |
| 重复答案的作用 | 容易被 `occurrence_count / sink_support` 放大 | 改为饱和特征和多样性特征，降低回声优势 |
| 监督结构 | 点式监督为主 | 点式 + pairwise 联合监督 |
| 数据集适配 | 容易在 `gsm8k` 过保守，在 `mbpp` 过激进 | 由每个样本内部的 pairwise 证据自适应决定 |

## 7. 预期影响

### 7.1 对 GSM8K

预期变化不是“覆盖率暴涨”，而是：

- 不再机械地把最像 stage1 的 candidate 永远打最高
- 即便最终大多数样本仍保留 anchor，也应是因为 pairwise 判断认为 anchor 更优，而不是 candidate 头先塌了
- reviewer 的质疑信号会更容易在少数 stage1 错样本上被识别出来

### 7.2 对 MBPP / HumanEval

预期更明显的变化是：

- challenger 不再需要先赢下“绝对重复度竞赛”才有资格翻案
- 代码任务里真正更优的新程序更容易被 pairwise 头选中
- reviewer 对错误 anchor 的高质量质疑更容易被转化为正信号

## 8. 实现原则

`v3.1` 在实现上坚持三条原则：

1. 不回退到旧版 `active node / edge_model / controller_model`
2. 不新增大量手工加权项，只保留任务合法性相关硬约束
3. 采用并行新版本目录，保留 `v3` 以便复现实验和对比

## 9. 落地计划

本次实现对应四步：

1. 先提交当前 `v3` 快照，保留问题版本基线
2. 新建 `mas_stage2_v3_1/` 与独立训练脚本
3. 实现 pairwise-based `v3.1`
4. 停止当前 `v3` 训练，重新启动 `v3.1` 串行训练


线性模型替换为moe/单一打分模型
debate引入强化学习