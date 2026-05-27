# Stage2-UCC Phase3a Unified MMLU-Pro Training Report

生成时间：2026-04-17  
实验目录：`outputs/stage2_ucc_phase3a_unified_mmlu_pro_3x_20260417_020010`

## 1. Scope

这份报告分析的是 `Stage2-UCC phase3a_unified_v1` 在 `MMLU-Pro` 上的当前训练态势。  
当前 run 尚未结束，因此本报告基于各 worker 的 `checkpoint.json`，而不是最终 `suite_report.json`。

主要证据文件：

- 判定逻辑：[Stage2-UCC/train_mas_stage2_phase3a_unified_target_suite.py](/mnt/nvme/projects/R-HAN/Stage2-UCC/train_mas_stage2_phase3a_unified_target_suite.py:58)
- `worker_0` 聚合统计：[worker_0 checkpoint](/mnt/nvme/projects/R-HAN/outputs/stage2_ucc_phase3a_unified_mmlu_pro_3x_20260417_020010/workers/worker_0/mmlu_pro/checkpoint.json:49589)
- `worker_1` 聚合统计：[worker_1 checkpoint](/mnt/nvme/projects/R-HAN/outputs/stage2_ucc_phase3a_unified_mmlu_pro_3x_20260417_020010/workers/worker_1/mmlu_pro/checkpoint.json:45133)
- `worker_2` 聚合统计：[worker_2 checkpoint](/mnt/nvme/projects/R-HAN/outputs/stage2_ucc_phase3a_unified_mmlu_pro_3x_20260417_020010/workers/worker_2/mmlu_pro/checkpoint.json:52051)

判定标准如下：

1. 先比较 `stage2_success` 与 `stage1_success`
2. 如果 success 打平，再比较 `stage2_task_score` 与 `stage1_task_score`
3. 由此给每个样本打 `better / worse / same`

## 2. Current Status

当前 run 还处于 `train` 阶段，没有 validation/test 结果。

| item | value |
|---|---:|
| workers | 3 |
| planned train samples | 1200 |
| completed train samples | 225 |
| progress | 18.75% |
| split analyzed | train only |
| stage2 version | `phase3a_unified_v1` |

各 worker 当前进度：

| worker | saved_at (UTC) | planned | completed | phase |
|---|---|---:|---:|---|
| `worker_0` | 2026-04-16T22:48:08Z | 400 | 75 | train |
| `worker_1` | 2026-04-16T22:37:50Z | 400 | 75 | train |
| `worker_2` | 2026-04-16T22:39:45Z | 400 | 75 | train |

## 3. Raw Data Table

### 3.1 Stage2 vs Stage1 outcome by worker

| worker | count | better | worse | same | stage1 success | stage2 success | delta success | stage1 task | stage2 task | delta task |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `worker_0` | 75 | 0 | 4 | 71 | 0.7600 | 0.7067 | -0.0533 | 0.7200 | 0.6800 | -0.0400 |
| `worker_1` | 75 | 0 | 5 | 70 | 0.8133 | 0.7467 | -0.0667 | 0.7600 | 0.7100 | -0.0500 |
| `worker_2` | 75 | 1 | 4 | 70 | 0.7467 | 0.7067 | -0.0400 | 0.7100 | 0.6800 | -0.0300 |
| `ALL` | 225 | 1 | 13 | 211 | 0.7733 | 0.7200 | -0.0533 | 0.7300 | 0.6900 | -0.0400 |

### 3.2 Stage2 train-side operating metrics

| worker | avg reward | avg task | avg success | avg latency | avg token | avg safety penalty | avg turns | avg memory records | avg structure reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `worker_0` | -2.8698 | 0.6800 | 0.7067 | 156.0595 | 0.0069 | 0.2053 | 5.0 | 52.1067 | 0.6374 |
| `worker_1` | -2.6662 | 0.7100 | 0.7467 | 150.5805 | 0.0071 | 0.1773 | 5.0 | 50.0667 | 0.6545 |
| `worker_2` | -2.7889 | 0.6800 | 0.7067 | 152.8239 | 0.0069 | 0.2053 | 5.0 | 50.4667 | 0.6455 |
| `ALL` | -2.7750 | 0.6900 | 0.7200 | 153.1546 | 0.0070 | 0.1960 | 5.0 | 50.8800 | 0.6458 |

附加结构指标：

| metric | overall mean |
|---|---:|
| coverage | 0.8904 |
| complementarity | 0.8941 |
| redundancy_quality | 0.2377 |
| candidate_count | 3.6000 |
| soft_class_count | 2.4622 |
| correction_attempt_count | 0.3911 |
| correction_accept_count | 0.0044 |

对应总量：

| metric | total |
|---|---:|
| total candidates | 810 |
| total soft classes | 554 |
| total correction attempts | 88 |
| total correction accepts | 1 |
| total correction improvements | 1 |
| total stage2 turns | 1125 |
| total stage2 memory records | 11448 |

## 4. Decision Analysis

### 4.1 Selection reason distribution

| selection reason | count | better | worse | same | stage1 success | stage2 success | delta success |
|---|---:|---:|---:|---:|---:|---:|---:|
| `phase3a_unified_anchor_wins_frontier` | 195 | 0 | 0 | 195 | 0.8000 | 0.8000 | 0.0000 |
| `phase3a_unified_override_frontier` | 23 | 1 | 13 | 9 | 0.5652 | 0.0435 | -0.5217 |
| `phase3a_unified_preserve_anchor_guard` | 7 | 0 | 0 | 7 | 0.7143 | 0.7143 | 0.0000 |

### 4.2 Final answer source type

| source type | count |
|---|---:|
| `stage1+stage2` | 194 |
| `stage2` | 23 |
| `stage1` | 8 |

解释：

1. 真正发生结果变化的几乎都是 `final_answer_source_type = stage2`
2. `stage1+stage2` 与 `stage1` 基本是保守型决策，当前没有造成回退
3. 当前损失几乎全部来自 `override_frontier`

### 4.3 Override quality

`override_frontier` 一共 23 次：

- 1 次把错题改对
- 13 次把对题改错
- 9 次改了但不改变最终判定

因此当前 override 的净收益是：

- `+1 better`
- `-13 worse`
- `net = -12`

如果只看发生变化的样本，坏变化占比为 `13 / 14 = 92.86%`。

## 5. Failure Pattern Analysis

### 5.1 Regression sample behavior

13 个 `worse` 样本有非常一致的形态：

1. 全部来自 `phase3a_unified_override_frontier`
2. 全部是 `final_answer_source_type = stage2`
3. 几乎全部表现为 `stage1_success = 1.0` 被翻成 `stage2_success = 0.0`
4. 对应 `task_score` 也从 `0.9` 掉到 `0.15`

也就是说，当前不是“stage2 没帮上忙”，而是“stage2 在少数 override case 中把正确 anchor 错翻了”。

### 5.2 Subject summary

| subject | count | better | worse | same |
|---|---:|---:|---:|---:|
| engineering | 19 | 0 | 4 | 15 |
| math | 34 | 0 | 2 | 32 |
| physics | 21 | 0 | 2 | 19 |
| other | 24 | 0 | 1 | 23 |
| chemistry | 21 | 1 | 1 | 19 |
| philosophy | 9 | 0 | 1 | 8 |
| business | 7 | 0 | 1 | 6 |
| computer science | 5 | 0 | 1 | 4 |
| law | 19 | 0 | 0 | 19 |
| health | 17 | 0 | 0 | 17 |
| biology | 16 | 0 | 0 | 16 |
| psychology | 15 | 0 | 0 | 15 |
| economics | 12 | 0 | 0 | 12 |
| history | 6 | 0 | 0 | 6 |

观察：

1. regression 不是均匀分布
2. `engineering` 最明显，19 个样本里有 4 个被错翻
3. `math` 和 `physics` 也出现了稳定负例
4. `law / health / biology / psychology / economics / history` 当前 0 regression，但也几乎没有 positive uplift

### 5.3 Source summary

高风险来源主要集中在以下 source：

| src | count | better | worse | same |
|---|---:|---:|---:|---:|
| `stemez-TransportPhenomena` | 5 | 0 | 3 | 2 |
| `stemez-Chemistry` | 14 | 1 | 1 | 12 |
| `theoremQA-Math` | 10 | 0 | 1 | 9 |
| `ori_mmlu-professional_accounting` | 5 | 0 | 1 | 4 |
| `stemez-Thermodynamics` | 3 | 0 | 1 | 2 |
| `theoremQA-EECS` | 3 | 0 | 1 | 2 |
| `ori_mmlu-conceptual_physics` | 2 | 0 | 1 | 1 |
| `ori_mmlu-formal_logic` | 2 | 0 | 1 | 1 |
| `scibench-diff` | 1 | 0 | 1 | 0 |
| `theoremQA-Finance` | 1 | 0 | 1 | 0 |
| `theoremQA-Physics` | 1 | 0 | 1 | 0 |

其中最值得优先检查的是 `stemez-TransportPhenomena`，5 个样本就有 3 个 regression。

### 5.4 Detailed regression list

| id | subject | src | candidate_count | soft_class_count | correction_attempts |
|---|---|---|---:|---:|---:|
| `mmlu_pro:11646` | engineering | `stemez-TransportPhenomena` | 2 | 2 | 1 |
| `mmlu_pro:10739` | computer science | `theoremQA-EECS` | 9 | 6 | 0 |
| `mmlu_pro:11340` | engineering | `stemez-TransportPhenomena` | 5 | 5 | 0 |
| `mmlu_pro:10780` | philosophy | `ori_mmlu-formal_logic` | 9 | 3 | 0 |
| `mmlu_pro:9841` | physics | `theoremQA-Physics` | 11 | 6 | 0 |
| `mmlu_pro:9738` | physics | `ori_mmlu-conceptual_physics` | 4 | 4 | 0 |
| `mmlu_pro:11957` | engineering | `stemez-TransportPhenomena` | 3 | 3 | 0 |
| `mmlu_pro:636` | business | `theoremQA-Finance` | 5 | 4 | 1 |
| `mmlu_pro:8638` | math | `scibench-diff` | 3 | 3 | 1 |
| `mmlu_pro:11444` | engineering | `stemez-Thermodynamics` | 3 | 3 | 1 |
| `mmlu_pro:8523` | math | `theoremQA-Math` | 11 | 5 | 0 |
| `mmlu_pro:5715` | other | `ori_mmlu-professional_accounting` | 2 | 2 | 0 |
| `mmlu_pro:4113` | chemistry | `stemez-Chemistry` | 4 | 4 | 0 |

唯一 positive case：

| id | subject | src | candidate_count | soft_class_count | correction_attempts |
|---|---|---|---:|---:|---:|
| `mmlu_pro:4304` | chemistry | `stemez-Chemistry` | 3 | 3 | 0 |

## 6. Training Dynamics

### 6.1 Graph execution dynamics

整体平均 active ratio：

| metric | turn1 | turn2 | turn3 | turn4 | turn5 |
|---|---:|---:|---:|---:|---:|
| active_edge_ratio_by_turn | 0.9574 | 0.9503 | 0.9506 | 0.9273 | 0.9302 |
| active_node_ratio_by_turn | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

分 outcome 看 edge ratio：

| outcome | turn1 | turn2 | turn3 | turn4 | turn5 |
|---|---:|---:|---:|---:|---:|
| same | 0.9592 | 0.9525 | 0.9529 | 0.9294 | 0.9332 |
| worse | 0.9250 | 0.9096 | 0.9096 | 0.8876 | 0.8766 |
| better | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

解释：

1. 当前 node 基本没有被裁掉，5 轮里始终是全开
2. edge 虽然有 prune，但整体仍然偏密
3. regression case 的 active edge 比 same case 更低一些，但仍然不构成“真正稀疏搜索”

### 6.2 Correction branch activity

当前 correction 机制几乎没有形成有效正反馈：

- 总 correction attempts: `88`
- 总 correction accepts: `1`
- 总 correction improvements: `1`
- `worse` 样本中的 `correction_accept_count` 平均值为 `0.0`
- `recovery_subgraph_size` 在当前聚合统计里始终为 `0.0`

这说明当前负例主要不是“错误修补失败”，而是“override 本身过于激进或校准失真”。

### 6.3 Learner state

| worker | LMPO mean reward | LMPO baseline | LMPO updates | utility steps | delta steps | halt steps | selector steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| `worker_0` | 0.6947 | 0.7159 | 75 | 252 | 23 | 75 | 0 |
| `worker_1` | 0.7302 | 0.6525 | 75 | 219 | 16 | 75 | 0 |
| `worker_2` | 0.6947 | 0.6654 | 75 | 267 | 19 | 75 | 0 |

补充说明：

1. 当前真正发生学习更新的是 `phase3a_utility_model / delta_model / halt_model`
2. `selector_model.steps = 0`，说明旧 selector 路线并未参与有效学习
3. 从当前结果看，phase3a 的学习已经足以改变少量决策，但方向还没有被校准到“宁可少翻，不要错翻”

## 7. Key Findings

### Finding 1

Observation：

- 当前 225 个 train 样本中，`better = 1`，`worse = 13`
- `stage1_success_avg = 0.7733`
- `stage2_success_avg = 0.7200`
- `delta success = -0.0533`

Interpretation：

- 当前 `stage2_ucc` 在 MMLU-Pro train 上不是“尚未显著提升”，而是已经出现可观测负迁移

Implication：

- 如果继续按当前 override 逻辑放大训练，模型更可能学到“敢翻案”，而不是“会翻对”

Next step：

- 在进入更深训练前，优先加 conservative override gate 或直接先关掉 MMLU 上的 aggressive override

### Finding 2

Observation：

- `override_frontier` 23 次里有 13 次回退，只有 1 次提升
- 发生变化的样本中，92.86% 是坏变化

Interpretation：

- 问题核心不在 stage2 生成了多少候选，而在 override 校准严重失真

Implication：

- 当前需要优先修“选谁覆盖 anchor”，而不是继续扩 candidate pool

Next step：

- 对 `override_frontier` 单独做 calibration study
- 比较 `candidate_count / class_count / reviewer signal / residual` 与最终正确性的关系

### Finding 3

Observation：

- 总 correction accepts 只有 1 次
- `recovery_subgraph_size = 0.0`
- regression 样本里 correction 没有任何成功挽救

Interpretation：

- 当前 MMLU 路线上，phase3a 的主要行为不是“局部修正”，而是“直接替换答案”

Implication：

- 这条线暂时不具备“先修后翻”的保护机制

Next step：

- 要么先让 MMLU 只走 `anchor_wins + preserve guard`
- 要么给 override 增加更强的 verifier-conditioned 审核和 abstain 通路

### Finding 4

Observation：

- regression 在 `engineering / math / physics` 更集中
- `stemez-TransportPhenomena` 5 个样本就出现 3 个 regression

Interpretation：

- 当前风险不是完全随机噪声，而是某些需要精确推导或专业计算的题型更容易被错翻

Implication：

- MMLU-Pro 里的理工题可能要求比当前更保守的覆盖条件

Next step：

- 按 `src` 和 `metadata.category` 做 per-domain override threshold
- 先从 `engineering / physics / theoremQA-like` 子集做对照实验

## 8. Recommended Next Experiments

1. 做一个 `no_override_on_mcq` 或 `very_conservative_override` 的对照组，确认负面是否几乎全部消失。
2. 单独抽取这 23 个 `override_frontier` 样本，做一个小型 calibration set，检查哪些信号在错误翻案前给出了虚高置信。
3. 对 `engineering / math / physics` 子集单独评估，确认是否需要 task-family-specific guard。
4. 在输出里补 `suite_report.json` 或中间 `report.json`，避免训练中途只能从 checkpoint 反推汇总。
5. 为每个 override case 保存 replay bundle，尤其要记录最终 why-override 的可解释信号。

## 9. Bottom Line

截至当前进度，`stage2_ucc` 在这轮 `MMLU-Pro` 训练中表现为**明确负向**：

- 整体 success 从 `0.7733` 降到 `0.7200`
- 整体 task score 从 `0.7300` 降到 `0.6900`
- 负面几乎完全由 `override_frontier` 驱动
- correction 分支基本没有形成保护作用

因此当前最合理的判断不是“stage2 还没学起来”，而是：

**它已经学到会在少量 case 上覆盖 stage1，但覆盖质量明显不够，当前策略对 MMLU-Pro 更像是有害 overturn，而不是有效纠错。**
