# MAS TreeSearch 与 MAS GFlowOpt 训练状态对比分析（2026-03-13）

## 1. 分析范围与数据源

本报告基于以下实际运行产物与代码实现：

- 日志：
  - `projects/R-HAN/logs/mas_gflowopt_dataset_train_full_run_20260313_104553.log`
  - `projects/R-HAN/logs/mas_treesearch_dataset_suite_full_run_20260312_153100.log`
- 结构化结果：
  - `projects/R-HAN/outputs/mas_treesearch_dataset_suite/suite_report.json`
  - `projects/R-HAN/outputs/mas_gflowopt_dataset_train/20260313_104555/run_config.json`
- 关键实现：
  - `demo_train_datasets.py`
  - `demo_train.py`
  - `demo_dataset_suite.py`
  - `mas_gflowopt/`
  - `mas_treesearch/`

本报告回答三件事：

1. 树搜索与 GFlow 两种方式目前收敛到什么程度、各数据集表现如何。
2. 两种方式当前实际训练架构是什么，每一步做什么，参数如何设置。
3. 两者目前是否可以直接横向比较，哪些结论可以下，哪些不能下。

## 2. 先给结论

### 2.1 当前运行完成度

`mas_treesearch` 当前这轮运行已经完成 5 个数据集，并在 `mmlu_pro` 上训练到 `880/910`：

- 已完成并有完整 `train + periodic_test + final_test` 结果：
  - `cqa`
  - `gaia`
  - `gsm8k`
  - `knowledge_crosswords`
  - `mmlu`
- 仍在日志中进行中：
  - `mmlu_pro`，最新日志快照到 `880/910`

`mas_gflowopt` 当前这轮运行只跑到了第一个数据集 `cqa`：

- 最新原始日志已开始处理 `841/1000`
- 但最新一条完整聚合统计还停在 `810/1000`
- 由于 `--eval-every 1000`，当前还没有任何 periodic test
- 因为 `cqa` 还未跑完，所以目前也没有 final test 结果

### 2.2 当前可下的核心判断

1. `mas_treesearch` 的训练已经呈现出明确的数据集分化：
   - `gsm8k` 最强，final test task/success 都在 `0.9222`
   - `mmlu` 次强，final test task `0.7189`
   - `cqa` 中等，final test task `0.5444`
   - `knowledge_crosswords` 偏弱，final test task `0.3000`
   - `gaia` 几乎没有学到有效策略，final test task 仅 `0.0476`
2. `mas_gflowopt` 当前在 `cqa` 上的训练已经表现出明显平台期：
   - 100 之后曾短暂到过 `score=0.5371`
   - 280 左右是较稳定高点，`acc=0.5286 / score=0.5302`
   - 400 到 810 长期徘徊在 `score≈0.512~0.523`
   - 最新聚合点 `810/1000` 为 `acc=0.5074 / score=0.5122`
3. 从墙钟时间看，当前 `mas_gflowopt` 在 `cqa` 上显著慢于 `mas_treesearch`：
   - `gflow` 在 `800` 条时约 `34.47 s/item`
   - `treesearch` 在 `800` 条时约 `19.49 s/item`
   - 同一数量级下，`gflow` 大约慢 `1.7x ~ 2.0x`
4. 这两轮运行不能把“准确率/成功率”当成严格同口径指标直接比较：
   - `treesearch` 当前训练与测试都带 gold answer，评测器会优先走基于参考答案的确定性打分
   - `gflow` 当前训练脚本没有把答案传给 evaluator，只能靠 LLM judge 自评
   - 因此 `treesearch` 的 `success/task_score` 更接近真实数据集指标，`gflow` 的 `acc/score` 更像 judge-based proxy

## 3. 两种方法当前运行配置

### 3.1 `mas_gflowopt` 当前运行参数

当前运行由 `run_mas_gflowopt_dataset_train_nohup.sh` 启动，实际命令是：

```bash
python demo_train_datasets.py \
  --all-datasets \
  --max-train-per-dataset 1000 \
  --test-ratio 0.1 \
  --agent-top-k 6 \
  --gflownet-train-epochs 1 \
  --gflownet-batch-size 1 \
  --num-sampled-dags 1 \
  --contribution-mode none \
  --one-eval-per-trajectory \
  --disable-refine \
  --batch-eval \
  --batch-eval-workers 12 \
  --batch-eval-size 24 \
  --eval-every 1000 \
  --periodic-test-size 100 \
  --embedding-api-base http://localhost:8018 \
  --embedding-model /mnt/nvme/Qwen3-Embedding-8B
```

从 `run_config.json` 可确认的关键参数：

| 参数 | 当前值 |
| --- | --- |
| 数据根目录 | `/mnt/nvme/projects/R-HAN/dataset/mas_gflowopt_processed` |
| 数据集选择 | `--all-datasets` |
| 每数据集训练上限 | `1000` |
| 测试比例 | `0.1` |
| agent top-k | `6` |
| GFlow 训练 epoch | `1` |
| GFlow batch size | `1` |
| run 阶段采样 DAG 数 | `1` |
| contribution mode | `none` |
| true eval 频率 | 被 `--one-eval-per-trajectory` 覆盖为“只在终态做一次” |
| refine | 关闭 |
| periodic test 触发 | 每 `1000` 训练样本一次 |
| periodic test 大小 | `100` |
| embedding API | `http://localhost:8018` |
| embedding 模型 | `/mnt/nvme/Qwen3-Embedding-8B` |

LLM evaluator 的服务地址没有写入 `run_config.json`，它由 `demo_train.py` 的 `build_llm_evaluator()` 从环境变量读取；如果环境变量未设置，则默认：

- `LLM_API_BASE=http://localhost:8039`
- `LLM_MAX_TOKENS=768`
- `LLM_JUDGE_MAX_TOKENS=256`

### 3.2 `mas_treesearch` 当前运行参数

这轮树搜索运行的“原始启动命令”日志里没有完整保存，但可以根据 `suite_report.json` 精确重建等效配置：

```bash
LLM_API_BASE=http://127.0.0.1:8045 \
EMBED_API_BASE=http://127.0.0.1:8021 \
LLM_MAX_TOKENS=512 \
LLM_JUDGE_MAX_TOKENS=128 \
python demo_dataset_suite.py \
  --all-datasets \
  --max-pool-size 1000 \
  --split-seed 7 \
  --log-every 10 \
  --periodic-eval-every 100 \
  --periodic-eval-size 10 \
  --search-iterations 4 \
  --candidate-core-k 4 \
  --candidate-explore-k 2 \
  --candidate-max-k 6 \
  --tier1-max-tokens 192 \
  --tier2-max-tokens 512 \
  --tier1-repeats 1 \
  --tier2-repeats 1
```

从 `suite_report.json` 读取到的当前有效配置：

| 参数 | 当前值 |
| --- | --- |
| 数据根目录 | `/mnt/nvme/projects/R-HAN/dataset/mas_treesearch_processed` |
| 输出根目录 | `/mnt/nvme/projects/R-HAN/outputs/mas_treesearch_dataset_suite` |
| max pool size | `1000` |
| split seed | `7` |
| periodic eval every | `100` |
| periodic eval size | `10` |
| search iterations | `4` |
| candidate core / explore / max | `4 / 2 / 6` |
| tier1 top fraction | `0.25` |
| tier2 top fraction | `0.20` |
| final top k | `3` |
| tier1 repeats | `1` |
| tier2 repeats | `1` |
| learned edit prior | 开启 |
| learned value model | 开启 |
| LLM API | `http://127.0.0.1:8045` |
| Embedding API | `http://127.0.0.1:8021` |
| Embedding 模型 | `/mnt/nvme/Qwen3-Embedding-8B` |

## 4. 重要口径差异：为什么现在不能把两边指标硬放一起

这是本报告里最重要的 caveat。

### 4.1 训练/测试切分协议不一样

`mas_gflowopt` 当前脚本的切分方式：

- 只从 processed dataset 的 `train` split 里取前 `1000` 条训练
- 只从 `test` split 里取前 `100` 条测试
- 没有把 `validation` 混入训练池

`mas_treesearch` 当前脚本的切分方式：

- 把 `train + validation + test` 全部合并成一个 pool
- 再随机打乱
- 截前 `1000` 条作为 sample pool
- 再按 `910/90` 切成 train/test

所以即便数据集名字都叫 `cqa`，两边当前看到的样本集合也不是同一批。

### 4.2 评测口径不一样

`mas_treesearch` 当前训练/测试时，会把 `reference_answer=item.get("answer")` 传进 evaluator。对于当前这些数据集：

- 选择题会走确定性的 option 对比
- 数值题会走数值对比
- 一般文本题会走 exact/numeric matching
- 只有缺少参考答案时才会退回 LLM judge

因此 `treesearch` 的 `task_score/success` 基本是“带标准答案”的指标。

`mas_gflowopt` 当前训练脚本中，`load_questions()` 只保留：

- `id`
- `question`
- `category`
- `source_dataset`

答案并没有传给 evaluator，所以当前 `gflow` 的 `task_score/success/acc` 来自 LLM judge 自评，不是 gold-based exact metric。

### 4.3 reward 定义完全不是同一个量纲

`mas_treesearch` reward 是线性组合：

- `task_score`
- `success`
- `token_cost`
- `latency`
- `safety_penalty`
- `size_penalty`
- `prompt_penalty`

所以它可以出现负值，也可以大于 `1`。

`mas_gflowopt` reward 则是：

- 先拼 `total_score`
- 再做 `exp(reward_temperature * total_score)`

因此它始终是正数，而且量纲与树搜索 reward 不可直接比较。

结论：当前可以比较的是“训练是否稳定”“是否平台”“墙钟速度”“大致趋势”，不能把两边的 `reward` 和 `acc` 视为同一指标。

## 5. `mas_treesearch` 当前训练表现与收敛情况

### 5.1 已完成数据集总览

| 数据集 | 样本协议 | train avg task | periodic avg task | final avg task | final avg success | final avg reward | final avg latency | final avg safety |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `gsm8k` | 1000 sampled -> 910/90 | 0.9066 | 0.9222 | 0.9222 | 0.9222 | 1.2479 | 0.4119 | 0.0467 |
| `mmlu` | 1000 sampled -> 910/90 | 0.7258 | 0.7189 | 0.7189 | 0.7111 | 0.9020 | 0.8756 | 0.1939 |
| `cqa` | 1000 sampled -> 910/90 | 0.5740 | 0.5444 | 0.5444 | 0.4889 | 0.5801 | 1.3021 | 0.3272 |
| `knowledge_crosswords` | 1000 sampled -> 910/90 | 0.3758 | 0.3111 | 0.3000 | 0.3000 | 0.2396 | 0.8621 | 0.4200 |
| `gaia` | 466 all -> 424/42 | 0.0283 | 0.0250 | 0.0476 | 0.0476 | -0.1651 | 0.8314 | 0.5714 |

已完成 5 个数据集上的总体统计：

- completed 数据集宏平均：
  - train avg task = `0.5221`
  - final avg task = `0.5066`
  - train avg reward = `0.5658`
  - final avg reward = `0.5609`
- completed 数据集按样本数加权平均：
  - train avg task = `0.5812`
  - final avg task = `0.5614`
  - final avg success = `0.5473`
  - final avg latency = `0.8596`
  - final avg safety = `0.2808`

### 5.2 各数据集收敛细节

#### `cqa`

| 阶段 | count | avg task | avg success | avg reward | avg s/item |
| --- | ---: | ---: | ---: | ---: | ---: |
| train 初始 | 10 | 0.4800 | 0.4000 | 0.4007 | 23.78 |
| train 峰值 | 120 | 0.6208 | 0.5833 | 0.6981 | 20.77 |
| train 最终 | 910 | 0.5740 | 0.5209 | 0.6090 | 19.38 |
| periodic 峰值 | 20 | 0.7425 | 0.7500 | 0.9380 | - |
| periodic 最终 | 90 | 0.5444 | 0.4889 | 0.5685 | - |
| final test | 90 | 0.5444 | 0.4889 | 0.5801 | 14.01 |

判断：

- 前 120 条上升很快，之后从峰值回落。
- 200 之后整体进入平稳区，最终训练均值大致稳定在 `0.57~0.59`。
- test 指标最终落在 `0.54` 左右，说明存在一定训练内乐观偏差，但不是完全崩塌。

#### `gaia`

| 阶段 | count | avg task | avg success | avg reward |
| --- | ---: | ---: | ---: | ---: |
| train 初始 | 10 | 0.1000 | 0.1000 | -0.0988 |
| train 峰值 | 10 | 0.1000 | 0.1000 | -0.0988 |
| train 最终 | 424 | 0.0283 | 0.0283 | -0.2220 |
| periodic 最终 | 40 | 0.0250 | 0.0250 | -0.2418 |
| final test | 42 | 0.0476 | 0.0476 | -0.1651 |

判断：

- 这是当前树搜索里最差的数据集。
- 从一开始的 `0.10` 很快滑到接近 `0.03`，说明现有模板空间和代理信号对 `gaia` 明显不适配。
- final test 略高于 periodic，但仍然非常低，不能视为有效收敛。

#### `gsm8k`

| 阶段 | count | avg task | avg success | avg reward |
| --- | ---: | ---: | ---: | ---: |
| train 初始 | 10 | 0.8000 | 0.8000 | 1.0516 |
| train 峰值 | 90 | 0.9556 | 0.9556 | 1.2988 |
| train 最终 | 910 | 0.9066 | 0.9066 | 1.2160 |
| periodic 峰值 | 60 | 0.9667 | 0.9667 | 1.3096 |
| periodic 最终 | 90 | 0.9222 | 0.9222 | 1.2370 |
| final test | 90 | 0.9222 | 0.9222 | 1.2479 |

判断：

- 这是当前树搜索最成功的数据集。
- 前 100 条内已经达到很高水平，后续保持高位小幅波动。
- final test 与 periodic 基本重合，泛化稳定。

#### `knowledge_crosswords`

| 阶段 | count | avg task | avg success | avg reward |
| --- | ---: | ---: | ---: | ---: |
| train 初始 | 10 | 0.6000 | 0.6000 | 0.6610 |
| train 峰值 | 10 | 0.6000 | 0.6000 | 0.6610 |
| train 最终 | 910 | 0.3758 | 0.3758 | 0.3237 |
| periodic 最终 | 90 | 0.3111 | 0.3111 | 0.2136 |
| final test | 90 | 0.3000 | 0.3000 | 0.2396 |

判断：

- 初期表现看起来不错，但之后持续回落。
- 最终训练、periodic、final test 都明显低于初始阶段。
- 这说明当前搜索策略在该任务上出现了负迁移或被 proxy/prior 带偏。

#### `mmlu`

| 阶段 | count | avg task | avg success | avg reward |
| --- | ---: | ---: | ---: | ---: |
| train 初始 | 10 | 0.8250 | 0.9000 | 1.1065 |
| train 峰值 | 10 | 0.8250 | 0.9000 | 1.1065 |
| train 最终 | 910 | 0.7258 | 0.7231 | 0.9023 |
| periodic 峰值 | 30 | 0.7833 | 0.8000 | 1.0100 |
| periodic 最终 | 90 | 0.7189 | 0.7111 | 0.8946 |
| final test | 90 | 0.7189 | 0.7111 | 0.9020 |

判断：

- 初始阶段就很强，后续回落并进入稳定平台。
- 最终 train/test 较为接近，说明虽然峰值没保持住，但最终解法是稳定的。

### 5.3 `mmlu_pro` 进行中快照

截至当前日志末尾，`mmlu_pro` 训练到 `880/910`，尚无 final test。

| 阶段 | count | avg task | avg success | avg reward |
| --- | ---: | ---: | ---: | ---: |
| train 初始 | 10 | 0.5250 | 0.5000 | 0.5429 |
| train 峰值 | 870 | 0.6828 | 0.7103 | 0.8526 |
| train 最新 | 880 | 0.6827 | 0.7102 | 0.8526 |
| periodic 初始 | 10 | 0.7500 | 0.8000 | 0.9716 |
| periodic 最新 | 80 | 0.6469 | 0.6625 | 0.7795 |

判断：

- 训练集上还在持续增益，到 `870/880` 基本已经接近平台。
- periodic test 从最早的 `0.75` 下滑到 `0.6469`，说明泛化比 train 弱一些。
- 目前看大概率会收敛在 `0.65` 左右的 test task 区间，但这只是根据当前日志做的趋势推断。

## 6. `mas_gflowopt` 当前训练表现与收敛情况

### 6.1 当前完成度

当前 `mas_gflowopt` 只在 `cqa` 上训练，还没进入：

- periodic test
- final test
- 第二个数据集

截至本次读取日志时：

- 最新原始进度：已开始 `841/1000`
- 最新一条完整聚合统计：`810/1000`

因此当前对 `gflow` 的结论只能是：

- `cqa` 训练中途收敛情况
- 不能给出跨数据集表现
- 不能给出真正的 final test 泛化结论

### 6.2 `cqa` 当前关键收敛节点

| n | score | acc | avg_reward | avg_s/item | 说明 |
| ---: | ---: | ---: | ---: | ---: | --- |
| 10 | 0.7000 | 0.7000 | 0.4058 | 83.16 | 极小样本，不稳定 |
| 20 | 0.6750 | 0.6500 | 0.4058 | 41.58 | 仍然偏高估 |
| 50 | 0.5260 | 0.5000 | 0.4059 | 49.55 | 很快掉回中等区间 |
| 100 | 0.5080 | 0.4800 | 0.4045 | 42.23 | 基本贴近 0.5 水平 |
| 140 | 0.5371 | 0.5214 | 0.4054 | 35.50 | 100 之后的最高 score |
| 200 | 0.5225 | 0.5150 | 0.4048 | 37.02 | 回落 |
| 280 | 0.5302 | 0.5286 | 0.4053 | 35.49 | 100 之后的最高 acc |
| 400 | 0.5211 | 0.5175 | 0.4057 | 35.01 | 进入平台 |
| 600 | 0.5228 | 0.5183 | 0.4057 | 33.91 | 平台继续 |
| 700 | 0.5184 | 0.5143 | 0.4050 | 34.95 | 平台下沿 |
| 800 | 0.5154 | 0.5112 | 0.4046 | 34.47 | 基本无提升 |
| 810 | 0.5122 | 0.5074 | 0.4046 | 34.04 | 当前最新聚合点 |

可以明确看到三个阶段：

1. `1~50` 条：从异常高的早期统计迅速回落。
2. `100~280` 条：有过两次短暂上冲，但没有持续。
3. `400~810` 条：长时间卡在 `0.51~0.52` 的平台附近。

### 6.3 当前 loss 与窗口表现

按日志里的 10 条滑窗统计：

| 指标 | 最佳 / 最新 |
| --- | --- |
| 最低 total_loss | `1.0287`，出现在 `530/1000` |
| 100 之后最高窗口 acc | `0.8000`，出现在 `200/1000` 的最近 10 条窗口 |
| 最新窗口 task_score | `0.2000` |
| 最新窗口 acc | `0.2000` |
| 最新窗口 total_loss | `1.8723` |

这说明：

- 训练 loss 曾经下降到较低位置
- 但最近 10 条样本的质量已经明显恶化
- loss 下降并没有带来持续的 judge-based performance 改善

### 6.4 当前 reward 几乎不动

这是当前 `gflow` 最值得注意的现象：

- 从 `100` 到 `810`，`avg_reward` 基本一直在 `0.4045 ~ 0.4063` 之间小幅抖动
- 但同时 `score/acc` 在 `0.48 ~ 0.53` 区间有明显波动

这表明当前 reward 对任务质量的分辨率非常差。

从代码实现看，这不是偶然现象，而是当前默认配置导致的：

- `MASConfig.use_task_score_as_bic = True`
- `MASRewardModel.score_and_reward()` 在该配置下会把 `task_utility` 塞进 `bic_score`
- 然后再做 `bic_term = tanh(bic_score / 1000.0)`
- 同时 `task` 项权重被直接置为 `0`

也就是说，在当前默认实现里，真实 evaluator 产生的任务效用并没有直接作为主要 reward 项进入优化目标，而是先被强烈压缩后才进入 `bic_term`。这与日志里 `avg_reward` 长期近似常数的现象是吻合的。

这也是为什么当前 `gflow` 日志里“reward 很稳”，但“score/acc 并不稳”的关键原因。

## 7. `cqa` 上两种方法的当前对照

再次强调：下面只适合做趋势级对照，不适合当严格公平 benchmark。

### 7.1 同训练条数下的速度对比

| n | gflow score | tree score | tree - gflow | gflow avg_s/item | tree avg_s/item | gflow / tree |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | 0.5260 | 0.5480 | +0.0220 | 49.55 | 22.71 | 2.18x |
| 100 | 0.5080 | 0.5975 | +0.0895 | 42.23 | 20.62 | 2.05x |
| 200 | 0.5225 | 0.5853 | +0.0628 | 37.02 | 20.18 | 1.83x |
| 280 | 0.5302 | 0.5963 | +0.0661 | 35.49 | 20.95 | 1.69x |
| 400 | 0.5211 | 0.5804 | +0.0593 | 35.01 | 20.86 | 1.68x |
| 600 | 0.5228 | 0.5903 | +0.0675 | 33.91 | 19.98 | 1.70x |
| 700 | 0.5184 | 0.5833 | +0.0649 | 34.95 | 19.82 | 1.76x |
| 800 | 0.5154 | 0.5779 | +0.0625 | 34.47 | 19.49 | 1.77x |

趋势上看：

- 在最初的 10~20 条里，`gflow` 因为样本太小，会出现“看起来更高”的虚高现象。
- 从 50 条以后，`treesearch` 的 `cqa` 训练统计一直更高。
- 速度上 `treesearch` 从头到尾都更快。

### 7.2 当前阶段谁更接近“已收敛”

`treesearch` 的 `cqa`：

- 已经跑完 train、periodic test、final test
- 最终稳定落在 `final task=0.5444`
- 可以说已经基本收敛

`gflow` 的 `cqa`：

- 还没跑完 train
- 没有 periodic test
- 没有 final test
- 但从 `400~810` 的聚合曲线看，已经很像“提前平台化”

所以当前更准确的说法是：

- `treesearch/cqa`：已完成且已收敛
- `gflow/cqa`：尚未完成，但从训练曲线看已经基本平台化

## 8. 两种方法当前实际架构：逐步拆解

### 8.1 `mas_gflowopt` 当前实际执行链路

### 数据级流程

对每个数据集，`demo_train_datasets.py` 做的是：

1. 从 `mas_gflowopt_processed/<dataset>/train.jsonl` 取前 `1000` 条作为 train。
2. 从 `.../test.jsonl` 取前 `100` 条作为 test。
3. 为该数据集新建一个 `MASGFlowPipeline`。
4. 逐条执行 `pipeline.train_and_run(...)`。
5. 数据集训练结束后，再对 test 样本执行 `run_test(...)`。

当前运行只完成了上面第 4 步中的 `cqa` 部分，还没到第 5 步。

### 单样本流程

每条训练样本执行的是 `pipeline.train_and_run(...)`，其内部实际是：

1. `_prepare_context()`
2. `_train_prepared()`
3. `_run_prepared()`

#### 第一步：`_prepare_context()`

这里做四件事：

1. 取全量 agent pool，并将 agent 描述向量化。
2. 对 question 做 embedding。
3. 按 question 相关性与多样性做 subset selection。
4. 只保留被选中的 top-k agent。

当前运行中的关键事实：

- gflow agent pool 总数是 `12`
- 当前 `agent_top_k=6`
- 也就是说每条样本先从 12 个 agent 里选 6 个
- 当前代码已经做了保守优化：
  - 全量 agent embedding 在 pipeline 生命周期内缓存一次
  - `train_and_run()` 复用同一份 prepared context，不再对同一条样本重复做 agent vectorize 与 question encode

#### 第二步：`_train_prepared()`

这里调用 `GFlowNetSampler.train()`，当前配置下等价于：

- `epochs = 1`
- `batch_size = 1`
- 只采样 1 条 trajectory
- action space 为 `edge_add`
- `gflownet_max_steps = 24`
- `allow_backtracking = False`

当前 reward/eval 触发方式：

- `--one-eval-per-trajectory` 会把
  - `true_eval_interval = 0`
  - `true_eval_budget_per_trajectory = 0`
  - `true_eval_terminal_always = True`
- 结果就是：轨迹中间不做 true eval，只在终态做一次 true eval

训练损失：

- `DB loss`
- `Contrastive loss`
- 总损失 = `1.0 * DB + 0.1 * CL`

当前 run 因为 `epochs=1`，所以 early stop 参数几乎没有实际作用。

#### 第三步：`_run_prepared()`

这里调用 `GFlowNetSampler.sample_batch()` 和 `ContinuousDiscreteOptimizer.optimize()`。

当前配置下：

- `num_sampled_dags = 1`
- 所以 run 阶段只会再采 1 个 DAG

后续 optimizer 的实际行为：

1. 用这 1 个样本训练 proxy MLP，默认 `proxy_train_epochs=80`
2. 在 latent space 上做 k-means
3. 从 cluster center 做 gradient ascent，默认 `gradient_steps=25`
4. 找回最近的离散 DAG
5. 因为 `--disable-refine`，离散 hill-climb refine 被跳过

在当前配置下，这里有两个结构性事实：

- `num_sampled_dags=1` 时，k-means 实际会退化成 `k=1`
- 也就是说当前 run 阶段仍然会做 proxy 训练、cluster、gradient ascent，但搜索空间其实已经极度退化

这部分不会额外增加 LLM 调用，但会增加不少本地计算时间。

### 当前 `gflow` 的 evaluator 调用方式

当前配置下，经过前面已经做的保守优化后，每条训练样本主链路里：

- `train()` 终态 true eval 1 次
- `run()` 终态 true eval 1 次
- `run_train()` 记录日志时复用了 `run()` 的 cached eval，不再做第三次 evaluator

所以当前每条训练样本是：

- `2` 次 evaluator

而当前 evaluator 是 `LLMExecutionMASTaskEvaluator`：

1. 对 DAG 中每个 active agent 顺序调用一次 LLM
2. 再调用一次 judge

当前 active agent 数固定为 `6`，因此每次 evaluator 约等于：

- `6` 次 agent chat
- `1` 次 judge chat
- 合计 `7` 次 chat completion

所以当前 `gflow` 每条训练样本大约是：

- `2 * 7 = 14` 次 chat completion

另外还有 embedding：

- 每条样本 1 次 question embedding
- 每个数据集开始时 12 个 agent embedding 一次性缓存

### 当前 `gflow` 架构里最重要的两个问题

#### 问题一：reward 对 task 质量不敏感

这在第 6.4 节已经展开。当前默认配置下，真实任务效用没有直接作为 reward 主项进入优化目标，导致：

- `avg_reward` 基本不动
- `score/acc` 波动却比较明显

这会直接削弱 GFlowNet 的训练信号。

#### 问题二：run 阶段的“优化器”在当前参数下退化

当前组合是：

- `num_sampled_dags=1`
- `disable_refine=True`

这意味着：

- proxy 只在一个点上训练
- cluster 实际退化成单簇
- top optimized representations 失去“多样候选”意义
- 离散精修又被关闭

所以当前 `run()` 阶段从算法上已经非常接近“对单个样本做一层形式上的连续优化包装”，而不是有效的结构搜索。

### 8.2 `mas_treesearch` 当前实际执行链路

### 数据级流程

`demo_dataset_suite.py` 对每个数据集做的是：

1. 读 `train/validation/test` 三个 split。
2. 合并成统一 pool。
3. 随机打乱后截前 `1000` 条。
4. 再切成 `910` train 和 `90` test。
5. 逐条调用 `pipeline.search(..., learn=True)` 做训练。
6. 每训练 `100` 条，在 test 集上抽 `10` 条做 periodic eval，`learn=False`。
7. 数据集训练完后，对全部 test 做 final test，`learn=False`。
8. 保存 dataset report 与 checkpoint。

### 单样本流程

`TreeSearchMASPipeline.search()` 的步骤是：

1. `TaskConditioner.select(question_text)`
2. `TreeSearchEngine.search(...)`

#### 第一步：TaskConditioner 选 agent

当前树搜索 agent pool 总数是 `10`，流程是：

1. 对所有 agent 文本做 embedding，初始化时缓存。
2. 对 question 做 embedding。
3. 按 relevance 先选 `candidate_core_k=4` 个核心 agent。
4. 再补 `candidate_explore_k=2` 个探索 agent。
5. 最终 candidate agent 上限 `candidate_max_k=6`。

也就是每条样本通常会得到一个最多 6 个 agent 的候选池。

#### 第二步：RootTemplateBuilder 生成初始架构

当前 root templates 是：

- `direct`
- `solve_verify`
- `parallel_vote`
- `critique_revise`

每个模板会按 role capability 偏好，从候选 agent 池里指派具体 agent。

模板空间定义如下：

| 模板 | 角色数 | 拓扑 |
| --- | ---: | --- |
| `direct` | 1 | `solver` |
| `solve_verify` | 2 | `solver -> verifier` |
| `parallel_vote` | 3 | `solver_a -> aggregator`, `solver_b -> aggregator` |
| `critique_revise` | 3 | `generator -> critic`, `generator -> reviser`, `critic -> reviser` |
| `debate_judge` | 3 | `debater_a -> judge`, `debater_b -> judge` |
| `route_solve` | 3 | `router -> solver -> aggregator` |

虽然 root 只从前 4 个模板起步，但搜索过程中允许编辑到全部模板。

#### 第三步：TreeSearchEngine 搜索

当前搜索配置：

- `search_iterations = 4`
- `top_k_selection = 5`
- `root_restart_prob = 0.2`
- `puct_c = 1.25`
- progressive widening:
  - `base = 2`
  - `alpha = 1.5`

每次迭代做的事情：

1. 在已有节点里选 parent：
   - 有 `20%` 概率从 root 重启
   - 否则按 `q_mean + puct bonus + proxy prior` 选
2. 枚举编辑动作：
   - `change_template`
   - `swap_agent`
   - `set_prompt_slot`
   - `stop`
3. 用 learned edit prior 对动作排序。
4. 按 progressive widening 限制展开若干 child。
5. 对 child 先做 tier-0 proxy 打分。

#### 第四步：多保真评估

当前评估是三层：

1. tier-0：`StaticProxyScorer`
   - 不调 LLM
   - 用 alignment/diversity/template prior/verification bonus/size/prompt complexity 给 cheap score
2. tier-1：`MultiFidelityEvaluator.evaluate(..., tier='tier1')`
   - 只对本轮 expanded child 里 top `25%` 做
   - 当前 `tier1.repeats = 1`
   - `tier1.max_tokens = 192`
3. tier-2：`MultiFidelityEvaluator.evaluate(..., tier='tier2')`
   - 只对 tier-1 结果里的 top `20%` 做
   - 当前 `tier2.repeats = 1`
   - `tier2.max_tokens = 512`

因此树搜索不会对所有候选都做昂贵评估，而是逐层筛选。

#### 第五步：在线学习 prior/value

当前两类在线模型都开启：

- `LearnableEditPrior`
- `LearnableValueModel`

更新逻辑：

- 对拿到 tier-2 结果的节点，用其真实 reward 更新 value model
- 如果 child 的 tier-2 reward 明显优于 parent proxy baseline，则把该 edit 记为正样本，更新 edit prior

训练阶段 `learn=True`，测试与 periodic eval 阶段 `learn=False`，避免 test leakage。

### 当前 `treesearch` evaluator 的关键特点

与 `gflow` 最大差异在于：

1. 它执行的是“模板化工作流”，不是 DAG 边搜索。
2. 它有 reference answer 时，优先走确定性评分，而不是 LLM judge。
3. 它有 chat cache 和 eval cache。
4. 当前 tier1/tier2 repeats 都是 `1`，开销已经压得比较低。

这也是为什么当前 `treesearch` 的 wall-clock 明显更好，且指标更稳定。

## 9. 最后的综合判断

### 9.1 当前 `mas_treesearch`

可以认为已经进入“可用但数据集敏感”的阶段：

- 对 `gsm8k` 和 `mmlu` 很有效
- 对 `cqa` 有中等效果
- 对 `knowledge_crosswords` 和 `gaia` 明显不足
- 当前搜索与评估链路是完整的，且已经能稳定产出 final report

### 9.2 当前 `mas_gflowopt`

可以认为仍处在“训练链路能跑，但当前配置的优化目标与搜索阶段都比较退化”的阶段：

- 只跑完了 `cqa` 的 84% 左右训练
- 从训练曲线看已经明显平台化
- 当前 reward 几乎是平的
- 当前 run 阶段在 `num_sampled_dags=1` 下退化很严重
- 现阶段还不足以证明它能在多数据集上稳定收敛

### 9.3 如果只看当前这两轮运行

当前最稳妥的结论是：

1. `mas_treesearch` 明显更成熟，至少已经能完整跑通多数据集并得到可解释的 final test 结果。
2. `mas_gflowopt` 当前这组参数更像是在“极端压缩调用成本后的可运行版本”，不是一个已经调顺的强基线。
3. 如果后续要继续比较两者，建议先统一三件事：
   - 相同的数据切分协议
   - 相同的评测口径（都带 gold，或者都只用 judge）
   - 相同的 LLM / embedding 服务与 token 限额

在这三件事统一之前，当前最可信的对照维度是：

- 完成度
- 训练是否平台化
- 每条样本墙钟时间
- 是否能产出可靠 final test


1.打分函数依赖llm judge太多（修改函数/换模型judge）
2.treesearch研究效果差的原因 修改后增大数据量重跑
3.gflownet提效率

