# R-HAN

当前仓库维护两条 MAS 架构搜索路线：

- 旧路线：`mas_gflowopt/`
  以 GFlowNet + proxy + refine 为主，保留用于对照与历史复现。
- 新路线：`mas_treesearch/`
  以 tree search + 多层评估 + 轻量 prompt 搜索为主，目标是显著减少与真实 LLM 的重复交互成本。

## 当前训练主线

当前实际在维护和运行的两套 dataset-separated 训练入口：

- TreeSearch：`train_mas_treesearch_target_suite.py`
- GFlowOpt：`train_mas_gflowopt_target_suite.py`

两者都直接读取 `dataset/mas_treesearch_target_suite_20260314/` 下的 `train / validation / test`，并按数据集单独训练、单独验证、单独测试，不再走“全数据集混合打乱”的旧流程。

### 1. TreeSearch 当前架构

代码主目录：

- `mas_treesearch/`
- 训练入口：`train_mas_treesearch_target_suite.py`

核心流程：

1. 用 embedding 对问题做任务条件化，先从 agent pool 里选候选 agent 子集。
2. 基于 root template 构造初始 MAS 架构状态。
3. 在离散结构空间里做 tree search：
   - 改 template
   - 换角色对应 agent
   - 改轻量 prompt slot
   - stop
4. 每个候选节点先走 proxy 打分，再走两层真实评估：
   - `tier1`: 便宜的粗筛
   - `tier2`: 更贵的最终评估
5. 训练时在线更新两个轻量学习模块：
   - `LearnableEditPrior`
   - `LearnableValueModel`
6. 最终对每题输出当前最优编排结构。

TreeSearch 的几个关键组件：

- `mas_treesearch/gating.py`: 任务条件化 agent 子集选择
- `mas_treesearch/search.py`: 主 tree search、动作展开、progressive widening、PUCT 风格父节点选择
- `mas_treesearch/evaluator.py`: 多层评估器
- `mas_treesearch/learning.py`: `LearnableEditPrior` / `LearnableValueModel`
- `mas_treesearch/pipeline.py`: 端到端 pipeline 与 checkpoint

当前 target-suite 训练脚本默认参数：

| 参数 | 当前值 |
| --- | --- |
| 数据入口 | `dataset/mas_treesearch_target_suite_20260314` |
| 输出目录 | `outputs/mas_treesearch_target_suite_train_20260314_212033` |
| `seed` | `7` |
| `search_iterations` | `6` |
| `candidate_core_k / explore_k / max_k` | `4 / 2 / 6` |
| `tier1_max_tokens` | `160` |
| `tier2_max_tokens` | `384` |
| `tier1_repeats / tier2_repeats` | `1 / 1` |
| learned edit prior | 开启 |
| learned value model | 开启 |
| `checkpoint_every` | `25` |
| `resume` | 开启时按数据集内进度继续 |

当前在线服务配置以环境变量注入：

- `LLM_API_BASE=http://127.0.0.1:8043`
- `LLM_MODEL=qwen3-8b-train`
- `LLM_JUDGE_API_BASE=http://127.0.0.1:8045`
- `LLM_JUDGE_MODEL=qwen3-32b-judge`
- `EMBED_API_BASE=http://127.0.0.1:8018`
- `EMBED_MODEL=/mnt/nvme/Qwen3-Embedding-8B`
- `LLM_MAX_TOKENS=512`
- `LLM_JUDGE_MAX_TOKENS=128`

说明：

- `SearchConfig` 的完整默认定义在 `mas_treesearch/config.py`
- 训练脚本只覆盖最核心的一组参数；其余仍走 `SearchConfig` / `TieredEvalConfig` 默认值
- dataset profile 会进一步覆盖 root templates、prompt 偏置和少量搜索约束，定义在 `mas_treesearch/profiles.py`

### 2. GFlowOpt 当前架构

代码主目录：

- `mas_gflowopt/`
- 训练入口：`train_mas_gflowopt_target_suite.py`

核心流程：

1. 先用 task conditioning 从 agent pool 中选 top-k agent 子集。
2. 在该子图上用可训练 GFlowNet 采样 DAG 轨迹。
3. 奖励函数综合：
   - task utility
   - BIC/proxy term
   - contribution term
   - question alignment
   - size penalty
4. 训练时优化：
   - detailed-balance loss
   - contrastive loss
5. 采样后再走离散优化器做 refine；当前 target-suite 训练默认关闭 refine。
6. 可学习 gating 会根据训练反馈更新，但当前策略偏保守，优先保证稳定性。

GFlowOpt 的几个关键组件：

- `mas_gflowopt/conditioning.py`: task-conditioned subset selection / gating
- `mas_gflowopt/gflownet.py`: GFlowNet sampler、DB loss、contrastive loss
- `mas_gflowopt/representation.py`: 图表示模型
- `mas_gflowopt/reward.py`: 组合奖励与贡献估计
- `mas_gflowopt/optimizer.py`: refine / 离散后处理
- `mas_gflowopt/pipeline.py`: 端到端 pipeline 与 checkpoint

当前 target-suite 训练脚本默认参数：

| 参数 | 当前值 |
| --- | --- |
| 数据入口 | `dataset/mas_treesearch_target_suite_20260314` |
| 输出目录 | `outputs/mas_gflowopt_target_suite_train_resume_fixidx_20260315_102622` |
| `seed` | `7` |
| `agent_top_k` | `6` |
| `gflownet_train_epochs` | `1` |
| `gflownet_batch_size` | `1` |
| `num_sampled_dags` | `1` |
| `contribution_mode` | `none` |
| `true_eval_interval` | `6`，但当前被 `--one-eval-per-trajectory` 覆盖 |
| `true_eval_budget` | `4`，但当前被 `--one-eval-per-trajectory` 覆盖 |
| `one_eval_per_trajectory` | 开启 |
| `disable_refine` | 开启 |
| `batch_eval_workers` | `12` |
| `early_stop_metric` | `total_loss` |
| `early_stop_patience` | `3` |
| `early_stop_min_delta` | `0.0001` |
| `early_stop_warmup_epochs` | `1` |
| `checkpoint_every` | `25` |

当前在线服务配置：

- `LLM_API_BASE=http://127.0.0.1:8039`
- `LLM_MODEL=qwen3-8b`
- embedding 由脚本参数指定：
  - `--embedding-api-base http://127.0.0.1:8018`
  - `--embedding-model /mnt/nvme/Qwen3-Embedding-8B`

说明：

- `MASConfig` 的完整默认定义在 `mas_gflowopt/types.py`
- target-suite 训练脚本通过 `_build_config()` 把一组实验参数写入 `MASConfig`
- 当前 target-suite 训练显式跳过了 `gsm8k`

### 3. 当前 checkpoint 设计

两套训练现在都支持：

- dataset-level checkpoint
- suite-level progress
- resume 后跳过已完成数据集
- resume 后在数据集内部继续训练

TreeSearch checkpoint 落点：

- suite 进度：`outputs/mas_treesearch_target_suite_train_20260314_212033/suite_progress.json`
- 每个数据集一个 checkpoint：
  - `outputs/mas_treesearch_target_suite_train_20260314_212033/<dataset>/checkpoint.json`

当前正在跑的数据集实例：

- `outputs/mas_treesearch_target_suite_train_20260314_212033/mbpp/checkpoint.json`

TreeSearch checkpoint 内容分两部分：

- `metadata`
  - 当前数据集计划
  - sampled ids
  - 已完成训练条数 `train_index_completed`
  - `validation_round`
  - `train_rows` / `periodic_validation_rows`
  - 统计量与阶段标记
- `pipeline_state`
  - `search_config`
  - `runtime_config`
  - `edit_prior`
  - `value_model`
  - `search_engine` 随机状态

GFlowOpt checkpoint 落点：

- suite 进度：`outputs/mas_gflowopt_target_suite_train_resume_fixidx_20260315_102622/suite_progress.json`
- 每个数据集一个 checkpoint：
  - `outputs/mas_gflowopt_target_suite_train_resume_fixidx_20260315_102622/<dataset>/checkpoint.pt`

当前正在跑的数据集实例：

- `outputs/mas_gflowopt_target_suite_train_resume_fixidx_20260315_102622/multiarith/checkpoint.pt`

GFlowOpt checkpoint 内容也分两部分：

- `metadata`
  - sampled ids
  - `train_index_completed`
  - `validation_round`
  - `train_rows` / `periodic_validation_rows`
  - 统计量与阶段标记
- `pipeline_state`
  - `config`
  - `repr_model`
  - `conditioner`
  - `sampler`
  - `python_random_state`
  - `torch_random_state`

### 4. 当前最重要的参数入口

如果你要改“实验设置”，优先看这几个文件：

- TreeSearch 主训练参数：`train_mas_treesearch_target_suite.py`
- TreeSearch 默认结构参数：`mas_treesearch/config.py`
- TreeSearch dataset-specific 偏置：`mas_treesearch/profiles.py`
- GFlowOpt 主训练参数：`train_mas_gflowopt_target_suite.py`
- GFlowOpt 默认总配置：`mas_gflowopt/types.py`
- 两套 pipeline 的 checkpoint 实现：
  - `mas_treesearch/pipeline.py`
  - `mas_gflowopt/pipeline.py`

目录归类：

- 旧栈运行入口：`runners/mas_gflowopt/`
- 新栈运行入口：`runners/mas_treesearch/`
- 旧栈测试：`tests/mas_gflowopt/`
- 新栈测试：`tests/mas_treesearch/`

## 后续更新约定

后续优化优先沿 `mas_treesearch/` 这条主线推进，推荐按下面顺序迭代：

1. 数据集拆分
   - 不再默认把所有数据集混合打乱统一训练。
   - 优先采用“每个数据集单独 train / 单独 test”的方式。
   - 统一从 `dataset/unified_mixed/` 派生出按数据集拆分的副本，原始数据保留不动。

2. 数据标准化
   - 每个数据集保留标准化后的 `question / answer / metadata`。
   - 在标准化阶段显式写入：
     - `mas_dataset_name`
     - `mas_task_type`
     - `mas_answer_format`
     - `mas_root_templates`
   - 让执行器、评测器、搜索器都直接消费这些字段，而不是每次临时猜题型。

3. 数据集 profile 驱动搜索
   - 不同数据集允许使用不同的：
     - root templates
     - 默认 prompt 槽位
     - 输出格式约束
     - reward 细则
   - 当前已接入 profile 机制，后续新增数据集时，优先补 `mas_treesearch/profiles.py`。

4. 先做软定制，再做硬定制
   - 第一阶段先用 dataset-specific profile 调整搜索偏置。
   - 第二阶段再按收益增量引入 dataset-specific operator。
   - 推荐优先级：
     - `mmlu / mmlu_pro`: `option_compare`, `option_eliminate`, `abstain_check`
     - `normad`: `judge_binary`
     - `knowledge_crosswords`: `fill_blanks_json`, `slotwise_verify`

5. 训练与测试分离
   - 训练阶段允许更新 `LearnableEditPrior / LearnableValueModel`。
   - 测试阶段复用训练后的在线模型，但禁止继续学习，避免 test leakage。

6. 控制真实 LLM 成本
   - 优先复用 embedding / chat cache。
   - 优先做单阶段搜索，而不是“训练一次再完整重复执行一次”。
   - 新功能接入时，先问两个问题：
     - 会不会增加重复 LLM 调用？
     - 能不能通过缓存、低保真评估或结构约束减少调用？

7. 新功能落地方式
   - 优先先加最小可用框架，不先铺完整实验。
   - 每次新增能力后，至少补：
     - 一个小规模真实模型回归测试
     - 一个轻量单元测试
     - README 中的使用说明和后续优化建议

## MAS-GFlowOpt (Trainable)

## 1. 现在已实现的“真实可训练”目标

### 1.0 问题条件化 + 子集选择（已实现）
- 输入问题文本 `question_text`，编码为问题向量 `q`。
- 先做轻量 gating：`score(agent_i | q)`，选 top-k agent 子集后再交给 GFlowNet。
- gating 基于“相关性 + q条件互补性 + 多样性”的确定性贪心选择。
- 额外加入可学习二阶打分器（按问题隔离的在线更新），建模 agent 组合交互价值。
- 同一套 GFlowNet 参数共享，不同 `q` 会产生不同策略分布（任务个性化）。

### 1.1 GFlowNet 主损失（已实现）
- **Detailed-Balance loss**（论文 Eq.(4) 风格）  
  基于转移 `G_t -> G_{t+1}` 计算：
  - `log R(G_t), log R(G_{t+1})`
  - `log P_F(G_{t+1}|G_t)`
  - `log P_F(stop|G_t), log P_F(stop|G_{t+1})`
  - 固定 backward policy `log P_B(G_t|G_{t+1}) = -log |E_{t+1}|`
- 使用手写梯度对策略参数（`w_src, w_dst, w_stop, b_edge, b_stop`）更新。
- 边动作策略显式引入全局上下文 `z`（`w_edge_ctx · z`），不再只靠 `(src,dst)` 偏好。
- 并注入问题向量 `q`（例如 `q⊙src`、`q⊙dst`、`q⊙z` 特征），实现任务条件化策略。

### 1.2 对比损失（已实现）
- **NT-Xent 风格 Contrastive loss**：使用轨迹相邻状态 `(z_t, z_{t+1})` 作为正样本，其余样本作为负样本。
- 使用可训练投影头 `proj_w` 优化对比目标。

### 1.3 代理模型损失（已实现）
- `ProxyModel` 从线性占位升级为 **MLP**。
- 训练目标：
  - `MSE(S(z), R(G))`
  - `Pairwise ranking hinge loss`（保持高奖励样本在预测上更高）
- 支持对 `z` 的梯度上升 `z <- z + eta * dS/dz`。

---

## 2. 奖励函数（已实现，可直接接真实 MAS）

`reward.py` 中实现了组合奖励：

- `BIC term`: `tanh(BIC / scale)`
- `Task utility`: 来自 MAS 完整任务执行结果（成功率、任务分、延迟、token 成本、安全惩罚的加权组合）
- `Contribution term`: 智能体贡献项（见下节）
- `Question alignment term`: 问题与已选 agent 子集的对齐项
- `Size penalty term`: 规模惩罚（关键角色豁免 + 分段惩罚）
- 最终：
  - `total_score = w_task*task_utility + w_bic*bic_term + w_contrib*contribution_term + w_q*question_align - lambda*active_agent_count`
  - `R(G) = exp(temperature * total_score)`（带数值裁剪）
- 代价控制：
  - 支持稀疏真评估（`interval / budget / terminal-always`）
  - 非真评估步骤优先走缓存/快层信号，避免每步都跑昂贵 MAS 消融
  - 真值缓存键包含：问题签名 + 节点身份映射 + 具名边，避免子集/顺序变化时错配复用

---

## 3. 智能体贡献评测（已实现）

`MASRewardModel.estimate_agent_contributions(...)` 支持：

1. `loo`（默认）：  
   `contrib_i = U(full) - U(without i)`
2. `shapley`（近似）：  
   Monte Carlo permutation 估计边际贡献。

你只需要提供一个实现 `MASTaskEvaluator` 协议的评测器：

```python
class MyEvaluator(MASTaskEvaluator):
    def evaluate(self, dag: DAGState, active_agent_ids=None, question_text=None, question_vector=None) -> TaskEvaluation:
        # 运行一次真实 MAS 任务，返回指标
        return TaskEvaluation(...)
```

框架会自动在奖励里使用该评测器并做贡献分解。

---

## 4. 训练入口

- 仅采样+优化（不训练）：
```bash
python3 demo_run.py
```

- 训练后再优化（含 DB+CL+Proxy）：
```bash
python3 demo_train.py
```

`pipeline.py` 入口：

- `MASGFlowPipeline.train(evaluator=...)`
- `MASGFlowPipeline.run(evaluator=...)`
- `MASGFlowPipeline.train_and_run(evaluator=...)`
- 这些入口均支持 `question_text` 和 `agent_top_k`。
- gating 更新默认 `train_only`（`run` 默认不更新），降低跨任务漂移。

示例：
```python
history, out = pipeline.train_and_run(
    evaluator=my_evaluator,
    question_text="Design a medically safe and cost-aware diagnosis workflow.",
    agent_top_k=4,
)
```
- 精修目标支持：
  - `refine_objective="bic"`：只看 BIC
  - `refine_objective="composite"`：与训练一致（任务+贡献+BIC）

---

## 5. 关键文件

- `mas_gflowopt/reward.py`: 奖励函数 + 贡献评测（LOO/Shapley）
- `mas_gflowopt/gflownet.py`: 可训练 GFlowNet（DB loss + CL loss）
- `mas_gflowopt/conditioning.py`: 问题向量编码 + agent subset gating
- `mas_gflowopt/proxy.py`: MLP 代理模型（MSE + ranking）
- `mas_gflowopt/scoring.py`: 占位 BIC + 离散数据真实 BIC (`DiscreteDataBICScorer`)
- `mas_gflowopt/pipeline.py`: train/run/train_and_run
- `mas_gflowopt/optimizer.py`: 复合目标精修 / BIC 精修切换

---

## 6. 你接入真实系统时需要替换/填充

1. `MASTaskEvaluator.evaluate()`：接你的大模型编排与任务执行。  
2. `DiscreteDataBICScorer` 数据输入：填入真实离散数据（或替换成你的评分器）。  
3. `agent_pool` 的 profile/prompt/metadata：替换为你的正式智能体配置。  
4. 如需更强表达能力，可把 `representation.py` 换成真实 GNN/Transformer encoder（当前可直接跑通）。  
