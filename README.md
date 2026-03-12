# R-HAN

当前仓库维护两条 MAS 架构搜索路线：

- 旧路线：`mas_gflowopt/`
  以 GFlowNet + proxy + refine 为主，保留用于对照与历史复现。
- 新路线：`mas_treesearch/`
  以 tree search + 多层评估 + 轻量 prompt 搜索为主，目标是显著减少与真实 LLM 的重复交互成本。

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
