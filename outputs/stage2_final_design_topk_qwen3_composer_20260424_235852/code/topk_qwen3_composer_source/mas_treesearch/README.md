# MAS Tree Search

独立于 `mas_gflowopt` 的最小可用树搜索 MAS 框架。

当前实现范围：

- 使用本地 OpenAI-compatible 大模型接口：默认 `LLM_API_BASE=http://127.0.0.1:8039`
- 使用本地 embedding 接口：默认 `EMBED_API_BASE=http://127.0.0.1:8018`
- 一次初始化后缓存 agent embedding，避免重复计算
- 架构搜索空间：`direct / solve_verify / parallel_vote / critique_revise / debate_judge / route_solve`
- 轻量 prompt 搜索：只搜索离散 prompt 槽位，不生成自由文本 prompt
- 多层评估：`tier0 proxy -> tier1 cheap LLM eval -> tier2 true LLM eval`
- 单阶段搜索：搜索过程直接返回 best architecture，不再 train/run 两段重复执行
- 已接入轻量可训练接口：
  - `LearnableEditPrior`：在线学习“什么 edit 更可能带来提升”
  - `LearnableValueModel`：在线拟合 tier2 reward，参与 tier0 排序融合
- 已支持按数据集独立处理和运行：
  - 从 `dataset/unified_mixed` 按 `source_dataset` 拆分为独立 train/validation/test
  - 为每个数据集附加标准化格式、任务类型、答案格式和推荐 root templates
  - dataset-aware 搜索：不同数据集可使用不同 root templates 和默认 prompt 槽位
  - 新增单数据集 `train -> test` 运行脚本，测试阶段不会继续更新在线模型

## 最小使用方式

```python
from mas_treesearch import TreeSearchMASPipeline

pipeline = TreeSearchMASPipeline()
result = pipeline.search("If 3 pens cost 9 dollars, how much do 5 pens cost?")
print(result.best_node.compiled.signature())
print(result.best_node.tier2.mean_task_score if result.best_node.tier2 else None)
```

## 处理后的数据集

先把混合数据集转换为按数据集拆分的副本，原始数据保持不变：

```bash
python /mnt/nvme/projects/R-HAN/prepare_mas_treesearch_data.py \
  --unified-root /mnt/nvme/projects/R-HAN/dataset/unified_mixed \
  --output-root /mnt/nvme/projects/R-HAN/dataset/mas_treesearch_processed
```

输出结构为：

```text
dataset/mas_treesearch_processed/
  manifest.json
  mmlu/train.jsonl
  mmlu/validation.jsonl
  mmlu/test.jsonl
  gsm8k/train.jsonl
  ...
```

## 单数据集训练与测试

```bash
LLM_API_BASE=http://127.0.0.1:8039 \
EMBED_API_BASE=http://127.0.0.1:8018 \
python /mnt/nvme/projects/R-HAN/demo_dataset_train_test.py \
  --data-root /mnt/nvme/projects/R-HAN/dataset/mas_treesearch_processed \
  --dataset gsm8k \
  --max-train 100 \
  --max-test 50 \
  --train-shuffle \
  --test-shuffle \
  --search-iterations 8
```

说明：

- 训练阶段会保留并更新 `LearnableEditPrior` 和 `LearnableValueModel`
- 测试阶段会复用训练后的在线模型，但不会继续更新，避免 test leakage
- 每个数据集会独立实例化一套 pipeline，不再像 `mas_gflowopt` 那样混合打乱后统一训练

## 当前未实现

- 实验脚本与数据集批量评测
- learnable gater 的训练逻辑
- 子图执行缓存与跨任务经验回放
- 更复杂的多 block workflow 组合搜索

## 二阶段说明

`mas_treesearch/` 当前仍然只负责第一阶段：

- tree search 拓扑生成
- top-k 互补结构选择
- `union graph` 合并
- 轻量 `union_runtime` 基线

正式的第二阶段三层图运行时已经单独拆到 `mas_stage2/`，避免影响当前一阶段训练。第二阶段代码默认以 `UnionGraph` 为输入，包含：

- 私有题内记忆
- latent-backed local composer
- graph-mediated exported messages
- global controller
- active subgraph pruning
- feedback event 记录与 replay bundle

因此如果后续继续推进“拓扑生成 + 权重优化解耦”的主线，当前推荐分工是：

- 第一阶段：`mas_treesearch/`
- 第二阶段：`mas_stage2/`
