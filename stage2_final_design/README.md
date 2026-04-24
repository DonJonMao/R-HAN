# stage2_final_design

`stage2_final_design/` 现在是当前可训练、可维护的正式实现 home，不再是历史快照容器。

这条线对应当前正在使用的主训练链：

```text
train_mas_stage2_v4_4_target_suite.py
  -> mas_stage2_v4_4.Stage2V44Pipeline
  -> Stage2-GCR+/stage2_gcr_plus/runtime_v44.py
  -> runtime_v2 graph loop + learned memory selector
```

## 当前保留的模块

这里只保留当前训练与回归测试实际使用的模块：

```text
stage2_final_design/
  Stage2-GCR+/
    stage2_gcr_plus/
    tests/
    run_stage2_gcr_plus_parallel.py
    run_stage2_gcr_plus_router.py

  mas_stage2/
    当前 Stage2 公共运行时基础模块
    (config / controller / learning / composer / gnn / global_node / structure_io / types ...)

  mas_stage2_v4_4/
    当前正式 pipeline 与 runtime 入口

  mas_treesearch/
    当前训练依赖的 stage1 / evaluator / profile / data 侧实现

  train_mas_stage2_v4_4_target_suite.py
  train_mas_stage2_target_suite.py
  .gitignore
```

## 当前主线特性

这版实现包含当前实际训练使用的两块核心改动：

1. 图稀疏：
   - incoming-edge activation 使用 `support_set_mode = "sparsemax"`
   - 每个目标节点的入边 support-set 由 sparsemax 几何决定

2. memory 学习化：
   - slot schema 使用 soft learned slot mask
   - slot 内 record 选择使用 learned scorer + sparsemax support-set
   - 当前训练已经在这两层同时开启

## 不再保留的东西

这次整理后，下面这些内容不再作为 `final_design` 的一部分保留：

- 作为历史基线说明存在的 snapshot/provenance 文档
- 仅为旧兼容链存在的 `mas_stage2_v4_1/`, `mas_stage2_v4_2/`, `mas_stage2_v4_3/`
- 与当前训练无关的旧测试与旧说明文档

## 回归检查

整理完成后，至少应通过：

```text
python -m pytest Stage2-GCR+/tests/test_memory.py \
  Stage2-GCR+/tests/test_v4_4.py \
  Stage2-GCR+/tests/test_orchestration.py
```

如果你要从这里继续做实验，后续就直接把 `stage2_final_design/` 当成项目根目录使用，不需要再经过 `source/` 包一层。
