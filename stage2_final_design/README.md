# stage2_final_design

This directory is an isolated snapshot of the Stage2-GCR+ code version that most closely matches the historical MBPP run with positive Stage2 gains:

```text
outputs/stage2_gcr_plus_phase3a_mbpp
```

The code snapshot is exported from git commit:

```text
08ec6ef  stage2-gcr phase2，recovery门槛明显放宽，如果candidate没有优于anchor的直接取一阶段最高分图对anchor做recovery
```

It is intentionally separate from the live repository code. Do not import it from production paths directly unless we explicitly decide to revive this version.

## Why This Commit

The MBPP run started at:

```text
2026-04-14T17:00:38Z
```

The relevant output directory reports:

```text
outputs/stage2_gcr_plus_phase3a_mbpp/workers/worker_*/mbpp/checkpoint.json
```

Those checkpoints contain the Stage2 config fields present in `08ec6ef`, and do not contain fields introduced later by `ff848d1`, such as:

```text
repair_self_check_enabled
```

Therefore `08ec6ef` is the safest committed approximation of the code that produced the historical MBPP improvement.

## Historical MBPP Result

Aggregating the three workers under `outputs/stage2_gcr_plus_phase3a_mbpp`:

```text
samples: 195
better: 4
worse: 0
same: 191
stage1_success: 0.8462
stage2_success: 0.8615
stage1_task_score: 0.9183
stage2_task_score: 0.9269
```

This was the run we remembered as "Stage2-GCR improved MBPP".

## Sparse Mode

This historical MBPP-improving run used:

```text
support_set_mode = "topk"
```

So this snapshot is the top-k SparseActivate / recovery-reinsert version, not the later forced-sparsemax code route.

The later commit `bae2452` is not this historical run. `bae2452` was created on 2026-04-21 as a baseline commit before the newer final-code-route edits.

## Workspace Dependency Note

The committed `08ec6ef` source tracks `mas_stage2_v4_4`, but that package imports earlier local stage2 variants. Those dependency directories were not tracked by the selected commit archive, while the historical run environment had them available in the workspace.

To keep this isolated snapshot importable, the following workspace dependency directories were copied into `source/` as compatibility dependencies:

```text
mas_stage2_v4_1/
mas_stage2_v4_2/
mas_stage2_v4_3/
```

The run-matching implementation files remain the `08ec6ef` snapshot. These added directories are compatibility support for the import chain, not evidence that the historical run used a later sparsemax route.

## Contents

```text
source/
  Stage2-GCR+/
  mas_stage2/
  mas_stage2_v4_1/
  mas_stage2_v4_2/
  mas_stage2_v4_3/
  mas_stage2_v4_4/
  mas_treesearch/
  train_mas_stage2_v4_4_target_suite.py
  train_mas_stage2_target_suite.py
  tests/
  test_stage2_v2.py

provenance/
  PROVENANCE.md
```

The historical training entrypoint imports `mas_stage2_v4_4`, but `source/mas_stage2_v4_4/runtime.py` is a wrapper that prepends `source/Stage2-GCR+` to `sys.path` and exposes:

```text
stage2_gcr_plus.runtime_v44.Stage2RuntimeV44
```

So the important MBPP-improving code path is the `Stage2-GCR+/stage2_gcr_plus/runtime_v44.py` recovery/reinsert implementation, reached through the `mas_stage2_v4_4` compatibility package.

## Reproduction Pointer

The original worker command shape was:

```text
python train_mas_stage2_v4_4_target_suite.py \
  --data-root outputs/stage2_gcr_plus_phase3a_mbpp_3x_manual_20260415_010034/sample_shards/shard_<i> \
  --output-root outputs/stage2_gcr_plus_phase3a_mbpp_3x_manual_20260415_010034/workers/worker_<i> \
  --dataset mbpp \
  --resume \
  --seed 7 \
  --search-iterations 8 \
  --candidate-core-k 4 \
  --candidate-explore-k 3 \
  --candidate-max-k 8 \
  --tier1-repeats 1 \
  --tier2-repeats 1 \
  --stage2-turn-count 5 \
  --memory-top-k 4 \
  --soft-prune-top-k 3 \
  --soft-prune-threshold 0.38 \
  --hard-prune-after-turn 4 \
  --periodic-every 1000000 \
  --periodic-size 0 \
  --max-train -1 \
  --max-validation 0 \
  --max-test -1 \
  --stage1-checkpoint-root outputs/stage1_selected_ckpts_v2_20260330 \
  --tier1-max-tokens 256 \
  --tier2-max-tokens 896
```
