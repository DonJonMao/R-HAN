# Provenance: `outputs/stage2_gcr_plus_phase3a_mbpp`

## Selected Source Commit

```text
08ec6ef
```

Reason:

- The historical MBPP run started at `2026-04-14T17:00:38Z`.
- Commit `08ec6ef` was created before that run.
- Commit `ff848d1` was created after that run and introduces `repair_self_check_enabled`, which is absent from the run checkpoints.
- The run logs contain `v4_4_code_reinsert_recollapse`, `repair_br`, and `repair_imp`, all present in `08ec6ef`.

## Historical Run Location

```text
outputs/stage2_gcr_plus_phase3a_mbpp
```

Worker checkpoints:

```text
outputs/stage2_gcr_plus_phase3a_mbpp/workers/worker_0/mbpp/checkpoint.json
outputs/stage2_gcr_plus_phase3a_mbpp/workers/worker_1/mbpp/checkpoint.json
outputs/stage2_gcr_plus_phase3a_mbpp/workers/worker_2/mbpp/checkpoint.json
```

## Runtime Entry Path

The worker command uses:

```text
train_mas_stage2_v4_4_target_suite.py
```

That script imports `mas_stage2_v4_4.Stage2V44Pipeline`. In this snapshot, `mas_stage2_v4_4/runtime.py` is a compatibility wrapper that exposes the actual GCR+ runtime:

```text
Stage2-GCR+/stage2_gcr_plus/runtime_v44.py
```

The MBPP-improving decision reasons `v4_4_code_override_verified_class` and `v4_4_code_reinsert_recollapse` are defined in that `Stage2-GCR+` runtime file.

## Worker Summaries

```text
worker_0: better=0, worse=0, same=65
worker_1: better=1, worse=0, same=64
worker_2: better=3, worse=0, same=62
```

Aggregate:

```text
count=195
better=4
worse=0
same=191
stage1_success_avg=0.8461538462
stage2_success_avg=0.8615384615
stage1_task_avg=0.9182564103
stage2_task_avg=0.9268717949
```

## Better Samples

```text
mbpp:454  stage1 0.0/0.76 -> stage2 1.0/1.0  reason=v4_4_code_reinsert_recollapse
mbpp:922  stage1 0.0/0.28 -> stage2 0.0/0.76  reason=v4_4_code_override_verified_class
mbpp:293  stage1 0.0/0.28 -> stage2 1.0/1.0  reason=v4_4_code_override_verified_class
mbpp:956  stage1 0.0/0.76 -> stage2 1.0/1.0  reason=v4_4_code_override_verified_class
```

## Important Distinction

This snapshot corresponds to:

```text
top-k SparseActivate + GCR recovery/reinsert/verified-class override
```

It does not correspond to:

```text
forced sparsemax code route + checkpoint-driven function rewrite
```

The latter is from the later `2743ef9` / `068aa70` line and did not yet reproduce the historical MBPP improvement.

## Compatibility Dependencies

`08ec6ef` contains the `mas_stage2_v4_4` code that matches the historical MBPP run, but `mas_stage2_v4_4/config.py` imports earlier local variants such as `mas_stage2_v4_3`. Those earlier variant directories are workspace dependencies rather than tracked files in the selected commit archive.

For practical reuse, this snapshot therefore includes copied compatibility directories:

```text
source/mas_stage2_v4_1/
source/mas_stage2_v4_2/
source/mas_stage2_v4_3/
```

This keeps `source/mas_stage2_v4_4` importable without changing the run-matching claim: the core selected implementation remains `08ec6ef`, and the historical MBPP gain was from the top-k recovery/reinsert route, not the later forced-sparsemax line.
