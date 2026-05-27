# Stage2 Final Design Union Top-2 Operator Ablation

Prepared on 2026-05-14.

This run keeps the current matched 8-dataset operator-ablation setup and changes the stage-1 final graph materialization to top-2 union:

```text
--stage1-final-graph-mode union
--selected-topology-k 2
```

All other run arguments mirror the immediately preceding best-graph matched 8ds run unless required for a fresh output root.

Dataset order:

```text
mbpp -> humaneval -> math_level5 -> math_full_math_ops -> mmlu_pro -> mmlu_as_mmlu_pro_ops -> knowledge_crosswords -> nlgraph
```

Counts with validation skipped:

| Dataset | Train | Test | Total |
|---|---:|---:|---:|
| `mbpp` | 195 | 779 | 974 |
| `humaneval` | 33 | 131 | 164 |
| `math_level5` | 123 | 494 | 617 |
| `math_full_math_ops` | 123 | 494 | 617 |
| `mmlu_pro` | 1200 | 1500 | 2700 |
| `mmlu_as_mmlu_pro_ops` | 1200 | 1500 | 2700 |
| `knowledge_crosswords` | 250 | 350 | 600 |
| `nlgraph` | 500 | 700 | 1200 |
