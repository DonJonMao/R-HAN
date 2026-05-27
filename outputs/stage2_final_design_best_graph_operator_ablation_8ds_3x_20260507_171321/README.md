# Stage2 Final Design Best-Graph Operator Ablation

Prepared on 2026-05-07.

This run removes the final stage-1 union graph with:

```text
--stage1-final-graph-mode best
```

The suite is intentionally ordered as:

```text
mbpp -> humaneval -> math_level5 -> math_full_math_ops -> mmlu_pro -> mmlu_as_mmlu_pro_ops -> knowledge_crosswords -> nlgraph
```

## Dataset Variants

| Dataset argument | Data source | Operator/profile source | Stage-1 checkpoint |
|---|---|---|---|
| `mbpp` | AFlow MBPP | `mbpp` | AFlow MBPP |
| `humaneval` | AFlow HumanEval | `humaneval` | AFlow HumanEval |
| `math_level5` | AFlow MATH Level-5 subset | `math` | AFlow MATH Level-5 |
| `math_full_math_ops` | Full MATH | `math` | AFlow MATH Level-5 |
| `mmlu_pro` | trainview MMLU-Pro | `mmlu_pro` | general three-way MMLU-Pro |
| `mmlu_as_mmlu_pro_ops` | Standard MMLU | `mmlu_pro` | general three-way MMLU-Pro |
| `knowledge_crosswords` | trainview Knowledge Crosswords | `knowledge_crosswords` | general three-way KC |
| `nlgraph` | trainview NLGraph | `nlgraph` | general three-way NLGraph |

`math_full_math_ops` keeps `source_dataset=math`, so it uses the same MATH profile/operators as the Level-5 subset while evaluating on full MATH.

`mmlu_as_mmlu_pro_ops` is generated from standard MMLU JSONL with `source_dataset=mmlu_pro`, so it uses the MMLU-Pro operator/profile path while evaluating on MMLU questions.

## Counts With Validation Skipped

Run arguments use `--max-validation 0`.

| Dataset | Train | Test | Total |
|---|---:|---:|---:|
| `mbpp` | 195 | 779 | 974 |
| `humaneval` | 33 | 131 | 164 |
| `math_level5` | 123 | 494 | 617 |
| `math_full_math_ops` | 6750 | 5000 | 11750 |
| `mmlu_pro` | 1200 | 1500 | 2700 |
| `mmlu_as_mmlu_pro_ops` | 28736 | 3565 | 32301 |
| `knowledge_crosswords` | 250 | 350 | 600 |
| `nlgraph` | 500 | 700 | 1200 |

Total train+test examples: 50306.
