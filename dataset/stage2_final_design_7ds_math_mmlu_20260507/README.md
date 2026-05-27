# Stage2 Final Design Data Root: Full MATH + MMLU

Prepared on 2026-05-07 for final-design experiments that should replace:

- `math` level-5 AFlow subset with full MATH.
- `mmlu_pro` with standard `mmlu`.

This directory uses symlinks only; source JSONL files are not copied.

## Paths

- Data root:
  `/mnt/nvme/projects/R-HAN/dataset/stage2_final_design_7ds_math_mmlu_20260507/data_root`
- Stage-1 checkpoint root:
  `/mnt/nvme/projects/R-HAN/dataset/stage2_final_design_7ds_math_mmlu_20260507/stage1_checkpoints`

## Dataset Order

Use this order when launching the seven-dataset run:

```bash
--dataset gsm8k \
--dataset humaneval \
--dataset math \
--dataset mbpp \
--dataset mmlu \
--dataset knowledge_crosswords \
--dataset nlgraph
```

## Counts

Counts below are raw stage2-supported counts. If the run keeps
`--max-validation 0`, validation is loaded but skipped by the plan.

| Dataset | Train | Validation | Test | Source |
|---|---:|---:|---:|---|
| `gsm8k` | 3781 | 0 | 15122 | `dataset/mas_treesearch_aflow_four_20260318/gsm8k` |
| `humaneval` | 33 | 0 | 131 | `dataset/mas_treesearch_aflow_four_20260318/humaneval` |
| `math` | 6750 | 750 | 5000 | `dataset/math` |
| `mbpp` | 195 | 0 | 779 | `dataset/mas_treesearch_aflow_four_20260318/mbpp` |
| `mmlu` | 28736 | 3668 | 3565 | `dataset/mas_treesearch_processed/mmlu` |
| `knowledge_crosswords` | 250 | 250 | 350 | `dataset/mas_treesearch_threeway_general_trainview_20260319/knowledge_crosswords` |
| `nlgraph` | 500 | 500 | 700 | `dataset/mas_treesearch_threeway_general_trainview_20260319/nlgraph` |

Full-run plan with `--max-train -1 --max-validation 0 --max-test -1`:

| Dataset | Train | Validation | Test |
|---|---:|---:|---:|
| `gsm8k` | 3781 | 0 | 15122 |
| `humaneval` | 33 | 0 | 131 |
| `math` | 6750 | 0 | 5000 |
| `mbpp` | 195 | 0 | 779 |
| `mmlu` | 28736 | 0 | 3565 |
| `knowledge_crosswords` | 250 | 0 | 350 |
| `nlgraph` | 500 | 0 | 700 |

## MATH Level Distribution

| Split | Level 1 | Level 2 | Level 3 | Level 4 | Level 5 | Other |
|---|---:|---:|---:|---:|---:|---:|
| train | 506 | 1215 | 1427 | 1521 | 2079 | 2 |
| validation | 58 | 133 | 165 | 169 | 225 | 0 |
| test | 437 | 894 | 1131 | 1214 | 1324 | 0 |

## Checkpoint Notes

`stage1_checkpoints/math/checkpoint.json` points to the older full-MATH target-suite checkpoint:

```text
outputs/mas_treesearch_target_suite_train_20260314_212033/math/checkpoint.json
```

`stage1_checkpoints/mmlu/checkpoint.json` points to the older standard-MMLU target-suite checkpoint:

```text
outputs/mas_treesearch_target_suite_train_20260314_212033/mmlu/checkpoint.json
```

The other checkpoint links follow the current AFlow/general roots used by the
running seven-dataset experiment.
