# Stage1 Selected Runs

This folder collects the three stage1 TreeSearch runs you asked to keep together:

1. `mbpp`: the candidate run associated with the remembered `0.8427` result
2. `humaneval`: the confirmed run with `success = 0.8855`
3. `gsm8k`: the confirmed run with `task_score = success = 0.9754`

Files in this folder:

- `mbpp_candidate_08427_manifest.json`
- `humaneval_08855_manifest.json`
- `gsm8k_09754_manifest.json`
- symlinks to the original checkpoints, reports, logs, and auxiliary evidence files

Important note for `mbpp`:

- The current workspace does not retain a standalone stage1 `report.json` or stage1 log that directly proves a full-run `mbpp` final test score of `0.8427`.
- The best surviving evidence is a later stage2 partial run where the rows that fell back to stage1 average to `task_score = 0.8424742268041239` over 97 rows.
- The associated stage1 checkpoint is preserved, but it is saved at `phase = post_train` and does not include a final `test_summary`.

Quick summary:

| dataset | status | checkpoint | score evidence |
|---|---|---|---|
| `mbpp` | candidate | `mbpp_candidate_08427_checkpoint.json` | stage2 partial fallback-to-stage1 subset mean `task_score = 0.8424742268041239` |
| `humaneval` | confirmed | `humaneval_08855_checkpoint.json` | stage1 `report.json` has `success = 0.8854961832061069`, `task_score = 0.9255725190839692` |
| `gsm8k` | confirmed | `gsm8k_09754_checkpoint.json` | stage1 `report.json` has `task_score = success = 0.9753554502369668` |
