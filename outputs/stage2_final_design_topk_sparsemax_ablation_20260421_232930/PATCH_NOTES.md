# Sparsemax-only Patch Notes

Baseline code: `code/topk_source`, copied from `/mnt/nvme/projects/R-HAN/stage2_final_design/source`.

Sparsemax code: `code/sparsemax_source`, same snapshot plus only these edits:

1. `mas_stage2/config.py`
   - Adds `Stage2GraphConfig.support_set_mode = "sparsemax"`.
2. `Stage2-GCR+/stage2_gcr_plus/runtime_v2.py`
   - Adds `_sparsemax_support(score_tensor)`.
   - In `_activate_edges_v2`, for every destination node, computes:
     `a_{u->v}^t = sparsemax_{u in N^-(v)}(gamma_{u->v}^t)`.
   - Keeps all incoming edges whose sparsemax support weight is positive.
   - Keeps a minimal `min_incoming_edges` fallback only if sparsemax returns fewer than the configured minimum.
   - Logs `support_set_mode` and `support_set_weight` into edge activation metadata.

No selector, recovery, reinsert, final guard, code rewrite, or self-check logic was changed.
