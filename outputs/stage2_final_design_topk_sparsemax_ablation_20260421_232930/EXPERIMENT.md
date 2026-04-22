# Stage2 Final Design Top-k vs Sparsemax Ablation

Purpose:
1. Re-run the historical `stage2_final_design/source` top-k GCR+ snapshot on MBPP and HumanEval.
2. Create a sparsemax-only copy that changes only per-destination sparse activation:
   `a_{u->v}^t = sparsemax_{u in N^-(v)}(gamma_{u->v}^t)`.
3. Store all outputs/logs/code snapshots under this run root.

Datasets: mbpp, humaneval
Backend: local 3x OpenAI-compatible endpoints 8041/8042/8043 via runner router.
