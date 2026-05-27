# TopK + Qwen3 Composer vs bdc1ae0

Baseline: `/mnt/nvme/projects/R-HAN/outputs/stage2_final_design_slot_mask_phase2_20260423_094157/slot_mask_phase2_mbpp_humaneval_3x`
Candidate: `/mnt/nvme/projects/R-HAN/outputs/stage2_final_design_topk_qwen3_composer_20260424_235852/topk_qwen3_composer_mbpp_humaneval_3x`

| Run | Dataset | n | wrong->right | right->wrong | stage2 correct | delta correct | task delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | mbpp | 195 | 6 | 0 | 169 | 6 | 2.880 |
| baseline | humaneval | 33 | 2 | 0 | 31 | 2 | 1.300 |
| baseline | total | 228 | 8 | 0 | 200 | 8 | 4.180 |
| candidate | mbpp | 195 | 2 | 0 | 166 | 2 | 0.480 |
| candidate | humaneval | 33 | 2 | 0 | 29 | 2 | 1.300 |
| candidate | total | 228 | 4 | 0 | 195 | 4 | 1.780 |

## Wrong-To-Right IDs

### baseline
- mbpp: mbpp:398, mbpp:949, mbpp:454, mbpp:33, mbpp:552, mbpp:293
- humaneval: HumanEval/5, HumanEval/4
### candidate
- mbpp: mbpp:454, mbpp:193
- humaneval: HumanEval/5, HumanEval/4
