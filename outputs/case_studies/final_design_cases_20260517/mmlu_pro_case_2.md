# mmlu_pro mmlu_pro:4757: safe_preserve_rejected_challenger

## 1. Problem
- Dataset / split / id / category: `mmlu_pro` / `test` / `mmlu_pro:4757` / `Knowledge`
- Question: When did the first pharaohs emerge in Egypt?

Options:
1) 3100 B.P.
2) 4100 B.P.
3) 5100 B.P.
4) 6100 B.P.

Answer with exactly one line: OPTION - <NUMBER>
- Gold answer: OPTION - 3
- Answer options:
- 1) 3100 B.P.
- 2) 4100 B.P.
- 3) 5100 B.P.
- 4) 6100 B.P.

## 2. Stage1 Anchor / Before / After
- Stage1 output: `OPTION - 3`
- Stage2 final output: `OPTION - 3`
- Gold option: `OPTION - 3`
- stage1_success -> stage2_success: `1.0` -> `1.0`
- Stage1 success: `1.0`
- Stage1 signature: `route_solve|aggregator:summarizer,router:verifier,solver:reasoner|aggregator:PromptSlots(reasoning_mode='direct', upstream_usage='summary', output_style='raw', verification_mode='light', finalization='answer_only'),router:PromptSlots(reasoning_mode='direct', upstream_usage='summary', output_style='raw', verification_mode='light', finalization='answer_only'),solver:PromptSlots(reasoning_mode='stepw ... [truncated]`
- Anchor verifier status: anchor_conflict_count=1; anchor_eval_status_hist={"missing": 0, "satisfied": 0, "unknown": 4, "violated": 1}

## 3. Stage2 Candidate / Repair / Probe / Why Stage2 Changed Or Preserved
- Stage2 route family: `mmlu_option_matrix_calibration`
- Candidate count / collapsed class count: `5` / `2`
- Selected candidate source: `stage1`
- Certificate kind: `fd_ccs_mmlu_vote_pair_contrast`
- Top candidate value: `{"answer": "3100 B.P."}`
- score_margin / vote_margin: `5.0` / `5.0`
- challenger_support_count vs anchor_conflict_count: `1` vs `1`
- Probe: triggered=True, winner=not found in rows/replay, confidence=not found in rows/replay
- Audit: triggered=True, winner=not found in rows/replay, confidence=not found in rows/replay, audit_agree=False

## 4. Final Selection
- Final output: `OPTION - 3`
- Final success: `1.0`
- Selection decision: `use_stage1_anchor`
- Selection reason: `v4_4_mmlu_option_matrix_calibration_preserve_anchor_no_certificate`
- Why override or preserve: Stage2 kept `OPTION - 3` despite challenger `{"answer": "3100 B.P."}` because accept_blocker=`reject_low_matrix_coverage` and selection_reason=`v4_4_mmlu_option_matrix_calibration_preserve_anchor_no_certificate`.

## 5. Graph and Memory Behavior
- active_edge_ratio_by_turn: `[1.0, 1.0, 1.0, 1.0, 1.0]`
- active_node_ratio_by_turn: `[1.0, 1.0, 1.0, 1.0, 1.0]`
- stage2_turns: `5.0`
- stage2_memory_records: `15.0`
- recovery_subgraph_size: `0.0`
- candidate_provenance_coverage: `0.6`
- Selected memory records:
- not found in rows/replay

## 6. Cost
- stage1 token cost: `0.00037000000000000005`
- stage2 token cost: `0.0022`
- total token cost: `0.0022`
- latency: `136.13755086436868`

## 7. Paper-ready takeaway
Stage2 explored an alternative answer but preserved the stage1 anchor because the option-level certificate was incomplete, the audit/probe did not confidently support the challenger, or the conflict/support margin was insufficient.

## Option Matrix
row-level counts only; raw matrix not found

| metric | value |
|---|---:|
| source | row-level counts only; raw matrix not found |
| candidate_bank_size | 3 |
| cert_bank_size | 2 |
| option_matrix_yes_votes | 5 |
| option_matrix_no_votes | 1 |
| evidence_atom_count | 2 |
| anchor_conflict_count | 1 |
| challenger_support_count | 1 |
| score_margin | 5.0 |
| vote_margin | 5.0 |
| audit_agree | False |
| calibrator_p_accept | 1.0 |
| accept_blocker | reject_low_matrix_coverage |
| certificate_kind | fd_ccs_mmlu_vote_pair_contrast |
| matrix_parse_status | json_repaired |
| matrix_option_covered_count | 4 |
| top_candidate_value | {"answer": "3100 B.P."} |

### Option Status Rows
| option | support_count | conflict_count | yes_votes | no_votes | evidence_atoms | final_status |
|---|---:|---:|---:|---:|---:|---|
| 1) 3100 B.P. | row-level | row-level | 5 | 1 | 2 | not selected |
| 2) 4100 B.P. | row-level | row-level | 5 | 1 | 2 | not selected |
| 3) 5100 B.P. | row-level | row-level | 5 | 1 | 2 | final, stage1_anchor, gold |
| 4) 6100 B.P. | row-level | row-level | 5 | 1 | 2 | not selected |
