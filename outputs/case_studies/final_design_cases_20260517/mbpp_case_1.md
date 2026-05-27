# mbpp mbpp:640: repair_wrong_to_correct

## 1. Problem
- Dataset / split / id / category: `mbpp` / `test` / `mbpp:640` / `Code`
- Function signature: `def remove_parenthesis(s)`
- Short task description: Write a function to remove the parenthesis area in a string.
- Visible tests:
- `assert remove_parenthesis(["python (chrome)"])==("python")`
- `assert remove_parenthesis(["string(.abc)"])==("string")`
- Gold answer: reference code is stored in the sanitized raw JSON; hidden tests are not shown.

## 2. Stage1 Anchor / Stage1 Failure
- Stage1 success: `0.0`
- Stage1 signature: `critique_revise|critic:verifier,generator:reasoner,reviser:summarizer|critic:PromptSlots(reasoning_mode='direct', upstream_usage='summary', output_style='raw', verification_mode='strict', finalization='answer_only'),generator:PromptSlots(reasoning_mode='stepwise', upstream_usage='summary', output_style='raw', verification_mode='off', finalization='answer_only'),reviser:PromptSlots(reasoning_mode=' ... [truncated]`
- Anchor visible tests passed / total: `2` / `3`
- Anchor failure_kind: `visible_test_failure`
- Stage1 code:
```python
def remove_parenthesis(s):
    return ''.join(word.split('(', 1)[0] for word in s)
```

## 3. Stage2 Candidate / Repair / Probe / Stage2 Repair
- Stage2 route family: `code_repair`
- Candidate count / collapsed class count: `7` / `4`
- Selected candidate source: `code_repair_branch`
- repair branch count: `3`
- selected_is_repair_branch: `True`
- selected visible tests passed / total: `3` / `3`
- selected failure_kind: `none`
- Code diff:
```diff
--- stage1_anchor.py
+++ stage2_selected.py
@@ -1,2 +1,2 @@
 def remove_parenthesis(s):
-    return ''.join(word.split('(', 1)[0] for word in s)
+    return ''.join(word.split('(', 1)[0].rstrip() for word in s)
```

## 4. Final Selection / Why Accepted
- Final success: `1.0`
- Selection decision: `use_stage2_protocol`
- Selection reason: `v4_4_code_reinsert_recollapse`
- Why accepted or preserved: Repair improved visible tests from `2/3` to `3/3` and failure kind from `visible_test_failure` to `none`.

## 5. Graph and Memory Behavior
- active_edge_ratio_by_turn: `[1.0, 1.0, 1.0, 1.0, 1.0]`
- active_node_ratio_by_turn: `[1.0, 1.0, 1.0, 1.0, 1.0]`
- stage2_turns: `5.0`
- stage2_memory_records: `20.0`
- recovery_subgraph_size: `3.0`
- candidate_provenance_coverage: `0.875`
- Selected memory records:
- not found in rows/replay

## 6. Cost
- stage1 token cost: `0.00041000000000000005`
- stage2 token cost: `0.00489`
- total token cost: `0.00489`
- latency: `127.40734945610166`

## 7. Paper-ready takeaway
The code route turned a failed solution into an executable residual, generated a local repair, and accepted it only after the repaired branch dominated the anchor on visible tests.
