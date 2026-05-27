# mmlu_pro_case_1 graph/memory timeline

Full graph replay not found; using row-level graph-faithfulness metrics.

| turn | active_edge_ratio | active_node_ratio | token_cost | token_estimate | candidate_count_after_turn | feedback/residual |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 1.0 | 1.0 | 0.008910000000000001 | 891 | not found in rows/replay | certificate=fd_ccs_contrastive_rescue; accept_blocker=none; support/conflict=1/1; score_margin=1.0; vote_margin=2.0 |
| 2 | 1.0 | 1.0 | 0.00924 | 924 | not found in rows/replay | certificate=fd_ccs_contrastive_rescue; accept_blocker=none; support/conflict=1/1; score_margin=1.0; vote_margin=2.0 |
| 3 | 1.0 | 1.0 | 0.0050100000000000006 | 501 | not found in rows/replay | certificate=fd_ccs_contrastive_rescue; accept_blocker=none; support/conflict=1/1; score_margin=1.0; vote_margin=2.0 |
| 4 | 1.0 | 1.0 | 0.008660000000000001 | 866 | not found in rows/replay | certificate=fd_ccs_contrastive_rescue; accept_blocker=none; support/conflict=1/1; score_margin=1.0; vote_margin=2.0 |
| 5 | 1.0 | 1.0 | 0.009210000000000001 | 921 | 6 | certificate=fd_ccs_contrastive_rescue; accept_blocker=none; support/conflict=1/1; score_margin=1.0; vote_margin=2.0 |

## Selected Memory Records

- not found in rows/replay

## Graph Details

- Stage1 union graph nodes: 6
- Stage1 union graph edges: 8
- node `generator@reasoner`: role=generator, agent=reasoner
- node `critic@verifier`: role=critic, agent=verifier
- node `reviser@summarizer`: role=reviser, agent=summarizer
- node `super_source`: role=super_source, agent=super_source
- node `global_controller`: role=global_controller, agent=global_controller
- node `super_sink`: role=super_sink, agent=super_sink
