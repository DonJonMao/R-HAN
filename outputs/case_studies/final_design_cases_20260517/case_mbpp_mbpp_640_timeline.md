# mbpp_case_1 graph/memory timeline

Full graph replay not found; using row-level graph-faithfulness metrics.

| turn | active_edge_ratio | active_node_ratio | token_cost | token_estimate | candidate_count_after_turn | feedback/residual |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 1.0 | 1.0 | 0.00071 | 71 | not found in rows/replay | visible_tests 2/3 -> 3/3; failure visible_test_failure -> none |
| 2 | 1.0 | 1.0 | 0.0008500000000000001 | 85 | not found in rows/replay | visible_tests 2/3 -> 3/3; failure visible_test_failure -> none |
| 3 | 1.0 | 1.0 | 0.0013100000000000002 | 131 | not found in rows/replay | visible_tests 2/3 -> 3/3; failure visible_test_failure -> none |
| 4 | 1.0 | 1.0 | 0.00101 | 101 | not found in rows/replay | visible_tests 2/3 -> 3/3; failure visible_test_failure -> none |
| 5 | 1.0 | 1.0 | 0.00101 | 101 | 7 | visible_tests 2/3 -> 3/3; failure visible_test_failure -> none |

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
