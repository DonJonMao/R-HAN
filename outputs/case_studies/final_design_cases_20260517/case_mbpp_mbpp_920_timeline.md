# mbpp_case_2 graph/memory timeline

Full graph replay not found; using row-level graph-faithfulness metrics.

| turn | active_edge_ratio | active_node_ratio | token_cost | token_estimate | candidate_count_after_turn | feedback/residual |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 1.0 | 1.0 | 0.0015600000000000002 | 156 | not found in rows/replay | visible_tests 3/3 -> 3/3; failure none -> none |
| 2 | 1.0 | 1.0 | 0.00143 | 143 | not found in rows/replay | visible_tests 3/3 -> 3/3; failure none -> none |
| 3 | 1.0 | 1.0 | 0.0017000000000000001 | 170 | not found in rows/replay | visible_tests 3/3 -> 3/3; failure none -> none |
| 4 | 1.0 | 1.0 | 0.0011300000000000001 | 113 | not found in rows/replay | visible_tests 3/3 -> 3/3; failure none -> none |
| 5 | 1.0 | 1.0 | 0.0015400000000000001 | 154 | 6 | visible_tests 3/3 -> 3/3; failure none -> none |

## Selected Memory Records

- not found in rows/replay

## Graph Details

- Stage1 union graph nodes: 6
- Stage1 union graph edges: 8
- node `generator@math`: role=generator, agent=math
- node `critic@verifier`: role=critic, agent=verifier
- node `reviser@planner`: role=reviser, agent=planner
- node `super_source`: role=super_source, agent=super_source
- node `global_controller`: role=global_controller, agent=global_controller
- node `super_sink`: role=super_sink, agent=super_sink
