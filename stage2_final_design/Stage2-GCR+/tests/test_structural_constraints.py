from __future__ import annotations

from stage2_gcr_plus.structural_constraints import (
    build_nlgraph_oracle_candidate,
    make_structural_eval,
    parse_kc_verify_all_result,
    parse_structural_artifact,
    parse_structural_problem,
    structural_dominates,
)


def test_parse_nlgraph_edges_and_shortest_path_task():
    q = "In an undirected graph, (0,1,2) (1,4,1) (0,2,1). Give the shortest path from node 0 to node 4."
    problem = parse_structural_problem(q, dataset_name="nlgraph")

    assert problem.object_kind == "graph"
    assert problem.task_kind == "shortest_path"
    assert problem.payload["weighted"] is True
    assert ("0", "1", 2.0) in problem.payload["edges"]
    assert problem.payload["source"] == "0"
    assert problem.payload["target"] == "4"


def test_nlgraph_invalid_path_edge_is_fatal():
    q = "In an undirected graph, (0,1) (1,2). Give the shortest path from node 0 to node 2."
    problem = parse_structural_problem(q, dataset_name="nlgraph")
    artifact = parse_structural_artifact('{"answer":["0","3","2"],"path":["0","3","2"]}', problem)
    ev = make_structural_eval(problem, artifact)

    assert "invalid_node" in ev.residual.fatal or "invalid_edge" in ev.residual.fatal
    assert ev.residual.certificate_ok is False


def test_nlgraph_oracle_shortest_path_candidate_is_clean():
    q = "In an undirected graph, (0,1,2) (1,4,1) (0,2,1) (2,4,5). Give the shortest path from node 0 to node 4."
    problem = parse_structural_problem(q, dataset_name="nlgraph")
    artifact = build_nlgraph_oracle_candidate(problem)
    assert artifact is not None
    ev = make_structural_eval(problem, artifact)

    assert ev.residual.certificate_ok is True
    assert ev.residual.certificate_kind in {"oracle_graph", "shortest_path"}


def test_topological_order_violation():
    q = "In a directed graph, (A,B) (B,C). Give a topological sort."
    problem = parse_structural_problem(q, dataset_name="nlgraph")
    artifact = parse_structural_artifact('{"answer":["C","B","A"],"order":["C","B","A"]}', problem)
    ev = make_structural_eval(problem, artifact)

    assert "order_violation" in ev.residual.fatal


def test_undirected_cycle_question_not_misread_as_directed_or_placeholder_edge():
    q = (
        "In an undirected graph, (i,j) means that node i and node j are connected with an undirected edge.\n"
        "The nodes are numbered from 0 to 3, and the edges are: (0,1) (1,2) (2,0) (2,3)\n"
        'Q: Is there a cycle in this graph?\n\nReturn only a JSON object using this schema: {"answer":"yes"}'
    )
    problem = parse_structural_problem(q, dataset_name="nlgraph", metadata={"task": "cycle"})
    artifact = build_nlgraph_oracle_candidate(problem)
    assert artifact is not None
    ev = make_structural_eval(problem, artifact)

    assert problem.payload["directed"] is False
    assert ("i", "j", 1.0) not in problem.payload["edges"]
    assert artifact.answer == "yes"
    assert ev.residual.certificate_ok is True


def test_long_invalid_witness_does_not_dominate_clean_anchor():
    q = "In an undirected graph, (0,1) (1,2). Is there a path between node 0 and node 2?"
    problem = parse_structural_problem(q, dataset_name="nlgraph")

    old = make_structural_eval(problem, parse_structural_artifact('{"answer":"yes","path":["0","1","2"]}', problem))
    new = make_structural_eval(problem, parse_structural_artifact('{"answer":"no","components":[["0"],["1"],["2"]]}', problem))

    assert not structural_dominates(new, old)


def test_kc_json_list_length_mismatch_is_fatal():
    q = "Instruction: Pick the correct answer for each blank. 1:? 2:? Options: Alice; Bob; Paris"
    problem = parse_structural_problem(q, dataset_name="knowledge_crosswords")
    artifact = parse_structural_artifact('["Alice"]', problem)
    ev = make_structural_eval(problem, artifact)

    assert "list_length_mismatch" in ev.residual.fatal


def test_kc_challenger_requires_verify_all_pass():
    result = parse_kc_verify_all_result(
        '{"status":"pass","failed_constraints":[],"answers":["Dana Scott","Harvard"],"confidence":"high"}'
    )
    assert result.status == "pass"
    assert result.confidence == "high"
