import json
import unittest

from mas_treesearch.compiler import compile_architecture
from mas_treesearch.data import standardize_record
from mas_treesearch.evaluator import MultiFidelityEvaluator
from mas_treesearch.profiles import resolve_dataset_profile
from mas_treesearch.result_utils import has_structure_output, resolve_structure_signature, resolve_structure_summary
from mas_treesearch.topology_set import TopologySetScorer
from mas_treesearch.union_runtime import GraphMerger
from mas_treesearch.config import SearchConfig, UnionRuntimeConfig
from mas_treesearch.types import (
    ArchitectureState,
    EvalSummary,
    PromptSlots,
    SearchNode,
    SearchResult,
    TaskEvaluation,
    WorkflowTemplate,
)


def _make_eval_summary(reward: float, task_score: float) -> EvalSummary:
    return EvalSummary(
        tier="tier2",
        mean_reward=reward,
        reward_std=0.0,
        mean_task_score=task_score,
        mean_success=1.0 if task_score >= 0.5 else 0.0,
        mean_latency=0.0,
        mean_token_cost=0.0,
        mean_safety_penalty=0.0,
        evaluations=[],
    )


def _make_node(template: WorkflowTemplate, role_to_agent: dict[str, str], reward: float, task_score: float) -> SearchNode:
    state = ArchitectureState(
        template=template,
        role_to_agent=role_to_agent,
        role_to_prompt={role: PromptSlots() for role in role_to_agent},
    )
    compiled = compile_architecture(state)
    return SearchNode(
        state=state,
        compiled=compiled,
        parent_signature=None,
        action_from_parent="root",
        proxy_score=reward,
        tier2=_make_eval_summary(reward, task_score),
    )


class ProcessedDatasetTests(unittest.TestCase):
    def test_mmlu_pro_question_includes_metadata_options(self) -> None:
        record = {
            "id": "mmlu_pro:1",
            "source_dataset": "mmlu_pro",
            "category": "Knowledge",
            "question": "Pick the right option.",
            "answer": "D",
            "metadata": {
                "options": ["alpha", "beta", "gamma", "delta"],
                "answer_index": 3,
            },
        }
        standardized = standardize_record(record, "train")
        self.assertIn("Options:", standardized["question"])
        self.assertIn("4) delta", standardized["question"])
        self.assertEqual(standardized["answer"], "OPTION - 4")

    def test_normad_answer_is_normalized_to_yes_no(self) -> None:
        record = {
            "id": "normad:1",
            "source_dataset": "normad",
            "category": "Safety",
            "question": "Is this acceptable?",
            "answer": "No",
            "metadata": {},
        }
        standardized = standardize_record(record, "test")
        self.assertTrue(standardized["question"].endswith("Answer with exactly one token: yes or no."))
        self.assertEqual(standardized["answer"], "no")

    def test_knowledge_crosswords_answer_is_json_list(self) -> None:
        record = {
            "id": "kc:1",
            "source_dataset": "knowledge_crosswords",
            "category": "Knowledge",
            "question": "Fill blanks",
            "answer": "[\"male\", \"Order\"]",
            "metadata": {
                "blanks": ["blank 1", "blank 2"],
                "options": {"blank 1": ["male", "female"], "blank 2": ["Order", "Prize"]},
            },
        }
        standardized = standardize_record(record, "validation")
        self.assertEqual(standardized["answer"], json.dumps(["male", "Order"], ensure_ascii=False, separators=(",", ":")))
        self.assertEqual(standardized["metadata"]["mas_answer_format"], "json_list")

    def test_dataset_profile_exposes_roots(self) -> None:
        profile = resolve_dataset_profile("gsm8k")
        self.assertIn("solve_verify", profile.root_templates)
        self.assertEqual(profile.answer_format, "number")

    def test_structure_priors_differ_between_code_and_numeric_tasks(self) -> None:
        humaneval = resolve_dataset_profile("humaneval")
        gsm8k = resolve_dataset_profile("gsm8k")
        self.assertGreater(humaneval.structure_prior.target_task_nodes, gsm8k.structure_prior.target_task_nodes)
        self.assertGreater(humaneval.structure_prior.selector_diversity_scale, gsm8k.structure_prior.selector_diversity_scale)
        self.assertGreater(humaneval.structure_prior.proxy_structure_weight, gsm8k.structure_prior.proxy_structure_weight)

    def test_humaneval_question_includes_visible_test_examples(self) -> None:
        record = {
            "id": "humaneval:1",
            "source_dataset": "humaneval",
            "category": "Code",
            "question": "def add(a, b):\n    pass",
            "answer": "def add(a, b):\n    return a + b",
            "metadata": {
                "entry_point": "add",
                "test": "def check(candidate):\n    assert candidate(1, 2) == 3\n    assert candidate(5, 7) == 12\n",
            },
        }
        standardized = standardize_record(record, "test")
        self.assertIn("Implement the Python function `add`.", standardized["question"])
        self.assertIn("Visible test examples:", standardized["question"])
        self.assertIn("assert candidate(1, 2) == 3", standardized["question"])

    def test_nlgraph_question_requires_json_only(self) -> None:
        record = {
            "id": "nlgraph:1",
            "source_dataset": "nlgraph",
            "category": "Graph",
            "question": "Return a topological order.",
            "answer": "{\"order\":[1,2,3]}",
            "metadata": {"task": "topology"},
        }
        standardized = standardize_record(record, "validation")
        self.assertIn("Return only a JSON object", standardized["question"])
        self.assertIn("Do not output explanations.", standardized["question"])

    def test_code_feedback_signal_uses_partial_execution_hints(self) -> None:
        summary = EvalSummary(
            tier="tier1",
            mean_reward=0.0,
            reward_std=0.0,
            mean_task_score=0.4,
            mean_success=0.0,
            mean_latency=0.0,
            mean_token_cost=0.0,
            mean_safety_penalty=0.0,
            evaluations=[
                TaskEvaluation(
                    task_score=0.4,
                    success=0.0,
                    latency=0.0,
                    token_cost=0.0,
                    safety_penalty=0.0,
                    debug_info={
                        "dataset_specific": {
                            "syntax_ok": True,
                            "entry_defined": True,
                            "passed": 1,
                            "total": 4,
                        }
                    },
                )
            ],
        )
        profile = resolve_dataset_profile("mbpp")
        signal = MultiFidelityEvaluator.feedback_signal(summary, dataset_profile=profile)
        self.assertGreater(signal, 0.3)

    def test_structure_summary_prefers_complementary_topologies(self) -> None:
        profile = resolve_dataset_profile("humaneval")
        node_a = _make_node(
            WorkflowTemplate.CRITIQUE_REVISE,
            {"generator": "planner", "critic": "verifier", "reviser": "coder"},
            reward=1.2,
            task_score=0.9,
        )
        node_b = _make_node(
            WorkflowTemplate.SOLVE_VERIFY,
            {"solver": "coder", "verifier": "verifier"},
            reward=1.0,
            task_score=0.8,
        )
        node_c = _make_node(
            WorkflowTemplate.ROUTE_SOLVE,
            {"router": "planner", "solver": "coder"},
            reward=0.9,
            task_score=0.7,
        )
        scorer = TopologySetScorer(SearchConfig(), UnionRuntimeConfig(selected_topology_k=2))
        selected = scorer.select([node_a, node_b, node_c], dataset_profile=profile, fallback_best=node_a)
        graph = GraphMerger(UnionRuntimeConfig(selected_topology_k=2)).merge(selected)
        structure = scorer.summarize(selected, graph, dataset_profile=profile, mode="structure_only")
        self.assertEqual(structure.mode, "structure_only")
        self.assertEqual(len(structure.selected_topology_signatures), 2)
        self.assertGreater(structure.metrics.total_reward, 0.0)
        self.assertGreaterEqual(structure.metrics.coverage, 0.5)
        self.assertGreaterEqual(structure.metrics.execution_probe, 0.5)

    def test_result_utils_expose_structure_output(self) -> None:
        node = _make_node(
            WorkflowTemplate.SOLVE_VERIFY,
            {"solver": "coder", "verifier": "verifier"},
            reward=1.1,
            task_score=0.85,
        )
        scorer = TopologySetScorer(SearchConfig(), UnionRuntimeConfig(selected_topology_k=1))
        graph = GraphMerger(UnionRuntimeConfig(selected_topology_k=1)).merge([node])
        structure = scorer.summarize([node], graph, dataset_profile=resolve_dataset_profile("mbpp"), mode="structure_only")
        result = SearchResult(
            question_text="q",
            selected_agents=["coder", "verifier"],
            root_signatures=[],
            best_node=node,
            top_nodes=[node],
            records=[],
            nodes={node.compiled.signature(): node},
            pipeline_mode="structure_only",
            selected_topology_nodes=[node],
            union_graph=graph,
            structure_summary=structure,
            final_signature=structure.signature,
        )
        self.assertTrue(has_structure_output(result))
        self.assertIsNotNone(resolve_structure_summary(result))
        self.assertEqual(resolve_structure_signature(result), structure.signature)


if __name__ == "__main__":
    unittest.main()
