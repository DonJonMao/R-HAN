from __future__ import annotations

import unittest

HAS_MAS_GFLOWOPT_DEPS = True
IMPORT_ERROR = ""
try:
    from mas_gflowopt.agent_pool import AgentPool
    from mas_gflowopt.evaluators import BatchLLMExecutionMASTaskEvaluator, LLMExecutionConfig
    from mas_gflowopt.types import AgentProfile, DAGState
except ModuleNotFoundError as exc:
    HAS_MAS_GFLOWOPT_DEPS = False
    IMPORT_ERROR = str(exc)
    BatchLLMExecutionMASTaskEvaluator = object  # type: ignore[assignment]


@unittest.skipUnless(HAS_MAS_GFLOWOPT_DEPS, f"mas_gflowopt test dependencies missing: {IMPORT_ERROR}")
class FakeBatchEvaluator(BatchLLMExecutionMASTaskEvaluator):
    def __init__(self, agent_pool: AgentPool):
        cfg = LLMExecutionConfig(model="fake-model", judge_model="fake-model", batch_max_workers=2)
        super().__init__(agent_pool=agent_pool, config=cfg, max_workers=2)
        self.calls: list[list[dict[str, str]]] = []

    def _post_chat(self, messages, model, temperature, max_tokens):
        self.calls.append(messages)
        system_content = messages[0].get("content", "") if messages else ""
        if "JSON对象" in system_content:
            return '{"task_score": 0.9, "success": 1, "safety_penalty": 0.0}'
        return "ok"


@unittest.skipUnless(HAS_MAS_GFLOWOPT_DEPS, f"mas_gflowopt test dependencies missing: {IMPORT_ERROR}")
class BatchEvaluatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.agent_pool = AgentPool(
            agents=[
                AgentProfile(
                    agent_id="a",
                    role="A",
                    profile="agent a",
                    system_prompt="a",
                ),
                AgentProfile(
                    agent_id="b",
                    role="B",
                    profile="agent b",
                    system_prompt="b",
                ),
            ]
        )
        self.dag = DAGState(nodes=["a", "b"], edges=[])

    def test_batch_evaluate_returns_results(self) -> None:
        evaluator = FakeBatchEvaluator(self.agent_pool)
        results = evaluator.evaluate_batch([self.dag, self.dag], question_texts=["q1", "q2"])
        self.assertEqual(len(results), 2)
        self.assertAlmostEqual(results[0].task_score, 0.9)
        self.assertAlmostEqual(results[1].success, 1.0)

    def test_batch_evaluate_validates_lengths(self) -> None:
        evaluator = FakeBatchEvaluator(self.agent_pool)
        with self.assertRaises(ValueError):
            evaluator.evaluate_batch([self.dag], question_texts=["q1", "q2"])


if __name__ == "__main__":
    unittest.main()
