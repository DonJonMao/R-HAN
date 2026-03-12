import json
import unittest

from mas_treesearch.data import standardize_record
from mas_treesearch.profiles import resolve_dataset_profile


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


if __name__ == "__main__":
    unittest.main()
