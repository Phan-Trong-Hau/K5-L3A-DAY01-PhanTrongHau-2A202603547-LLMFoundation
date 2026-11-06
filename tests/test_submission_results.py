"""Regression checks for missing labels, API failures and exported evidence."""
import ast
import json
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def functions_from_notebook():
    notebook = json.loads((ROOT / "src/day01_spam_ml_dl_embedding_llm.ipynb").read_text())
    wanted = {"parse_label", "llm_classify_batch", "llm_stats", "json_safe", "sigmoid"}
    namespace = {"np": np, "time": time, "re": __import__("re"),
                 "OPENAI_MINI_MODEL": "unit-test-model", "CLASSIFY_MAX_TOKENS": 64,
                 "LLM_REQUEST_DELAY": 0.0, "PRICE_PER_1M": {"unit-test-model": (0.1, 0.5)},
                 "build_classification_prompt": lambda text, examples: text,
                 "retry_with_backoff": lambda fn: fn()}
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "\n".join(line for line in "".join(cell["source"]).splitlines()
                           if not line.lstrip().startswith(("%", "!")))
        for node in ast.parse(source).body:
            if isinstance(node, ast.FunctionDef) and node.name in wanted:
                exec(compile(ast.Module(body=[node], type_ignores=[]), "notebook", "exec"), namespace)
    return namespace


class SubmissionResultsTest(unittest.TestCase):
    def setUp(self):
        self.ns = functions_from_notebook()

    def test_all_failed_batch_has_no_quality_or_speed_measurement(self):
        self.ns["call_llm"] = MagicMock(side_effect=RuntimeError("missing configuration"))
        r = self.ns["llm_classify_batch"](["one", "two"], model="unit-test-model")
        self.assertEqual(r["n_invalid"], 2)
        self.assertEqual(r["attempted"], 1)
        self.assertEqual(r["records"][1]["error_type"], "NotAttemptedAfterFatalError")
        self.assertGreater(r["latency_s"], 0)
        stats = self.ns["llm_stats"](r)
        self.assertIsNone(stats["p95_ms"])
        self.assertIsNone(stats["msg_per_s"])
        self.assertIsNone(stats["usd_per_1k"])

    def test_successful_batch_has_real_labels_and_timing(self):
        self.ns["call_llm"] = MagicMock(side_effect=[("spam", 0.01), ("Not spam", 0.01)])
        r = self.ns["llm_classify_batch"](["one", "two"], model="unit-test-model")
        self.assertEqual(r["preds"].tolist(), [1, 0])
        self.assertEqual(r["n_invalid"], 0)
        self.assertGreater(self.ns["llm_stats"](r)["msg_per_s"], 0)

    def test_invalid_response_is_not_silently_classified_as_ham(self):
        self.ns["call_llm"] = MagicMock(return_value=("I do not know", 0.01))
        r = self.ns["llm_classify_batch"](["one"], model="unit-test-model")
        self.assertTrue(np.isnan(r["preds"][0]))
        self.assertEqual(r["records"][0]["error_type"], "InvalidLabel")

    def test_authentication_error_stops_without_retrying_all_messages(self):
        error = RuntimeError("redacted")
        error.status_code = 401
        self.ns["call_llm"] = MagicMock(side_effect=error)
        r = self.ns["llm_classify_batch"](["one", "two", "three"], model="unit-test-model")
        self.assertEqual(self.ns["call_llm"].call_count, 1)
        self.assertEqual(r["records"][0]["http_status"], 401)

    def test_json_serializer_handles_nested_nonfinite_numpy_values(self):
        data = {"x": np.array([np.nan, np.inf, -np.inf, 1.0]), "n": np.int64(4)}
        clean = self.ns["json_safe"](data)
        self.assertEqual(clean, {"x": [None, None, None, 1.0], "n": 4})
        json.dumps(clean, allow_nan=False)

    def test_sigmoid_extreme_values_are_finite(self):
        with np.errstate(over="raise", invalid="raise"):
            result = self.ns["sigmoid"](np.array([-1e6, 0, 1e6]))
        np.testing.assert_allclose(result, [0, 0.5, 1])

    def test_notebooks_and_strict_export_are_consistent(self):
        source = ROOT / "src/day01_spam_ml_dl_embedding_llm.ipynb"
        submission = ROOT / "report/individual/day01_lab.ipynb"
        self.assertEqual(source.read_bytes(), submission.read_bytes())
        def reject(value):
            raise ValueError(f"Non-standard JSON constant: {value}")
        report = json.loads((ROOT / "report/individual/lab1_report.json").read_text(), parse_constant=reject)
        self.assertEqual(report["seed"], 4388)
        self.assertEqual(set(report["challenge_results"]), {"4", "512"})
        self.assertEqual(report["hard_eval"]["n_cases"], 50)
        self.assertEqual(len(report["hard_eval"]["cases"]), 50)
        for result in report["results"].values():
            if result.get("status") != "ok" and result.get("status") is not None:
                self.assertIsNone(result["f1"])
                self.assertIsNone(result["p95_ms"])


if __name__ == "__main__":
    unittest.main()
