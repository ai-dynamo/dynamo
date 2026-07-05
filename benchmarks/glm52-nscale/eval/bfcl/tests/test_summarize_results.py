# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "summarize_results.py"
SPEC = importlib.util.spec_from_file_location("summarize_results", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
summarize_results = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summarize_results)


class OverallRowTests(unittest.TestCase):
    def test_requires_one_exact_campaign_model_row(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            score = run_dir / "score"
            score.mkdir()
            path = score / "data_overall.csv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=("Model", "Overall Acc"))
                writer.writeheader()
                writer.writerow(
                    {
                        "Model": summarize_results.OFFICIAL_MODEL,
                        "Overall Acc": "73.05%",
                    }
                )
            self.assertEqual(
                summarize_results.overall_row(run_dir)["Overall Acc"], "73.05%"
            )

            path.write_text("Model,Overall Acc\nGLM-5.2 impostor,99.00%\n")
            with self.assertRaisesRegex(ValueError, "exactly one exact"):
                summarize_results.overall_row(run_dir)


if __name__ == "__main__":
    unittest.main()
