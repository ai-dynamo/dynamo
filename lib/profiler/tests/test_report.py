# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for report_generator — verify HTML structure from demo data."""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dynamo_profiler import report_generator


DEMO_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "sysprofile-demo-output"
)


class TestReportGeneratorEmpty(unittest.TestCase):
    def test_empty_report_produces_valid_html(self):
        html = report_generator.generate_report()
        self.assertIn("<!DOCTYPE html>", html)
        self.assertIn("</html>", html)

    def test_report_with_no_merge_result_still_renders(self):
        html = report_generator.generate_report(merge_result=None)
        self.assertIn("<!DOCTYPE html>", html)


class TestEscaping(unittest.TestCase):
    def test_esc_html_entities(self):
        self.assertEqual(report_generator._esc("<script>"), "&lt;script&gt;")
        self.assertEqual(report_generator._esc('x"y'), "x&quot;y")
        self.assertEqual(report_generator._esc("a&b"), "a&amp;b")


@unittest.skipUnless(
    os.path.isdir(DEMO_DIR), "sysprofile-demo-output not found"
)
class TestReportWithDemoData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(os.path.join(DEMO_DIR, "merge_result.json")) as f:
            cls.merge_result = json.load(f)

    def test_all_dep_sections_present(self):
        html = report_generator.generate_report(merge_result=self.merge_result)
        for marker in [
            "p99 Critical-Path Attribution",
            "Top-10 Slowest Requests",
            "View A",
            "View B",
            "View C",
            "View D",
            "Causality DAG",
        ]:
            self.assertIn(marker, html, f"Missing section: {marker}")

    def test_plotly_charts_present(self):
        html = report_generator.generate_report(merge_result=self.merge_result)
        self.assertEqual(html.count("Plotly.newPlot"), 3)
        self.assertIn("heatmap-a", html)
        self.assertIn("heatmap-b", html)
        self.assertIn("dag-d", html)

    def test_kpi_cards_have_values(self):
        html = report_generator.generate_report(merge_result=self.merge_result)
        self.assertIn("45.4", html)  # p99 TTFT
        self.assertIn("27.9", html)  # p50 TTFT
        self.assertIn("40", html)    # request count

    def test_critical_path_bar_present(self):
        html = report_generator.generate_report(merge_result=self.merge_result)
        self.assertIn("cp-bar", html)
        self.assertIn("cp-seg", html)
        self.assertIn("prefill.compute", html)

    def test_deep_analysis_sections_graceful_without_data(self):
        html = report_generator.generate_report(merge_result=self.merge_result)
        self.assertNotIn("Deep Analysis", html)

    def test_deep_analysis_with_stage_attr(self):
        stage_attr = {
            "report": {
                "per_stage_percentiles": {
                    "tokenize": {"p50": 1.0, "p95": 2.0, "p99": 3.0},
                    "prefill_compute": {"p50": 10.0, "p95": 15.0, "p99": 20.0},
                },
                "stage_order": ["tokenize", "prefill_compute"],
            }
        }
        html = report_generator.generate_report(
            merge_result=self.merge_result,
            stage_attr=stage_attr,
        )
        self.assertIn("Deep Analysis", html)
        self.assertIn("Per-Stage Latency Distribution", html)
        self.assertIn("tokenize", html)
        self.assertIn("prefill_compute", html)


if __name__ == "__main__":
    unittest.main()
