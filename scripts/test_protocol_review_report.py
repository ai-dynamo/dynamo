# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import unittest

from protocol_review_report import payload, reference, render_review


class ReviewReportTests(unittest.TestCase):
    def setUp(self):
        self.report = {
            "previous_upstream_commit": "a" * 40,
            "candidate_upstream_commit": "b" * 40,
            "dynamo_commit": "c" * 40,
            "generated_at": "2026-10-04T00:00:00Z",
            "endpoints": ["/v1/chat/completions", "/v1/completions"],
            "platforms": ["cpu", "xpu"],
            "limitations": ["Static source only."],
            "triage_complete": False,
            "triage": "missing: decision file",
            "changes": [
                {
                    "id": "test-id",
                    "kind": "added",
                    "path": ["vllm/protocol.py", "classes", "Request", "fields", "new"],
                    "after": {"annotation": "bool", "default": "True"},
                    "status": "Unverified",
                    "next_step": "Run a probe",
                }
            ],
        }

    def render(self, gate="triage"):
        return render_review([("pair.json", self.report)], gate)

    def test_complete_developer_workflow_and_source_links(self):
        result = self.render()
        for expected in [
            "Command gate: FAIL",
            "Triage coverage: incomplete",
            "cpu, xpu",
            "test-id",
            "Before",
            "Not present.",
            '"annotation": "bool"',
            "Owner: Not recorded",
            "Runtime evidence: Not recorded",
            "Next action: Run a probe",
            "[Full JSON evidence](pair.json)",
            f"https://github.com/vllm-project/vllm/blob/{'b' * 40}/vllm/protocol.py",
        ]:
            self.assertIn(expected, result)
        self.assertNotIn("Before source", result)

    def test_recorded_unresolved_disposition_can_pass_without_parity(self):
        self.report["triage_complete"] = True
        self.report["changes"][0].update(
            owner="frontend",
            rationale="Needs testing",
            tracking_issue="https://example.com/issues/1",
        )
        result = self.render()
        self.assertIn("Command gate: PASS", result)
        self.assertIn("Unverified: 1", result)
        self.assertIn(
            "not that humans approved them or runtime compatibility was proven", result
        )
        self.assertIn("https://example.com/issues/1", result)

    def test_no_drift_and_unenforced_gate(self):
        self.assertIn("Command gate: Not enforced", self.render("none"))
        self.report["changes"] = []
        self.report["triage_complete"] = True
        result = self.render()
        self.assertIn("Command gate: PASS", result)
        self.assertIn("No source changes detected in scanned modules", result)

    def test_non_material_and_runtime_evidence(self):
        item = self.report["changes"][0]
        item.update(
            material=False,
            scope_endpoints=self.report["endpoints"],
            decision="No action",
            evidence=["relative/source.py", "https://example.com/source"],
        )
        item.pop("status")
        result = self.render()
        self.assertIn("Non-material for recorded scope", result)
        self.assertIn("Recorded scope", result)
        self.assertIn("relative/source.py", result)
        self.assertNotIn("Unverified", result)
        item.pop("material")
        item.update(status="Compatible", runtime_evidence=["https://example.com/run"])
        self.assertIn("Runtime evidence: [https://example.com/run]", self.render())

    def test_payloads_preserve_missing_null_and_fences(self):
        self.assertEqual(payload({}, "before"), "Not present.\n")
        self.assertIn("\nnull\n", payload({"before": None}, "before"))
        self.assertTrue(
            payload({"before": "```\n<script>"}, "before").startswith("````json")
        )

    def test_decision_text_is_inert_and_relative_or_unsafe_urls_not_linked(self):
        self.report["changes"][0][
            "rationale"
        ] = "<script>\n[claim](javascript:alert(1)) | fake"
        result = self.render()
        self.assertNotIn("<script>", result)
        self.assertNotIn("[claim](javascript", result)
        self.assertNotIn("](javascript", reference("javascript:alert(1)"))
        self.assertNotIn("](", reference("../evidence.json"))
        self.assertNotIn("](", reference("https://[malformed"))

    def test_large_payloads_are_expandable_without_dropping_evidence(self):
        value = "long " * 1000
        result = payload({"after": value}, "after")
        self.assertIn("<details>", result)
        self.assertIn(value, result)
        self.assertTrue(result.endswith("</details>\n"))

    def test_multiple_pairs_and_stable_nonmutating_output(self):
        original = copy.deepcopy(self.report)
        results = [("one.json", self.report), ("two.json", self.report)]
        output = render_review(results, "triage")
        self.assertIn("2 source-change candidates across 2 comparison pairs", output)
        self.assertIn('id="comparison-2-change-1"', output)
        self.assertEqual(output, render_review(results, "triage"))
        self.assertEqual(original, self.report)

    def test_incomplete_coverage_gate_even_without_source_changes(self):
        self.report.update(
            changes=[], triage_complete=True, require_complete_coverage=True
        )
        coverage = {
            "complete": False,
            "scope": "Repository-local types",
            "module_consumers": {},
            "primitive_leaves": [],
            "unresolved": [
                {
                    "source": "protocol.py",
                    "symbol": "sdk.Model",
                    "reason": "external dependency",
                    "roots": ["Request.value"],
                }
            ],
        }
        self.report["dependency_coverage"] = {"before": coverage, "after": coverage}
        result = self.render("none")
        self.assertIn("Command gate: FAIL", result)
        self.assertIn("Triage coverage: complete", result)
        self.assertIn("external dependency", result)
        self.assertIn("1 unresolved references", result)


if __name__ == "__main__":
    unittest.main()
