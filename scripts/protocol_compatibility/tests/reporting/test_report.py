# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import unittest

from scripts.protocol_compatibility.assessment.upstream_changes import (
    behavior_findings,
    selected_behavior,
)
from scripts.protocol_compatibility.reporting.report import (
    evidence_link,
    fact_summary,
    render_assessment,
)
from scripts.protocol_compatibility.tests.assessment.test_comparison import (
    assessment,
    contracts,
    registry,
)


class AssessmentReportTests(unittest.TestCase):
    def test_nested_locations_are_readable_without_expanding_inventory_facts(self):
        value = {"nested_inputs": [{"wire_path": ["nvext", "literal.key/~"]}]}
        self.assertEqual(
            fact_summary(value, "placement_candidate"),
            "nested public locations (JSON Pointer): /nvext/literal.key~1~0",
        )

    def test_large_facts_expand_and_evidence_is_clickable(self):
        native, dynamo = contracts()
        dynamo.endpoints["/v1/chat/completions"].fields = {}
        first = assessment(native, dynamo)
        report = assessment(native, dynamo, decisions=registry(first))
        rendered = render_assessment(report)
        self.assertIn("<details>", rendered)
        self.assertIn("temperature: number; required=False", rendered)
        self.assertIn(
            "[https://example.org/source](https://example.org/source)", rendered
        )
        self.assertIn(first["findings"][0]["fingerprint"], rendered)

    def test_evidence_cannot_inject_executable_markdown_link(self):
        self.assertNotIn("](javascript:", evidence_link("javascript:alert(1)"))
        self.assertNotIn("](//", evidence_link("//evil.example"))

    def test_readable_baseline_and_gate_distinctions(self):
        native, dynamo = contracts()
        dynamo.endpoints["/v1/chat/completions"].fields = {}
        report = assessment(native, dynamo)
        original = copy.deepcopy(report)
        rendered = render_assessment(report)
        self.assertIn("No Dynamo input or handling path identified", rendered)
        self.assertIn("Initial baseline", rendered)
        self.assertIn("Runtime conformance", rendered)
        self.assertIn("do **not** establish runtime parity", rendered)
        self.assertIn(native.revision, rendered)
        self.assertIn(dynamo.revision, rendered)
        self.assertEqual(report, original)

    def test_source_payload_is_escaped(self):
        native, dynamo = contracts()
        native.endpoints["/v1/chat/completions"].fields["temperature"].default = {
            "kind": "value",
            "value": "<script>alert(1)</script> [click](javascript:bad)",
        }
        output = render_assessment(assessment(native, dynamo))
        self.assertNotIn("<script>", output)
        self.assertIn("&lt;script&gt;", output)

    def test_validator_body_change_with_identical_fields_is_selected(self):
        snapshot = {
            "modules": {
                "vllm/entrypoints/openai/protocol.py": {
                    "contract": {
                        "classes": {
                            "ChatCompletionRequest": {
                                "fields": {"n": {}},
                                "methods": {"validate_n": "old"},
                            }
                        },
                        "functions": {"unrelated_helper": "old"},
                    }
                }
            }
        }
        baseline = selected_behavior(snapshot)
        snapshot["modules"]["vllm/entrypoints/openai/protocol.py"]["contract"][
            "classes"
        ]["ChatCompletionRequest"]["methods"]["validate_n"] = "new"
        current = selected_behavior(snapshot)
        findings = behavior_findings(
            current, {"selected_behavior": baseline, "findings": []}
        )
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].category, "behavior")
        self.assertEqual(findings[0].native["affected_fields"], ["n"])
        self.assertNotIn("unrelated_helper", str(current))


if __name__ == "__main__":
    unittest.main()
