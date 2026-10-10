# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hermetic assessment semantics; these fixtures are not runtime parity evidence."""

from __future__ import annotations

import copy
import unittest

from scripts.protocol_compatibility.assessment.policy import build_assessment
from scripts.protocol_compatibility.common.contracts import (
    Contract,
    CoverageDiagnostic,
    EndpointContract,
    FieldContract,
    Handling,
)

ENDPOINT = "/v1/chat/completions"


def contracts():
    native_field = FieldContract(
        "temperature",
        ["temperature"],
        {"type": "number"},
        False,
        False,
        {"kind": "value", "value": 1.0},
        {},
    )
    local_field = copy.deepcopy(native_field)
    local_field.handling = Handling(["interpret", "forward"], complete=True)
    return (
        Contract(
            "vllm",
            "a" * 40,
            {ENDPOINT: EndpointContract({"temperature": native_field}, True)},
        ),
        Contract(
            "dynamo",
            "b" * 40,
            {ENDPOINT: EndpointContract({"temperature": local_field}, True)},
        ),
    )


def assessment(native, dynamo, **kwargs):
    return build_assessment(
        native, dynamo, scope={"backend": "vllm", "pipeline": "token"}, **kwargs
    )


def registry(report, **overrides):
    return {
        "schema": "dynamo-native-decisions/v2",
        "layer": "contract",
        "decisions": [
            {
                "identity": item["identity"],
                "fingerprint": item["fingerprint"],
                "scope": report["scope"],
                "reviewed_revisions": report["revisions"],
                "disposition": "tracked_gap",
                "owner": "frontend",
                "rationale": "Fixture divergence.",
                "tracking": "https://example.org/issues/1",
                "next_action": "Align default.",
                "evidence": [{"kind": "source", "url": "https://example.org/source"}],
                **overrides,
            }
            for item in report["findings"]
        ],
    }


class ComparisonTests(unittest.TestCase):
    def test_contract_complete_without_handling_or_behavior_tests(self):
        native, dynamo = contracts()
        dynamo.endpoints[ENDPOINT].fields["temperature"].handling = None
        dynamo.endpoints[ENDPOINT].additional_properties = Handling()
        report = assessment(native, dynamo)
        self.assertEqual(report["exit_code"], 0)
        self.assertEqual(report["behavioral_conformance"]["status"], "not_assessed")
        self.assertEqual(report["investigation"]["status"], "not_implemented")
        self.assertEqual(report["investigation"]["notes"], [])

    def test_contract_only_preserves_prior_advisory_history(self):
        native, dynamo = contracts()
        previous = assessment(native, dynamo)
        note = {"identity": "old-note", "fingerprint": "a" * 64}
        previous["investigation"]["notes"] = [note]
        current = assessment(native, dynamo, previous=previous)
        self.assertEqual(current["investigation"]["notes"], [])
        self.assertEqual(current["investigation"]["unobserved_notes"], [note])
        self.assertEqual(current["gates"], previous["gates"])

    def test_missing_schema_blocks_even_with_reviewed_disposition(self):
        native, dynamo = contracts()
        dynamo.endpoints[ENDPOINT].fields["temperature"].wire_type = None
        first = assessment(native, dynamo)
        report = assessment(native, dynamo, decisions=registry(first))
        self.assertEqual(report["gates"]["review"], "complete")
        self.assertEqual(report["gates"]["extraction"], "incomplete")
        self.assertEqual(report["exit_code"], 1)

    def test_legacy_report_migration_preserves_notes_without_approval(self):
        from dataclasses import asdict

        from scripts.protocol_compatibility.common.contracts import finding

        native, dynamo = contracts()
        dynamo.endpoints[ENDPOINT].fields["temperature"].default = {
            "kind": "value",
            "value": 0.5,
        }
        old = assessment(native, dynamo)
        old["schema"] = "dynamo-native-assessment/v1"
        old["findings"][0]["dynamo"]["handling"] = {"effects": ["forward"]}
        old["findings"].append(
            asdict(
                finding(
                    ENDPOINT,
                    "temperature",
                    "handling",
                    "Unknown old handling",
                    {},
                    {},
                    category="coverage",
                )
            )
        )
        old["findings"][0]["decision_status"] = "applicable"
        report = assessment(native, dynamo, previous=old)
        self.assertEqual(report["findings"][0]["lifecycle"], "unchanged")
        self.assertEqual(report["findings"][0]["decision_status"], "missing")
        self.assertEqual(report["retired_findings"], [])
        self.assertEqual(len(report["history_migration"]["legacy_investigation"]), 1)
        self.assertTrue(
            all(
                not item["approval_carried"]
                for item in report["history_migration"]["mappings"]
            )
        )
        legacy = registry(report)
        legacy["schema"] = "dynamo-native-decisions/v1"
        with self.assertRaisesRegex(ValueError, "re-review"):
            assessment(native, dynamo, decisions=legacy)

    def test_lost_nested_coverage_does_not_resolve_placement_candidate(self):
        native, dynamo = contracts()
        dynamo.endpoints[ENDPOINT].fields = {}
        dynamo.endpoints[ENDPOINT].nested_inputs = [
            {"wire_path": ["nvext", "temperature"], "source": [], "declaration": {}}
        ]
        previous = assessment(native, dynamo)
        dynamo.endpoints[ENDPOINT].nested_inputs = []
        dynamo.diagnostics = [
            CoverageDiagnostic(
                ENDPOINT, "nvext", "structural", "unresolved nested schema"
            )
        ]
        current = assessment(native, dynamo, previous=previous)
        retired = next(
            item
            for item in current["retired_findings"]
            if item["aspect"] == "placement_candidate"
        )
        self.assertEqual(retired["lifecycle"], "no_longer_assessable")

    def test_nested_same_name_does_not_override_existing_root_input(self):
        native, dynamo = contracts()
        dynamo.endpoints[ENDPOINT].nested_inputs = [
            {"wire_path": ["nvext", "temperature"], "source": [], "declaration": {}}
        ]
        report = assessment(native, dynamo)
        self.assertEqual(report["findings"], [])

    def test_previous_scope_and_malformed_gate_inputs_rejected(self):
        native, dynamo = contracts()
        report = assessment(native, dynamo)
        report["scope"]["pipeline"] = "text"
        with self.assertRaisesRegex(ValueError, "scope differs"):
            assessment(native, dynamo, previous=report)
        for policy in (
            [],
            {
                "schema": "dynamo-native-support-policy/v1",
                "required_findings_absent": "field",
            },
        ):
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                assessment(native, dynamo, policy=policy)

    def test_stale_facts_do_not_retain_current_runtime_evidence(self):
        native, dynamo = contracts()
        dynamo.endpoints[ENDPOINT].fields = {}
        report = assessment(native, dynamo)
        decisions = registry(
            report,
            evidence=[
                {
                    "kind": "runtime",
                    "url": "https://example.org/run/1",
                    "revisions": report["revisions"],
                    "scope": report["scope"],
                }
            ],
        )
        decisions["decisions"][0]["fingerprint"] = "0" * 64
        changed = assessment(native, dynamo, decisions=decisions)
        item = changed["findings"][0]
        self.assertEqual(item["decision_status"], "stale_facts")
        self.assertEqual(item["decision"]["current_runtime_evidence"], [])

    def test_untyped_source_rejection_does_not_fill_contract_coverage(self):
        native, dynamo = contracts()
        endpoint = dynamo.endpoints[ENDPOINT]
        endpoint.fields = {}
        endpoint.additional_properties = Handling()
        endpoint.untyped_handling["temperature"] = Handling(["reject"], complete=True)
        report = assessment(native, dynamo)
        self.assertEqual(report["findings"][0]["category"], "coverage")
        self.assertEqual(report["findings"][0]["aspect"], "input_slot")
        self.assertEqual(report["investigation"]["notes"], [])

    def test_baseline_finds_old_gap_even_without_upstream_change(self):
        native, dynamo = contracts()
        dynamo.endpoints[ENDPOINT].fields = {}
        report = assessment(native, dynamo)
        self.assertEqual(report["findings"][0]["aspect"], "input_slot")
        self.assertEqual(report["exit_code"], 1)
        again = assessment(native, dynamo, previous=report)
        self.assertEqual(again["findings"][0]["lifecycle"], "unchanged")

    def test_dynamo_only_change_detected_and_revision_pairs_recorded(self):
        native, dynamo = contracts()
        old = assessment(native, dynamo)
        dynamo.revision = "c" * 40
        dynamo.endpoints[ENDPOINT].fields["temperature"].default = {
            "kind": "value",
            "value": 0.7,
        }
        new = assessment(native, dynamo, previous=old)
        self.assertEqual(new["previous_revisions"], old["revisions"])
        self.assertEqual(new["revisions"]["vllm"], old["revisions"]["vllm"])
        self.assertEqual(new["findings"][0]["aspect"], "default")
        self.assertEqual(new["findings"][0]["lifecycle"], "new")

    def test_all_structural_differences_are_observations(self):
        for aspect, value in (
            ("wire_names", ["nvext.temperature"]),
            ("wire_type", {"type": "integer"}),
            ("required", True),
            ("nullable", True),
            ("default", {"kind": "value", "value": 2}),
            ("constraints", {"le": 1}),
        ):
            with self.subTest(aspect=aspect):
                native, dynamo = contracts()
                setattr(dynamo.endpoints[ENDPOINT].fields["temperature"], aspect, value)
                report = assessment(native, dynamo)
                self.assertEqual(len(report["findings"]), 1)
                self.assertIsNone(report["findings"][0]["decision"])

    def test_added_native_field_and_dynamo_specific_not_confused(self):
        native, dynamo = contracts()
        dynamo.endpoints[ENDPOINT].fields["nvext"] = FieldContract("nvext", ["nvext"])
        report = assessment(native, dynamo)
        self.assertEqual(report["findings"][0]["category"], "dynamo_specific")
        self.assertEqual(report["gates"]["review"], "complete")
        self.assertEqual(report["gates"]["extraction"], "incomplete")

    def test_handling_effects_and_unknown_passthrough(self):
        for effects in (["interpret"], ["forward"], ["interpret", "forward"]):
            native, dynamo = contracts()
            dynamo.endpoints[ENDPOINT].fields["temperature"].handling = Handling(
                effects, complete=True
            )
            report = assessment(native, dynamo)
            self.assertEqual(report["findings"], [])
            self.assertEqual(
                report["gates"]["runtime_conformance"],
                "not_established_by_static_assessment",
            )
        native, dynamo = contracts()
        dynamo.endpoints[ENDPOINT].fields = {}
        dynamo.endpoints[ENDPOINT].additional_properties = Handling(
            ["forward"], complete=False
        )
        report = assessment(native, dynamo)
        self.assertEqual(report["findings"][0]["category"], "coverage")
        self.assertIn("passthrough", report["findings"][0]["observation"])

    def test_source_rejection_is_not_a_contract_verdict(self):
        native, dynamo = contracts()
        dynamo.endpoints[ENDPOINT].fields["temperature"].handling = Handling(
            ["reject"],
            [{"stream": True}],
            complete=True,
        )
        report = assessment(native, dynamo)
        self.assertEqual(report["findings"], [])
        self.assertEqual(report["investigation"]["notes"], [])
        self.assertEqual(report["exit_code"], 0)
        self.assertEqual(report["behavioral_conformance"]["status"], "not_assessed")

    def test_static_decision_matches_facts_and_survives_unrelated_commit_opt_in(self):
        native, dynamo = contracts()
        dynamo.endpoints[ENDPOINT].fields = {}
        first = assessment(native, dynamo)
        decisions = registry(first, carry_static_disposition=True)
        dynamo.revision = "c" * 40
        second = assessment(native, dynamo, previous=first, decisions=decisions)
        self.assertEqual(second["findings"][0]["lifecycle"], "unchanged")
        self.assertEqual(second["findings"][0]["decision_status"], "applicable")
        self.assertEqual(second["gates"]["review"], "complete")

    def test_contract_decision_is_independent_of_handling_change(self):
        for change in ("default", "implementation"):
            with self.subTest(change=change):
                native, dynamo = contracts()
                item = dynamo.endpoints[ENDPOINT].fields["temperature"]
                item.default = {"kind": "value", "value": 0.7}
                old = assessment(native, dynamo)
                decisions = registry(old)
                if change == "default":
                    item.default = {"kind": "value", "value": 0.8}
                else:
                    item.handling.evidence = [{"semantic_sha256": "different body"}]
                report = assessment(native, dynamo, previous=old, decisions=decisions)
                self.assertEqual(
                    report["findings"][0]["lifecycle"],
                    "changed" if change == "default" else "unchanged",
                )
                self.assertEqual(
                    report["findings"][0]["decision_status"],
                    "stale_facts" if change == "default" else "applicable",
                )

    def test_runtime_evidence_never_carried_between_revisions(self):
        native, dynamo = contracts()
        dynamo.endpoints[ENDPOINT].fields = {}
        first = assessment(native, dynamo)
        decisions = registry(
            first,
            carry_static_disposition=True,
            evidence=[
                {
                    "kind": "runtime",
                    "revisions": first["revisions"],
                    "scope": first["scope"],
                    "url": "https://example.org/run/1",
                }
            ],
        )
        same = assessment(native, dynamo, decisions=decisions)
        self.assertEqual(
            len(same["findings"][0]["decision"]["current_runtime_evidence"]), 1
        )
        dynamo.revision = "c" * 40
        changed = assessment(native, dynamo, decisions=decisions)
        self.assertEqual(
            changed["findings"][0]["decision"]["current_runtime_evidence"], []
        )

    def test_lost_coverage_not_resolved_even_across_multiple_assessments(self):
        native, dynamo = contracts()
        dynamo.endpoints[ENDPOINT].fields["temperature"].default = {
            "kind": "value",
            "value": 0.7,
        }
        first = assessment(native, dynamo)
        dynamo.endpoints[ENDPOINT].fields = {}
        dynamo.endpoints[ENDPOINT].fields_complete = False
        second = assessment(native, dynamo, previous=first)
        self.assertEqual(
            second["retired_findings"][0]["lifecycle"], "no_longer_assessable"
        )
        third = assessment(native, dynamo, previous=second)
        self.assertEqual(
            third["retired_findings"][0]["lifecycle"], "no_longer_assessable"
        )
        self.assertEqual(third["gates"]["review"], "action_required")

    def test_resolved_when_contract_fixed_and_coverage_retained(self):
        native, dynamo = contracts()
        dynamo.endpoints[ENDPOINT].fields["temperature"].default = {
            "kind": "value",
            "value": 0.7,
        }
        first = assessment(native, dynamo)
        dynamo.endpoints[ENDPOINT].fields["temperature"].default = {
            "kind": "value",
            "value": 1.0,
        }
        second = assessment(native, dynamo, previous=first)
        self.assertEqual(second["retired_findings"][0]["lifecycle"], "resolved")

    def test_reviewed_gap_can_still_block_release_policy(self):
        native, dynamo = contracts()
        dynamo.endpoints[ENDPOINT].fields = {}
        first = assessment(native, dynamo)
        policy = {
            "schema": "dynamo-native-support-policy/v1",
            "required_findings_absent": [first["findings"][0]["identity"]],
        }
        report = assessment(native, dynamo, decisions=registry(first), policy=policy)
        self.assertEqual(report["gates"]["review"], "complete")
        self.assertEqual(report["gates"]["release"]["status"], "blocked")
        self.assertEqual(report["exit_code"], 1)

    def test_old_b1_decisions_and_reports_rejected(self):
        native, dynamo = contracts()
        with self.assertRaises(ValueError):
            assessment(native, dynamo, decisions={"format_version": 1})
        with self.assertRaises(ValueError):
            assessment(native, dynamo, previous={"format_version": 1})

    def test_same_field_source_move_does_not_invalidate(self):
        native, dynamo = contracts()
        dynamo.endpoints[ENDPOINT].fields = {}
        old = assessment(native, dynamo)
        native.endpoints[ENDPOINT].fields["temperature"].source = [
            {"path": "moved.py", "line": 42}
        ]
        new = assessment(native, dynamo, previous=old, decisions=registry(old))
        self.assertEqual(new["findings"][0]["decision_status"], "applicable")


if __name__ == "__main__":
    unittest.main()
