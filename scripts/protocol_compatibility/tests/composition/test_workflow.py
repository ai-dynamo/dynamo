# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check assessment outputs and exit status independently of oasdiff behavior.

Synthetic inputs and mocked comparator results exercise file retention, alias
probes, response evidence, and failure reporting. tests/integration/test_oasdiff.py
covers the real executable; these tests do not establish comparator correctness.
"""

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml

from scripts.protocol_compatibility.__main__ import main
from scripts.protocol_compatibility.assessment.openapi_workflow import assess
from scripts.protocol_compatibility.inputs import write_json
from scripts.protocol_compatibility.tests.assessment import test_openapi as fixtures
from scripts.protocol_compatibility.tests.assessment.test_openapi import document
from scripts.protocol_compatibility.tests.assessment.test_responses import (
    stream_selection,
    streaming_document,
)


class WorkflowTests(unittest.TestCase):
    @patch(
        "scripts.protocol_compatibility.assessment.openapi_workflow.compare",
        return_value=({}, ["mock"]),
    )
    def test_missing_sse_declaration_remains_an_evidence_gap(self, compare):
        """A JSON-only declaration must not be interpreted as proof that streaming is absent."""
        self.args.dynamo = self.root / "streaming.json"
        write_json(self.args.dynamo, streaming_document())
        self.assertEqual(assess(self.args), 1)
        report = json.loads((self.args.output_dir / "report.json").read_text())
        event_gaps = [
            gap for gap in report["coverage_gaps"] if gap.get("layer") == "event"
        ]
        self.assertEqual(len(event_gaps), 2)
        self.assertEqual({gap["side"] for gap in event_gaps}, {"framework"})

    @patch(
        "scripts.protocol_compatibility.assessment.openapi_workflow.compare",
        return_value=({}, ["mock"]),
    )
    def test_chunk_only_workflow_keeps_event_evidence_and_layer_gaps(self, compare):
        """Chunk agreement never becomes whole-event agreement; retain hashes and mappings."""
        native = streaming_document()
        native["components"]["schemas"]["Response"]["properties"]["choices"] = {
            "type": "array",
            "items": {"type": "object"},
        }
        chunk = copy.deepcopy(native)
        chunk["openapi"] = "3.1.0"
        chunk["paths"] = {
            key.removeprefix("/v1"): value for key, value in chunk["paths"].items()
        }
        for path in chunk["paths"].values():
            path["post"]["responses"]["200"]["content"]["text/event-stream"] = {
                "schema": {"$ref": "#/components/schemas/Response"}
            }
        settings = {"version": 1, "sources": {}}
        for side, raw, mode in (
            ("dynamo", native, "event-item-schema"),
            ("framework", chunk, "decoded-chunk"),
        ):
            path = self.root / f"{side}.json"
            write_json(path, raw)
            setattr(self.args, side, path)
            selection = stream_selection(raw, mode, chunk=side == "dynamo")
            selection["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            if side == "framework":
                selection["path_mappings"] = {key: "/v1" + key for key in raw["paths"]}
            settings["sources"][side] = selection
        self.args.response_adapters = self.root / "adapters.yaml"
        self.args.response_adapters.write_text(yaml.safe_dump(settings))
        self.assertEqual(assess(self.args), 1)
        report = json.loads((self.args.output_dir / "report.json").read_text())
        self.assertEqual(compare.call_count, 4)
        self.assertEqual(
            report["response_chunk_probes"][0]["status"],
            "no_detected_chunk_differences",
        )
        self.assertEqual(
            {gap["layer"] for gap in report["coverage_gaps"] if gap.get("layer")},
            {"envelope", "encoding", "error", "terminator", "extensions"},
        )
        self.assertEqual(
            report["response_sources"]["framework"]["streams"][0]["pointer"],
            settings["sources"]["framework"]["streams"][0]["pointer"],
        )
        full = json.loads((self.args.output_dir / "dynamo.responses.json").read_text())
        normalized = json.loads(
            (self.args.output_dir / "dynamo.responses.normalized.json").read_text()
        )
        self.assertIn(
            "properties",
            full["paths"]["/v1/chat/completions"]["post"]["responses"]["200"][
                "content"
            ]["text/event-stream"]["schema"],
        )
        self.assertEqual(
            normalized["paths"]["/v1/chat/completions"]["post"]["responses"]["200"][
                "content"
            ]["text/event-stream"]["schema"],
            {},
        )
        self.assertEqual(
            (self.args.output_dir / "framework/openapi.raw").read_bytes(),
            self.args.framework.read_bytes(),
        )
        self.assertIn(
            "not event equivalence", (self.args.output_dir / "report.md").read_text()
        )

    def test_duplicate_adapter_keys_and_unknown_sources_fail_before_output(self):
        """Configuration typos cannot silently choose a different interpretation."""
        self.args.response_adapters = self.root / "adapters.yaml"
        for content in (
            "",
            "version: 1\nsources: {framework: null}\n",
            "version: 1\nversion: 1\nsources: {}\n",
            "version: 1\nsources: {other: {}}\n",
        ):
            self.args.response_adapters.write_text(content)
            with self.subTest(content=content), self.assertRaises(ValueError):
                assess(self.args)
            self.assertFalse(self.args.output_dir.exists())

    def setUp(self):
        """Create local input files and CLI arguments in a temporary directory."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for side in ("dynamo", "framework"):
            write_json(self.root / f"{side}.json", document())
        self.baseline = self.root / "openai.yaml"
        self.baseline.write_text("components: {schemas: {}}\n")
        self.manifest = self.root / "manifest.yaml"
        self.manifest.write_text(
            yaml.safe_dump(
                {
                    "openai": {
                        "sha256": hashlib.sha256(self.baseline.read_bytes()).hexdigest()
                    },
                    "dependencies": {"fixture": "1"},
                    "imports": {},
                    "exclusions": [
                        {"field": "nvext", "reason": "Dynamo-only contents"}
                    ],
                }
            )
        )
        self.binary = self.root / "oasdiff"
        self.binary.write_bytes(b"mock binary, not executable")
        self.args = SimpleNamespace(
            dynamo=self.root / "dynamo.json",
            framework=self.root / "framework.json",
            dynamo_metadata=None,
            framework_metadata=None,
            framework_name="example-engine",
            openai=self.baseline,
            manifest=self.manifest,
            oasdiff=self.binary,
            output_dir=self.root / "report",
            response_adapters=None,
        )

    @patch("scripts.protocol_compatibility.assessment.openapi_workflow.compare")
    def test_response_only_difference_is_reported_separately_and_fails_gate(
        self, compare
    ):
        """An unchanged request diff cannot hide a response mismatch in the workflow."""
        delta = {"deleted": ["200"]}
        response_diff = {
            "paths": {
                "modified": {
                    "/v1/chat/completions": {
                        "operations": {"modified": {"POST": {"responses": delta}}}
                    }
                }
            }
        }
        compare.side_effect = [
            ({}, ["mock-request"]),
            (response_diff, ["mock-response"]),
        ]
        self.assertEqual(assess(self.args), 1)
        report = json.loads((self.args.output_dir / "report.json").read_text())
        self.assertEqual(report["differences"], [])
        self.assertEqual(report["response_differences"][0]["delta"], delta)
        self.assertEqual(report["comparator"]["response_command"], ["mock-response"])
        inventory = report["finding_inventory"]
        self.assertFalse((self.args.output_dir / "findings.json").exists())
        self.assertTrue(any(row["status"] == "200" for row in inventory["findings"]))
        self.assertEqual(
            json.loads((self.args.output_dir / "response-oasdiff.json").read_text()),
            response_diff,
        )
        self.assertIn(
            "Categories in review order",
            (self.args.output_dir / "report.md").read_text(),
        )

    @patch(
        "scripts.protocol_compatibility.assessment.openapi_workflow.compare",
        return_value=({}, ["mock"]),
    )
    def test_equal_schemas_do_not_hide_known_coverage_gaps(self, compare):
        """Retain inputs and declared gaps even when the comparator finds no differences."""
        self.assertEqual(assess(self.args), 1)
        result = json.loads((self.args.output_dir / "report.json").read_text())
        self.assertEqual(result["status"], "incomplete_coverage")
        self.assertEqual(result["framework_name"], "example-engine")
        self.assertEqual(result["differences"], [])
        self.assertEqual(result["behavioral_conformance"], "not_assessed")
        self.assertEqual(
            (self.args.output_dir / "dynamo/openapi.raw").read_bytes(),
            self.args.dynamo.read_bytes(),
        )
        self.assertTrue((self.args.output_dir / "dynamo.composed.json").exists())
        markdown = (self.args.output_dir / "report.md").read_text()
        self.assertIn("Intentional exclusions", markdown)
        self.assertIn("Coverage gaps", markdown)
        self.assertIn("Dynamo versus example-engine", markdown)
        with self.assertRaises(FileExistsError):
            assess(self.args)
        self.assertEqual(compare.call_count, 2)

    def test_openai_checksum_still_fails_before_writing(self):
        self.baseline.write_text("changed baseline")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            assess(self.args)
        self.assertFalse(self.args.output_dir.exists())

    def test_wrong_or_invalid_framework_identity_fails_before_writing(self):
        """Reject Dynamo as its own target and labels outside the CLI's identity format."""
        for name in ("dynamo", "../escape"):
            with self.subTest(name=name):
                self.args.framework_name = name
                with self.assertRaises(ValueError):
                    assess(self.args)
                self.assertFalse(self.args.output_dir.exists())

    @patch(
        "scripts.protocol_compatibility.__main__.assess",
        side_effect=AttributeError("'NoneType' object has no attribute 'get'"),
    )
    def test_unexpected_exception_is_a_tool_failure(self, assess):
        """Exit 1 means a completed assessment, so a crash must return 2."""
        argv = ["assess", "--dynamo", str(self.args.dynamo)]
        argv += ["--framework", str(self.args.framework)]
        argv += ["--framework-name", "example-engine", "--openai", str(self.baseline)]
        argv += ["--oasdiff", str(self.binary), "--output-dir", str(self.root / "out")]
        self.assertEqual(main(argv), 2)

    @patch("scripts.protocol_compatibility.__main__.assess", return_value=0)
    def test_cli_accepts_framework_identity_without_backend_specific_flags(
        self, assess
    ):
        """Forward local file annotations independently for each comparison input."""
        self.assertEqual(
            main(
                [
                    "assess",
                    "--dynamo",
                    str(self.args.dynamo),
                    "--framework",
                    str(self.args.framework),
                    "--framework-name",
                    "example-engine",
                    "--dynamo-metadata",
                    "dynamo-metadata.json",
                    "--framework-metadata",
                    "framework-metadata.json",
                    "--openai",
                    str(self.baseline),
                    "--oasdiff",
                    str(self.binary),
                    "--output-dir",
                    str(self.args.output_dir),
                ]
            ),
            0,
        )
        self.assertEqual(assess.call_args.args[0].framework_name, "example-engine")
        self.assertEqual(
            assess.call_args.args[0].dynamo_metadata, Path("dynamo-metadata.json")
        )
        self.assertEqual(
            assess.call_args.args[0].framework_metadata, Path("framework-metadata.json")
        )

    @patch(
        "scripts.protocol_compatibility.assessment.openapi_workflow.compare",
        return_value=({}, ["mock"]),
    )
    def test_file_metadata_is_retained_without_narrowing_declarations(self, compare):
        """Scope/configuration labels are evidence, not filters on the declared contract."""
        labels = {
            "source_revision": "caller claim",
            "dependencies": {"fixture": "different from manifest"},
            "scope": {"enabled_fields": ["value"]},
            "configuration": {"dispatch_events": False},
        }
        metadata = self.root / "metadata.json"
        write_json(metadata, labels)
        for side in ("dynamo", "framework"):
            setattr(self.args, side + "_metadata", metadata)
        self.assertEqual(assess(self.args), 1)
        report = json.loads((self.args.output_dir / "report.json").read_text())
        for side in ("dynamo", "framework"):
            self.assertEqual(report["inputs"][side]["caller_metadata"], labels)
            self.assertFalse(report["inputs"][side]["caller_metadata_verified"])
            projection = json.loads(
                (self.args.output_dir / f"{side}.requests.normalized.json").read_text()
            )
            self.assertEqual(
                set(projection["components"]["schemas"]["Request"]["properties"]),
                {"value", "recursive", "$ref"},
            )

    @patch(
        "scripts.protocol_compatibility.assessment.openapi_workflow.compare",
        return_value=({}, ["mock"]),
    )
    def test_ambiguous_alias_coverage_can_be_rendered(self, compare):
        """An ambiguous alias must produce a request gap, not crash report generation."""
        own, native = fixtures.AliasMatchingTests().pair()
        own["components"]["schemas"]["Request"]["properties"]["alternate"] = {}
        self.args.dynamo, self.args.framework = (
            self.root / "own.json",
            self.root / "native.json",
        )
        write_json(self.args.dynamo, own)
        write_json(self.args.framework, native)
        self.assertEqual(assess(self.args), 1)
        report = json.loads((self.args.output_dir / "report.json").read_text())
        gaps = [
            gap for gap in report["coverage_gaps"] if "Ambiguous alias" in gap["reason"]
        ]
        self.assertEqual(len(gaps), 2)
        self.assertTrue(all(gap["kind"] == "request" for gap in gaps))
        self.assertTrue((self.args.output_dir / "report.md").is_file())

    @patch(
        "scripts.protocol_compatibility.assessment.openapi_workflow.compare",
        return_value=({}, ["mock"]),
    )
    def test_alias_audit_and_value_probes_do_not_claim_compatibility(self, compare):
        """Retain alias probes and parent constraints without asserting runtime parity."""
        own, native = fixtures.AliasMatchingTests().pair()
        write_json(self.args.dynamo, own)
        write_json(self.args.framework, native)
        self.assertEqual(assess(self.args), 1)
        result = json.loads((self.args.output_dir / "report.json").read_text())
        self.assertEqual(result["status"], "incomplete_coverage")
        self.assertEqual(compare.call_count, 4)
        self.assertEqual(len(result["alias_matches"]), 2)
        for match in result["alias_matches"]:
            self.assertEqual(match["compatibility"], "not_established_by_name_match")
            self.assertEqual(match["value_comparison"], "no_detected_differences")
            self.assertTrue((self.args.output_dir / match["evidence"]).exists())
            self.assertTrue(match["parent_context"]["backend"])
        self.assertEqual(
            json.loads((self.args.output_dir / "alias-matches.json").read_text()),
            result["alias_matches"],
        )
        self.assertIn(
            "Input alias matches", (self.args.output_dir / "report.md").read_text()
        )

    @patch(
        "scripts.protocol_compatibility.assessment.openapi_workflow.compare",
        return_value=({}, ["mock"]),
    )
    def test_normalized_inputs_are_compared_and_originals_retained(self, compare):
        """Compare normalized projections while retaining the original schema and rewrite log."""
        raw = document()
        union = {"anyOf": [{"type": "string"}, {"type": "null"}]}
        raw["components"]["schemas"]["Request"]["properties"]["value"] = union
        write_json(self.args.framework, raw)
        assess(self.args)
        original = json.loads(
            (self.args.output_dir / "framework.requests.json").read_text()
        )
        normalized_path = self.args.output_dir / "framework.requests.normalized.json"
        normalized = json.loads(normalized_path.read_text())
        self.assertEqual(
            original["components"]["schemas"]["Request"]["properties"]["value"], union
        )
        self.assertEqual(
            normalized["components"]["schemas"]["Request"]["properties"]["value"],
            {"type": ["string", "null"]},
        )
        self.assertEqual(compare.call_args_list[0].args[1], normalized_path)
        report = json.loads((self.args.output_dir / "report.json").read_text())
        self.assertEqual(
            report["representation_normalizations"][0]["side"], "framework"
        )
        self.assertEqual(
            json.loads((self.args.output_dir / "normalization.json").read_text()),
            report["representation_normalizations"],
        )
        self.assertIn(
            "Equivalent representations",
            (self.args.output_dir / "report.md").read_text(),
        )


if __name__ == "__main__":
    unittest.main()
