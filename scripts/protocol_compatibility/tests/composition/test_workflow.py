# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check CLI orchestration, retained evidence, and failure status with a mocked comparator."""

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
from scripts.protocol_compatibility.acquisition.http import load_capture, write_json
from scripts.protocol_compatibility.assessment.openapi_workflow import assess
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
        """Create checksummed captures and CLI inputs in an automatically removed directory."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for side in ("dynamo", "framework"):
            directory = self.root / side
            directory.mkdir()
            write_json(directory / "openapi.raw.json", document())
            write_json(
                directory / "acquisition.json",
                {
                    "schema": "protocol-http-acquisition/v1",
                    "server": "example-engine" if side == "framework" else side,
                    "sha256": hashlib.sha256(
                        (directory / "openapi.raw.json").read_bytes()
                    ).hexdigest(),
                    "image": "fixture@sha256:" + "0" * 64,
                    "provenance": {
                        "source_revision": "a" * 40,
                        "server_version": "fixture",
                        "dependencies": {"fixture": "1"},
                    },
                },
            )
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
            dynamo=self.root / "dynamo",
            framework=self.root / "framework",
            framework_name="example-engine",
            openai=self.baseline,
            manifest=self.manifest,
            oasdiff=self.binary,
            output_dir=self.root / "report",
            response_adapters=None,
        )

    def test_modified_capture_is_rejected(self):
        (self.root / "dynamo/openapi.raw.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "changed after acquisition"):
            load_capture(self.root / "dynamo", "dynamo")

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
        inventory = json.loads((self.args.output_dir / "findings.json").read_text())
        self.assertEqual(inventory, report["finding_inventory"])
        self.assertTrue(any(row["status"] == "200" for row in inventory["findings"]))
        self.assertEqual(inventory["accounting"]["unaccounted"], [])
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
        self.assertEqual(assess(self.args), 1)
        result = json.loads((self.args.output_dir / "report.json").read_text())
        self.assertEqual(result["status"], "incomplete_coverage")
        self.assertEqual(result["framework_name"], "example-engine")
        self.assertEqual(result["differences"], [])
        self.assertEqual(result["behavioral_conformance"], "not_assessed")
        self.assertEqual(
            (self.args.output_dir / "dynamo/openapi.raw.json").read_bytes(),
            (self.args.dynamo / "openapi.raw.json").read_bytes(),
        )
        self.assertTrue((self.args.output_dir / "dynamo.composed.json").exists())
        markdown = (self.args.output_dir / "report.md").read_text()
        self.assertIn("Intentional exclusions", markdown)
        self.assertIn("Coverage gaps", markdown)
        self.assertIn("Dynamo versus example-engine", markdown)
        with self.assertRaises(FileExistsError):
            assess(self.args)
        self.assertEqual(compare.call_count, 2)

    @patch(
        "scripts.protocol_compatibility.assessment.openapi_workflow.compare",
        return_value=({}, ["mock"]),
    )
    def test_caller_dependency_annotations_do_not_gate_comparison(self, compare):
        manifest = yaml.safe_load(self.manifest.read_text())
        manifest["dependencies"] = {"fixture": "2"}
        self.manifest.write_text(yaml.safe_dump(manifest))
        self.assertEqual(assess(self.args), 1)
        report = json.loads((self.args.output_dir / "report.json").read_text())
        self.assertEqual(report["deployment_provenance"], "not_verified")
        self.assertEqual(
            report["composition_dependencies"]["expected"], {"fixture": "2"}
        )
        self.assertEqual(
            report["inputs"]["dynamo"]["provenance"]["dependencies"], {"fixture": "1"}
        )
        self.assertEqual(compare.call_count, 2)

    @patch(
        "scripts.protocol_compatibility.assessment.openapi_workflow.compare",
        return_value=({}, ["mock"]),
    )
    def test_bare_json_and_yaml_files_need_no_deployment_metadata(self, compare):
        self.args.dynamo = self.root / "dynamo/openapi.raw.json"
        self.args.framework = self.root / "published.yaml"
        self.args.framework.write_text(yaml.safe_dump(document()))
        self.assertEqual(assess(self.args), 1)
        result = json.loads((self.args.output_dir / "report.json").read_text())
        for side, source in (
            ("dynamo", self.args.dynamo),
            ("framework", self.args.framework),
        ):
            self.assertEqual(
                (self.args.output_dir / side / "openapi.raw").read_bytes(),
                source.read_bytes(),
            )
            self.assertIsNone(result["inputs"][side]["caller_metadata"])
            self.assertFalse(result["inputs"][side]["caller_metadata_verified"])
        self.assertEqual(result["differences"], [])
        self.assertIn("unverified", (self.args.output_dir / "report.md").read_text())

    def test_openai_checksum_still_fails_before_writing(self):
        self.baseline.write_text("changed baseline")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            assess(self.args)
        self.assertFalse(self.args.output_dir.exists())

    def test_wrong_or_invalid_framework_identity_fails_before_writing(self):
        for name in ("different-engine", "dynamo", "../escape", "", "**heading**"):
            with self.subTest(name=name):
                self.args.framework_name = name
                with self.assertRaises(ValueError):
                    assess(self.args)
                self.assertFalse(self.args.output_dir.exists())

    @patch("scripts.protocol_compatibility.__main__.assess", return_value=0)
    def test_cli_accepts_framework_identity_without_backend_specific_flags(
        self, assess
    ):
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

    @patch(
        "scripts.protocol_compatibility.assessment.openapi_workflow.compare",
        return_value=({}, ["mock"]),
    )
    def test_second_framework_uses_the_same_pipeline_and_fixed_artifact_names(
        self, compare
    ):
        self.args.framework_name = "another-engine"
        metadata_path = self.args.framework / "acquisition.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["server"] = self.args.framework_name
        write_json(metadata_path, metadata)
        self.assertEqual(assess(self.args), 1)
        report = json.loads((self.args.output_dir / "report.json").read_text())
        self.assertEqual(report["framework_name"], "another-engine")
        self.assertEqual(report["inputs"]["framework"]["server"], "another-engine")
        self.assertTrue(
            (self.args.output_dir / "framework.requests.normalized.json").exists()
        )
        self.assertEqual(compare.call_count, 2)

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
        own, native = fixtures.AliasMatchingTests().pair()
        for directory, raw in ((self.args.dynamo, own), (self.args.framework, native)):
            path = directory / "openapi.raw.json"
            write_json(path, raw)
            metadata_path = directory / "acquisition.json"
            metadata = json.loads(metadata_path.read_text())
            metadata["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            write_json(metadata_path, metadata)
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
        raw = document()
        union = {"anyOf": [{"type": "string"}, {"type": "null"}]}
        raw["components"]["schemas"]["Request"]["properties"]["value"] = union
        path = self.args.framework / "openapi.raw.json"
        write_json(path, raw)
        metadata_path = self.args.framework / "acquisition.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        write_json(metadata_path, metadata)
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
