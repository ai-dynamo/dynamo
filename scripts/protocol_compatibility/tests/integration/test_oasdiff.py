# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check that projections remove representation noise without hiding schema changes.

These synthetic request, alias, event, and chunk cases execute the pinned oasdiff
binary. Set OASDIFF_BIN to its path; without it, the suite skips. The tests cover
the Python-to-comparator boundary, not compatibility of real framework releases
or server behavior. test_workflow.py separately checks orchestration with mocks.
"""

import copy
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.protocol_compatibility.assessment.openapi import (
    alias_value_document,
    compare,
    match_input_aliases,
    normalize_nullable_strings,
    request_differences,
    request_document,
)
from scripts.protocol_compatibility.assessment.openapi_workflow import compare_responses
from scripts.protocol_compatibility.assessment.responses import (
    prepare_response_source,
    response_differences,
    response_document,
)
from scripts.protocol_compatibility.inputs import write_json
from scripts.protocol_compatibility.tests.assessment import test_openapi as fixtures
from scripts.protocol_compatibility.tests.assessment.test_openapi import document
from scripts.protocol_compatibility.tests.assessment.test_responses import (
    stream_selection,
    streaming_document,
)


class ComparatorIntegrationTests(unittest.TestCase):
    def setUp(self):
        """Use OASDIFF_BIN and a temporary output directory; skip when the binary is unspecified."""
        configured = os.environ.get("OASDIFF_BIN")
        if not configured:
            self.skipTest(
                "Set OASDIFF_BIN to validate the real pinned comparator (required in CI)"
            )
        self.binary = Path(configured).resolve()
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def differences(self, native, dynamo):
        """Project gap-free request fixtures and return real target-to-Dynamo deltas."""
        paths = []
        for name, raw in (("native", native), ("dynamo", dynamo)):
            scoped, gaps = request_document(raw)
            self.assertFalse(gaps)
            scoped, _ = normalize_nullable_strings(scoped)
            path = self.root / f"{name}.json"
            write_json(path, scoped)
            paths.append(path)
        result, _ = compare(self.binary, *paths, flatten=True)
        return request_differences(result)

    def response_deltas(self, native, dynamo, native_selection=None):
        """Compare decoded responses, optionally qualifying the native SSE form; require no gaps."""
        paths = []
        for name, raw in (("native", native), ("dynamo", dynamo)):
            streams = {}
            if name == "native" and native_selection is not None:
                raw, streams, _ = prepare_response_source(
                    raw, native_selection, native_selection["sha256"]
                )
            scoped, gaps = response_document(raw, streams)
            self.assertFalse(gaps)
            path = self.root / f"{name}.responses.json"
            write_json(path, scoped)
            paths.append(path)
        result, _ = compare(self.binary, *paths, flatten=True)
        return response_differences(result)

    def test_qualified_31_envelopes_compare_without_losing_material_changes(self):
        """The actual comparator removes representation-only deltas, not payload/event changes."""
        dynamo = streaming_document()
        for path in dynamo["paths"].values():
            event = path["post"]["responses"]["200"]["content"]["text/event-stream"][
                "itemSchema"
            ]
            event["properties"].update(
                {
                    "event": {"type": "string", "const": "message"},
                    "id": {"type": "string"},
                    "retry": {"type": "integer", "minimum": 0},
                }
            )
            event["properties"]["data"]["anyOf"].append(
                {
                    "contentMediaType": "application/json",
                    "contentSchema": {
                        "type": "object",
                        "required": ["error"],
                        "properties": {"error": {"type": "string"}},
                    },
                }
            )
        native = copy.deepcopy(dynamo)
        native["openapi"] = "3.1.0"
        for path in native["paths"].values():
            media = path["post"]["responses"]["200"]["content"]["text/event-stream"]
            media["schema"] = media.pop("itemSchema")
        selected = stream_selection(native, "event-schema")
        for path in dynamo["paths"].values():
            media = path["post"]["responses"]["200"]["content"]["text/event-stream"]
            media["itemSchema"] = {"anyOf": [media["itemSchema"]]}
        self.assertEqual(self.response_deltas(native, dynamo, selected), [])
        for change in (
            "type",
            "required",
            "error",
            "encoding",
            "event",
            "retry",
            "extension",
        ):
            changed = copy.deepcopy(dynamo)
            media = changed["paths"]["/v1/chat/completions"]["post"]["responses"][
                "200"
            ]["content"]["text/event-stream"]
            event = media["itemSchema"]["anyOf"][0]
            data = event["properties"]["data"]["anyOf"]
            if change == "type":
                data[1]["contentSchema"] = {"type": "integer"}
            elif change == "required":
                changed["components"]["schemas"]["Response"]["required"] = []
            elif change == "error":
                data[2]["contentSchema"]["properties"]["error"]["type"] = "integer"
            elif change == "encoding":
                data[0] = {
                    "contentMediaType": "application/json",
                    "contentSchema": {"const": "[DONE]"},
                }
            elif change == "event":
                del event["properties"]["event"]["const"]
            elif change == "retry":
                event["properties"]["retry"]["minimum"] = 1
            else:
                extension = copy.deepcopy(event)
                extension["required"].append("event")
                extension["properties"]["event"]["const"] = "tool_call_dispatch"
                media["itemSchema"] = {"anyOf": [event, extension]}
            with self.subTest(change=change):
                self.assertTrue(self.response_deltas(native, changed, selected))

    def test_real_chunk_probes_preserve_constraints_without_event_equivalence(self):
        """Compare equivalent, changed and unsupported chunk declarations with real oasdiff."""
        for change in ("equal", "type", "required", "unsupported"):
            own = streaming_document()
            own["components"]["schemas"]["Response"]["properties"]["choices"] = {
                "type": "array",
                "items": {"type": "object"},
            }
            other = copy.deepcopy(own)
            other["openapi"] = "3.1.0"
            for path in other["paths"].values():
                path["post"]["responses"]["200"]["content"]["text/event-stream"] = {
                    "schema": {"$ref": "#/components/schemas/Response"}
                }
            payload = own["components"]["schemas"]["Response"]
            if change == "type":
                payload["properties"]["id"]["type"] = "integer"
            elif change == "required":
                payload["required"] = []
            elif change == "unsupported":
                other["components"]["schemas"]["Response"] = {"type": "string"}
            documents, selections = {}, {}
            for side, raw, mode in (
                ("framework", other, "decoded-chunk"),
                ("dynamo", own, "event-item-schema"),
            ):
                config = stream_selection(raw, mode, chunk=side == "dynamo")
                documents[side], selections[side], _ = prepare_response_source(
                    raw, config, config["sha256"]
                )
            output = self.root / change
            output.mkdir()
            with self.subTest(change=change):
                _, gaps, probes, _ = compare_responses(
                    SimpleNamespace(oasdiff=self.binary, output_dir=output),
                    documents,
                    selections,
                )
                self.assertTrue(gaps)
                self.assertEqual(
                    probes[0]["status"],
                    "not_compared"
                    if change == "unsupported"
                    else "no_detected_chunk_differences"
                    if change == "equal"
                    else "differences_found",
                )

    def test_identical_json_and_sse_contracts_have_no_response_delta(self):
        """The decoded projection must not manufacture a difference by server identity."""
        raw = streaming_document()
        self.assertEqual(self.response_deltas(raw, copy.deepcopy(raw)), [])

    def test_sse_content_schema_is_compared_even_when_unary_schema_is_unchanged(self):
        """A change hidden inside SSE data.contentSchema must reach the actual comparator."""
        native = streaming_document()
        dynamo = copy.deepcopy(native)
        data = dynamo["paths"]["/v1/chat/completions"]["post"]["responses"]["200"][
            "content"
        ]["text/event-stream"]["itemSchema"]["properties"]["data"]
        data["anyOf"][1]["contentSchema"] = {"type": "integer"}
        self.assertTrue(self.response_deltas(native, dynamo))

    def test_sse_literal_terminator_change_is_not_ignored_as_an_annotation(self):
        """Changing the literal data marker must produce a candidate, not a false pass."""
        native = streaming_document()
        dynamo = copy.deepcopy(native)
        data = dynamo["paths"]["/v1/chat/completions"]["post"]["responses"]["200"][
            "content"
        ]["text/event-stream"]["itemSchema"]["properties"]["data"]
        data["anyOf"][0]["const"] = "[FINISHED]"
        self.assertTrue(self.response_deltas(native, dynamo))

    def test_response_status_and_media_removal_are_compared(self):
        """Status and media selection are contracts, not only the nested JSON payload."""
        for change in ("status", "media"):
            native = streaming_document()
            dynamo = copy.deepcopy(native)
            responses = dynamo["paths"]["/v1/chat/completions"]["post"]["responses"]
            if change == "status":
                responses["201"] = responses.pop("200")
            else:
                del responses["200"]["content"]["text/event-stream"]
            with self.subTest(change=change):
                self.assertTrue(self.response_deltas(native, dynamo))

    def test_equivalent_event_reference_is_not_a_response_difference(self):
        """Inlining versus referencing the same SSE envelope must compare equivalently."""
        native = streaming_document()
        dynamo = copy.deepcopy(native)
        for path in dynamo["paths"].values():
            media = path["post"]["responses"]["200"]["content"]["text/event-stream"]
            dynamo["components"]["schemas"]["Event"] = media["itemSchema"]
            media["itemSchema"] = {"$ref": "#/components/schemas/Event"}
        self.assertEqual(self.response_deltas(native, dynamo), [])

    def test_response_and_unreachable_components_are_outside_scope(self):
        native = document()
        dynamo = copy.deepcopy(native)
        dynamo["components"]["schemas"]["Response"] = {"type": "string"}
        dynamo["components"]["schemas"]["Unused"] = {"type": "number"}
        self.assertEqual(self.differences(native, dynamo), [])

    def test_simple_allof_representation_does_not_invent_a_difference(self):
        native = document()
        # Remove recursion here: this test is specifically the normal flattened
        # object-export case, not the library's handling of recursive allOf.
        del native["components"]["schemas"]["Request"]["properties"]["recursive"]
        dynamo = copy.deepcopy(native)
        dynamo["components"]["schemas"]["Request"] = {
            "allOf": [dynamo["components"]["schemas"]["Request"]]
        }
        self.assertEqual(self.differences(native, dynamo), [])

    def test_nullable_string_representation_is_equivalent_in_both_directions(self):
        native = document()
        native["components"]["schemas"]["Request"]["properties"]["value"] = {
            "anyOf": [{"type": "string"}, {"type": "null"}],
            "default": None,
        }
        dynamo = copy.deepcopy(native)
        dynamo["components"]["schemas"]["Request"]["properties"]["value"] = {
            "type": ["null", "string"],
            "default": None,
        }
        self.assertEqual(self.differences(native, dynamo), [])
        self.assertEqual(self.differences(dynamo, native), [])

    def test_alias_value_probe_preserves_real_differences_and_raw_name_delta(self):
        own, native = fixtures.AliasMatchingTests().pair()
        matches, gaps = match_input_aliases(own, native)
        self.assertFalse(gaps)
        match = matches[0]
        self.assertTrue(self.differences(native, own))
        for after in (
            {"type": "string"},
            {"type": "integer"},
            {"type": "string", "minLength": 2},
            {"type": "string", "default": "x"},
        ):
            with self.subTest(after=after):
                own, _ = fixtures.AliasMatchingTests().pair()
                own["components"]["schemas"]["Request"]["properties"]["value"].update(
                    after
                )
                differences = self.differences(
                    alias_value_document(native, match["backend_location"]),
                    alias_value_document(own, match["dynamo_location"]),
                )
                self.assertEqual(bool(differences), after != {"type": "string"})

    def test_nullable_string_real_constraints_still_differ(self):
        for after in (
            {"type": "string", "default": "ab"},
            {"type": ["string", "null"], "default": "cd"},
            {"type": ["string", "null"], "default": "ab", "minLength": 2},
            {"type": ["string", "null"], "default": "ab", "enum": ["ab"]},
        ):
            with self.subTest(after=after):
                native = document()
                del native["components"]["schemas"]["Request"]["default"]
                native["components"]["schemas"]["Request"]["properties"]["value"] = {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                    "default": "ab",
                }
                dynamo = copy.deepcopy(native)
                dynamo["components"]["schemas"]["Request"]["properties"][
                    "value"
                ] = after
                self.assertTrue(self.differences(native, dynamo))

    def test_changes_behind_nested_references_and_contract_constraints_are_detected(
        self,
    ):
        pairs = [
            (True, False),
            ({"type": "string", "const": "a"}, {"type": "string", "const": "b"}),
            ({"type": "string", "format": "email"}, {"type": "string"}),
            (
                {"type": "object", "dependentRequired": {"x": ["y"]}},
                {"type": "object"},
            ),
            (
                {"type": "array", "items": {"type": "string"}, "minItems": 1},
                {"type": "array", "items": {"type": "string"}, "minItems": 2},
            ),
            (
                {"type": "number", "exclusiveMinimum": 0},
                {"type": "number", "exclusiveMinimum": 1},
            ),
            (
                {"type": "object", "propertyNames": {"pattern": "^a"}},
                {"type": "object", "propertyNames": {"pattern": "^b"}},
            ),
            ({"not": {"type": "string"}}, {"not": {"type": "number"}}),
            ({"type": "string"}, {"type": "integer"}),
            ({"type": "integer", "minimum": 0}, {"type": "integer", "minimum": 1}),
            ({"type": "string", "enum": ["a", "b"]}, {"type": "string", "enum": ["a"]}),
            ({"type": ["string", "null"]}, {"type": "string"}),
            (
                {"type": "array", "items": {"type": "string"}},
                {"type": "array", "items": {"type": "integer"}},
            ),
            (
                {"type": "object", "additionalProperties": True},
                {"type": "object", "additionalProperties": False},
            ),
            (
                {"type": "object", "properties": {"x": {"type": "string"}}},
                {
                    "type": "object",
                    "properties": {"x": {"type": "string"}},
                    "required": ["x"],
                },
            ),
        ]
        for before, after in pairs:
            with self.subTest(before=before, after=after):
                native = document()
                native["components"]["schemas"]["Request"]["properties"]["nested"] = {
                    "$ref": "#/components/schemas/Alias"
                }
                native["components"]["schemas"]["Alias"] = {
                    "$ref": "#/components/schemas/Leaf"
                }
                native["components"]["schemas"]["Leaf"] = before
                dynamo = copy.deepcopy(native)
                dynamo["components"]["schemas"]["Leaf"] = after
                self.assertEqual(len(self.differences(native, dynamo)), 2)


if __name__ == "__main__":
    unittest.main()
