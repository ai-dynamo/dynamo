# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Guard explicit SSE decoding and response coverage independently of inference."""

import copy
import hashlib
import json
import unittest

import yaml
from jsonschema import Draft202012Validator, ValidationError

from scripts.protocol_compatibility.assessment.responses import (
    chunk_document,
    decoded_data_schema,
    decoded_event_schema,
    prepare_response_source,
    response_differences,
    response_document,
    validate_sse_event,
)
from scripts.protocol_compatibility.inputs import parse_document
from scripts.protocol_compatibility.tests.assessment.test_openapi import document


def streaming_document():
    """Return a fresh two-endpoint fixture with JSON and OpenAPI 3.2 SSE contracts."""
    raw = document()
    raw["openapi"] = "3.2.0"
    raw["components"]["schemas"]["Response"] = {
        "type": "object",
        "required": ["id"],
        "properties": {"id": {"type": "string"}},
    }
    for operation in raw["paths"].values():
        operation["post"]["responses"]["200"]["content"]["text/event-stream"] = {
            "itemSchema": {
                "type": "object",
                "required": ["data"],
                "properties": {
                    "data": {
                        "type": "string",
                        "anyOf": [
                            {"const": "[DONE]"},
                            {
                                "contentMediaType": "application/json",
                                "contentSchema": {
                                    "$ref": "#/components/schemas/Response"
                                },
                            },
                        ],
                    }
                },
            }
        }
    return raw


def stream_selection(raw, mode, *, chunk=False):
    """Qualify both fixture SSE media pointers with a deterministic raw-byte hash."""
    keyword = "itemSchema" if mode == "event-item-schema" else "schema"
    entries = []
    for endpoint in raw["paths"]:
        location = (
            "#/paths/"
            + endpoint.replace("/", "~1")
            + "/post/responses/200/content/text~1event-stream/"
            + keyword
        )
        entry = {"pointer": location, "interpretation": mode}
        if chunk:
            entry["chunk_pointer"] = location + "/properties/data/anyOf/1/contentSchema"
        entries.append(entry)
    return {
        "sha256": hashlib.sha256(json.dumps(raw).encode()).hexdigest(),
        "source": "test-fixture",
        "revision": "fixture-v1",
        "streams": entries,
    }


class ResponseProjectionTests(unittest.TestCase):
    def test_yaml_integer_statuses_preserve_response_contracts(self):
        """Unquoted YAML codes match quoted codes without changing the source."""
        quoted = streaming_document()
        for path in quoted["paths"].values():
            path["post"]["responses"].update(
                {
                    "204": {"description": "No content"},
                    "304": {"description": "Not modified"},
                    "default": {"description": "Undeclared error body"},
                    "x-note": "Ignored extension",
                }
            )
        unquoted = copy.deepcopy(quoted)
        for path in unquoted["paths"].values():
            responses = path["post"]["responses"]
            for status in ("200", "204", "304"):
                responses[int(status)] = responses.pop(status)
        raw = parse_document(yaml.safe_dump(unquoted).encode())
        self.assertEqual(response_document(raw), response_document(quoted))
        self.assertEqual(raw, unquoted)

    def test_qualified_envelope_reuses_decoder_and_keeps_original_input(self):
        """Equivalent 3.1 envelope and 3.2 itemSchema forms produce equal projections."""
        native = streaming_document()
        legacy = copy.deepcopy(native)
        legacy["openapi"] = "3.1.0"
        for path in legacy["paths"].values():
            media = path["post"]["responses"]["200"]["content"]["text/event-stream"]
            media["schema"] = media.pop("itemSchema")
        before = copy.deepcopy(legacy)
        config = stream_selection(legacy, "event-schema")
        selected, streams, evidence = prepare_response_source(
            legacy, config, config["sha256"]
        )
        self.assertEqual(
            response_document(selected, streams), response_document(native)
        )
        self.assertEqual(legacy, before)
        self.assertEqual(
            evidence["streams"][0]["pointer"], config["streams"][0]["pointer"]
        )
        self.assertTrue(response_document(legacy)[1])
        for change in ("hash", "pointer", "duplicate", "mode"):
            invalid = copy.deepcopy(config)
            if change == "hash":
                invalid["sha256"] = "0" * 64
            elif change == "pointer":
                invalid["streams"][0]["pointer"] += "/properties"
            elif change == "duplicate":
                invalid["streams"].append(invalid["streams"][0])
            else:
                invalid["streams"][0]["interpretation"] = "infer"
            with self.subTest(change=change), self.assertRaises(ValueError):
                prepare_response_source(legacy, invalid, config["sha256"])

    def test_chunk_selection_cannot_establish_missing_event_layers(self):
        """Chunk references/constraints survive; absent envelope layers remain explicit gaps."""
        raw = streaming_document()
        raw["openapi"] = "3.1.0"
        raw["components"]["schemas"]["Response"]["properties"]["choices"] = {
            "type": "array",
            "items": {"type": "object"},
        }
        for path in raw["paths"].values():
            path["post"]["responses"]["200"]["content"]["text/event-stream"] = {
                "schema": {"$ref": "#/components/schemas/Response"}
            }
        config = stream_selection(raw, "decoded-chunk")
        selected, streams, _ = prepare_response_source(raw, config, config["sha256"])
        projected, gaps = response_document(selected, streams)
        self.assertEqual(
            {gap["layer"] for gap in gaps},
            {"envelope", "encoding", "error", "terminator", "extensions"},
        )
        self.assertEqual(
            projected["components"]["schemas"],
            {"Response": raw["components"]["schemas"]["Response"]},
        )
        probe, missing = chunk_document(selected, next(iter(streams.values())))
        self.assertFalse(missing)
        self.assertEqual(
            probe["components"]["schemas"]["Response"],
            raw["components"]["schemas"]["Response"],
        )
        raw["components"]["schemas"]["Response"] = {"type": "string"}
        self.assertTrue(response_document(raw, streams)[1])

    def test_path_mapping_is_explicit_collision_safe_and_records_server_context(self):
        """Map only the two retained OpenAI routes; never infer deployed endpoint identity."""
        raw = streaming_document()
        raw["paths"] = {
            key.removeprefix("/v1"): value for key, value in raw["paths"].items()
        }
        raw["servers"] = [{"url": "https://example.invalid/v1"}]
        config = stream_selection(raw, "event-item-schema")
        config["path_mappings"] = {key: "/v1" + key for key in raw["paths"]}
        before = copy.deepcopy(raw)
        selected, _, evidence = prepare_response_source(raw, config, config["sha256"])
        self.assertEqual(
            set(selected["paths"]), {"/v1/chat/completions", "/v1/completions"}
        )
        self.assertEqual(raw, before)
        self.assertEqual(
            evidence["path_mappings"][0]["server_context"]["document"], raw["servers"]
        )
        with self.assertRaises(ValueError):
            prepare_response_source(
                raw, {**config, "path_mappings": {}}, config["sha256"]
            )
        for change in ("collision", "unknown", "missing"):
            candidate, settings = copy.deepcopy(raw), copy.deepcopy(config)
            if change == "collision":
                candidate["paths"]["/v1/completions"] = candidate["paths"][
                    "/completions"
                ]
            elif change == "unknown":
                settings["path_mappings"]["/other"] = "/v1/completions"
            else:
                del candidate["paths"]["/completions"]["post"]
            with self.subTest(change=change), self.assertRaises(ValueError):
                prepare_response_source(candidate, settings, config["sha256"])

    def test_chunk_probe_rejects_wrong_selector_and_encoded_string_constraints(self):
        """Selecting a payload cannot discard material constraints in its containing event."""
        raw = streaming_document()
        config = stream_selection(raw, "event-item-schema", chunk=True)
        selected, streams, _ = prepare_response_source(raw, config, config["sha256"])
        entry = next(iter(streams.values()))
        self.assertFalse(chunk_document(selected, entry)[1])
        bad = {
            **entry,
            "projection_chunk_pointer": entry["projection_pointer"]
            + "/properties/other/contentSchema",
        }
        with self.assertRaises(ValueError):
            chunk_document(selected, bad)
        selected["paths"]["/v1/chat/completions"]["post"]["responses"]["200"][
            "content"
        ]["text/event-stream"]["itemSchema"]["properties"]["data"]["maxLength"] = 100
        with self.assertRaisesRegex(ValueError, "encoded-string"):
            chunk_document(selected, entry)

    def test_content_validation_cannot_accept_an_unresolved_import_as_an_object(self):
        """A raw native placeholder must fail closed rather than accept arbitrary chunks."""
        raw = streaming_document()
        raw["components"]["schemas"]["Response"] = {
            "type": "object",
            "x-dynamo-schema-import": {"type": "Missing"},
        }
        item = raw["paths"]["/v1/chat/completions"]["post"]["responses"]["200"][
            "content"
        ]["text/event-stream"]["itemSchema"]
        with self.assertRaisesRegex(ValueError, "Unresolved dependency"):
            validate_sse_event(raw, item, {"data": "{}"})

    def test_content_aware_validation_decodes_json_and_rejects_malformed_wire_data(
        self,
    ):
        """Unlike a generic string validator, event validation asserts embedded JSON."""
        raw = streaming_document()
        item = raw["paths"]["/v1/chat/completions"]["post"]["responses"]["200"][
            "content"
        ]["text/event-stream"]["itemSchema"]
        for data in ('{"id":"chunk"}', "[DONE]"):
            validate_sse_event(raw, item, {"data": data})
        for data in ("{", "{}", '{"id":1}', '"[DONE]"', "NaN", '{"id":NaN}', 1):
            with self.subTest(data=data), self.assertRaises(ValidationError):
                validate_sse_event(raw, item, {"data": data})

    """Missing coverage must remain visible; decoded payload assertions must survive."""

    def test_projection_retains_status_media_and_only_response_components(self):
        """Request changes cannot leak into the independent response comparison."""
        raw = streaming_document()
        before = copy.deepcopy(raw)
        projected, gaps = response_document(raw)
        self.assertEqual(raw, before)
        self.assertEqual(gaps, [])
        self.assertEqual(projected["openapi"], "3.1.0")
        self.assertEqual(set(projected["components"]["schemas"]), {"Response"})
        for operation in projected["paths"].values():
            self.assertNotIn("requestBody", operation["post"])
            content = operation["post"]["responses"]["200"]["content"]
            self.assertEqual(set(content), {"application/json", "text/event-stream"})
            self.assertIn("schema", content["text/event-stream"])
            self.assertNotIn("itemSchema", content["text/event-stream"])

    def test_decoded_data_rejects_invalid_payloads_and_distinguishes_done_encoding(
        self,
    ):
        """Generic string validation is insufficient: validate the decoded JSON too."""
        raw = streaming_document()
        projected, _ = response_document(raw)
        projected[
            "$ref"
        ] = "#/paths/~1v1~1chat~1completions/post/responses/200/content/text~1event-stream/schema"
        validator = Draft202012Validator(projected)
        self.assertTrue(
            validator.is_valid({"data": {"encoding": "json", "value": {"id": "x"}}})
        )
        self.assertTrue(
            validator.is_valid({"data": {"encoding": "literal", "value": "[DONE]"}})
        )
        for data in (
            {"encoding": "json", "value": {}},
            {"encoding": "json", "value": "[DONE]"},
            {"encoding": "literal", "value": "garbage"},
            {"encoding": "json", "value": {"id": 5}},
        ):
            with self.subTest(data=data):
                self.assertFalse(validator.is_valid({"data": data}))

    def test_named_event_envelopes_keep_discriminators(self):
        """A valid payload under the wrong event name must not validate."""
        schema = {
            "type": "object",
            "required": ["event", "data"],
            "properties": {
                "event": {"const": "reasoning_dispatch"},
                "data": {
                    "type": "string",
                    "contentMediaType": "application/json",
                    "contentSchema": {"type": "string"},
                },
            },
        }
        validator = Draft202012Validator(decoded_event_schema({}, schema))
        event = {
            "event": "reasoning_dispatch",
            "data": {"encoding": "json", "value": "reason"},
        }
        self.assertTrue(validator.is_valid(event))
        event["event"] = "tool_call_dispatch"
        self.assertFalse(validator.is_valid(event))

    def test_unsupported_data_constraints_and_missing_item_schema_report_gaps(self):
        """No omitted constraint or legacy payload schema can silently pass coverage."""
        for change in (
            "pattern",
            "contentEncoding",
            "missing",
            "legacy",
            "legacy-item",
            "external",
        ):
            raw = streaming_document()
            contract = raw["paths"]["/v1/chat/completions"]["post"]["responses"]["200"][
                "content"
            ]["text/event-stream"]
            data = contract["itemSchema"]["properties"]["data"]
            if change == "legacy-item":
                raw["openapi"] = "3.1.0"
            elif change == "missing":
                del contract["itemSchema"]
            elif change == "legacy":
                contract["schema"] = contract.pop("itemSchema")
            elif change == "external":
                contract["itemSchema"] = {"$ref": "https://example.invalid/schema"}
            else:
                data[change] = "unsupported"
            with self.subTest(change=change):
                _, gaps = response_document(raw)
                self.assertTrue(gaps)
                self.assertEqual(gaps[0]["endpoint"], "/v1/chat/completions")

    def test_missing_error_response_body_is_not_proven_empty(self):
        """Description-only errors are coverage gaps, while explicit 204 is bodyless."""
        raw = streaming_document()
        responses = raw["paths"]["/v1/chat/completions"]["post"]["responses"]
        responses["400"] = {"description": "Bad request"}
        responses["204"] = {"description": "No content"}
        projected, gaps = response_document(raw)
        self.assertEqual(len(gaps), 1)
        self.assertTrue(gaps[0]["location"].endswith("/400"))
        self.assertIn(
            "204", projected["paths"]["/v1/chat/completions"]["post"]["responses"]
        )

    def test_recursive_event_reference_is_a_gap_not_infinite_recursion(self):
        """Payload recursion is separate from unsupported recursive event envelopes."""
        raw = streaming_document()
        raw["components"]["schemas"]["Event"] = {"$ref": "#/components/schemas/Event"}
        with self.assertRaisesRegex(ValueError, "Unsupported event reference"):
            decoded_event_schema(raw, {"$ref": "#/components/schemas/Event"})

    def test_only_explicit_json_or_literal_data_can_be_projected(self):
        """Other encodings and unconstrained strings require a separate coverage decision."""
        for schema in (
            {"type": "string"},
            {"type": "object"},
            {"contentMediaType": "text/plain", "contentSchema": {}},
        ):
            with self.subTest(schema=schema), self.assertRaises(ValueError):
                decoded_data_schema(schema)

    def test_response_deltas_exclude_request_noise(self):
        """Preserve nested response differences without relabeling request changes."""
        delta = {"modified": {"200": {"content": {"deleted": ["text/event-stream"]}}}}
        diff = {
            "paths": {
                "modified": {
                    "/v1/chat/completions": {
                        "operations": {
                            "modified": {
                                "POST": {
                                    "responses": delta,
                                    "requestBody": {"changed": True},
                                }
                            }
                        }
                    }
                }
            }
        }
        findings = response_differences(diff)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["delta"], delta)
