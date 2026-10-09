# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import json
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from jsonschema import Draft202012Validator

from scripts.protocol_compatibility.assessment.openapi import (
    ENDPOINTS,
    alias_value_document,
    compare,
    match_input_aliases,
    normalize_nullable_strings,
    request_differences,
    request_document,
    request_pointer,
)


def document():
    return {
        "openapi": "3.1.0",
        "info": {"title": "Fixture", "version": "1"},
        "paths": {
            endpoint: {
                "post": {
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/Request"},
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Not compared",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/Response"}
                                },
                            },
                        }
                    },
                }
            }
            for endpoint in ENDPOINTS
        },
        "components": {
            "schemas": {
                "Request": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "string"},
                        "recursive": {"$ref": "#/components/schemas/Request"},
                        "$ref": {"type": "string"},
                    },
                    "required": ["value"],
                    "default": {"value": "x", "$ref": "literal"},
                },
                "Response": {"type": "integer"},
                "Unused": {"type": "boolean"},
            }
        },
    }


class RequestProjectionTests(unittest.TestCase):
    def test_custom_dialect_is_not_silently_replaced_by_default(self):
        raw = document()
        raw["jsonSchemaDialect"] = "https://example.invalid/custom-dialect"
        with self.assertRaisesRegex(ValueError, "Custom OpenAPI schema dialects"):
            request_document(raw)

    def test_only_scoped_requests_and_reference_closure_survive(self):
        raw = document()
        raw["paths"]["/unrelated/{id}"] = {"get": {"responses": {}}}
        before = copy.deepcopy(raw)
        scoped, gaps = request_document(raw)
        self.assertEqual(raw, before)
        self.assertFalse(gaps)
        self.assertEqual(set(scoped["paths"]), set(ENDPOINTS))
        self.assertEqual(set(scoped["components"]["schemas"]), {"Request"})
        self.assertEqual(
            scoped["components"]["schemas"]["Request"]["default"],
            {"value": "x", "$ref": "literal"},
        )
        self.assertNotIn(
            "content", scoped["paths"][ENDPOINTS[0]]["post"]["responses"]["200"]
        )

    def test_external_refs_and_unresolved_slots_fail_visibly(self):
        raw = document()
        raw["components"]["schemas"]["Request"]["properties"]["value"] = {
            "$ref": "https://example.com/schema"
        }
        with self.assertRaisesRegex(ValueError, "Unsupported request reference"):
            request_document(raw)
        raw = document()
        raw["components"]["schemas"]["Request"]["x-dynamo-schema-import"] = {
            "type": "Missing"
        }
        with self.assertRaisesRegex(ValueError, "Unresolved dependency"):
            request_document(raw)

    def test_unhandled_constraints_remain_visible_and_report_gaps(self):
        raw = document()
        raw["components"]["schemas"]["Request"]["if"] = {"required": ["conditional"]}
        raw["components"]["schemas"]["Request"]["then"] = {"required": ["value"]}
        scoped, gaps = request_document(raw)
        self.assertEqual(len(gaps), 4)
        self.assertIn("if", scoped["components"]["schemas"]["Request"])
        self.assertTrue(
            all(gap["location"] == "#/components/schemas/Request" for gap in gaps)
        )

    def test_missing_endpoint_or_media_type_is_not_empty_success(self):
        raw = document()
        del raw["paths"][ENDPOINTS[0]]
        with self.assertRaises(KeyError):
            request_document(raw)
        raw = document()
        raw["paths"][ENDPOINTS[0]]["post"]["requestBody"]["content"]["text/plain"] = {}
        with self.assertRaisesRegex(ValueError, "media type"):
            request_document(raw)

    def test_differences_keep_nested_details_and_ignore_response_noise(self):
        delta = {
            "content": {
                "modified": {
                    "application/json": {
                        "schema": {
                            "properties": {
                                "modified": {
                                    "nested": {
                                        "type": {"from": "string", "to": "integer"}
                                    }
                                }
                            },
                        }
                    }
                }
            }
        }
        diff = {
            "components": {"schemas": {"added": ["Renamed"]}},
            "paths": {
                "modified": {
                    ENDPOINTS[0]: {
                        "operations": {"modified": {"POST": {"requestBody": delta}}}
                    },
                    ENDPOINTS[1]: {
                        "operations": {
                            "modified": {"POST": {"responses": {"deleted": ["400"]}}}
                        }
                    },
                }
            },
        }
        self.assertEqual(
            request_differences(diff),
            [
                {
                    "endpoint": ENDPOINTS[0],
                    "location": request_pointer(ENDPOINTS[0]),
                    "delta": delta,
                }
            ],
        )
        self.assertEqual(request_differences({}), [])

    @patch("scripts.protocol_compatibility.assessment.openapi.subprocess.run")
    def test_comparator_pin_network_restriction_and_flatten_guard(self, run):
        run.side_effect = [
            subprocess.CompletedProcess([], 0, "oasdiff version 1.33.0\n"),
            subprocess.CompletedProcess([], 0, json.dumps({})),
        ]
        result, command = compare(
            Path("/fixture/oasdiff"), Path("v.json"), Path("d.json"), flatten=False
        )
        self.assertEqual(result, {})
        self.assertIn("--allow-external-refs=false", command)
        self.assertNotIn("--flatten-allof", command)
        self.assertEqual(command[-2:], ["v.json", "d.json"])
        run.side_effect = [
            subprocess.CompletedProcess([], 0, "oasdiff version 9.0.0\n")
        ]
        with self.assertRaisesRegex(ValueError, "Expected oasdiff"):
            compare(
                Path("/fixture/oasdiff"), Path("v.json"), Path("d.json"), flatten=True
            )


class AliasMatchingTests(unittest.TestCase):
    def pair(self):
        own = document()
        del own["components"]["schemas"]["Request"]["default"]
        own["components"]["schemas"]["Request"]["properties"]["value"].update(
            {
                "x-dynamo-input-aliases": ["alternate"],
            }
        )
        native = copy.deepcopy(own)
        request = native["components"]["schemas"]["Request"]
        request["properties"]["alternate"] = {"type": "string"}
        del request["properties"]["value"]
        request["required"] = ["alternate"]
        return own, native

    def test_match_keeps_requiredness_and_original_documents(self):
        own, native = self.pair()
        before = copy.deepcopy((own, native))
        matches, gaps = match_input_aliases(own, native)
        self.assertEqual((own, native), before)
        self.assertFalse(gaps)
        self.assertEqual(len(matches), 2)
        self.assertEqual(matches[0]["path"], ["alternate"])
        self.assertEqual(matches[0]["compatibility"], "not_established_by_name_match")
        self.assertEqual(
            matches[0]["simultaneous_names"],
            "Dynamo rejects canonical and alias together",
        )
        self.assertEqual(
            matches[0]["parent_context"]["dynamo"][0]["constraints"]["required"],
            ["value"],
        )
        probe = alias_value_document(own, matches[0]["dynamo_location"])
        self.assertEqual(probe["components"], own["components"])

    def test_legacy_conflict_marker_cannot_override_alias_semantics(self):
        own, native = self.pair()
        expected = match_input_aliases(own, native)
        field = own["components"]["schemas"]["Request"]["properties"]["value"]
        field["x-dynamo-alias-conflict"] = "reject"
        self.assertEqual(match_input_aliases(own, native), expected)
        for conflict in ("prefer_canonical", "allow_equal", None):
            with self.subTest(conflict=conflict):
                field["x-dynamo-alias-conflict"] = conflict
                with self.assertRaisesRegex(ValueError, "alias metadata"):
                    match_input_aliases(own, native)

    def test_nested_refs_match_only_same_instance_path(self):
        own, native = self.pair()
        for doc in (own, native):
            doc["components"]["schemas"]["Root"] = {
                "properties": {
                    "messages": {
                        "type": "array",
                        "items": {"$ref": "#/components/schemas/Request"},
                    }
                }
            }
            for endpoint in ENDPOINTS:
                doc["paths"][endpoint]["post"]["requestBody"]["content"][
                    "application/json"
                ]["schema"] = {"$ref": "#/components/schemas/Root"}
        matches, gaps = match_input_aliases(own, native)
        self.assertFalse(gaps)
        self.assertEqual(matches[0]["path"], ["messages", None, "alternate"])
        native["components"]["schemas"]["Root"]["properties"]["elsewhere"] = native[
            "components"
        ]["schemas"]["Root"]["properties"].pop("messages")
        self.assertEqual(match_input_aliases(own, native), ([], []))

    def test_ambiguous_and_colliding_aliases_are_gaps(self):
        for side in ("dynamo", "backend"):
            with self.subTest(side=side):
                own, native = self.pair()
                if side == "dynamo":
                    own["components"]["schemas"]["Request"]["properties"][
                        "alternate"
                    ] = {"type": "integer"}
                else:
                    schema = native["components"]["schemas"]["Request"]
                    native["components"]["schemas"]["Request"] = {
                        "anyOf": [schema, copy.deepcopy(schema)]
                    }
                matches, gaps = match_input_aliases(own, native)
                self.assertFalse(matches)
                self.assertEqual(len(gaps), 2)

    def test_no_metadata_no_inference_and_invalid_metadata_fails(self):
        own, native = self.pair()
        field = own["components"]["schemas"]["Request"]["properties"]["value"]
        del field["x-dynamo-input-aliases"]
        self.assertEqual(match_input_aliases(own, native), ([], []))
        for aliases in ("alternate", [], [1], ["value"], ["alternate", "alternate"]):
            field["x-dynamo-input-aliases"] = aliases
            with self.assertRaisesRegex(ValueError, "alias metadata"):
                match_input_aliases(own, native)


class NullableStringNormalizationTests(unittest.TestCase):
    def normalize_field(self, field):
        raw = document()
        del raw["components"]["schemas"]["Request"]["default"]
        raw["components"]["schemas"]["Request"]["properties"]["value"] = field
        scoped, _ = request_document(raw)
        before = copy.deepcopy(scoped)
        normalized, changes = normalize_nullable_strings(scoped)
        self.assertEqual(scoped, before)
        again, again_changes = normalize_nullable_strings(normalized)
        self.assertEqual(again, normalized)
        self.assertEqual(again_changes, [])
        return (
            normalized["components"]["schemas"]["Request"]["properties"]["value"],
            changes,
        )

    def test_both_orders_are_equivalent_and_shared_refs_are_recorded_once(self):
        for field in (
            {"anyOf": [{"type": "string"}, {"type": "null"}]},
            {"anyOf": [{"type": "null"}, {"type": "string"}]},
            {"type": ["null", "string"]},
        ):
            with self.subTest(field=field):
                normalized, changes = self.normalize_field(field)
                self.assertEqual(normalized, {"type": ["string", "null"]})
                self.assertEqual(len(changes), 1)
                self.assertEqual(changes[0]["before"], field)
                self.assertEqual(changes[0]["after"], normalized)
                self.assertEqual(
                    changes[0]["location"],
                    "#/components/schemas/Request/properties/value",
                )

    def test_sibling_constraints_annotations_and_literal_data_survive(self):
        union = {"anyOf": [{"type": "string"}, {"type": "null"}]}
        for constraints in (
            {},
            {"minLength": 2},
            {"enum": ["ab", None]},
            {"const": None},
        ):
            with self.subTest(constraints=constraints):
                field = {
                    **union,
                    **constraints,
                    "title": "Value",
                    "description": "Keep",
                    "default": None,
                    "examples": [union],
                    "x-example": union,
                }
                normalized, _ = self.normalize_field(field)
                self.assertEqual(
                    normalized,
                    {k: v for k, v in field.items() if k != "anyOf"}
                    | {"type": ["string", "null"]},
                )
                for value in (None, "", "a", "ab", 0, False, [], {}):
                    self.assertEqual(
                        Draft202012Validator(field).is_valid(value),
                        Draft202012Validator(normalized).is_valid(value),
                    )

    def test_complex_branches_and_other_types_are_not_simplified(self):
        for field in (
            {"anyOf": [{"type": "string", "minLength": 1}, {"type": "null"}]},
            {"anyOf": [{"type": "string", "description": "Keep"}, {"type": "null"}]},
            {"anyOf": [{"$ref": "#/components/schemas/Request"}, {"type": "null"}]},
            {"oneOf": [{"type": "string"}, {"type": "null"}]},
            {"anyOf": [{"type": "boolean"}, {"type": "null"}]},
            {"anyOf": [{"type": "string"}, {"type": "null"}, {"type": "integer"}]},
            {"type": "string", "anyOf": [{"type": "string"}, {"type": "null"}]},
        ):
            with self.subTest(field=field):
                normalized, changes = self.normalize_field(field)
                self.assertEqual(normalized, field)
                self.assertEqual(changes, [])

    def test_inline_schemas_nested_arrays_and_escaped_property_names(self):
        raw = document()
        union = {"anyOf": [{"type": "string"}, {"type": "null"}]}
        for endpoint in ENDPOINTS:
            raw["paths"][endpoint]["post"]["requestBody"]["content"][
                "application/json"
            ]["schema"] = {
                "type": "object",
                "properties": {
                    "a/b~": {"type": "array", "items": union},
                    "metadata": {},
                },
                "required": ["a/b~"],
                "default": {"a/b~": [], "metadata": union},
            }
        scoped, _ = request_document(raw)
        normalized, changes = normalize_nullable_strings(scoped)
        self.assertEqual(len(changes), 2)
        self.assertTrue(
            all(c["location"].endswith("/properties/a~1b~0/items") for c in changes)
        )
        for endpoint in ENDPOINTS:
            schema = normalized["paths"][endpoint]["post"]["requestBody"]["content"][
                "application/json"
            ]["schema"]
            self.assertEqual(
                schema["properties"]["a/b~"]["items"], {"type": ["string", "null"]}
            )
            self.assertEqual(schema["default"], {"a/b~": [], "metadata": union})
            self.assertEqual(schema["required"], ["a/b~"])


if __name__ == "__main__":
    unittest.main()
