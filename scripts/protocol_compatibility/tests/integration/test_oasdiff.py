# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Actual pinned comparator checks. CI must set OASDIFF_BIN; no mock engine."""

import copy
import os
import tempfile
import unittest
from pathlib import Path

from scripts.protocol_compatibility.acquisition.http import write_json
from scripts.protocol_compatibility.assessment.openapi import (
    alias_value_document,
    compare,
    match_input_aliases,
    normalize_nullable_strings,
    request_differences,
    request_document,
)
from scripts.protocol_compatibility.tests.assessment import test_openapi as fixtures
from scripts.protocol_compatibility.tests.assessment.test_openapi import document


class ComparatorIntegrationTests(unittest.TestCase):
    def setUp(self):
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
