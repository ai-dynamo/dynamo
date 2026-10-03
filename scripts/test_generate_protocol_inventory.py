# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the source-derived native-field admission vocabulary."""

import unittest

from generate_protocol_inventory import request_fields, rust_vocabulary, wire_names
from protocol_drift import module_contract


def inventory(source):
    return {"modules": {"protocol.py": {"contract": module_contract(source)}}}


class ProtocolInventoryTests(unittest.TestCase):
    def test_resolves_inherited_fields_and_child_override(self):
        upstream = inventory(
            "class Parent(BaseModel):\n"
            "    inherited: int = 0\n"
            "    changed: int = 1\n"
            "class Request(Parent):\n"
            "    changed: int = 2\n"
            "    own: bool = False\n"
        )
        fields = request_fields(upstream, "Request")
        self.assertEqual(sorted(fields), ["changed", "inherited", "own"])
        self.assertEqual(fields["inherited"]["declared_in"], "Parent")
        self.assertEqual(fields["changed"]["default"], "2")
        self.assertEqual(fields["own"]["default"], "False")

    def test_metadata_and_private_attributes_not_request_fields(self):
        upstream = inventory(
            "class Request(BaseModel):\n"
            "    field_names: ClassVar[set[str] | None] = None\n"
            "    metadata: typing.ClassVar[int] = 1\n"
            "    _cache: int = 0\n"
            "    value: int = 0\n"
        )
        self.assertEqual(list(request_fields(upstream, "Request")), ["value"])

    def test_literal_aliases_preserved_and_dynamic_alias_fails_closed(self):
        declaration = {
            "annotation": "str",
            "default": "Field(alias='old', validation_alias=AliasChoices('a', 'b'))",
        }
        self.assertEqual(wire_names("value", declaration), ["a", "b", "old", "value"])
        declaration["default"] = "Field(alias=compute_alias())"
        with self.assertRaisesRegex(ValueError, "unresolved input alias"):
            wire_names("value", declaration)

    def test_unresolved_base_and_cycles_fail_closed(self):
        for source in [
            "class Request(Unknown): pass",
            "class Request(Parent): pass\nclass Parent(Request): pass",
        ]:
            with self.assertRaises(ValueError):
                request_fields(inventory(source), "Request")

    def test_ambiguous_class_fails_closed(self):
        upstream = inventory("class Request(BaseModel): pass")
        upstream["modules"]["other.py"] = upstream["modules"]["protocol.py"]
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            request_fields(upstream, "Request")

    def test_generated_vocabulary_deduplicates_endpoints_and_versions(self):
        profile = {
            "version": "test",
            "endpoints": {"/v1/completions": {"value": {"wire_names": ["z", "a"]}}},
        }
        result = rust_vocabulary({"profiles": [profile, profile]})
        self.assertEqual(result.count('    "a",'), 1)
        self.assertLess(result.index('    "a",'), result.index('    "z",'))


if __name__ == "__main__":
    unittest.main()
