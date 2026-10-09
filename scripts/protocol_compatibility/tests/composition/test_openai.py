# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import hashlib
import json
import unittest

import jsonpatch
import jsonpointer

from scripts.protocol_compatibility.composition.openai import MARKER, compose


class CompositionTests(unittest.TestCase):
    def setUp(self):
        self.baseline = {
            "components": {
                "schemas": {
                    "External": {
                        "type": "object",
                        "properties": {
                            "child": {"$ref": "#/components/schemas/Child"},
                            "$ref": {"type": "string"},
                        },
                        "default": {"$ref": "literal-not-a-reference"},
                    },
                    "Child": {
                        "type": "object",
                        "properties": {
                            "parent": {"$ref": "#/components/schemas/External"},
                        },
                    },
                    "Unrelated": {"type": "integer"},
                }
            }
        }
        self.raw = {
            "openapi": "3.1.0",
            "paths": {"/v1/chat/completions": {}},
            "components": {
                "schemas": {
                    "async_openai.External": {
                        MARKER: {"crate": "async-openai", "type": "External"}
                    },
                    "Owned": {"type": "string"},
                }
            },
        }
        self.manifest = {
            "openai": {"revision": "a" * 40},
            "dependencies": {"async-openai": "0.41.1", "dynamo-protocols": "6.1.0"},
            "imports": {"External": {"pointer": "/components/schemas/External"}},
            "exclusions": [{"field": "nvext", "reason": "Dynamo-only contents"}],
        }

    def run_composition(self):
        data = json.dumps(self.baseline).encode()
        self.manifest["openai"]["sha256"] = hashlib.sha256(data).hexdigest()
        return compose(self.raw, self.manifest, baseline_bytes=data)

    def test_closure_namespaces_cycles_and_preserves_raw(self):
        before = copy.deepcopy(self.raw)
        output = self.run_composition()
        self.assertEqual(self.raw, before)
        self.assertEqual(output["paths"], before["paths"])
        schemas = output["components"]["schemas"]
        self.assertNotIn("OpenAIBaseline.Unrelated", schemas)
        self.assertEqual(schemas["Owned"], {"type": "string"})
        imported = schemas["async_openai.External"]
        self.assertEqual(imported["default"], {"$ref": "literal-not-a-reference"})
        self.assertEqual(imported["properties"]["$ref"], {"type": "string"})
        child = schemas["OpenAIBaseline.Child"]
        self.assertEqual(
            child["properties"]["parent"]["$ref"],
            "#/components/schemas/async_openai.External",
        )
        self.assertEqual(len(output["x-dynamo-schema-composition"]["resolved"]), 1)

    def test_hash_mismatch_is_not_partial_success(self):
        self.manifest["openai"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "checksum"):
            compose(self.raw, self.manifest, baseline_bytes=b"{}")

    def test_unmapped_slot_is_not_a_generic_object(self):
        self.manifest["imports"] = {}
        with self.assertRaisesRegex(ValueError, "Unmapped"):
            self.run_composition()

    def test_collision_never_overwrites_owned_component(self):
        self.raw["components"]["schemas"]["OpenAIBaseline.Child"] = {"type": "boolean"}
        with self.assertRaisesRegex(ValueError, "collides"):
            self.run_composition()

    def test_no_network_or_filesystem_reference_resolution(self):
        for reference in (
            "https://example.com/schema",
            "file:///etc/passwd",
            "#/other/schema",
        ):
            with self.subTest(reference=reference):
                self.baseline["components"]["schemas"]["External"] = {"$ref": reference}
                with self.assertRaisesRegex(
                    ValueError, "Unsupported imported reference"
                ):
                    self.run_composition()

    def test_missing_reference_is_an_error(self):
        del self.baseline["components"]["schemas"]["Child"]
        with self.assertRaises(jsonpointer.JsonPointerException):
            self.run_composition()

    def test_subschema_reference_keeps_pointer_suffix(self):
        self.baseline["components"]["schemas"]["External"] = {
            "$ref": "#/components/schemas/Child/properties/parent"
        }
        output = self.run_composition()
        self.assertEqual(
            output["components"]["schemas"]["async_openai.External"]["$ref"],
            "#/components/schemas/OpenAIBaseline.Child/properties/parent",
        )

    def test_guarded_correction_and_stale_guard(self):
        mapping = self.manifest["imports"]["External"]
        mapping.update(
            rationale="Fixture demonstrates a reviewed type correction",
            patch=[
                {"op": "test", "path": "/type", "value": "object"},
                {"op": "replace", "path": "/type", "value": "string"},
            ],
        )
        self.assertEqual(
            self.run_composition()["components"]["schemas"]["async_openai.External"][
                "type"
            ],
            "string",
        )
        mapping["patch"][0]["value"] = "integer"
        with self.assertRaises(jsonpatch.JsonPatchTestFailed):
            self.run_composition()

    def test_unguarded_or_wrong_guarded_correction_rejected(self):
        mapping = self.manifest["imports"]["External"]
        mapping["patch"] = [{"op": "replace", "path": "/type", "value": "string"}]
        with self.assertRaisesRegex(ValueError, "Unguarded"):
            self.run_composition()
        mapping["patch"].insert(
            0, {"op": "test", "path": "/description", "value": "irrelevant"}
        )
        with self.assertRaisesRegex(ValueError, "wrong location"):
            self.run_composition()

    def test_transitive_component_correction_is_guarded_and_recorded(self):
        self.manifest["components"] = {
            "Child": {
                "rationale": "Fixture constraint differs from dependency contract",
                "patch": [
                    {"op": "test", "path": "/type", "value": "object"},
                    {"op": "replace", "path": "/type", "value": ["object", "null"]},
                ],
            }
        }
        result = self.run_composition()
        self.assertEqual(
            result["components"]["schemas"]["OpenAIBaseline.Child"]["type"],
            ["object", "null"],
        )
        self.assertIn(
            "Child", result["x-dynamo-schema-composition"]["component_corrections"]
        )
        self.manifest["components"]["Child"]["patch"][0]["value"] = "string"
        with self.assertRaises(jsonpatch.JsonPatchTestFailed):
            self.run_composition()

    def test_struct_mapping_does_not_replace_tagged_baseline_references(self):
        self.manifest["imports"]["External"]["redirect_references"] = False
        result = self.run_composition()
        parent = result["components"]["schemas"]["OpenAIBaseline.Child"]["properties"][
            "parent"
        ]
        self.assertEqual(parent["$ref"], "#/components/schemas/OpenAIBaseline.External")
        self.assertIn("async_openai.External", result["components"]["schemas"])

    def test_discriminator_mapping_uses_same_reference_closure(self):
        self.baseline["components"]["schemas"]["External"]["discriminator"] = {
            "propertyName": "kind",
            "mapping": {"child": "#/components/schemas/Child"},
        }
        result = self.run_composition()
        discriminator = result["components"]["schemas"]["async_openai.External"][
            "discriminator"
        ]
        self.assertEqual(
            discriminator["mapping"]["child"],
            "#/components/schemas/OpenAIBaseline.Child",
        )

    def test_dynamic_reference_is_not_silently_imported(self):
        self.baseline["components"]["schemas"]["External"]["$dynamicRef"] = "#node"
        with self.assertRaisesRegex(ValueError, "Dynamic references"):
            self.run_composition()


if __name__ == "__main__":
    unittest.main()
