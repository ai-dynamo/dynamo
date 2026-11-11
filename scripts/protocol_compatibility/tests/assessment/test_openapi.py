# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check request projection, aliases, and evidence-linked reporting with synthetic schemas."""

import copy
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import jsonpointer
from jsonschema import Draft202012Validator

from scripts.protocol_compatibility.assessment.openapi import (
    ENDPOINTS,
    alias_value_document,
    compare,
    match_input_aliases,
    normalize_nullable_strings,
    pointer,
    request_differences,
    request_document,
    request_pointer,
)
from scripts.protocol_compatibility.reporting.findings import (
    build_findings,
    related_aliases,
)
from scripts.protocol_compatibility.reporting.openapi import (
    CATEGORY_GUIDE,
    regenerate,
    render,
)


def document():
    """Return independent two-route request/response fixtures with recursive and unused schemas."""
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
    def test_default_reference_is_literal_data_and_never_fetched(self):
        """Schema validation must not resolve a literal $ref in a default object."""
        raw = document()
        literal = {"value": "x", "$ref": "https://example.invalid/not-a-schema"}
        raw["components"]["schemas"]["Request"]["default"] = literal
        with patch(
            "urllib.request.OpenerDirector.open", side_effect=AssertionError("network")
        ):
            projected, gaps = request_document(raw)
            normalized, _ = normalize_nullable_strings(projected)
        self.assertFalse(gaps)
        self.assertEqual(
            normalized["components"]["schemas"]["Request"]["default"], literal
        )
        self.assertEqual(raw["components"]["schemas"]["Request"]["default"], literal)

    def test_openapi32_request_projection_preserves_constraints_not_export_version(
        self,
    ):
        """A 3.2 input remains retained; only the bounded comparator copy uses 3.1."""
        raw = document()
        expected, _ = request_document(raw)
        raw["openapi"] = "3.2.0"
        raw[
            "jsonSchemaDialect"
        ] = "https://spec.openapis.org/oas/3.2/dialect/2025-09-17"
        actual, gaps = request_document(raw)
        self.assertFalse(gaps)
        self.assertEqual(actual, expected)
        self.assertEqual(raw["openapi"], "3.2.0")

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

    @patch("scripts.protocol_compatibility.assessment.openapi.subprocess.run")
    def test_comparator_failure_keeps_bounded_stderr(self, run):
        """Retain a failed comparator's diagnostic tail without dumping unbounded output."""
        run.side_effect = [
            subprocess.CompletedProcess([], 0, "oasdiff version 1.33.0\n"),
            subprocess.CalledProcessError(
                102, ["oasdiff"], stderr="x" * 5000 + 'map key "examples" not found'
            ),
        ]
        with self.assertRaisesRegex(ValueError, "examples.*not found") as raised:
            compare(
                Path("/fixture/oasdiff"), Path("v.json"), Path("d.json"), flatten=False
            )
        self.assertLess(len(str(raised.exception)), 2100)


class AliasMatchingTests(unittest.TestCase):
    def pair(self):
        """Return separate Dynamo/target documents with an explicit alias and different required names."""
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


class FindingReportTests(unittest.TestCase):
    """Exercise retained-delta explanation without reimplementing or running schema comparison."""

    def fixture(self, delta):
        """Return one request delta and independent reference/Dynamo projection fixtures."""
        endpoint = ENDPOINTS[0]
        report = {
            "framework_name": "example-engine",
            "status": "differences_found",
            "differences": [
                {
                    "endpoint": endpoint,
                    "location": request_pointer(endpoint),
                    "delta": {
                        "content": {"modified": {"application/json": {"schema": delta}}}
                    },
                }
            ],
            "response_differences": [],
            "coverage_gaps": [],
            "alias_matches": [],
            "exclusions": [],
        }
        projections = {
            f"{side}.{kind}.normalized.json": document()
            for side in ("framework", "dynamo")
            for kind in ("requests", "responses")
        }
        return report, projections

    def test_category_guide_and_details_follow_review_order(self):
        """Render each category once in review order without changing evidence or dropping unknowns."""
        order = [
            "target_only",
            "both_declared_mismatch",
            "representation_ambiguity",
            "missing_coverage",
            "dynamo_only",
            "alias_evidence",
            "intentional_exclusion",
        ]
        self.assertEqual([entry[0] for entry in CATEGORY_GUIDE], order)
        report, sources = self.fixture({"properties": {"deleted": ["value"]}})
        inventory = build_findings(report, sources)
        self.assertNotIn(
            "**Next check:**", render({**report, "finding_inventory": inventory})
        )
        template = inventory["findings"][0]
        # Presentation fixture: reverse order to prove rendering does not inherit it.
        inventory["findings"] = [
            {**template, "id": name, "category": name}
            for name in reversed(order + ["future_category"])
        ]
        inventory["counts"]["categories"] = {
            row["category"]: 1 for row in inventory["findings"]
        }
        report["finding_inventory"] = inventory
        before = copy.deepcopy(report)
        markdown = render(report)
        introduction = markdown.split("## Target-only declarations\n", 1)[1].split(
            "### ", 1
        )[0]
        self.assertIn("missing input declarations do not prove rejection", introduction)
        self.assertIn("probe its declared boundary and backend effect", introduction)
        self.assertIn("serializer and client decoder", introduction)
        self.assertEqual(introduction.count("**Next action:**"), 2)
        self.assertEqual(markdown.count(introduction), 1)
        table_positions, section_positions = [], []
        for name, title, description in CATEGORY_GUIDE:
            escaped = name.replace("_", "\\_")
            table_positions.append(markdown.index(f"| {escaped} | 1 | {description} |"))
            section_positions.append(markdown.index("## " + title))
            self.assertEqual(markdown.count(f"### {name} —"), 1)
        self.assertEqual(table_positions, sorted(table_positions))
        self.assertEqual(section_positions, sorted(section_positions))
        self.assertGreater(
            markdown.index("## future\\_category"), section_positions[-1]
        )
        self.assertIn("not to assign severity", markdown)
        self.assertNotIn("raw candidate terminals", markdown)
        self.assertIn("Exact declared alias matches appear as alias evidence", markdown)
        self.assertEqual(report, before)
        empty, sources = self.fixture({})
        empty["finding_inventory"] = build_findings(empty, sources)
        self.assertIn("| target\\_only | 0 |", render(empty))
        self.assertNotIn("## Target-only declarations", render(empty))

    def test_declared_aliases_replace_one_sided_spelling_categories(self):
        """Known spellings share alias evidence; value differences and raw IDs survive."""
        target_name, dynamo_name = "chat_template_kwargs", "chat_template_args"
        report, sources = self.fixture(
            {
                "properties": {
                    "added": [dynamo_name],
                    "deleted": [target_name],
                }
            }
        )
        target = sources["framework.requests.normalized.json"]
        dynamo = sources["dynamo.requests.normalized.json"]
        target["components"]["schemas"]["Request"]["properties"] = {
            target_name: {"type": "string"},
        }
        dynamo["components"]["schemas"]["Request"]["properties"] = {
            dynamo_name: {"type": "integer", "x-dynamo-input-aliases": [target_name]},
        }
        baseline = build_findings(report, sources)
        matches, gaps = match_input_aliases(dynamo, target)
        self.assertFalse(gaps)
        match = next(item for item in matches if item["endpoint"] == ENDPOINTS[0])
        match["value_comparison"] = "differences_found"
        report["alias_matches"] = [match]
        before = copy.deepcopy((report, sources))
        result = build_findings(report, sources)
        spelling_rows = [row for row in result["findings"] if row["kind"] == "request"]
        self.assertEqual(len(spelling_rows), 2)
        for row in spelling_rows:
            self.assertEqual(row["category"], "alias_evidence")
            self.assertIn("not a missing field", row["interpretation"])
            self.assertNotIn("only declaration", row["interpretation"])
            self.assertEqual(row["alias_report_pointers"], ["#/alias_matches/0"])
            self.assertTrue(
                all(side["evidence"] for side in row["declarations"].values())
            )
        self.assertEqual(
            {row["id"]: row["delta"] for row in baseline["findings"]},
            {row["id"]: row["delta"] for row in spelling_rows},
        )
        alias_row = next(
            row for row in result["findings"] if row["kind"] == "alias_evidence"
        )
        self.assertIn(
            "Value comparison: differences_found", alias_row["interpretation"]
        )
        self.assertEqual((report, sources), before)
        report["alias_matches"] = [{**match, "name_coverage": "unknown"}]
        categories = {
            row["category"]
            for row in build_findings(report, sources)["findings"]
            if row["kind"] == "request"
        }
        self.assertEqual(categories, {"target_only", "dynamo_only"})
        report["alias_matches"] = [
            {**match, "dynamo_location": "#/components/schemas/Missing"}
        ]
        self.assertTrue(
            all(
                row["category"] == "missing_coverage"
                for row in build_findings(report, sources)["findings"]
                if row["kind"] == "request"
            )
        )

    def test_alias_links_require_matching_instance_path_and_endpoint(self):
        """A nested alias must not reclassify its parent or an unrelated same-name field."""
        match = {
            "endpoint": ENDPOINTS[0],
            "path": ["messages", None, "reasoning"],
            "backend_name": "reasoning",
            "dynamo_name": "reasoning_content",
        }
        for name in ("reasoning", "reasoning_content"):
            steps = ("properties", "messages", "items", "properties", name)
            self.assertEqual(
                related_aliases([match], ENDPOINTS[0], steps, exact=True), [0]
            )
            self.assertEqual(
                related_aliases([match], ENDPOINTS[1], steps, exact=True), []
            )
            self.assertEqual(
                related_aliases([match], ENDPOINTS[0], ("properties", name)), []
            )
        parent = ("properties", "messages")
        self.assertEqual(related_aliases([match], ENDPOINTS[0], parent), [0])
        self.assertEqual(related_aliases([match], ENDPOINTS[0], parent, exact=True), [])
        self.assertEqual(related_aliases([match], ENDPOINTS[0], ()), [])
        self.assertEqual(
            related_aliases([match], ENDPOINTS[0], ("additionalProperties",)), []
        )

    def test_declaration_categories_follow_source_presence(self):
        """Split request/response declarations; missing evidence is not field absence."""
        cases = (
            ({"type": "string"}, None, "target_only"),
            ({"type": "string"}, {"type": "integer"}, "both_declared_mismatch"),
            (None, {}, "dynamo_only"),
            (None, None, "missing_coverage"),
        )
        for kind in ("requests", "responses"):
            for target, dynamo, expected in cases:
                with self.subTest(kind=kind, category=expected):
                    change = "deleted" if dynamo is None else "added"
                    delta = {"properties": {change: ["value"]}}
                    if target is not None and dynamo is not None:
                        delta = {
                            "properties": {
                                "modified": {
                                    "value": {
                                        "type": {
                                            "deleted": ["string"],
                                            "added": ["integer"],
                                        }
                                    }
                                }
                            }
                        }
                    report, sources = self.fixture(delta)
                    if kind == "responses":
                        request = report["differences"].pop()
                        report["response_differences"] = [
                            {
                                "endpoint": ENDPOINTS[0],
                                "location": request["location"].split("/requestBody")[0]
                                + "/responses",
                                "delta": {"modified": {"200": request["delta"]}},
                            }
                        ]
                    for side, schema in (("framework", target), ("dynamo", dynamo)):
                        name = "Request" if kind == "requests" else "Response"
                        sources[f"{side}.{kind}.normalized.json"]["components"][
                            "schemas"
                        ][name] = {
                            "type": "object",
                            "properties": {} if schema is None else {"value": schema},
                        }
                    result = build_findings(report, sources)
                    self.assertEqual(result["findings"][0]["category"], expected)
                    self.assertNotIn(
                        "declaration_mismatch", result["counts"]["categories"]
                    )
                    del sources[f"framework.{kind}.normalized.json"]
                    self.assertEqual(
                        build_findings(report, sources)["findings"][0]["category"],
                        "missing_coverage",
                    )

    def test_missing_default_is_not_null_and_constraints_remain_readable(self):
        """Read annotation presence from retained sources, not oasdiff's null sentinel."""
        delta = {
            "properties": {
                "modified": {
                    "value": {
                        "default": {"from": 1, "to": None},
                        "min": {"from": None, "to": 0},
                        "type": {"added": ["integer"], "deleted": ["string"]},
                    }
                }
            },
            "required": {"added": ["model"], "deleted": ["value"]},
        }
        report, sources = self.fixture(delta)
        reference = sources["framework.requests.normalized.json"]["components"][
            "schemas"
        ]["Request"]
        dynamo = sources["dynamo.requests.normalized.json"]["components"]["schemas"][
            "Request"
        ]
        reference["properties"]["value"] = {"type": "string", "default": 1}
        dynamo["properties"]["value"] = {"type": ["integer", "null"], "minimum": 0}
        del dynamo["required"]
        dynamo["allOf"] = [{"required": ["model"]}]
        before = copy.deepcopy((report, sources))
        result = build_findings(report, sources)
        by_change = {row["change"]: row for row in result["findings"]}
        self.assertTrue(
            all(
                row["category"] == "both_declared_mismatch"
                for row in result["findings"]
            )
        )
        self.assertIn(
            "default not directly declared",
            by_change["default"]["declarations"]["dynamo"]["summary"],
        )
        self.assertEqual(
            by_change["min"]["declarations"]["dynamo"]["summary"], "minimum=0"
        )
        self.assertIn('"null"', by_change["type"]["declarations"]["dynamo"]["summary"])
        self.assertIn(
            "required not directly declared",
            by_change["required"]["declarations"]["dynamo"]["summary"],
        )
        self.assertEqual((report, sources), before)
        self.assertIn(
            "**Next check:**", render({**report, "finding_inventory": result})
        )
        self.assertIn(
            "Reported required-member change: added model; deleted value.",
            render({**report, "finding_inventory": result}),
        )
        dynamo["properties"]["value"]["default"] = None
        explicit = next(
            row
            for row in build_findings(report, sources)["findings"]
            if row["change"] == "default"
        )
        self.assertEqual(explicit["declarations"]["dynamo"]["summary"], "default=null")
        self.assertEqual(explicit["id"], by_change["default"]["id"])

    def test_union_and_allof_movement_stay_one_ambiguous_group(self):
        """A root field removed beside added allOf is not reported as a removed wire field."""
        delta = {
            "allOf": {"added": [{"properties": {"value": {"type": "string"}}}]},
            "properties": {"deleted": ["value"]},
            "required": {"deleted": ["value"]},
        }
        report, sources = self.fixture(delta)
        result = build_findings(report, sources)
        self.assertEqual(len(result["findings"]), 1)
        row = result["findings"][0]
        self.assertEqual(row["category"], "representation_ambiguity")
        self.assertEqual(row["delta"], delta)
        self.assertIn("not established", row["interpretation"])
        for keyword in ("anyOf", "oneOf"):
            report, sources = self.fixture(
                {"properties": {"modified": {"value": {keyword: {"deleted": [0]}}}}}
            )
            self.assertEqual(
                build_findings(report, sources)["findings"][0]["category"],
                "representation_ambiguity",
            )

    def test_nested_fields_unknown_keys_and_empty_fragments_are_accounted(self):
        """Unknown sibling keys must have distinct stable IDs, including empty deltas."""
        delta = {
            "properties": {
                "modified": {
                    "value": {
                        "items": {
                            "unknown-a": {"added": [1]},
                            "unknown-b": {},
                            "minLength": {"from": 0, "to": 2},
                        }
                    }
                }
            }
        }
        report, sources = self.fixture(delta)
        result = build_findings(report, sources)
        self.assertEqual(len(result["findings"]), 3)
        self.assertEqual(len({row["id"] for row in result["findings"]}), 3)
        self.assertTrue(all(row["field"] == "value[]" for row in result["findings"]))
        self.assertEqual(
            result["accounting"],
            {"raw_terminal_count": 4, "owned_terminal_count": 4},
        )
        report["differences"][0]["delta"] = {
            "content": {"modified": {"application/json": {}}}
        }
        self.assertEqual(len(build_findings(report, sources)["findings"]), 1)

    def test_gap_transport_context_uses_operation_structure(self):
        """Only operation containers supply status/media; property names and truncated paths do not."""
        endpoint = ENDPOINTS[0]
        operation = pointer("paths", endpoint, "post")
        component = "#/components/schemas/Request/properties"
        cases = [
            (component + "/content", None, None, "JSON"),
            (component + "/responses", None, None, "JSON"),
            (component + "/content/properties/responses/type", None, None, "JSON"),
            (operation, None, None, "JSON"),
            (operation + "/requestBody/content", None, None, "JSON"),
            (operation + "/responses", None, None, "HTTP"),
            (operation + "/responses/400", "400", None, "HTTP"),
            (operation + "/responses/200/content", "200", None, "HTTP"),
            (
                operation
                + "/responses/200/content/text~1event-stream/schema/properties/content",
                "200",
                "text/event-stream",
                "SSE",
            ),
            (
                operation
                + "/requestBody/content/application~1json/schema/properties/responses",
                None,
                "application/json",
                "JSON",
            ),
            (
                pointer(
                    "paths",
                    "/other",
                    "post",
                    "responses",
                    "200",
                    "content",
                    "text/event-stream",
                ),
                None,
                None,
                "JSON",
            ),
        ]
        for location, status, media, transport in cases:
            with self.subTest(location=location):
                report, sources = self.fixture({})
                report["differences"] = []
                report["coverage_gaps"] = [
                    {
                        "kind": "response"
                        if location.startswith(operation + "/responses")
                        else "request",
                        "endpoint": endpoint,
                        "location": location,
                        "reason": "Unsupported declaration",
                    }
                ]
                (row,) = build_findings(report, sources)["findings"]
                self.assertEqual(
                    (row["status"], row["media"], row["transport"]),
                    (status, media, transport),
                )

    def test_response_sse_error_alias_and_gap_evidence_remain_separate(self):
        """Retain transport/status context and aliases without promoting them to runtime proof."""
        report, sources = self.fixture({"properties": {"deleted": ["value"]}})
        endpoint = ENDPOINTS[0]
        location = request_pointer(endpoint)
        report["response_differences"] = [
            {
                "endpoint": endpoint,
                "location": location.split("/requestBody")[0] + "/responses",
                "delta": {
                    "added": ["503"],
                    "modified": {
                        "200": {
                            "content": {
                                "modified": {
                                    "text/event-stream": {
                                        "schema": {"anyOf": {"added": [1]}}
                                    }
                                }
                            }
                        }
                    },
                },
            }
        ]
        report["coverage_gaps"] = [
            {
                "kind": "response",
                "side": "framework",
                "endpoint": endpoint,
                "location": report["response_differences"][0]["location"] + "/400",
                "reason": "Body not declared",
            }
        ]
        report["alias_matches"] = [
            {
                "endpoint": endpoint,
                "path": ["value"],
                "backend_name": "value",
                "dynamo_name": "alias",
                "backend_location": "#/components/schemas/Request/properties/value",
                "dynamo_location": "#/components/schemas/Request/properties/value",
            }
        ]
        report["exclusions"] = [
            {"field": "nvext", "reason": "Dynamo-specific extension"}
        ]
        result = build_findings(report, sources)
        self.assertTrue(
            any(
                row["transport"] == "SSE" and row["status"] == "200"
                for row in result["findings"]
            )
        )
        self.assertTrue(
            any(
                row["status"] == "503"
                and "Extra possible output" in row["interpretation"]
                for row in result["findings"]
            )
        )
        self.assertTrue(any(row["alias_report_pointers"] for row in result["findings"]))
        self.assertEqual(result["counts"]["categories"]["alias_evidence"], 1)
        self.assertEqual(result["counts"]["categories"]["intentional_exclusion"], 1)

    def test_reported_change_text_is_escaped(self):
        """Keep schema-provided member names from adding HTML or headings to the report."""
        report, sources = self.fixture(
            {"required": {"added": ["<script>x</script>\n## fake"]}}
        )
        inventory = build_findings(report, sources)
        markdown = render({**report, "finding_inventory": inventory})
        self.assertNotIn("<script>", markdown)
        self.assertNotIn("\n## fake", markdown)

    def test_accounting_rejects_double_ownership(self):
        """Reject findings when the walker assigns one raw delta twice."""
        report, sources = self.fixture({"properties": {"added": ["value"]}})

        def twice(func, *bound):
            """Replace partial with a callback that emits each finding twice."""
            return lambda *args: (func(*bound, *args), func(*bound, *args))

        with patch("scripts.protocol_compatibility.reporting.findings.partial", twice):
            with self.assertRaisesRegex(ValueError, "exactly once"):
                build_findings(report, sources)

    def test_accounting_rejects_unowned_terminals(self):
        """Reject findings when the walker leaves a raw delta unassigned."""
        report, sources = self.fixture({"properties": {"added": ["value"]}})

        def never(func, *bound):
            """Replace partial with a callback that emits no findings."""
            return lambda *args: None

        with patch("scripts.protocol_compatibility.reporting.findings.partial", never):
            with self.assertRaisesRegex(ValueError, "exactly once"):
                build_findings(report, sources)

    def test_regeneration_is_portable_escaped_and_preserves_input(self):
        """Rendering copies frozen evidence, rejects overwrite/nesting, and escapes schema text."""
        report, sources = self.fixture({"properties": {"added": ["value"]}})
        report[
            "framework_name"
        ] = "<script>alert(1)</script> | [x](javascript:bad)\n# heading"
        report_before = copy.deepcopy(report)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frozen, output = root / "frozen", root / "rendered"
            frozen.mkdir()
            (frozen / "report.json").write_text(json.dumps(report))
            for name, value in sources.items():
                (frozen / name).write_text(json.dumps(value))
            before = {path.name: path.read_bytes() for path in frozen.iterdir()}
            result = regenerate(frozen, output)
            self.assertFalse((output / "findings.json").exists())
            self.assertEqual(
                {path.name: path.read_bytes() for path in frozen.iterdir()}, before
            )
            self.assertEqual(
                {k: v for k, v in result.items() if k != "finding_inventory"},
                report_before,
            )
            markdown = (output / "report.md").read_text()
            self.assertIn(
                "[Finding details](report.json#/finding_inventory/findings/0)", markdown
            )
            self.assertIn("[Target declaration 1]", markdown)
            self.assertIn("[Raw delta 1]", markdown)
            self.assertNotIn("Location: #/", markdown)
            self.assertNotIn("<script>", markdown)
            self.assertNotIn("[x](javascript:", markdown)
            self.assertIn("evidence/dynamo.requests.normalized.json#", markdown)
            for row in result["finding_inventory"]["findings"]:
                for raw in row["raw_report_pointers"]:
                    jsonpointer.resolve_pointer(result, raw[1:])
                for side in row["declarations"].values():
                    for record in side["evidence"]:
                        retained = json.loads(
                            (output / "evidence" / record["file"]).read_text()
                        )
                        self.assertEqual(
                            jsonpointer.resolve_pointer(
                                retained, record["pointer"][1:]
                            ),
                            record["declaration"],
                        )
            with self.assertRaises(FileExistsError):
                regenerate(frozen, output)
            with self.assertRaisesRegex(ValueError, "inside"):
                regenerate(frozen, frozen / "nested")
        self.assertIn("projection unavailable", render(report))


if __name__ == "__main__":
    unittest.main()
