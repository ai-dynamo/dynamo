# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from schema_validation import (  # noqa: E402
    SchemaValidationUnavailable,
    validate_json_schema,
)

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


def test_decoded_string_constraints_and_extra_arguments_are_checked():
    schema = {
        "type": "object",
        "properties": {
            "city": {"type": "string", "minLength": 3, "pattern": "^[A-Z]+$"}
        },
        "required": ["city"],
        "additionalProperties": False,
    }
    violations = validate_json_schema({"city": "a", "extra": "b"}, schema)
    assert len(violations) == 3
    assert any("too short" in item for item in violations)
    assert any("does not match" in item for item in violations)
    assert any("Additional properties" in item for item in violations)
    assert validate_json_schema({"city": "ROME"}, schema) == []


@pytest.mark.parametrize("value", ("<a", "東京", "🌧a"))
def test_string_bounds_count_unicode_characters_after_parsing(value):
    schema = {"type": "string", "minLength": 2, "maxLength": 2}
    assert validate_json_schema(value, schema) == []
    assert validate_json_schema(value + "b", schema)
    assert validate_json_schema(value[0], schema)


def test_quotes_do_not_count_after_the_parser_decodes_a_string():
    schema = {"type": "string", "minLength": 3}
    assert validate_json_schema('"a"', schema) == []
    assert validate_json_schema("a", schema)


def test_escaped_reference_is_resolved_once_and_required_cannot_be_bypassed():
    schema = {
        "$ref": "#/properties/a~00b",
        "properties": {
            "a~0b": {
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
            },
            "a~b": {"type": "object"},
        },
    }
    assert validate_json_schema({}, schema)
    assert validate_json_schema({"value": 42}, schema) == []


def test_local_reference_composition_and_boolean_schemas():
    schema = {
        "$defs": {"Count": {"type": "integer", "minimum": 1}},
        "type": "object",
        "properties": {
            "count": {"$ref": "#/$defs/Count"},
            "result": {"anyOf": [{"type": "null"}, {"const": "ok"}]},
        },
        "required": ["count", "result"],
    }
    assert validate_json_schema({"count": 1, "result": None}, schema) == []
    assert len(validate_json_schema({"count": 0, "result": "bad"}, schema)) == 2
    assert validate_json_schema({}, True) == []
    assert validate_json_schema({}, False)


def test_declared_supported_dialect_is_honored():
    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "array",
        "items": [{"type": "integer"}, {"type": "string"}],
        "additionalItems": False,
    }
    assert validate_json_schema([1, "one"], schema) == []
    assert validate_json_schema([1, "one", True], schema)


@pytest.mark.parametrize(
    "schema",
    (
        {"$ref": "https://example.invalid/schema"},
        {"$ref": "file:///etc/passwd"},
        {"$ref": "#/$defs/Missing"},
        {"$schema": "https://example.invalid/dialect", "type": "string"},
        {"$id": "https://example.invalid/base", "$ref": "#/properties/value"},
        {"type": "not-a-type"},
        {"$ref": "#"},
        {"$ref": "#/enum/0", "enum": [{"type": "not-a-type"}]},
        {
            "$ref": "#/enum/0",
            "enum": [{"$schema": "https://example.invalid/dialect"}],
        },
    ),
)
def test_unverifiable_schema_is_visible_without_network_retrieval(schema):
    with pytest.raises(SchemaValidationUnavailable):
        validate_json_schema({}, schema)


def test_schema_shaped_literal_data_does_not_trigger_reference_rejection():
    value = {"$ref": "https://example.invalid/not-a-schema"}
    assert validate_json_schema(value, {"enum": [value]}) == []


def test_nested_error_keeps_the_callers_path():
    assert validate_json_schema(
        {"items": [False]},
        {"properties": {"items": {"items": {"type": "integer"}}}},
        path="$.arguments",
    ) == ["$.arguments.items[0]: False is not of type 'integer'"]
