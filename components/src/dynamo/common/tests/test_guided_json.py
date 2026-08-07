# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for guided JSON schema validation."""

import json

import pytest

from dynamo.common.utils.guided_json import validate_guided_json_schema
from dynamo.llm import HttpError

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
]


@pytest.mark.parametrize(
    "schema",
    [
        {"$ref": "#"},
        json.dumps({"$ref": "#"}),
        {"$ref": "#", "type": "object"},
        {
            "$defs": {
                "A": {"$ref": "#/$defs/B"},
                "B": {"$ref": "#/$defs/A"},
            },
            "$ref": "#/$defs/A",
        },
        {
            "$defs": {
                "A": {"$ref": "#/$defs/B"},
                "B": {"$ref": "#/$defs/C"},
                "C": {"$ref": "#/$defs/A"},
            },
            "$ref": "#/$defs/A",
        },
        {
            "$defs": {
                "A": {"allOf": [{"$ref": "#/$defs/A"}]},
            },
            "$ref": "#/$defs/A",
        },
        {
            "type": "string",
            "$defs": {
                "A": {"$ref": "#/$defs/B"},
                "B": {"$ref": "#/$defs/A"},
            },
        },
    ],
    ids=[
        "root-self-reference",
        "serialized-root-self-reference",
        "root-self-reference-with-sibling",
        "two-definition-cycle",
        "three-definition-cycle",
        "cycle-through-all-of",
        "unused-definition-cycle",
    ],
)
def test_rejects_unproductive_local_reference_cycles(schema):
    with pytest.raises(
        HttpError,
        match="unproductive circular.*ref",
    ) as error:
        validate_guided_json_schema(schema)

    assert error.value.code == 400


@pytest.mark.parametrize(
    "schema",
    [
        {
            "type": "object",
            "properties": {"city": {"type": "string"}},
        },
        {
            "$defs": {"Name": {"type": "string"}},
            "$ref": "#/$defs/Name",
        },
        {
            "$defs": {
                "Node": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "string"},
                        "next": {"$ref": "#/$defs/Node"},
                    },
                }
            },
            "$ref": "#/$defs/Node",
        },
        {
            "$defs": {
                "Node": {
                    "anyOf": [
                        {"type": "null"},
                        {
                            "type": "array",
                            "items": {"$ref": "#/$defs/Node"},
                        },
                    ]
                }
            },
            "$ref": "#/$defs/Node",
        },
        {
            "$defs": {"a/b~c": {"type": "integer"}},
            "$ref": "#/$defs/a~1b~0c",
        },
        {
            "type": "object",
            "properties": {"$ref": {"type": "string"}},
        },
    ],
    ids=[
        "ordinary-object",
        "acyclic-definition",
        "recursive-object-property",
        "recursive-array-items",
        "escaped-json-pointer",
        "property-named-ref",
    ],
)
def test_accepts_productive_or_acyclic_schemas(schema):
    validate_guided_json_schema(schema)


@pytest.mark.parametrize(
    "schema",
    [
        {"$ref": "#/$defs/Missing"},
        {"$defs": {"A": {"type": "string"}}, "$ref": "#/$defs/A~2B"},
        {"$defs": {"A": 1}, "$ref": "#/$defs/A"},
        {"$ref": 123},
    ],
    ids=[
        "missing-target",
        "invalid-pointer-escape",
        "non-schema-target",
        "non-string-ref",
    ],
)
def test_rejects_malformed_local_references(schema):
    with pytest.raises(HttpError) as error:
        validate_guided_json_schema(schema)

    assert error.value.code == 400


def test_leaves_external_references_to_the_backend():
    validate_guided_json_schema({"$ref": "https://example.com/schema.json"})
