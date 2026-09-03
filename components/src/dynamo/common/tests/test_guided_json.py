# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from dynamo.common.utils.guided_json import reject_nonprogressing_guided_json_ref_cycles
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
        {
            "$defs": {
                "A": {"$ref": "#/$defs/B"},
                "B": {"$ref": "#/$defs/A"},
            },
            "$ref": "#/$defs/A",
        },
        {
            "$defs": {
                "A": {"$id": "urn:example:A", "$ref": "#"},
            },
            "$ref": "#/$defs/A",
        },
        {
            "$defs": {
                "a/b": {"$ref": "#/$defs/c~0d"},
                "c~d": {"$ref": "#/$defs/a~1b"},
            },
            "$ref": "#/$defs/a~1b",
        },
    ],
)
def test_rejects_cyclic_root_ref_chains(schema):
    with pytest.raises(HttpError, match=r"non-progressing local \$ref cycle") as error:
        reject_nonprogressing_guided_json_ref_cycles(schema)

    assert error.value.code == 400


@pytest.mark.parametrize(
    "schema",
    [
        pytest.param(
            {
                "$defs": {
                    "A": {"allOf": [{"$ref": "#/$defs/A"}]},
                },
                "$ref": "#/$defs/A",
            },
            id="allof",
        ),
        pytest.param(
            {
                "type": "object",
                "properties": {"value": {"$ref": "#/$defs/A"}},
                "required": ["value"],
                "$defs": {"A": {"$ref": "#/$defs/A"}},
            },
            id="required-property",
        ),
    ],
)
def test_rejects_nonprogressing_ref_cycles(schema):
    with pytest.raises(HttpError, match=r"non-progressing local \$ref cycle") as error:
        reject_nonprogressing_guided_json_ref_cycles(schema)

    assert error.value.code == 400


@pytest.mark.parametrize(
    "schema",
    [
        {
            "type": "string",
            "$defs": {
                "A": {"$ref": "#/$defs/B"},
                "B": {"$ref": "#/$defs/A"},
            },
        },
        {
            "$defs": {
                "Node": {
                    "type": "object",
                    "properties": {
                        "next": {"$ref": "#/$defs/Node"},
                    },
                }
            },
            "$ref": "#/$defs/Node",
        },
        {"anyOf": [{"type": "string"}, {"$ref": "#"}]},
        {
            "$defs": {
                "A": {
                    "$id": "urn:example:A",
                    "$defs": {"B": {"type": "string"}},
                    "$ref": "#/$defs/B",
                }
            },
            "$ref": "#/$defs/A",
        },
    ],
)
def test_allows_schemas_outside_cyclic_root_ref_chains(schema):
    reject_nonprogressing_guided_json_ref_cycles(schema)


@pytest.mark.parametrize(
    "schema",
    [
        {"$ref": "#/$defs/Missing"},
        {"$ref": "https://example.com/schema.json"},
        {"$ref": 123},
    ],
)
def test_leaves_unresolved_references_to_backend(schema):
    reject_nonprogressing_guided_json_ref_cycles(schema)
