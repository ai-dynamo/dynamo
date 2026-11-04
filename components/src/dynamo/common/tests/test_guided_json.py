# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import json

import pytest

from dynamo.common.utils.guided_json import (
    build_required_tool_schema,
    reject_nonprogressing_guided_json_ref_cycles,
)
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


def _resolve_pointer(document, ref):
    """Follow a ``#/a/b`` reference inside ``document``; None when it dangles."""
    assert ref.startswith("#")
    target = document
    for token in ref[1:].split("/")[1:]:
        token = token.replace("~1", "/").replace("~0", "~")
        if not isinstance(target, dict) or token not in target:
            return None
        target = target[token]
    return target


def _assert_backend_accepts(schema):
    reject_nonprogressing_guided_json_ref_cycles(schema)
    xgrammar = pytest.importorskip("xgrammar")
    xgrammar.Grammar.from_json_schema(json.dumps(schema))


_PLAIN_PARAMETERS = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
}


# Mirrors required_choice_preserves_local_references_without_defs in
# lib/llm/src/protocols/openai/tools.rs.
@pytest.mark.parametrize(
    ("parameters", "reference_path", "target"),
    [
        pytest.param(
            {
                "type": "object",
                "properties": {"value": {"$ref": "#/definitions/value"}},
                "definitions": {"value": {"type": "string"}},
            },
            ("properties", "value", "$ref"),
            {"type": "string"},
            id="legacy-definitions",
        ),
        pytest.param(
            {
                "type": "object",
                "properties": {
                    "value": {"type": "string"},
                    "alias": {"$ref": "#/properties/value"},
                },
            },
            ("properties", "alias", "$ref"),
            {"type": "string"},
            id="property-pointer",
        ),
        pytest.param(
            {"type": "object", "properties": {"child": {"$ref": "#"}}},
            ("properties", "child", "$ref"),
            {
                "type": "object",
                "properties": {"child": {"$ref": "#/$defs/tool_0"}},
            },
            id="recursive-root",
        ),
    ],
)
def test_required_tool_schema_preserves_local_references(
    parameters, reference_path, target
):
    original = copy.deepcopy(parameters)

    schema = build_required_tool_schema([("lookup", parameters)], None)

    entry = schema["items"]["anyOf"][0]
    assert entry["properties"]["name"] == {"type": "string", "enum": ["lookup"]}
    assert entry["properties"]["parameters"] == {"$ref": "#/$defs/tool_0"}
    reference = schema["$defs"]["tool_0"]
    for key in reference_path:
        reference = reference[key]
    assert _resolve_pointer(schema, reference) == target
    assert parameters == original
    _assert_backend_accepts(schema)


# Mirrors required_choice_isolates_conflicting_definitions.
def test_required_tool_schema_isolates_conflicting_definitions():
    tools = [
        (
            name,
            {
                "type": "object",
                "properties": {"value": {"$ref": "#/$defs/shared"}},
                "$defs": {"shared": {"type": kind}},
            },
        )
        for name, kind in [("as_text", "string"), ("as_number", "number")]
    ]

    schema = build_required_tool_schema(tools, None)

    assert set(schema["$defs"]) == {"tool_0", "tool_1"}
    for index, kind in enumerate(["string", "number"]):
        root = f"#/$defs/tool_{index}"
        assert schema["items"]["anyOf"][index]["properties"]["parameters"] == {
            "$ref": root
        }
        parameters = _resolve_pointer(schema, root)
        reference = parameters["properties"]["value"]["$ref"]
        assert reference == f"{root}/$defs/shared"
        assert _resolve_pointer(schema, reference) == {"type": kind}
    assert tools[0][1]["properties"]["value"]["$ref"] == "#/$defs/shared"
    assert tools[0][1]["$defs"] == {"shared": {"type": "string"}}
    _assert_backend_accepts(schema)


# Mirrors required_choice_rebases_only_document_local_schema_references.
def test_required_tool_schema_rebases_only_document_local_schema_references():
    parameters = {
        "type": "object",
        "properties": {
            "value": {"$ref": "#/$defs/a~1b~0c"},
            "child": {"$ref": "#"},
            "encoded": {"$ref": "#%2F$defs%2Fa~1b~0c"},
            "remote": {"$ref": "urn:example:remote"},
            "relative": {"$ref": "a%2Fremote"},
            "scoped": {
                "$id": "urn:example:scoped",
                "$ref": "#/$defs/local",
                "$defs": {"local": {"type": "string"}},
            },
            "literal": {"const": {"$ref": "#/$defs/a~1b~0c"}},
        },
        "$defs": {
            "a/b~c": {"anyOf": [{"type": "string"}, {"$ref": "#/$defs/a~1b~0c"}]}
        },
        "default": {"$ref": "#"},
    }
    original = copy.deepcopy(parameters)

    schema = build_required_tool_schema([("lookup", parameters)], None)

    rebased = schema["$defs"]["tool_0"]
    properties = rebased["properties"]
    assert properties["value"]["$ref"] == "#/$defs/tool_0/$defs/a~1b~0c"
    assert _resolve_pointer(schema, properties["value"]["$ref"]) == (
        rebased["$defs"]["a/b~c"]
    )
    assert properties["child"]["$ref"] == "#/$defs/tool_0"
    assert properties["encoded"]["$ref"] == "#/$defs/tool_0%2F$defs%2Fa~1b~0c"
    assert (
        rebased["$defs"]["a/b~c"]["anyOf"][1]["$ref"] == "#/$defs/tool_0/$defs/a~1b~0c"
    )
    assert properties["remote"]["$ref"] == "urn:example:remote"
    assert properties["relative"]["$ref"] == "a%2Fremote"
    assert properties["scoped"] == original["properties"]["scoped"]
    assert properties["literal"] == {"const": {"$ref": "#/$defs/a~1b~0c"}}
    assert rebased["default"] == {"$ref": "#"}
    assert parameters == original


@pytest.mark.parametrize("parameters", [None, {}])
def test_required_tool_schema_keeps_plain_tools_inline(parameters):
    schema = build_required_tool_schema(
        [("get_weather", _PLAIN_PARAMETERS), ("ping", parameters)], None
    )

    assert schema == {
        "type": "array",
        "minItems": 1,
        "items": {
            "type": "object",
            "anyOf": [
                {
                    "properties": {
                        "name": {"type": "string", "enum": ["get_weather"]},
                        "parameters": _PLAIN_PARAMETERS,
                    },
                    "required": ["name", "parameters"],
                },
                {
                    "properties": {
                        "name": {"type": "string", "enum": ["ping"]},
                        "parameters": {"type": "object", "properties": {}},
                    },
                    "required": ["name", "parameters"],
                },
            ],
        },
    }
    assert (
        schema["items"]["anyOf"][0]["properties"]["parameters"] is not _PLAIN_PARAMETERS
    )


def test_required_tool_schema_keeps_id_resource_inline():
    parameters = {
        "$id": "urn:example:args",
        "type": "object",
        "properties": {"value": {"$ref": "#/definitions/value"}},
        "definitions": {"value": {"type": "string"}},
    }

    schema = build_required_tool_schema([("lookup", parameters)], None)

    assert "$defs" not in schema
    assert schema["items"]["anyOf"][0]["properties"]["parameters"] == parameters


@pytest.mark.parametrize(
    ("parallel_tool_calls", "max_items"),
    [(None, None), (True, None), (False, 1)],
)
def test_required_tool_schema_caps_only_disabled_parallel_calls(
    parallel_tool_calls, max_items
):
    schema = build_required_tool_schema(
        [("get_weather", _PLAIN_PARAMETERS)], parallel_tool_calls
    )

    assert schema.get("maxItems") == max_items


def test_required_tool_schema_rejects_non_object_defs():
    with pytest.raises(
        ValueError, match=r"`\$defs` .* tool 'lookup' must be an object"
    ):
        build_required_tool_schema(
            [("lookup", {"type": "object", "$defs": ["shared"]})], None
        )
