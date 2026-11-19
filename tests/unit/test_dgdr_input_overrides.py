# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Planner interprets the operator's null aliases when checking Run inputs."""

import importlib.util
import json
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.core,
    pytest.mark.planner,
]


@pytest.fixture
def expand_nulls():
    # Load this JSON-only module without the native runtime imported by planner.__init__.
    path = (
        Path(__file__).resolve().parents[2]
        / "components/src/dynamo/planner/retune/live/input_overrides.py"
    )
    module_spec = importlib.util.spec_from_file_location("dgdr_input_overrides", path)
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module.expand_input_override_nulls


@pytest.mark.parametrize(
    ("raw", "want"),
    [
        (
            '{"engine":{"workers":{"prefill":{"$null":true},"decode":null}}}',
            {"engine": {"workers": {"prefill": None, "decode": None}}},
        ),
        (
            '{"values":[null,{"nested":null},[1]]}',
            {"values": [None, {"nested": None}, [1]]},
        ),
        ('{"values":{"$null":true}}', {"values": None}),
        ("null", None),
        ('{"value":{"other":true}}', {"value": {"other": True}}),
        (
            '{"optimizer":{"seed":9007199254740993}}',
            {"optimizer": {"seed": 9007199254740993}},
        ),
    ],
    ids=[
        "worker-deletions",
        "native-array-nulls",
        "whole-array-field",
        "native-null-document",
        "ordinary-object",
        "large-integer",
    ],
)
def test_expand_input_override_nulls(expand_nulls, raw, want):
    document = json.loads(raw)

    assert expand_nulls(document) == want
    assert document == json.loads(raw)


@pytest.mark.parametrize(
    ("raw", "path", "message"),
    [
        (
            '{"value":{"$null":false}}',
            '["value"]',
            "$null must be the boolean true and the only object field",
        ),
        (
            '{"value":{"$null":1}}',
            '["value"]',
            "$null must be the boolean true and the only object field",
        ),
        (
            '{"value":{"$null":"true"}}',
            '["value"]',
            "$null must be the boolean true and the only object field",
        ),
        (
            '{"value":{"$null":null}}',
            '["value"]',
            "$null must be the boolean true and the only object field",
        ),
        (
            '{"value":{"$null":{}}}',
            '["value"]',
            "$null must be the boolean true and the only object field",
        ),
        (
            '{"value":{"$null":[]}}',
            '["value"]',
            "$null must be the boolean true and the only object field",
        ),
        (
            '{"value":{"$null":true,"other":1}}',
            '["value"]',
            "$null must be the boolean true and the only object field",
        ),
        ('{"$null":true}', "", "$null is not allowed at the document root"),
        (
            '{"values":[{"$null":true}]}',
            '["values"][0]',
            "$null is not allowed inside arrays",
        ),
        (
            '{"values":[{"nested":{"$null":true}}]}',
            '["values"][0]["nested"]',
            "$null is not allowed inside arrays",
        ),
        (
            '{"values":[[{"$null":true}]]}',
            '["values"][0][0]',
            "$null is not allowed inside arrays",
        ),
        (
            '{"a.b":{"c/d":{"$null":false}}}',
            '["a.b"]["c/d"]',
            "$null must be the boolean true and the only object field",
        ),
        (
            '{"z":{"$null":false},"a":{"$null":"true"}}',
            '["a"]',
            "$null must be the boolean true and the only object field",
        ),
    ],
    ids=[
        "false",
        "number",
        "string",
        "null",
        "object",
        "list",
        "extra-keys",
        "root",
        "array-element",
        "object-in-array",
        "nested-array",
        "map-key-path",
        "deterministic-error",
    ],
)
def test_reject_invalid_null_sentinels(expand_nulls, raw, path, message):
    document = json.loads(raw)

    with pytest.raises(ValueError) as caught:
        expand_nulls(document)

    assert str(caught.value) == f"spec.aisim.inputOverrides{path}: {message}"
    assert document == json.loads(raw)
