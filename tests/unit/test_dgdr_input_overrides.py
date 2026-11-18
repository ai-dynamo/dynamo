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
            '{"values":[{"$null":true},{"nested":{"$null":true}},null]}',
            {"values": [None, {"nested": None}, None]},
        ),
        ('{"$null":true}', None),
        ('{"value":{"$null":false}}', {"value": {"$null": False}}),
        ('{"value":{"$null":1}}', {"value": {"$null": 1}}),
        ('{"value":{"$null":"true"}}', {"value": {"$null": "true"}}),
        ('{"value":{"$null":true,"other":1}}', {"value": {"$null": True, "other": 1}}),
        (
            '{"optimizer":{"seed":9007199254740993}}',
            {"optimizer": {"seed": 9007199254740993}},
        ),
    ],
    ids=[
        "worker-deletions",
        "array-nulls",
        "whole-document",
        "false",
        "number",
        "string",
        "extra-keys",
        "large-integer",
    ],
)
def test_expand_input_override_nulls(expand_nulls, raw, want):
    document = json.loads(raw)

    assert expand_nulls(document) == want
    assert document == json.loads(raw)
