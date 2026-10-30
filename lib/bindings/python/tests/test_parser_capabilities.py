# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from dynamo._core import get_structural_tag_parser_names, get_tool_parser_names
from dynamo._core.backend import EngineConfig

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.parallel,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.timeout(30),
]


def test_grammar_inventory_is_sorted_unique_and_selectable():
    names = get_structural_tag_parser_names()
    assert names == sorted(set(names))
    assert set(names) <= set(get_tool_parser_names())


@pytest.mark.parametrize(
    "family", ["hermes", "deepseek_v4", "deepseek_v41", "glm45", "kimi_k25"]
)
def test_registered_grammar_is_available(family):
    assert family in get_structural_tag_parser_names()


def test_selectable_output_parser_does_not_imply_grammar_availability():
    assert "gemma4" in get_tool_parser_names()
    assert "gemma4" not in get_structural_tag_parser_names()
    assert "no_such_parser" not in get_structural_tag_parser_names()


@pytest.mark.parametrize("value", [512, 0.5, True, {"limits": [-3, 0, 2**63 + 1]}])
def test_parser_dependency_preserves_runtime_metadata_numbers(value):
    # Parser dependency features also affect serializers used by runtime metadata.
    config = EngineConfig(model="test", runtime_data={"value": value})
    actual = config.runtime_data["value"]
    assert type(actual) is type(value)
    if isinstance(value, dict):
        assert [type(item) for item in actual["limits"]] == [
            type(item) for item in value["limits"]
        ]
    assert actual == value
