# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for endpoint type parsing."""

import pytest

from dynamo.common.utils.endpoint_types import parse_endpoint_types
from dynamo.llm import ModelType

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def test_parse_endpoint_types_allows_topology_only_workers():
    assert parse_endpoint_types("none") == ModelType.Empty


@pytest.mark.parametrize("value", ["chat,none", "none,completions"])
def test_parse_endpoint_types_rejects_mixing_none_with_public_surfaces(value):
    with pytest.raises(ValueError, match="'none' cannot be combined"):
        parse_endpoint_types(value)


@pytest.mark.parametrize("value", ["", "   ", ",,,"])
def test_parse_endpoint_types_rejects_empty_input(value):
    with pytest.raises(ValueError):
        parse_endpoint_types(value)


def test_parse_endpoint_types_invalid_option_lists_none():
    with pytest.raises(ValueError, match="'chat', 'completions', 'none'"):
        parse_endpoint_types("responses")
