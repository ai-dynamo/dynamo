# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Operator-compatible source ordering, without runtime or engine dependencies."""

import copy

import pytest

from dynamo.common.environment import compose_environment, ordered_environment_enabled

pytestmark = [pytest.mark.gpu_0, pytest.mark.unit, pytest.mark.pre_merge]


@pytest.mark.parametrize(
    "origin, enabled",
    [
        ("", False),
        ("invalid", False),
        ("1.7.99", False),
        ("1.8.0-rc.1", False),
        ("1.8.0", True),
        ("v1.8", True),
        ("1.8.0+build", True),
        ("1.8.1-dev", True),
        ("1.9.0-01", False),
        ("18446744073709551616.0.0", False),
    ],
)
def test_origin_boundary(origin, enabled):
    assert (
        ordered_environment_enabled(
            {"nvidia.com/dynamo-operator-origin-version": origin}
        )
        is enabled
    )


def test_cross_source_order_and_whole_entry_override():
    layers = [
        [
            {"name": "MODEL", "value": "$(Z_ROOT)/model"},
            {
                "name": "Z_ROOT",
                "valueFrom": {"secretKeyRef": {"name": "old", "key": "path"}},
            },
        ],
        [
            {"name": "Z_ROOT", "value": "/models"},
            {"name": "A", "value": "1"},
            {"name": "A", "value": "$(A)2"},
        ],
    ]
    original = copy.deepcopy(layers)
    assert compose_environment(*layers) == [layers[1][0], layers[0][0], *layers[1][1:]]
    assert layers == original


def test_local_forward_references_and_escaped_cross_source_references():
    layer = [{"name": "X", "value": "$(Y)"}, {"name": "Y", "value": "$(X)"}]
    escaped = [{"name": "A", "value": "$$(Z)"}, {"name": "B", "value": "$(" * 10000}]
    assert compose_environment(layer) == layer
    assert compose_environment(escaped, [{"name": "Z", "value": "$(A)"}])[:2] == escaped


def test_cross_source_order_conflict_reports_original_positions():
    with pytest.raises(ValueError, match=r"layer\[0\].env\[1\].*B"):
        compose_environment(
            [{"name": "A", "value": "$(C)"}, {"name": "B", "value": "1"}],
            [{"name": "C", "value": "$(B)"}],
        )
