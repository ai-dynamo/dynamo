# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402
# Optional-dependency preflight must run before the AISimulate imports.

"""Finite expansion of AISimulate value domains (``dynamo.aisimulate.domains``)."""

from __future__ import annotations

import pytest

pytest.importorskip(
    "aisimulate.config.common",
    reason="AI Simulate is an optional Dynamo simulation dependency",
)

from aisimulate.config.common import Choices, IntegerRange, NumericRange

from dynamo.aisimulate.domains import (
    ContinuousDomain,
    domain_size,
    expand_domain,
    finite_values,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.planner,
    pytest.mark.parallel,
]


def _float_range(**spec) -> NumericRange:
    return NumericRange.model_validate({"range": spec})


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, [0.0, 0.5]),
        (0.25, [0.25]),
        (Choices[float](choices=[1.0, 0.5]), [1.0, 0.5]),
        # Decimal steps: float accumulation would end at 0.30000000000000004
        # and drop the endpoint.
        (_float_range(min=0.0, max=0.3, step=0.1), [0.0, 0.1, 0.2, 0.3]),
        (_float_range(min=0, max=1, step=0.3), [0.0, 0.3, 0.6, 0.9]),
        (_float_range(min=0.5, max=0.5), [0.5]),
        (
            IntegerRange.model_validate({"range": {"min": 1, "max": 7, "step": 3}}),
            [1, 4, 7],
        ),
        (_float_range(min=0.0, max=1.0), ContinuousDomain(0.0, 1.0, False)),
        (
            _float_range(min=0.25, max=32.0, scale="log"),
            ContinuousDomain(0.25, 32.0, True),
        ),
    ],
)
def test_expand_domain(value, expected) -> None:
    domain = expand_domain(value, (0.0, 0.5))
    assert domain == expected
    if isinstance(expected, ContinuousDomain):
        assert domain_size(value, None) is None
    else:
        assert [type(item) for item in domain] == [type(item) for item in expected]
        assert domain_size(value, (0.0, 0.5)) == len(expected)


def test_size_limit_is_checked_before_building_values() -> None:
    huge = _float_range(min=0.0, max=1000.0, step=1e-6)
    with pytest.raises(ValueError, match="knob expands to 1000000001 values"):
        finite_values("knob", huge, None, limit=512)
    # Too many values for Decimal to count is over any limit.
    beyond = _float_range(min=0.0, max=1000.0, step=1e-30)
    with pytest.raises(ValueError, match="knob expands to over 10\\^28 values"):
        finite_values("knob", beyond, None, limit=512)
    with pytest.raises(ValueError, match="expands to 3 values, more than the limit"):
        finite_values("knob", None, (1, 2, 3), limit=2)
    assert finite_values("knob", None, (1, 2, 3), limit=3) == [1, 2, 3]
