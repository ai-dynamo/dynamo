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
from pydantic import BaseModel, ConfigDict

from dynamo.aisimulate.domains import (
    ContinuousDomain,
    domain_size,
    expand_domain,
    finite_values,
    require_enumerable,
    stepped_numeric_values,
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


def _int_range(**spec) -> IntegerRange:
    return IntegerRange.model_validate({"range": spec})


def test_stepped_values_are_decimal_exact_and_floor_the_endpoint() -> None:
    # Float accumulation gives 0.30000000000000004 and drops the endpoint.
    assert stepped_numeric_values(0.0, 0.3, 0.1) == [0.0, 0.1, 0.2, 0.3]
    assert stepped_numeric_values(0.1, 0.3, 0.1) == [0.1, 0.2, 0.3]
    assert stepped_numeric_values(0, 1, 0.3) == [0.0, 0.3, 0.6, 0.9]
    assert stepped_numeric_values(2, 2, 0.5) == [2.0]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, [0.0, 0.5]),
        (0.25, [0.25]),
        ("kv_router", ["kv_router"]),
        (Choices[float](choices=[1.0, 0.5]), [1.0, 0.5]),
        (_float_range(min=0.0, max=0.2, step=0.1), [0.0, 0.1, 0.2]),
        (_float_range(min=0.5, max=0.5), [0.5]),
        (_float_range(min=2.0, max=2.0, scale="log"), [2.0]),
        (_int_range(min=1, max=7, step=3), [1, 4, 7]),
        (_int_range(min=4, max=4, scale="log"), [4]),
    ],
)
def test_expand_domain_finite_forms(value, expected) -> None:
    domain = expand_domain(value, (0.0, 0.5))
    assert domain == expected
    assert [type(item) for item in domain] == [type(item) for item in expected]
    assert domain_size(value, (0.0, 0.5)) == len(expected)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (_float_range(min=0.0, max=1.0), ContinuousDomain(0.0, 1.0, False)),
        (
            _float_range(min=0.25, max=32.0, scale="log"),
            ContinuousDomain(0.25, 32.0, True),
        ),
        (_int_range(min=1, max=64, scale="log"), ContinuousDomain(1, 64, True)),
    ],
)
def test_expand_domain_continuous_forms(value, expected) -> None:
    assert expand_domain(value, None) == expected
    assert domain_size(value, None) is None


def test_expand_domain_without_value_or_default_is_an_error() -> None:
    with pytest.raises(ValueError, match="no default domain"):
        expand_domain(None, None)
    with pytest.raises(ValueError, match="retune.router.x has no value"):
        finite_values("retune.router.x", None, None, limit=10)


@pytest.mark.parametrize(
    ("value", "match"),
    [
        (_float_range(min=0.0, max=1.0), r"range without step over \[0.0, 1.0\]"),
        (
            _float_range(min=0.25, max=32.0, scale="log"),
            r"log range over \[0.25, 32.0\]",
        ),
        (_int_range(min=1, max=64, scale="log"), r"log range over \[1, 64\]"),
    ],
    ids=["linear-without-step", "float-log", "integer-log"],
)
def test_finite_values_rejects_continuous_ranges(value, match) -> None:
    with pytest.raises(ValueError, match=f"knob is a {match}") as error:
        finite_values("knob", value, None, limit=100)
    assert "step" in str(error.value) and "choices" in str(error.value)


def test_finite_values_checks_the_limit_before_building_values() -> None:
    huge = _float_range(min=0.0, max=1000.0, step=1e-6)
    assert domain_size(huge, None) == 1_000_000_001
    with pytest.raises(ValueError, match="knob expands to 1000000001 values"):
        finite_values("knob", huge, None, limit=512)
    with pytest.raises(ValueError, match="expands to 3 values, more than"):
        finite_values("knob", None, (1, 2, 3), limit=2)
    assert finite_values("knob", None, (1, 2, 3), limit=3) == [1, 2, 3]
    assert finite_values("knob", 0.5, None, limit=1) == [0.5]


def test_size_check_counts_ranges_beyond_the_default_decimal_precision() -> None:
    # 10**33 + 1 values: more digits than the default 28-digit Decimal
    # context, where the quotient raises decimal.InvalidOperation.
    huge = _float_range(min=0.0, max=1000.0, step=1e-30)
    assert domain_size(huge, None) == 10**33 + 1
    with pytest.raises(ValueError, match=r"knob expands to 1000000000000000000000"):
        finite_values("knob", huge, None, limit=512)
    widest = _float_range(min=0.0, max=1.7976931348623157e308, step=5e-324)
    assert (domain_size(widest, None) or 0) > 10**600


class _Knobs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    a: float | NumericRange | None = None
    b: int | IntegerRange | None = None
    c: float | NumericRange | None = None


def test_require_enumerable_checks_only_explicit_ranges() -> None:
    require_enumerable(_Knobs())
    require_enumerable(
        _Knobs.model_validate(
            {
                "a": {"range": {"min": 0.0, "max": 1.0, "step": 0.5}},
                "b": {"range": {"min": 3, "max": 3, "scale": "log"}},
                "c": 0.5,
            }
        )
    )
    with pytest.raises(ValueError, match="retune.planner.b is a log range"):
        require_enumerable(
            _Knobs.model_validate(
                {"b": {"range": {"min": 1, "max": 8, "scale": "log"}}}
            ),
            prefix="retune.planner.",
        )
    with pytest.raises(ValueError, match="c is a range without step"):
        require_enumerable(
            _Knobs.model_validate({"c": {"range": {"min": 0.0, "max": 1.0}}})
        )
