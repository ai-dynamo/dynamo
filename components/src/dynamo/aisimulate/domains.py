# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Value-domain expansion shared by the Dynamo AISimulate adapters and retune.

An adapter knob is written as a scalar, ``{choices: [...]}`` or
``{range: {min, max, step?, scale?}}``; after schema validation these are a
plain value or an AISimulate ``Choices``, ``NumericRange`` or
``IntegerRange``. ``expand_domain`` turns one validated knob into its values,
or into a ``ContinuousDomain`` for a range AISimulate samples continuously (a
linear range without ``step``, or a log range). A recommendation sweep hands
continuous domains to the AISimulate sampler; an exhaustive grid such as the
planner retune sweep uses ``finite_values`` and ``require_enumerable``, which
reject them.

``SCALAR``, ``CHOICES`` and ``RANGE`` tag the members of a knob's domain
union, and ``KNOB_FORM`` (or ``CHOICE_FORM`` for a knob without ranges)
selects the member from the form the value is written in, so pydantic reports
only that member's errors.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

from aisimulate.config.common import Choices, IntegerRange, NumericRange
from pydantic import BaseModel, Discriminator, Tag

SCALAR = Tag("scalar")
CHOICES = Tag("choices")
RANGE = Tag("range")


def _knob_form(value: Any) -> str | None:
    """The tag of the domain form ``value`` is written in; ``None`` for none."""

    if isinstance(value, Choices) or (
        isinstance(value, Mapping) and "choices" in value
    ):
        return "choices"
    if isinstance(value, (NumericRange, IntegerRange)) or (
        isinstance(value, Mapping) and "range" in value
    ):
        return "range"
    if isinstance(value, (Mapping, list)):
        return None
    return "scalar"


def _choice_form(value: Any) -> str | None:
    form = _knob_form(value)
    return None if form == "range" else form


KNOB_FORM = Discriminator(
    _knob_form,
    custom_error_type="domain_form",
    custom_error_message=(
        "write a scalar, {choices: [...]} or {range: {min, max, step}}"
    ),
)
CHOICE_FORM = Discriminator(
    _choice_form,
    custom_error_type="domain_form",
    custom_error_message="write a scalar or {choices: [...]}",
)


@dataclass(frozen=True)
class ContinuousDomain:
    """A range without a finite value list: linear without ``step``, or log."""

    minimum: float
    maximum: float
    log: bool


def _stepped_count(
    minimum: int | float, maximum: int | float, step: int | float
) -> int:
    """Values of a stepped range.

    Raises ``decimal.InvalidOperation`` for a count of more than 28 digits,
    the precision of the default ``Decimal`` context.
    """

    start = Decimal(str(minimum))
    stop = Decimal(str(maximum))
    increment = Decimal(str(step))
    return int((stop - start) // increment) + 1


def stepped_numeric_values(
    minimum: int | float,
    maximum: int | float,
    step: int | float,
) -> list[float]:
    """Expand a decimal-facing YAML range without cumulative float drift.

    Values are ``minimum + i * step`` computed in ``Decimal`` from the YAML
    literals, so ``0.1..0.3 step 0.1`` ends exactly at ``0.3``; a ``maximum``
    not on the step grid is not included.
    """

    start = Decimal(str(minimum))
    increment = Decimal(str(step))
    return [
        float(start + increment * index)
        for index in range(_stepped_count(minimum, maximum, step))
    ]


def domain_size(value: Any, defaults: Sequence[Any] | None) -> int | float | None:
    """Number of values ``expand_domain`` returns, without building them.

    ``None`` for a continuous range; ``math.inf`` for a stepped float range
    with too many values to count, which is over any limit.
    """

    if value is None:
        if defaults is None:
            raise ValueError("domain has no value and no default domain")
        return len(defaults)
    if isinstance(value, Choices):
        return len(value.choices)
    if isinstance(value, (NumericRange, IntegerRange)):
        spec = value.range
        if spec.min == spec.max:
            return 1
        if spec.step is None:
            return None
        if isinstance(value, IntegerRange):
            return (spec.max - spec.min) // spec.step + 1
        try:
            return _stepped_count(spec.min, spec.max, spec.step)
        except InvalidOperation:
            return math.inf
    return 1


def expand_domain(
    value: Any, defaults: Sequence[Any] | None
) -> list[Any] | ContinuousDomain:
    """Expand one validated knob domain.

    - ``None`` (knob omitted): ``defaults``.
    - ``Choices``: its choices, in order.
    - ``NumericRange`` / ``IntegerRange``: ``[min]`` when ``min == max``; the
      stepped values when ``step`` is set (``range`` for integers,
      ``stepped_numeric_values`` for floats); otherwise ``ContinuousDomain``.
    - Any other value is a scalar: ``[value]``.
    """

    if value is None:
        if defaults is None:
            raise ValueError("domain has no value and no default domain")
        return list(defaults)
    if isinstance(value, Choices):
        return list(value.choices)
    if isinstance(value, (NumericRange, IntegerRange)):
        spec = value.range
        if spec.min == spec.max:
            return [spec.min]
        if spec.step is None:
            # AISimulate rejects a step on log ranges, so every log range
            # with distinct bounds lands here.
            return ContinuousDomain(spec.min, spec.max, spec.scale == "log")
        if isinstance(value, IntegerRange):
            return list(range(spec.min, spec.max + 1, spec.step))
        return stepped_numeric_values(spec.min, spec.max, spec.step)
    return [value]


def _continuous_error(name: str, minimum: Any, maximum: Any, log: bool) -> ValueError:
    kind = "a log range" if log else "a range without step"
    return ValueError(
        f"{name} is {kind} over [{minimum}, {maximum}], which is continuous and "
        "cannot be enumerated; give a linear range with step or list the values "
        "as {choices: [...]}"
    )


def check_domain_size(
    name: str, value: Any, defaults: Sequence[Any] | None, *, limit: int
) -> None:
    """Raise ``ValueError`` naming ``name`` for a domain of more than ``limit``
    values, without building them; a continuous range passes."""

    size = domain_size(value, defaults)
    if size is not None and size > limit:
        count = "over 10^28" if size == math.inf else size
        raise ValueError(
            f"{name} expands to {count} values, more than the limit of {limit}"
        )


def finite_values(
    name: str, value: Any, defaults: Sequence[Any] | None, *, limit: int
) -> list[Any]:
    """``expand_domain`` for an exhaustive grid.

    Raises ``ValueError`` naming ``name`` for a continuous range, and for a
    domain of more than ``limit`` values; the size is checked before any
    value is built, so a range such as ``0..1000 step 1e-6`` fails fast.
    """

    if value is None and defaults is None:
        raise ValueError(f"{name} has no value and no default domain")
    check_domain_size(name, value, defaults, limit=limit)
    values = expand_domain(value, defaults)
    if isinstance(values, ContinuousDomain):
        raise _continuous_error(name, values.minimum, values.maximum, values.log)
    return values


def require_enumerable(model: BaseModel, *, prefix: str = "") -> None:
    """Reject every explicitly set range field of ``model`` that is continuous.

    Checks the top-level fields in ``model.model_fields_set`` only. Use it
    before handing a validated model to code that enumerates ranges itself:
    the planner's preset expansion turns an integer log range into every
    integer in it and a float log range into its two endpoints.
    """

    for name in type(model).model_fields:
        if name not in model.model_fields_set:
            continue
        value = getattr(model, name)
        if (
            isinstance(value, (NumericRange, IntegerRange))
            and domain_size(value, None) is None
        ):
            spec = value.range
            raise _continuous_error(
                prefix + name, spec.min, spec.max, spec.scale == "log"
            )
