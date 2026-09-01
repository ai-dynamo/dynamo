# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pool priority declaration and resolution for the GlobalPlanner.

The GlobalPlanner is a *scaling mediator*: when several pools compete for one
GPU budget it has to decide who is served. Priority is the operator's way of
encoding that business decision. This module owns the declaration surface and
the resolution logic; it is pure and infrastructure-free, and nothing here
reads or writes cluster state.

**Polarity: higher numbers are more important.** This matches Kubernetes
``PriorityClass`` and ``nvext.agent_hints.priority``, so an operator carrying
over intuitions from either lands in the right place. Note it is the *opposite*
of the plugin-stage ``priority`` in ``dynamo.planner``, where smaller is more
authoritative -- that one orders pipeline stages, this one orders capacity
allocation between pools. Use :func:`outranks` rather than open-coding a
comparison.

Declaration is coarse, resolution is fine
-----------------------------------------
Business priority is usually a property of a deployment, but the unit the budget
arbitration actually manipulates is a *pool* -- one ``sub_type`` within one
participant. So a selector may name either, and the most specific match wins:

.. code-block:: yaml

    priority:
      default: 100                     # pools this GlobalPlanner has not seen
      pools:
        - selector: "prod/chat"          # participant: every pool under it
          priority: 900
        - selector: "prod/chat/prefill"  # one pool: overrides the above
          priority: 950
        - selector: "dev/*"              # any deployment in the dev namespace
          priority: 10

Static priorities are degenerate conditionals
---------------------------------------------
A policy is *always* an ordered list of rules resolved first-match-wins, whose
final rule is unconditional, so ``priority: <n>`` is simply the one-rule case:

.. code-block:: yaml

    - selector: "prod/batch"
      rules:
        - when: {predicted_requests_below: 50}   # quiet: yield to everyone
          priority: 800
        - priority: 100                          # busy: normal standing

Conditions are evaluated against the pool's own :class:`PriorityContext`, not
the requester's -- a partner's context comes from the intent it published, so
"while *its* traffic is low" means what it says.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dynamo.global_planner.pool_selectors import (
    PoolSelector,
    order_by_specificity,
    reject_duplicate_selectors,
)

logger = logging.getLogger(__name__)

#: Floor of the priority range. Nothing is less important than this.
LOWEST_PRIORITY = 0

#: Applied to any pool no selector matches. Deliberately mid-range so operators
#: can declare both more- and less-important pools without renumbering.
DEFAULT_POOL_PRIORITY = 100

#: Reported as the matched selector when a pool fell through to the default.
DEFAULT_SELECTOR = "<default>"


def outranks(a: int, b: int) -> bool:
    """Whether priority ``a`` is more important than priority ``b``.

    Exists so call sites read as intent rather than as a bare comparison whose
    direction has to be re-derived, and so the polarity lives in exactly one
    place if it ever moves again.
    """
    return a > b


# ---------------------------------------------------------------------------- #
# Runtime context                                                              #
# ---------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PriorityContext:
    """Signals a conditional rule may test when resolving a pool's priority.

    Populated from the ``predicted_load`` a local planner already attaches to
    every ``ScaleRequest``, so no protocol change was needed to reach them.
    Each field is ``None`` when the caller did not supply it or supplied
    something unusable (wrong type, negative, NaN, infinity); a condition that
    tests a missing signal does **not** match, so a pool always falls through
    to its unconditional rule rather than being silently reclassified on absent
    or malformed data.

    ``predicted_isl`` and ``predicted_osl`` are carried but no condition tests
    them yet -- the context is the full signal set, conditions are the subset
    we have designed predicates for.
    """

    predicted_num_requests: Optional[float] = None
    predicted_isl: Optional[float] = None
    predicted_osl: Optional[float] = None

    @classmethod
    def from_predicted_load(cls, predicted_load: Optional[dict]) -> "PriorityContext":
        """Build a context from a ``ScaleRequest.predicted_load`` payload.

        Tolerates a missing payload and unexpected keys: this crosses a trust
        boundary from a caller-supplied dict, and a malformed prediction must
        degrade to "no signal" rather than fail a scale request.
        """
        if not predicted_load:
            return cls()

        def _number(key: str) -> Optional[float]:
            value = predicted_load.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return None
            number = float(value)
            # NaN and infinity are not usable signals, and a negative rate is
            # not a rate. NaN in particular is dangerous: every comparison
            # against it is False, so it would slip past both predicate checks
            # and satisfy any condition rather than degrading to "no signal".
            if not math.isfinite(number) or number < 0:
                return None
            return number

        return cls(
            predicted_num_requests=_number("num_requests"),
            predicted_isl=_number("isl"),
            predicted_osl=_number("osl"),
        )


@dataclass(frozen=True)
class ResolvedPriority:
    """A pool's effective priority plus where it came from.

    ``selector`` and ``rule_index`` are provenance for logs: when a scale
    request is denied on priority grounds the operator needs to see *which*
    line of their config produced the number, not just the number.
    """

    priority: int
    selector: str
    rule_index: int


# ---------------------------------------------------------------------------- #
# Declaration surface                                                          #
# ---------------------------------------------------------------------------- #


class PriorityCondition(BaseModel):
    """Predicates gating a :class:`PriorityRule`.

    All declared predicates must hold for the condition to match. A predicate
    whose signal is absent from the context does **not** hold, so a rule never
    fires on missing data. ``extra="forbid"`` keeps a typo'd predicate name
    from parsing into a condition that silently matches everything.
    """

    model_config = ConfigDict(extra="forbid")

    predicted_requests_at_least: Optional[float] = Field(
        default=None,
        ge=0,
        description=(
            "Match when the caller's predicted request rate is at or above "
            "this value -- 'this priority while traffic is high'."
        ),
    )
    predicted_requests_below: Optional[float] = Field(
        default=None,
        ge=0,
        description=(
            "Match when the caller's predicted request rate is strictly below "
            "this value -- 'this priority while traffic is low'."
        ),
    )

    @model_validator(mode="after")
    def _validate_range(self) -> "PriorityCondition":
        low = self.predicted_requests_at_least
        high = self.predicted_requests_below
        if low is not None and high is not None and low >= high:
            raise ValueError(
                f"predicted_requests_at_least ({low}) must be below "
                f"predicted_requests_below ({high}); as written no request "
                f"rate can satisfy both"
            )
        return self

    def matches(self, ctx: PriorityContext) -> bool:
        """Whether every declared predicate holds for ``ctx``."""
        rate = ctx.predicted_num_requests
        if rate is not None and not math.isfinite(rate):
            # Treat a non-finite rate as absent. Comparing against NaN yields
            # False in both directions, which would otherwise let it satisfy
            # every predicate instead of none of them.
            rate = None
        if self.predicted_requests_at_least is not None:
            if rate is None or rate < self.predicted_requests_at_least:
                return False
        if self.predicted_requests_below is not None:
            if rate is None or rate >= self.predicted_requests_below:
                return False
        return True


class PriorityRule(BaseModel):
    """One priority value, optionally gated by a condition."""

    model_config = ConfigDict(extra="forbid")

    when: Optional[PriorityCondition] = Field(
        default=None,
        description="Condition gating this rule. None means it always applies.",
    )
    priority: int = Field(
        ge=LOWEST_PRIORITY,
        description="Priority value. Higher is more important.",
    )

    def applies(self, ctx: PriorityContext) -> bool:
        return self.when is None or self.when.matches(ctx)


class PoolPriorityPolicy(PoolSelector):
    """A selector bound to an ordered, first-match-wins list of rules.

    ``priority`` is shorthand for a single unconditional rule; it and ``rules``
    are mutually exclusive. Validation normalizes the shorthand into ``rules``
    and clears it, so downstream code -- and any ``model_dump()`` round trip --
    sees exactly one shape.
    """

    priority: Optional[int] = Field(default=None, ge=LOWEST_PRIORITY)
    rules: Optional[list[PriorityRule]] = Field(default=None)

    @model_validator(mode="after")
    def _normalize(self) -> "PoolPriorityPolicy":
        if (self.priority is None) == (self.rules is None):
            raise ValueError(
                f"selector {self.selector!r} must set exactly one of "
                f"'priority' (a constant) or 'rules' (a conditional list)"
            )

        if self.priority is not None:
            # Normalize the shorthand away entirely rather than leaving both
            # fields populated: GlobalPlannerConfig round-trips through
            # model_dump() -> model_validate() when merging CLI flags, and a
            # dump carrying both would fail the "exactly one of" check above.
            self.rules = [PriorityRule(priority=self.priority)]
            self.priority = None

        assert self.rules is not None
        if not self.rules:
            raise ValueError(f"selector {self.selector!r} has an empty 'rules' list")
        if self.rules[-1].when is not None:
            raise ValueError(
                f"selector {self.selector!r} must end with an unconditional rule "
                f"so every context resolves to a priority"
            )

        return self

    def resolve(self, ctx: PriorityContext) -> Optional[tuple[int, int]]:
        """First applicable rule as ``(priority, rule_index)``, else ``None``."""
        assert self.rules is not None
        for index, rule in enumerate(self.rules):
            if rule.applies(ctx):
                return rule.priority, index
        return None


class PriorityConfig(BaseModel):
    """Declarative pool priorities for one GlobalPlanner process."""

    model_config = ConfigDict(extra="forbid")

    default: int = Field(
        default=DEFAULT_POOL_PRIORITY,
        ge=LOWEST_PRIORITY,
        description=(
            "Priority for pools no selector matches, including ones this "
            "GlobalPlanner has never received a scale request from."
        ),
    )
    pools: list[PoolPriorityPolicy] = Field(
        default_factory=list,
        description="Selector-scoped policies, most specific match wins.",
    )

    @model_validator(mode="after")
    def _reject_duplicate_selectors(self) -> "PriorityConfig":
        reject_duplicate_selectors(self.pools)
        return self


# ---------------------------------------------------------------------------- #
# Resolution                                                                   #
# ---------------------------------------------------------------------------- #


class PriorityResolver:
    """Resolves a pool's effective priority from a :class:`PriorityConfig`.

    Policies are ordered once at construction by specificity, then by
    declaration order, so resolution is deterministic and independent of how
    the config file happened to be written.
    """

    def __init__(self, config: Optional[PriorityConfig] = None):
        self.config = config or PriorityConfig()
        self._ordered = order_by_specificity(self.config.pools)

    def resolve(
        self,
        participant_id: str,
        sub_type: str,
        ctx: Optional[PriorityContext] = None,
    ) -> ResolvedPriority:
        """Effective priority for one pool.

        Falls back to :attr:`PriorityConfig.default` when no selector matches,
        which is the normal path for a pool the operator never named.
        """
        context = ctx if ctx is not None else PriorityContext()
        for policy in self._ordered:
            if not policy.matches(participant_id, sub_type):
                continue
            outcome = policy.resolve(context)
            if outcome is None:
                # Unreachable while every policy ends unconditional, but a
                # future condition set could make it reachable; prefer falling
                # through to the next selector over raising mid-arbitration.
                logger.debug(
                    f"Selector {policy.selector!r} matched "
                    f"{participant_id}/{sub_type} but no rule applied"
                )
                continue
            priority, rule_index = outcome
            return ResolvedPriority(priority, policy.selector, rule_index)

        return ResolvedPriority(self.config.default, DEFAULT_SELECTOR, 0)
