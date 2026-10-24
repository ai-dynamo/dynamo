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
final rule is unconditional. Today every rule is unconditional, so a policy is
just a constant -- but :meth:`PriorityResolver.resolve` already takes a
:class:`PriorityContext`, so adding real predicates later is a new
:class:`PriorityCondition` field and nothing else. Callers do not change.

``PriorityCondition`` forbids unknown fields, so a config that tries to declare
a condition today fails loudly at startup instead of silently always-matching.
"""

from __future__ import annotations

import fnmatch
import logging
from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

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

    Empty today: every rule is unconditional, so resolution ignores it. It is
    threaded through :meth:`PriorityResolver.resolve` from the start precisely
    so that populating it later does not touch any call site.
    """


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

    No predicate fields exist yet. ``extra="forbid"`` means a config declaring
    one is rejected at startup rather than parsed into a condition that
    silently matches everything.
    """

    model_config = ConfigDict(extra="forbid")

    def matches(self, ctx: PriorityContext) -> bool:
        """Whether this condition holds for ``ctx``. Vacuously true today."""
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


class PoolPriorityPolicy(BaseModel):
    """A selector bound to an ordered, first-match-wins list of rules.

    ``priority`` is shorthand for a single unconditional rule; it and ``rules``
    are mutually exclusive. Validation normalizes the shorthand into ``rules``
    and clears it, so downstream code -- and any ``model_dump()`` round trip --
    sees exactly one shape.
    """

    model_config = ConfigDict(extra="forbid")

    selector: str = Field(
        description=(
            "Slash-separated path matched against '<participant_id>/<sub_type>'. "
            "'*' matches exactly one segment, '**' matches any number, and a "
            "selector shorter than the pool path covers everything beneath it -- "
            "so 'prod/chat' selects every pool of that deployment while "
            "'prod/chat/prefill' selects one."
        )
    )
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

        segments = self.selector.split("/")
        if not segments or any(not seg for seg in segments):
            raise ValueError(
                f"selector {self.selector!r} must be a slash-separated path with "
                f"no empty segments"
            )
        return self

    # ------------------------------------------------------------------ #
    # Matching                                                           #
    # ------------------------------------------------------------------ #

    @property
    def _segments(self) -> tuple[str, ...]:
        return tuple(self.selector.split("/"))

    @property
    def _pattern(self) -> tuple[str, ...]:
        """Declared segments, implicitly covering everything beneath them.

        A selector shorter than a pool path should select every pool under it,
        which is exactly "match the declared segments, then anything" -- so a
        trailing ``**`` is appended unless one is already there.
        """
        segments = self._segments
        if segments[-1] == "**":
            return segments
        return segments + ("**",)

    @staticmethod
    def _match_segments(pattern: tuple[str, ...], path: tuple[str, ...]) -> bool:
        """Glob a segment path, where ``**`` spans any number of segments.

        Wildcards other than ``**`` never span a ``/``, so ``a/*/prefill``
        matches ``a/b/prefill`` but not ``a/b/c/prefill``; ``a/**/prefill``
        matches both.
        """
        if not pattern:
            return not path
        head, rest = pattern[0], pattern[1:]
        if head == "**":
            # Try consuming 0, 1, 2 ... segments here.
            return any(
                PoolPriorityPolicy._match_segments(rest, path[i:])
                for i in range(len(path) + 1)
            )
        if not path:
            return False
        if not fnmatch.fnmatchcase(path[0], head):
            return False
        return PoolPriorityPolicy._match_segments(rest, path[1:])

    def matches(self, participant_id: str, sub_type: str) -> bool:
        """Whether this selector covers the pool ``participant_id``/``sub_type``."""
        path = tuple(participant_id.split("/")) + (sub_type,)
        return self._match_segments(self._pattern, path)

    @property
    def specificity(self) -> tuple[int, int]:
        """Sort key ranking selectors most-specific first (smaller wins).

        Ranked by how many segments are named exactly, then by depth. So
        ``prod/chat/prefill`` beats ``prod/chat``, which beats ``prod/*``,
        which beats ``**``. Ties fall back to declaration order in
        :class:`PriorityResolver`.
        """
        exact = sum(1 for seg in self._segments if not any(c in seg for c in "*?["))
        return (-exact, -len(self._segments))

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
        seen: set[str] = set()
        for policy in self.pools:
            if policy.selector in seen:
                raise ValueError(
                    f"duplicate selector {policy.selector!r}: merge the entries, "
                    f"since only one of them could ever take effect"
                )
            seen.add(policy.selector)
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
        self._ordered = sorted(
            enumerate(self.config.pools),
            key=lambda pair: (pair[1].specificity, pair[0]),
        )

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
        for _, policy in self._ordered:
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
