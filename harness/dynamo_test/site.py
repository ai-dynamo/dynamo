# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Where the system under test lives, and who owns its lifecycle.

These are two independent questions, and conflating them is the mistake this
module exists to prevent.

*Where it lives* decides what is physically possible. If pytest and the workers
share a process namespace, a test can send SIGSTOP to an engine subprocess. If
they are in different containers, that same call is meaningless — there is no
such process to signal.

*Who owns it* decides what is permitted. A test that applied a
DynamoGraphDeployment may tear it down. A test that attached to a deployment
somebody else is running may not, even though the API call would succeed.

The four topologies the suite needs to span:

======================  ================================================
``all-in-one``          pytest and every Dynamo process in one container.
                        What CI does today.
``sidecar``             pytest in its own container, the stack in another.
``compose``             pytest, a stock upstream engine, a Dynamo sidecar and
                        a Dynamo frontend, each in its own container.
``kubernetes``          a DynamoGraphDeployment on a cluster.
======================  ================================================

## Capabilities are computed, never declared

A site does not get to *say* it can signal an inner process. It says where it is
and who owns it, and the capability set follows. This matters because a
declaration can be wrong, and a wrong declaration fails in the worst possible
way: the verb is attempted, the platform quietly does nothing resembling what
was asked, and the test passes having measured nothing.

So the flow is one-directional — facts in, capabilities out — and a site
definition may only ever **narrow** the result. There is no way to widen it,
because widening is exactly the lie that produces a vacuous pass.

## Why every removal carries a reason

:func:`why` returns the specific rung that removed a capability, in the author's
words rather than as a boolean:

    shares_network=False: this site cannot reach a test-owned sink

A skip that says "not supported here" teaches nothing and gets copied. A skip
that names the property responsible tells the reader whether to fix the site,
pick another one, or accept that the test belongs to one tier.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Mapping

__all__ = [
    "Topology",
    "Ownership",
    "Capability",
    "Site",
    "UnknownSite",
    "capabilities",
    "why",
    "LOCAL",
    "BUILTIN_SITES",
    "resolve",
    "Refused",
    "Mismatch",
    "Divergence",
]


class Topology(str, Enum):
    """Where the system under test runs, relative to the test process.

    A **reporting label**. Nothing in the harness branches on it, deliberately:
    the moment behaviour keys off the topology name, adding a fifth arrangement
    means auditing every branch. Behaviour keys off the three physical
    properties on :class:`Site` instead, and the label exists so a run record and
    a failure message can say which arrangement was in play.
    """

    ALL_IN_ONE = "all-in-one"
    SIDECAR = "sidecar"
    COMPOSE = "compose"
    KUBERNETES = "kubernetes"

    def __str__(self) -> str:
        return self.value


class Ownership(str, Enum):
    """What bringing the system up actually does.

    Verbs, not adjectives — each names an action the harness will take, so a
    reader can tell what happens on entry and on exit without consulting a table.
    """

    IMPOSE = "impose"
    """Create it, wait for it, destroy it after the evidence is sealed.

    The declared stack is a **recipe**: whatever it says, the harness makes true.
    """

    VERIFY = "verify"
    """Bind to something already running and check it is what was declared.

    The declared stack is an **expectation**, not a recipe. Divergence raises at
    bind time rather than surfacing later as a confusing assertion failure. This
    is the mode that makes the existing all-in-one CI job usable without
    rewriting it: the container already started everything, and the test's job is
    to confirm it got what it expected and then measure it.
    """

    ADOPT = "adopt"
    """Create it if absent, and never destroy it.

    For expensive shared dependencies. The precedent is already in the suite:
    ``tests/serve/lora_utils.py`` health-probes MinIO, sets ``_owns_container =
    False`` when it finds one running, and makes ``stop()`` a no-op.
    """

    def __str__(self) -> str:
        return self.value


class Capability(str, Enum):
    """Something a test might need the site to be able to do."""

    RESTART_ROLE = "restart_role"
    STOP_ROLE = "stop_role"
    SCALE_REPLICAS = "scale_replicas"
    SIGNAL_INNER_PROCESS = "signal_inner_process"
    READ_LOG_FILE = "read_log_file"
    SINK = "sink"
    EXEC_IN = "exec_in"
    SCRUB = "scrub"

    def __str__(self) -> str:
        return self.value


# Verbs that change the deployment. VERIFY holds none of them: a site that does
# not own what it is looking at may observe it and nothing more.
_MUTATORS = frozenset(
    {
        Capability.RESTART_ROLE,
        Capability.STOP_ROLE,
        Capability.SCALE_REPLICAS,
    }
)

# Available everywhere, because they need nothing beyond being able to reach the
# system at all.
_ALWAYS = frozenset({Capability.EXEC_IN})

# The ceiling is the same for every arrangement, on purpose. An earlier draft
# subtracted SIGNAL_INNER_PROCESS per topology, which was redundant -- the
# shares_process_tree rung already removes it -- and actively worse, because the
# topology-shaped exclusion fired first and `why()` then answered
# "this arrangement never supports it" instead of naming the property. Two
# mechanisms expressing one rule will eventually disagree, and the disagreement
# surfaces as a skip message that contradicts the skip.
#
# So there is exactly one place capabilities are removed: the rungs below, each
# keyed on a physical property. That is what makes Topology a pure label.
_CEILING = frozenset(Capability)


class UnknownSite(KeyError):
    def __init__(self, name: object, known: Iterable[str]) -> None:
        known = sorted(known)
        super().__init__(
            f"no site named {name!r}; known sites: {', '.join(known) or '<none>'}"
        )


@dataclass(frozen=True)
class Site:
    """Where the system under test is, and what may be done to it.

    The three booleans are the whole model. They are *physical properties of the
    arrangement*, answerable by looking at it, and every capability decision is
    derived from them.
    """

    name: str
    topology: Topology
    ownership: Ownership

    shares_process_tree: bool
    """pytest and the workers are in one process namespace.

    True only for all-in-one. It is what makes ``kill -STOP <engine pid>`` and
    reading a worker's log file off the local filesystem meaningful.
    """

    shares_network: bool
    """The system can dial a port the test process bound.

    Distinct from *the test can reach the system*, which is true everywhere and
    is an addressing question. This is the reverse direction, and it is what an
    SSRF or callback test depends on. Getting it wrong is unusually bad: a
    canary that binds ``127.0.0.1`` and is never dialled looks exactly like a
    canary that correctly refused a request.
    """

    substrate_owned: bool
    """The namespace, compose project or container belongs to this run.

    Authorises destruction that reaches beyond the deployment itself. False for
    a shared cluster namespace, where scrubbing deletes other people's work.
    """

    provider: str = "local"
    narrow: frozenset = frozenset()
    """Capabilities this site declines even though they are possible.

    Narrowing only. There is no widening field, because the whole point of
    computing capabilities is that a site cannot claim one it does not have.
    """

    settings: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("a site needs a name; it appears in every failure message")
        if self.shares_process_tree and self.topology is not Topology.ALL_IN_ONE:
            raise ValueError(
                f"site {self.name!r} claims a shared process tree on "
                f"{self.topology} — only all-in-one puts pytest and the workers "
                "in one process namespace, and believing otherwise makes every "
                "signal-based fault test vacuous"
            )
        unknown = set(self.narrow) - set(Capability)
        if unknown:
            raise ValueError(
                f"site {self.name!r} narrows unknown capabilities: {unknown}"
            )

    def capabilities(self, provider_caps: Iterable[Capability] | None = None):
        return capabilities(self, provider_caps)

    def can(self, capability: Capability, provider_caps=None) -> bool:
        return capability in capabilities(self, provider_caps)

    def why(self, capability: Capability, provider_caps=None) -> str | None:
        return why(self, capability, provider_caps)

    def to_record(self) -> dict:
        return {
            "name": self.name,
            "topology": self.topology.value,
            "ownership": self.ownership.value,
            "shares_process_tree": self.shares_process_tree,
            "shares_network": self.shares_network,
            "substrate_owned": self.substrate_owned,
            "provider": self.provider,
            "narrow": sorted(c.value for c in self.narrow),
            "capabilities": sorted(c.value for c in capabilities(self)),
            "settings": dict(self.settings),
        }

    def __str__(self) -> str:
        return f"{self.name} ({self.topology}, {self.ownership})"


# The rungs, in the order they are applied. Each is (predicate, capabilities it
# removes, the sentence a test author reads). Keeping them as data rather than
# as an if-chain is what lets `why()` answer without duplicating the logic --
# two copies of this reasoning would eventually disagree, and the disagreement
# would show up as a skip message that contradicts the skip.
_RUNGS = (
    (
        lambda s: not s.shares_process_tree,
        frozenset({Capability.SIGNAL_INNER_PROCESS, Capability.READ_LOG_FILE}),
        "shares_process_tree=False: the system is in another process namespace, "
        "so there is no local pid to signal and no local log file to read",
    ),
    (
        lambda s: not s.shares_network,
        frozenset({Capability.SINK}),
        "shares_network=False: the system cannot dial a listener this test "
        "process bound, so a callback would never arrive and the test would "
        "pass without measuring anything",
    ),
    (
        lambda s: s.ownership is Ownership.VERIFY,
        _MUTATORS,
        "ownership=verify: this site attached to a deployment it does not own, "
        "so it may observe but not mutate it",
    ),
    (
        lambda s: not s.substrate_owned,
        frozenset({Capability.SCRUB}),
        "substrate_owned=False: the namespace or compose project is shared, so "
        "destroying everything in it would take other runs with it",
    ),
)


def capabilities(site: Site, provider_caps: Iterable[Capability] | None = None):
    """What this site can actually do.

    Computed. Facts in, capabilities out — a site has no way to assert one it
    does not have.

    ``provider_caps`` is physics rather than policy: whatever the arrangement
    permits, a provider that has not implemented a verb cannot perform it.
    """
    caps = _ALWAYS | _CEILING
    for predicate, removed, _reason in _RUNGS:
        if predicate(site):
            caps -= removed
    if provider_caps is not None:
        caps &= frozenset(provider_caps)
    return frozenset(caps) - frozenset(site.narrow)


def why(
    site: Site,
    capability: Capability,
    provider_caps: Iterable[Capability] | None = None,
) -> str | None:
    """Why ``capability`` is unavailable here, or ``None`` if it is available.

    Returns the *first* rung that removed it, because that is the one worth
    fixing: later rungs would have removed it anyway, and reporting all of them
    buries the actionable one.
    """
    if capability in capabilities(site, provider_caps):
        return None

    for predicate, removed, reason in _RUNGS:
        if predicate(site) and capability in removed:
            return reason
    if capability in site.narrow:
        return f"site {site.name!r} narrows {capability} explicitly"
    if provider_caps is not None and capability not in frozenset(provider_caps):
        return (
            f"provider={site.provider!r} does not implement {capability}; the "
            "site permits it but nothing can carry it out"
        )
    return f"{capability} is unavailable on {site.name}"


# The default, in code rather than in a file: a site definition should never be
# required to run the suite the way it runs today.
LOCAL = Site(
    name="local",
    topology=Topology.ALL_IN_ONE,
    ownership=Ownership.IMPOSE,
    shares_process_tree=True,
    shares_network=True,
    substrate_owned=True,
    provider="local",
)

BUILTIN_SITES: Mapping[str, Site] = {LOCAL.name: LOCAL}


def resolve(name: str | None, extra: Mapping[str, Site] | None = None) -> Site:
    """Look up a site by name, falling back to the built-in ``local``."""
    known = {**BUILTIN_SITES, **(extra or {})}
    if not name:
        return LOCAL
    try:
        return known[name]
    except KeyError:
        raise UnknownSite(name, known) from None


class Refused(PermissionError):
    """A verb was called that this site cannot or may not perform.

    **Raised, never returned, and never quietly satisfied.** That is the whole
    point. A ``stop()`` that no-ops because the site does not own the deployment
    turns a fault-tolerance test into a test of nothing: the fault is never
    injected, the system stays healthy, and the assertion that it recovered
    passes. A raise is noisy and correct; a no-op is quiet and wrong.

    The message names the property responsible, so the reader can tell whether
    to change the site, pick another, or accept the test belongs to one tier.
    """

    def __init__(self, verb: str, site: "Site", why_not: str | None = None) -> None:
        self.verb = verb
        self.site = site
        self.why_not = why_not
        super().__init__(
            f"{verb}() is not available on site {site.name!r} "
            f"({site.topology}, {site.ownership})" + (f": {why_not}" if why_not else "")
        )


@dataclass(frozen=True)
class Divergence:
    """One field where the running system differs from what was declared.

    Both sides are :class:`~dynamo_test.facts.Fact`, which is what lets a
    comparison record *unverified* instead of guessing. ``declared=KNOWN,
    observed=UNKNOWN`` means the check could not be made — scoring that as
    either agreement or divergence would be inventing a result.
    """

    role: str
    field: str
    declared: object  # Fact[Any] -- typed loosely to keep this module import-light
    observed: object

    @property
    def is_unverified(self) -> bool:
        return getattr(self.observed, "is_unknown", False)

    def describe(self) -> str:
        d = getattr(self.declared, "value", self.declared)
        o = getattr(self.observed, "value", self.observed)
        if self.is_unverified:
            detail = getattr(self.observed, "detail", "")
            return f"{self.role}.{self.field}: declared {d!r}, could not observe ({detail})"
        return f"{self.role}.{self.field}: declared {d!r}, observed {o!r}"


class Mismatch(AssertionError):
    """A ``VERIFY`` site bound to something that is not what was declared.

    Raised at bind time, on purpose. The alternative is that the divergence
    surfaces much later as a confusing symptom: attach to a deployment serving a
    different model and the first sign is a 404 at query time, which reads like a
    routing bug rather than "you are looking at the wrong deployment".
    """

    def __init__(self, divergences: "Iterable[Divergence]") -> None:
        self.divergences = tuple(divergences)
        unverified = [d for d in self.divergences if d.is_unverified]
        lines = "\n  ".join(d.describe() for d in self.divergences)
        super().__init__(
            f"the running system does not match what was declared "
            f"({len(self.divergences)} divergence(s)"
            + (f", {len(unverified)} unverified" if unverified else "")
            + f"):\n  {lines}"
        )
