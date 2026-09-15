# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Making the system available, three ways, through one mechanism.

Every tier needs the system running before a test can measure it, but *how* it
gets there differs by who owns it — and that is a property of the site, not of
the test. So a test never says "start the workers"; it says "give me the system",
and the site decides whether that means creating it, checking it, or joining it.

:data:`~dynamo_test.site.Ownership` names the three:

``IMPOSE``
    Create it, wait, tear it down afterwards. The stack is a **recipe**.

``VERIFY``
    Bind to what is already running and confirm it matches. The stack is an
    **expectation**. Nothing is created and nothing is destroyed.

``ADOPT``
    Create it only if it is absent, and never destroy it. For shared,
    expensive dependencies.

## Why VERIFY exists

It is what lets the existing all-in-one CI job run unchanged. That container
already started everything before pytest began; the test's job is to confirm it
got what it expected and then measure it. Without VERIFY the only options are to
re-start a system that is already running, or to skip the check entirely and
hope.

And the check has to happen **at bind time**. Attach to a deployment serving a
different model and, without a comparison, the first symptom is a 404 at query
time — which reads like a routing bug and sends the reader looking in the wrong
place. :class:`~dynamo_test.site.Mismatch` moves that failure to the moment the
wrong system was adopted, and names the field.

## Why ADOPT is transcribed, not invented

The suite already solved this once, for one dependency:
``tests/serve/lora_utils.py:140-167`` health-probes MinIO, and on success sets
``_owns_container = False`` so ``stop()`` becomes a no-op. That is exactly
ADOPT. This module generalises the shape rather than inventing a new one.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Sequence

from .facts import Fact
from .roles import Role, Sel, at
from .site import Divergence, Mismatch, Ownership, Refused, Site
from .stack import Stack

__all__ = ["BringUp", "bring_up", "observe"]


@dataclass(frozen=True)
class BringUp:
    """What bringing the system up actually did.

    ``owns`` is the field that matters at teardown: it is false whenever the run
    joined something it did not create, and a caller that tears down anyway is
    destroying somebody else's deployment.
    """

    site: Site
    owns: bool
    started: tuple[str, ...] = ()
    adopted: tuple[str, ...] = ()
    checked: tuple[str, ...] = ()
    divergences: tuple[Divergence, ...] = ()

    def describe(self) -> str:
        parts = []
        if self.started:
            parts.append(f"started {', '.join(self.started)}")
        if self.adopted:
            parts.append(f"adopted {', '.join(self.adopted)}")
        if self.checked:
            parts.append(f"verified {', '.join(self.checked)}")
        return (
            f"{self.site.ownership} on {self.site.name}: "
            + ("; ".join(parts) or "nothing to do")
            + (
                " (owned)"
                if self.owns
                else " (not owned — teardown must not destroy it)"
            )
        )

    def to_record(self) -> dict:
        return {
            "site": self.site.name,
            "ownership": self.site.ownership.value,
            "owns": self.owns,
            "started": list(self.started),
            "adopted": list(self.adopted),
            "checked": list(self.checked),
            "divergences": [d.describe() for d in self.divergences],
        }


def observe(provider: Any, sel: Sel, timeout: float = 0.0) -> Fact[Any]:
    """What the running system says it is serving, or why that is unknown.

    Deliberately returns a :class:`Fact`. "Nothing is listening" and "something
    is listening and serving a different model" are different answers, and a
    comparison that cannot tell them apart will score a connection failure as a
    disagreement.
    """
    deadline = time.monotonic() + timeout
    while True:
        got = provider.request(sel, "/v1/models", timeout=min(5.0, timeout or 5.0))
        if got.is_known or time.monotonic() >= deadline:
            return got
        time.sleep(0.2)


def _served_models(models: Any) -> tuple[str, ...]:
    if isinstance(models, dict):
        return tuple(
            str(entry.get("id"))
            for entry in models.get("data", [])
            if isinstance(entry, dict) and entry.get("id")
        )
    return ()


def bring_up(
    stack: Stack,
    site: Site,
    provider: Any,
    *,
    roles: Sequence[Role] | None = None,
    timeout: float = 120.0,
) -> BringUp:
    """Make the system available in the way this site's ownership requires."""
    wanted = list(roles or [c.role for c in stack.plan])

    if site.ownership is Ownership.IMPOSE:
        return _impose(stack, site, provider, wanted, timeout)
    if site.ownership is Ownership.ADOPT:
        return _adopt(stack, site, provider, wanted, timeout)
    if site.ownership is Ownership.VERIFY:
        return _verify(stack, site, provider, wanted, timeout)
    raise ValueError(f"unhandled ownership {site.ownership!r}")


def _impose(stack, site, provider, wanted, timeout) -> BringUp:
    started = []
    for role in wanted:
        provider.start(at(role))
        started.append(str(role))
    return BringUp(site=site, owns=True, started=tuple(started))


def _adopt(stack, site, provider, wanted, timeout) -> BringUp:
    """Create only what is missing, and own none of it.

    Transcribed from ``tests/serve/lora_utils.py:148-152``: probe first, and on
    success record that this run did not create it, so teardown leaves it alone.
    """
    started, adopted = [], []
    for role in wanted:
        running = provider.replicas(at(role))
        if running.or_else(0) > 0:
            adopted.append(str(role))
            continue
        provider.start(at(role))
        started.append(str(role))
    # Deliberately False even when this run started something. ADOPT promises
    # never to destroy; a run that created half a stack and tore down that half
    # would leave the other half orphaned and the next run confused.
    return BringUp(
        site=site, owns=False, started=tuple(started), adopted=tuple(adopted)
    )


def _verify(stack, site, provider, wanted, timeout) -> BringUp:
    """Bind to what is running and confirm it is what was declared."""
    divergences: list[Divergence] = []
    checked = []

    for role in wanted:
        sel = at(role)
        running = provider.replicas(sel)
        if running.or_else(0) < 1:
            divergences.append(
                Divergence(
                    role=str(role),
                    field="replicas",
                    declared=Fact.known(
                        1, str(stack.plan.source), "the stack expects it"
                    ),
                    observed=running,
                )
            )
            continue
        checked.append(str(role))

    frontends = [r for r in wanted if r is Role.FRONTEND]
    if frontends and not divergences:
        declared = _declared_model(stack)
        observed = observe(provider, at(Role.FRONTEND), timeout=timeout)
        if declared.is_known:
            served = _served_models(observed.or_else(None))
            if observed.is_known and declared.require() not in served:
                divergences.append(
                    Divergence(
                        role="frontend",
                        field="model",
                        declared=declared,
                        observed=Fact.known(
                            served[0] if served else "<none>",
                            "/v1/models",
                            f"serving {list(served)}",
                        ),
                    )
                )
            elif not observed.is_known:
                divergences.append(Divergence("frontend", "model", declared, observed))

    if [d for d in divergences if not d.is_unverified]:
        raise Mismatch(divergences)

    return BringUp(
        site=site,
        owns=False,
        checked=tuple(checked),
        divergences=tuple(divergences),
    )


def _declared_model(stack: Stack) -> Fact[str]:
    """The model this stack says it serves, from whichever component names one.

    Reads the command line directly rather than through the engine dialect. A
    frontend declares ``--model`` but runs no engine, so a dialect-only lookup
    returns UNKNOWN for it — and an UNKNOWN declaration means the comparison is
    skipped, which silently turns VERIFY into a mode that always succeeds. That
    is exactly what the negative control caught.
    """
    for component in stack.plan:
        got = component.argv.model()
        if got.is_known:
            return got
    for component in stack.plan:
        got = component.read("model")
        if got.is_known:
            return got
    return Fact.absent(stack.plan.source, "no component declares a model")


def refuse_if_not_owned(site: Site, verb: str) -> None:
    """Raise if a destructive verb is being attempted on a system we joined.

    A helper rather than a decorator so the call site reads plainly. It raises
    rather than returning, because the whole hazard is a quiet no-op.
    """
    if site.ownership is Ownership.IMPOSE:
        return
    raise Refused(
        verb,
        site,
        f"ownership={site.ownership}: this run joined the system rather than "
        "creating it, so tearing it down would destroy something it does not own",
    )
