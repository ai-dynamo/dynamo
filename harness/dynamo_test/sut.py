# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The system under test, and the phases you may act on it in.

``Sut`` is the ACT receiver: the object a test does things to. It is the verb
façade from :mod:`dynamo_test.verbs` made concrete — ``sut.query(...)`` and
``sut.restart(at("frontend"))`` rather than ``sut.frontend.restart()``, with the
component spelling still available through ``REGISTRY.bind``.

## Why phases are enforced rather than documented

A test that acts on the deployment, tears it down, and *then* asserts cannot fail
for the right reason: whatever it reads afterwards is a property of the teardown,
not of the system. The usual form is a check that reads logs after the context
manager has exited and finds none, which reads as "no errors".

So leaving the ``with`` block is a state transition, not just cleanup:

* inside — ACT verbs work; evidence is being collected
* on exit — COLLECT runs **unconditionally**, including when the test has already
  failed, because a failed run is when its evidence matters most; then the bundle
  is sealed
* after — ACT verbs raise :class:`PhaseError` naming the assertion to use
  instead; only :meth:`Sut.evidence` is legal

The error message matters more than the error. ``kill()`` after teardown is
almost always someone reaching for an in-timeline assertion, so the exception
says which CHECK-phase equivalent to reach for.

## Grants

Nothing destructive happens implicitly. A ``Sut`` is constructed with the set of
:class:`~dynamo_test.verbs.Grant` values the suite has opted into, and a verb
whose grant is not held raises before the provider is touched — at the call, not
halfway through a fault.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

from . import catalog as _catalog  # noqa: F401  (registers the standard verbs)
from .evidence import Evidence, Recorder
from .facts import Fact
from .roles import Role, RoleTable, Sel, at
from .verbs import REGISTRY, Grant, Phase, VerbSpec

# `catalog` is imported for its import-time side effect: it is what puts the
# standard verbs in REGISTRY. Without it every verb lookup here raises
# UnknownVerb against an empty registry, which is a confusing way to discover
# that a module was never imported.

__all__ = [
    "Phase",
    "PhaseError",
    "NotGranted",
    "Handle",
    "Provider",
    "Sut",
]


class PhaseError(RuntimeError):
    """A verb was called in a phase where it cannot mean what it says."""


class NotGranted(PermissionError):
    """A verb was called that this suite has not opted into."""

    def __init__(self, verb: str, needed: Grant, held: Iterable[Grant]) -> None:
        held = sorted(g.value for g in held)
        super().__init__(
            f"{verb}() needs the {needed.value!r} grant; this Sut holds "
            f"{', '.join(held) or '<none>'}. Grants are opt-in so a suite cannot "
            "acquire the ability to break things by importing something."
        )


@dataclass(frozen=True)
class Handle:
    """What a verb did, when, and what it resolved to.

    Returned by every ACT verb. Being a value rather than ``None`` is what lets a
    later check say *since this point* — ``expect_restarted(at("worker"),
    since=h)`` — instead of comparing against wall-clock guesses.
    """

    verb: str
    selector: Sel | None
    started_ns: int
    ended_ns: int
    resolved: Mapping[str, Any] = field(default_factory=dict)
    value: Any = None

    @property
    def elapsed_s(self) -> float:
        return (self.ended_ns - self.started_ns) / 1e9

    def to_record(self) -> dict:
        return {
            "verb": self.verb,
            "selector": self.selector.describe() if self.selector else None,
            "started_ns": self.started_ns,
            "ended_ns": self.ended_ns,
            "elapsed_s": round(self.elapsed_s, 6),
            "resolved": dict(self.resolved),
        }


class Provider(Protocol):
    """What a platform must do for the verbs to work.

    Narrow on purpose: a new platform is a handful of small methods rather than a
    fork of a lifecycle module. Everything richer — waiting, probing, evidence —
    is built on top of these in :class:`Sut`, so it is written once.
    """

    def address(self, sel: Sel) -> Fact[str]:
        ...

    def start(self, sel: Sel) -> Mapping[str, Any]:
        ...

    def stop(self, sel: Sel, *, graceful: bool = True) -> Mapping[str, Any]:
        ...

    def restart(self, sel: Sel, **settings: Any) -> Mapping[str, Any]:
        ...

    def replicas(self, sel: Sel) -> Fact[int]:
        ...

    def restart_count(self, sel: Sel) -> Fact[int]:
        ...

    def logs(self, sel: Sel) -> Fact[str]:
        ...

    def request(
        self,
        sel: Sel,
        path: str,
        *,
        method: str = "GET",
        body: Any = None,
        timeout: float = 30.0,
    ) -> Fact[Any]:
        ...


class Sut:
    """The system under test, acted on through verbs.

    Use as a context manager. Leaving the block runs COLLECT and seals the
    bundle; after that only :meth:`evidence` is legal.
    """

    def __init__(
        self,
        provider: Provider,
        roles: RoleTable,
        recorder: Recorder,
        *,
        grants: Iterable[Grant] = (Grant.READ,),
        collectors: Sequence[Callable[["Sut", Recorder], None]] = (),
    ) -> None:
        self.provider = provider
        self.roles = roles
        self.recorder = recorder
        self.grants = frozenset(grants)
        self._collectors = list(collectors)
        self.phase = Phase.PLAN
        self.timeline: list[Handle] = []
        self._evidence: Evidence | None = None

    # ------------------------------------------------------------- phases

    def __enter__(self) -> "Sut":
        self.phase = Phase.ACT
        self.recorder.note("roles", self.roles.to_record())
        self.recorder.note("grants", sorted(g.value for g in self.grants))
        self.recorder.declare("timeline.jsonl", "sut", min_rows=0)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        # COLLECT runs whether or not the body raised. A failed run is exactly
        # when its evidence matters, and a collector that only runs on success
        # collects nothing at the only time anyone needs it.
        self.phase = Phase.COLLECT
        try:
            self.recorder.write_rows(
                "timeline.jsonl", [h.to_record() for h in self.timeline]
            )
            for collect in self._collectors:
                try:
                    collect(self, self.recorder)
                except Exception as collector_error:  # noqa: BLE001
                    self.recorder.note(
                        f"collector_error/{getattr(collect, '__name__', 'anonymous')}",
                        repr(collector_error),
                    )
        finally:
            self.recorder.seal()
            self.phase = Phase.CHECK
        return False  # never swallow the body's exception

    def evidence(self) -> Evidence:
        """The sealed bundle. Legal only after the ``with`` block exits."""
        if self.phase is not Phase.CHECK:
            raise PhaseError(
                f"evidence() is a CHECK-phase call and this Sut is in {self.phase.value}. "
                "Leave the with-block first; judging a bundle still being written "
                "gives a different answer on a replay."
            )
        if self._evidence is None:
            self._evidence = self.recorder.evidence()
        return self._evidence

    # -------------------------------------------------------------- guards

    def _act(self, name: str) -> VerbSpec:
        spec = REGISTRY.require(name)
        if self.phase is not Phase.ACT:
            raise PhaseError(
                f"{name}() is an ACT verb and this Sut is in {self.phase.value}. "
                + (
                    "Assert over the evidence instead — the system may not exist "
                    "any more, so whatever this read would return is a property of "
                    "the teardown, not of the deployment."
                    if self.phase is Phase.CHECK
                    else "Enter the with-block first."
                )
            )
        if spec.grant not in self.grants and spec.grant is not Grant.READ:
            raise NotGranted(name, spec.grant, self.grants)
        return spec

    def _record(
        self,
        name: str,
        sel: Sel | None,
        started: int,
        resolved: Mapping[str, Any] | None = None,
        value: Any = None,
    ) -> Handle:
        handle = Handle(
            verb=name,
            selector=sel,
            started_ns=started,
            ended_ns=time.monotonic_ns(),
            resolved=dict(resolved or {}),
            value=value,
        )
        self.timeline.append(handle)
        return handle

    # --------------------------------------------------------------- reach

    def address(self, sel: Sel = at(Role.FRONTEND)) -> Fact[str]:
        self._act("address")
        return self.provider.address(sel)

    def url(self, sel: Sel = at(Role.FRONTEND)) -> Fact[str]:
        self._act("url")
        return self.provider.address(sel)

    # ----------------------------------------------------------- inference

    def query(
        self, payload: Any, sel: Sel = at(Role.FRONTEND), *, timeout: float = 60.0
    ) -> Handle:
        """One completion. The default role is why most tests just say query()."""
        self._act("query")
        started = time.monotonic_ns()
        body = {"prompt": payload} if isinstance(payload, str) else payload
        answer = self.provider.request(
            sel, "/v1/completions", method="POST", body=body, timeout=timeout
        )
        return self._record(
            "query", sel, started, {"ok": answer.is_known}, value=answer
        )

    def models(self, sel: Sel = at(Role.FRONTEND)) -> Handle:
        self._act("models")
        started = time.monotonic_ns()
        got = self.provider.request(sel, "/v1/models")
        return self._record("models", sel, started, {"ok": got.is_known}, value=got)

    # ------------------------------------------------------------ waiting

    def wait(self, seconds: float) -> Handle:
        self._act("wait")
        started = time.monotonic_ns()
        time.sleep(seconds)
        return self._record("wait", None, started, {"seconds": seconds})

    def wait_serving(
        self,
        sel: Sel = at(Role.FRONTEND),
        *,
        timeout: float = 120.0,
        interval: float = 0.2,
    ) -> Handle:
        """Wait until the endpoint actually answers, not until a pod is ready.

        Readiness and serving are different facts, and a test that waits for the
        first and asserts on the second is measuring a race.
        """
        self._act("wait_serving")
        started = time.monotonic_ns()
        deadline = time.monotonic() + timeout
        last = "never attempted"
        while time.monotonic() < deadline:
            got = self.provider.request(sel, "/v1/models", timeout=min(5.0, timeout))
            if got.is_known:
                return self._record(
                    "wait_serving", sel, started, {"ready": True, "models": got.value}
                )
            last = got.detail
            time.sleep(interval)
        self._record("wait_serving", sel, started, {"ready": False, "last": last})
        raise TimeoutError(
            f"{sel.describe()} did not serve within {timeout}s; last attempt: {last}"
        )

    # ---------------------------------------------------------- lifecycle

    def restart(self, sel: Sel, **settings: Any) -> Handle:
        self._act("restart")
        started = time.monotonic_ns()
        resolved = self.provider.restart(sel, **settings)
        return self._record("restart", sel, started, resolved)

    def stop(self, sel: Sel, *, graceful: bool = True) -> Handle:
        self._act("stop")
        started = time.monotonic_ns()
        resolved = self.provider.stop(sel, graceful=graceful)
        return self._record("stop", sel, started, resolved)

    def start(self, sel: Sel) -> Handle:
        self._act("start")
        started = time.monotonic_ns()
        resolved = self.provider.start(sel)
        return self._record("start", sel, started, resolved)

    # -------------------------------------------- in-timeline assertions

    def require_restarted(
        self, sel: Sel, *, since: Handle, within: float = 60.0
    ) -> Handle:
        """Assert now that a role restarted since ``since``, and record it.

        ``require_*`` raises immediately, unlike ``expect_*`` which is evaluated
        after the fact over collected evidence. Both exist because some facts —
        a restart count — are only observable while the system is up.
        """
        self._act("require_restarted")
        started = time.monotonic_ns()
        deadline = time.monotonic() + within
        before = since.resolved.get("restart_count")
        seen = None
        while time.monotonic() < deadline:
            got = self.provider.restart_count(sel)
            if got.is_known:
                seen = got.require()
                if before is None or seen > before:
                    return self._record(
                        "require_restarted",
                        sel,
                        started,
                        {"before": before, "after": seen},
                    )
            time.sleep(0.1)
        self._record(
            "require_restarted", sel, started, {"before": before, "after": seen}
        )
        raise AssertionError(
            f"{sel.describe()} did not restart within {within}s "
            f"(restart count {before} -> {seen})"
        )

    # ------------------------------------------------------------ reading

    def logs(self, sel: Sel) -> Fact[str]:
        """Live logs. In CHECK phase, read them from the bundle instead."""
        self._act("logs")
        return self.provider.logs(sel)

    def component(self, role: Role | str):
        """The component-first view: ``sut.component('frontend').restart()``.

        Generated from the verb registry, so it cannot drift from the verb
        surface — it is not separately written down.
        """
        return REGISTRY.bind(self, role)
