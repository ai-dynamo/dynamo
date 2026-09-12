# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Roles, selectors, and the one place a role becomes a concrete service name.

A Dynamo deployment names the same thing several ways at once. The frontend is
``Frontend`` in a v1alpha1 manifest, ``frontend`` as a log directory, ``dynamo``
in a Prometheus label, and ``dynamo.frontend`` as a process. Every layer that
re-derives one spelling from another is a place where a lookup can miss.

When a lookup misses, what happens next is decided by the polarity of whoever
asked. Measured in the scenario suite this design converges with: log reads
resolve a service name through a lowercase directory listing plus a hard-coded
alias list of three names, and every consumer then does
``logs.get(name, "") or ""``. Names outside that list — ``VllmWorker``,
``TRTLLMPrefillWorker``, ``TRTLLMDecodeWorker``, all of which are passed to
checks today — resolve to the empty string with no error. A check asserting a
pattern *is* present then fails loudly; a check asserting one is *not* present
passes vacuously. Which of those a given deployment gets is luck.

So this module makes two rules structural:

1. **Resolution happens once, here.** :class:`RoleTable` is derived from the
   deployment plan, and every later layer asks it rather than reconstructing a
   name. ``log_key`` is one string used for both the bundle directory and the
   log-scrape key, so the two cannot drift apart.
2. **An unresolved role raises at plan time**, naming the roles that do exist.
   It never returns a default, because an empty result is indistinguishable from
   a real one — an empty log stream from a wrong-but-valid key is
   *present-and-empty*, not absent, so no amount of absence checking downstream
   recovers it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterator

__all__ = [
    "Role",
    "Process",
    "Policy",
    "PortName",
    "Sel",
    "at",
    "RoleBinding",
    "RoleTable",
    "UnknownRole",
]


class Role(str, Enum):
    """What a component *is*, independent of what a manifest calls it."""

    FRONTEND = "frontend"
    WORKER = "worker"
    PREFILL = "prefill"
    DECODE = "decode"
    ENCODE = "encode"
    ROUTER = "router"
    PLANNER = "planner"
    OPERATOR = "operator"
    KVBM = "kvbm"
    LOAD = "load"
    ETCD = "etcd"
    NATS = "nats"
    GATEWAY = "gateway"
    SELF = "self"

    def __str__(self) -> str:
        return self.value


class Process(str, Enum):
    """A process *within* a role, named semantically.

    ``ENGINE`` is ``VLLM::EngineCore`` on vLLM and a ``sglang::scheduler`` child
    on SGLang. Tests say what they mean; the engine dialect knows the spelling.
    """

    MAIN = "main"
    ENGINE = "engine"
    WORKER = "worker"
    RANK = "rank"


class Policy(str, Enum):
    """How to pick among replicas when a selector does not name one."""

    FIRST = "first"
    ALL = "all"
    RANDOM = "random"
    HOTTEST = "hottest"


class PortName(str, Enum):
    SERVICE = "service"
    SYSTEM = "system"
    METRICS = "metrics"
    GRPC = "grpc"


class UnknownRole(KeyError):
    """A role was requested that this deployment does not have.

    Raised when the role table is built or first consulted — before any fault is
    injected — so the failure names a configuration mistake rather than surfacing
    later as a check that mysteriously found nothing.
    """

    def __init__(self, role: object, known: tuple[str, ...]) -> None:
        self.role = role
        self.known = known
        super().__init__(
            f"no binding for role {role!r}; this deployment declares: "
            f"{', '.join(known) if known else '<none>'}"
        )


@dataclass(frozen=True)
class Sel:
    """Which instance of a role a verb acts on.

    Selection is hoisted out of every verb signature. Without this, each verb
    grows its own ``rank=``/``process=``/``pod_indices=``/``fraction=`` argument
    set and they drift; with it, ``restart(at("decode", replica=1))`` and
    ``stall(at("decode", replica=1))`` mean the same thing by construction.
    """

    role: Role
    replica: int | None = None
    policy: Policy | None = None
    fraction: float | None = None
    rank: int | None = None
    process: Process | None = None
    port: PortName = PortName.SERVICE

    def __post_init__(self) -> None:
        if self.replica is not None and self.policy is not None:
            raise ValueError(
                f"{self}: replica and policy both given; a selector either names "
                "a replica or says how to choose one"
            )
        if self.replica is not None and self.replica < 0:
            raise ValueError(f"replica must be non-negative, got {self.replica}")
        if self.fraction is not None and not 0 < self.fraction <= 1:
            raise ValueError(f"fraction must be in (0, 1], got {self.fraction}")

    def describe(self) -> str:
        """A short, stable string for log lines and evidence keys."""
        parts = [str(self.role)]
        if self.replica is not None:
            parts.append(f"replica={self.replica}")
        elif self.policy is not None:
            parts.append(f"policy={self.policy.value}")
        if self.rank is not None:
            parts.append(f"rank={self.rank}")
        if self.process is not None:
            parts.append(f"process={self.process.value}")
        return "/".join(parts)

    def __str__(self) -> str:
        return self.describe()


def at(role: Role | str, **kwargs: Any) -> Sel:
    """Build a :class:`Sel`. The only constructor callers should use.

    Accepts a plain string so scenario documents can say ``role: decode``
    without importing the enum; an unrecognised string fails here, at the call
    site, rather than as a lookup miss much later.
    """
    if not isinstance(role, Role):
        try:
            role = Role(str(role).lower())
        except ValueError:
            raise UnknownRole(role, tuple(r.value for r in Role)) from None
    return Sel(role=role, **kwargs)


@dataclass(frozen=True)
class RoleBinding:
    """Every spelling of one role, resolved once.

    ``log_key`` is deliberately a single field rather than a directory name and
    a scrape key that some helper relates by lowercasing. That relationship is
    exactly what silently drops logs for services whose name is not in an alias
    list.
    """

    role: Role
    service: str
    log_key: str
    metric_labels: Mapping[str, str] = field(default_factory=dict)
    argv_target: str = "main"
    processes: Mapping[Process, str] = field(default_factory=dict)
    ports: Mapping[PortName, int] = field(default_factory=dict)

    def process_pattern(self, process: Process) -> str:
        """The ``ps`` substring identifying ``process`` in this role."""
        try:
            return self.processes[process]
        except KeyError:
            raise KeyError(
                f"role {self.role} has no {process.value!r} process; it declares: "
                f"{', '.join(p.value for p in self.processes) or '<none>'}"
            ) from None

    def port(self, name: PortName = PortName.SERVICE) -> int:
        try:
            return self.ports[name]
        except KeyError:
            raise KeyError(
                f"role {self.role} exposes no {name.value!r} port; it declares: "
                f"{', '.join(p.value for p in self.ports) or '<none>'}"
            ) from None


class RoleTable(Mapping):
    """The resolved roles of one deployment.

    Built once, then consulted — never rebuilt from a manifest by a later layer.
    Serialise it into the run record so a check can prove which service each
    role resolved to instead of inferring it from whether a read returned
    anything.
    """

    def __init__(self, bindings: Mapping[Role, RoleBinding]) -> None:
        self._bindings = dict(bindings)
        for role, binding in self._bindings.items():
            if binding.role != role:
                raise ValueError(
                    f"binding filed under {role} declares role {binding.role}"
                )
        duplicates = _duplicates(b.log_key for b in self._bindings.values())
        if duplicates:
            raise ValueError(
                f"log_key must identify exactly one role; reused: {sorted(duplicates)}. "
                "Two roles sharing a key means one role's evidence overwrites the other's."
            )

    def __getitem__(self, role: Role | str) -> RoleBinding:
        return self.require(role)

    def __iter__(self) -> Iterator[Role]:
        return iter(self._bindings)

    def __len__(self) -> int:
        return len(self._bindings)

    def require(self, role: Role | str) -> RoleBinding:
        """The binding for ``role``, or raise naming the roles that exist.

        Never returns a placeholder. A default here reappears downstream as an
        empty log stream or an unmatched metric label, where it is
        indistinguishable from a real measurement of nothing.
        """
        if not isinstance(role, Role):
            try:
                role = Role(str(role).lower())
            except ValueError:
                raise UnknownRole(role, self.known()) from None
        try:
            return self._bindings[role]
        except KeyError:
            raise UnknownRole(role, self.known()) from None

    def known(self) -> tuple[str, ...]:
        return tuple(sorted(str(r) for r in self._bindings))

    def resolve(self, sel: Sel) -> RoleBinding:
        """The binding a selector refers to."""
        return self.require(sel.role)

    def to_record(self) -> dict:
        """A JSON-safe view for the run record.

        Recording resolution is what makes ``role_bindings_resolved`` provable
        rather than assumed.
        """
        return {
            str(role): {
                "service": b.service,
                "log_key": b.log_key,
                "metric_labels": dict(b.metric_labels),
                "argv_target": b.argv_target,
                "processes": {p.value: v for p, v in b.processes.items()},
                "ports": {p.value: v for p, v in b.ports.items()},
            }
            for role, b in sorted(self._bindings.items(), key=lambda kv: str(kv[0]))
        }

    def __repr__(self) -> str:
        return f"RoleTable({', '.join(self.known())})"


def _duplicates(values) -> set:
    seen, dupes = set(), set()
    for value in values:
        if value in seen:
            dupes.add(value)
        seen.add(value)
    return dupes
