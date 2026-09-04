# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The verb registry: one name per action, and what that name promises.

## Why verbs take the role as an argument

``dynamo.restart("frontend")`` rather than ``dynamo.frontend.restart()``. The
reason is not taste, it is that a scenario written as a document has to become a
call:

    - kind: StallProcess          ->   stall_process(at("worker", rank=0), seconds=30)
      role: worker
      rank: 0
      seconds: 30

With the role as an argument that mapping is mechanical — the document names a
verb and a selector, and both are values. With a component-first surface the
runner needs a dispatch table from event kind to attribute path, which is a
second source of truth that drifts from the first.

The component-first spelling is still available: :meth:`VerbRegistry.bind` makes
``sut.frontend.restart()`` out of the same registry entry, so there is one
definition and two spellings, rather than two definitions.

## Why the name carries the gating policy

A check that can be switched from asserting to observing by an argument reports
the same result either way. Measured in the scenario suite this converges with:
five of its 49 checks assert *and* contain a bare early return, so an argument
turns the gate off — and the report still says ``PASSED``. There is no way, from
the outside, to tell a check that held from a check that was asked not to look.

So the policy lives in the name, where the result can see it:

===============  ========  ================================================
prefix           receiver  contributes to
===============  ========  ================================================
``expect_``      JUDGE     the verdict — must hold
``observe_``     JUDGE     ``OBSERVED``; measured on purpose, never gates
``require_``     ACT       must hold *now*; raises now, and records
``require_valid_`` JUDGE   run validity — is this measurement admissible?
``report_``      JUDGE     an artifact; runs first, never gates
everything else  ACT       does a thing
===============  ========  ================================================

``observe_worker_panics`` and ``expect_no_worker_panics`` are different verbs
with different verdicts. The branch that is invisible today becomes a name.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterator, Mapping, Sequence

from .roles import Role, Sel, at

__all__ = [
    "Phase",
    "Receiver",
    "Grant",
    "Contribution",
    "VerbSpec",
    "VerbCall",
    "VerbRegistry",
    "REGISTRY",
    "verb",
    "DuplicateVerb",
    "NamingLaw",
    "UnknownVerb",
    "MissingProofContract",
]


class Phase(str, Enum):
    """When a verb may run. Enforced by the receiver, not by convention."""

    ARRANGE = "arrange"
    ACT = "act"
    COLLECT = "collect"
    CHECK = "check"


class Receiver(str, Enum):
    ACT = "act"
    JUDGE = "judge"


class Grant(str, Enum):
    """What a verb is allowed to do. A suite opts in; nothing is implicit.

    ``LIFECYCLE`` and ``FAULT`` are separate because a suite that may restart a
    worker to change a flag is not thereby permitted to kill one, and
    ``INFRA`` is separate again because scaling ``etcd`` to zero in a shared
    namespace breaks tests that are not yours.
    """

    READ = "read"
    INFER = "infer"
    LIFECYCLE = "lifecycle"
    FAULT = "fault"
    INFRA = "infra"


class Contribution(str, Enum):
    """What a verb's outcome contributes to the run result.

    Deliberately not called ``Verdict``. A *verdict* is what a run concluded
    (:class:`dynamo_test.evidence.Verdict`); this is what one verb's outcome
    feeds into that conclusion. They are different questions and sharing a name
    made it possible to import the wrong one.
    """

    GATES = "gates"
    OBSERVED = "observed"
    VALIDITY = "validity"
    ARTIFACT = "artifact"
    NONE = "none"


class DuplicateVerb(ValueError):
    """A verb name was registered twice."""


class NamingLaw(ValueError):
    """A verb name does not match the policy its receiver requires."""


class MissingProofContract(ValueError):
    """A fault verb did not declare what its effect proves.

    A fault whose effect is not observable is indistinguishable from a fault
    that silently did nothing, and a test built on one passes for the wrong
    reason.
    """


class UnknownVerb(KeyError):
    def __init__(self, name: object, known: Sequence[str]) -> None:
        self.name = name
        near = [k for k in known if isinstance(name, str) and name.lower() in k]
        super().__init__(
            f"no verb named {name!r}"
            + (f"; did you mean {', '.join(near)}?" if near else "")
            + f" ({len(known)} verbs registered)"
        )


# Pure readers are exempt from the JUDGE prefix law: they return data and assert
# nothing, so there is no gating policy for a prefix to carry.
PURE_READERS = frozenset(
    {
        "series",
        "logs",
        "timeline",
        "metrics",
        "spans",
        "requests",
        "records",
        "pod_state",
        "artifacts",
        "proofs",
    }
)

_PREFIX_CONTRIBUTION = {
    "expect_": Contribution.GATES,
    "observe_": Contribution.OBSERVED,
    "require_valid_": Contribution.VALIDITY,
    "report_": Contribution.ARTIFACT,
}

_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True)
class VerbSpec:
    """One verb: what it does, when it may run, and what it promises."""

    name: str
    receiver: Receiver
    phase: Phase
    grant: Grant
    summary: str
    takes_selector: bool = True
    default_role: Role | None = None
    params: Mapping[str, str] = field(default_factory=dict)
    proves: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not _NAME.match(self.name):
            raise NamingLaw(
                f"{self.name!r} is not a valid verb name: lowercase, digits and "
                "underscores, starting with a letter"
            )
        if self.receiver is Receiver.JUDGE:
            if self.name not in PURE_READERS and not any(
                self.name.startswith(p) for p in _PREFIX_CONTRIBUTION
            ):
                raise NamingLaw(
                    f"JUDGE verb {self.name!r} must start with one of "
                    f"{', '.join(sorted(_PREFIX_CONTRIBUTION))} — the prefix is how a "
                    "reader tells a gate from an observation — or be one of the "
                    f"pure readers ({', '.join(sorted(PURE_READERS))})."
                )
        elif self.name.startswith(("expect_", "observe_", "report_")):
            raise NamingLaw(
                f"ACT verb {self.name!r} uses a JUDGE prefix. ACT-phase assertions "
                "are spelled require_*, which raise immediately and record."
            )
        if self.grant is Grant.FAULT and not self.proves:
            raise MissingProofContract(
                f"fault verb {self.name!r} declares no proves=(...). A fault whose "
                "effect cannot be observed is indistinguishable from one that "
                "silently did nothing."
            )

    @property
    def contributes(self) -> Contribution:
        """What this verb's outcome contributes to the run result."""
        if self.receiver is Receiver.ACT:
            return Contribution.NONE
        for prefix, contribution in _PREFIX_CONTRIBUTION.items():
            if self.name.startswith(prefix):
                return contribution
        return Contribution.NONE

    @property
    def gates(self) -> bool:
        return self.contributes is Contribution.GATES

    def call(self, **kwargs: Any) -> "VerbCall":
        """Build a call to this verb, validating the selector."""
        return VerbCall.build(self, kwargs)


@dataclass(frozen=True)
class VerbCall:
    """A verb plus its arguments — the value a scenario document becomes.

    Being a value rather than a bound method is what lets a timeline be
    serialised, replayed, diffed against a previous run, and printed in a plan
    without executing anything.
    """

    spec: VerbSpec
    selector: Sel | None
    kwargs: Mapping[str, Any]

    @classmethod
    def build(cls, spec: VerbSpec, kwargs: Mapping[str, Any]) -> "VerbCall":
        kwargs = dict(kwargs)
        selector = kwargs.pop("sel", None)

        if spec.takes_selector:
            role = kwargs.pop("role", None)
            if selector is None:
                selector_fields = {
                    k: kwargs.pop(k)
                    for k in (
                        "replica",
                        "policy",
                        "rank",
                        "process",
                        "port",
                        "fraction",
                    )
                    if k in kwargs
                }
                if role is not None:
                    selector = at(role, **selector_fields)
                elif spec.default_role is not None:
                    selector = at(spec.default_role, **selector_fields)
                elif selector_fields:
                    raise TypeError(
                        f"{spec.name}: {', '.join(sorted(selector_fields))} given "
                        "without a role to apply them to"
                    )
        elif selector is not None or "role" in kwargs:
            raise TypeError(f"{spec.name} does not act on a role")

        unknown = set(kwargs) - set(spec.params)
        if unknown and spec.params:
            raise TypeError(
                f"{spec.name}: unexpected argument(s) {', '.join(sorted(unknown))}; "
                f"accepts {', '.join(sorted(spec.params)) or '<none>'}"
            )
        return cls(spec=spec, selector=selector, kwargs=kwargs)

    @classmethod
    def from_document(
        cls, event: Mapping[str, Any], registry: "VerbRegistry | None" = None
    ) -> "VerbCall":
        """Read one scenario-document event into a call.

        The whole point of role-as-argument: this is a lookup and a splat, not a
        dispatch table that has to be kept in step with the verb surface.
        """
        registry = registry or REGISTRY
        event = dict(event)
        kind = event.pop("kind", None) or event.pop("verb", None)
        if kind is None:
            raise ValueError(f"event has no 'kind' or 'verb': {sorted(event)}")
        return registry.require(str(kind)).call(**event)

    def to_document(self) -> dict:
        """The inverse of :meth:`from_document`."""
        out: dict[str, Any] = {"kind": self.spec.name}
        if self.selector is not None:
            out["role"] = str(self.selector.role)
            for name in ("replica", "rank", "fraction"):
                value = getattr(self.selector, name)
                if value is not None:
                    out[name] = value
            if self.selector.policy is not None:
                out["policy"] = self.selector.policy.value
            if self.selector.process is not None:
                out["process"] = self.selector.process.value
        out.update(self.kwargs)
        return out

    def describe(self) -> str:
        parts = []
        if self.selector is not None:
            parts.append(f"at({self.selector.describe()!r})")
        parts += [f"{k}={v!r}" for k, v in sorted(self.kwargs.items())]
        return f"{self.spec.name}({', '.join(parts)})"

    def __str__(self) -> str:
        return self.describe()


class VerbRegistry:
    """Every verb, and the rules they must satisfy.

    Validation runs at registration, so a violation is an import error in the
    module that declared it rather than a surprise at run time.
    """

    def __init__(self) -> None:
        self._specs: dict[str, VerbSpec] = {}
        self._aliases: dict[str, str] = {}

    def register(self, spec: VerbSpec) -> VerbSpec:
        if spec.name in self._specs or spec.name in self._aliases:
            raise DuplicateVerb(
                f"{spec.name!r} is already registered; a verb name must exist on "
                "at most one receiver"
            )
        for alias in spec.aliases:
            if alias in self._specs or alias in self._aliases:
                raise DuplicateVerb(f"alias {alias!r} collides with an existing verb")
        self._specs[spec.name] = spec
        for alias in spec.aliases:
            self._aliases[alias] = spec.name
        return spec

    def require(self, name: str) -> VerbSpec:
        if name in self._specs:
            return self._specs[name]
        if name in self._aliases:
            return self._specs[self._aliases[name]]
        raise UnknownVerb(name, sorted(self._specs))

    def __contains__(self, name: object) -> bool:
        return name in self._specs or name in self._aliases

    def __iter__(self) -> Iterator[VerbSpec]:
        return iter(self._specs.values())

    def __len__(self) -> int:
        return len(self._specs)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._specs))

    def for_receiver(self, receiver: Receiver) -> tuple[VerbSpec, ...]:
        return tuple(s for s in self if s.receiver is receiver)

    def grants_needed(self, calls: Sequence[VerbCall]) -> frozenset[Grant]:
        """Every grant a timeline requires, computable before it runs.

        A suite that will kill a worker can be refused at plan time rather than
        halfway through, when half the faults have already landed.
        """
        return frozenset(c.spec.grant for c in calls)

    def bind(self, target: Any, role: Role | str) -> "_RoleView":
        """The component-first spelling, generated from this registry.

        ``registry.bind(sut, "frontend").restart()`` is exactly
        ``sut.restart(at("frontend"))``. One definition, two spellings — the
        component surface cannot drift from the verb surface because it is not
        separately written down.
        """
        return _RoleView(self, target, at(role).role)

    def to_record(self) -> list[dict]:
        """A JSON-safe catalogue, for ``dynamo-test list verbs``."""
        return [
            {
                "name": s.name,
                "receiver": s.receiver.value,
                "phase": s.phase.value,
                "grant": s.grant.value,
                "contributes": s.contributes.value,
                "gates": s.gates,
                "takes_selector": s.takes_selector,
                "default_role": str(s.default_role) if s.default_role else None,
                "params": dict(s.params),
                "proves": list(s.proves),
                "aliases": list(s.aliases),
                "summary": s.summary,
            }
            for s in sorted(self, key=lambda s: s.name)
        ]

    def __repr__(self) -> str:
        return f"VerbRegistry({len(self._specs)} verbs)"


class _RoleView:
    """``sut.frontend.restart()`` over ``sut.restart(at('frontend'))``."""

    def __init__(self, registry: VerbRegistry, target: Any, role: Role) -> None:
        self._registry = registry
        self._target = target
        self._role = role

    def __getattr__(self, name: str) -> Callable[..., Any]:
        spec = self._registry.require(name)
        method = getattr(self._target, spec.name)
        if not spec.takes_selector:
            return method

        def bound(*args: Any, **kwargs: Any) -> Any:
            if "sel" in kwargs or (args and isinstance(args[0], Sel)):
                raise TypeError(
                    f"{self._role}.{name}() already names its role; pass replica= "
                    "or rank= instead of a selector"
                )
            selector_fields = {
                k: kwargs.pop(k)
                for k in ("replica", "policy", "rank", "process", "port", "fraction")
                if k in kwargs
            }
            return method(*args, sel=at(self._role, **selector_fields), **kwargs)

        return bound

    def __dir__(self) -> list[str]:
        return sorted(self._registry.names())

    def __repr__(self) -> str:
        return f"<{self._role} view of {type(self._target).__name__}>"


REGISTRY = VerbRegistry()


def verb(
    name: str,
    *,
    receiver: Receiver = Receiver.ACT,
    phase: Phase = Phase.ACT,
    grant: Grant = Grant.READ,
    summary: str = "",
    takes_selector: bool = True,
    default_role: Role | None = None,
    params: Mapping[str, str] | None = None,
    proves: Sequence[str] = (),
    aliases: Sequence[str] = (),
    registry: VerbRegistry | None = None,
) -> VerbSpec:
    """Declare a verb. Returns the spec so a module can reference it."""
    return (registry or REGISTRY).register(
        VerbSpec(
            name=name,
            receiver=receiver,
            phase=phase,
            grant=grant,
            summary=summary,
            takes_selector=takes_selector,
            default_role=default_role,
            params=dict(params or {}),
            proves=tuple(proves),
            aliases=tuple(aliases),
        )
    )
