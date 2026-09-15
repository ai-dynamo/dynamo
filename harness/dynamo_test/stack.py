# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Turning a manifest into something you could actually run.

A ``DynamoGraphDeployment`` is not a complete description of how to start
anything. On Kubernetes it does not need to be: the operator fills in the gaps
on the way to a Pod. Measured over ``recipes/``:

* **57 of 427 components (13%)** declare neither ``command`` nor ``args``, and
  **27 of 164 frontends (16%)** declare no command at all. The operator supplies
  ``python3 -m dynamo.frontend``
  (``deploy/operator/internal/dynamo/component_frontend.go:31-32``).
* **0 of 427 components declare ``NATS_SERVER`` or ``ETCD_ENDPOINTS``.** Not
  "few" — none. The operator injects both from its own configuration
  (``graph.go:1570,1577``). Nothing discovers anything without them.
* Almost no component declares a port. The operator uses
  ``DynamoServicePort = 8000`` and ``DynamoSystemPort = 9090``
  (``consts.go:13,20``).

So a manifest plus an operator is runnable, and a manifest alone is not. Every
tier below Kubernetes has to supply what the operator would have — which is what
this module does, by **transcribing** the operator's own defaults rather than
inventing plausible ones. Where the two disagree, the operator is right and this
file is a bug.

## Everything carries its provenance

Each resolved field is a :class:`~dynamo_test.facts.Fact` whose ``source`` says
which rung produced it:

    port 8000 from "operator default consts.go:13 DynamoServicePort"
    port 8080 from "recipes/x/deploy.yaml[Frontend] --http-port"

That matters more here than almost anywhere else in the harness. A test that
fails to connect needs to know whether the port came from the manifest, from an
argument, or from a default this file guessed — those have three different
fixes, and a bare integer tells you nothing about which.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Mapping

from .facts import Fact
from .manifest import Component, Plan
from .roles import PortName, Role

__all__ = ["Defaults", "Render", "Stack", "DYNAMO_DEFAULTS"]


@dataclass(frozen=True)
class Defaults:
    """What the operator would have supplied, transcribed rather than invented.

    Every field cites the operator source it mirrors. When the operator changes,
    this is the file that has to change with it, and the citations are how a
    reader checks whether it has.
    """

    service_port: int = 8000
    """``consts.go:13`` ``DynamoServicePort``, container port name ``http``."""

    system_port: int = 9090
    """``consts.go:20`` ``DynamoSystemPort``, container port name ``system``."""

    frontend_command: tuple[str, ...] = ("python3", "-m", "dynamo.frontend")
    """``component_frontend.go:31-32``: ``Command=["python3"]``,
    ``Args=["-m","dynamo.frontend"]``."""

    nats_server: str = "nats://localhost:4222"
    etcd_endpoints: str = "http://localhost:2379"
    """``graph.go:1570,1577``. The values come from operator configuration, so
    these are only a shape; a site is expected to override them with addresses
    that resolve wherever it puts the discovery plane. A compose site points them
    at service names, which is the entire reason the frontend can leave the
    all-in-one container at all."""

    engine_module: Mapping[str, str] = field(
        default_factory=lambda: {
            "vllm": "dynamo.vllm",
            "sglang": "dynamo.sglang",
            "trtllm": "dynamo.trtllm",
        }
    )

    def command_for(
        self, component: Component, plan_backend: str | None = None
    ) -> Fact[tuple[str, ...]]:
        """The command this component would run, or why it cannot be determined."""
        argv = component.argv
        if not argv.is_parseable:
            return Fact.unknown(
                component.name,
                f"the declared command could not be tokenised: {argv.parse_error}",
            )
        declared = argv.invocation()
        if declared:
            return Fact.known(
                tuple(declared), f"{component.name}", "declared in the manifest"
            )
        if component.role is Role.FRONTEND:
            return Fact.known(
                self.frontend_command,
                "operator default",
                "component_frontend.go:31-32",
            )
        engine = self.engine_for(component, plan_backend)
        if engine.is_known:
            module = self.engine_module.get(engine.require())
            if module:
                return Fact.known(
                    ("python3", "-m", module),
                    "operator default",
                    f"engine {engine.require()!r} from {engine.source}",
                )
        return Fact.absent(
            component.name,
            "no command in the manifest, not a frontend, and nothing declares "
            "which engine this is — not the command, not spec.backendFramework. "
            "On Kubernetes the operator resolves this from the image; off it, a "
            "site must supply the engine",
        )

    def engine_for(
        self, component: Component, plan_backend: str | None = None
    ) -> Fact[str]:
        """Which engine a component runs, in descending order of authority.

        The command line first, because it is what actually executes. Then
        ``spec.backendFramework`` — the CRD's own declared field
        (``dynamocomponentdeployment_types.go:52``), present in 139 of the 178
        recipe plans. Nothing is inferred from the component's *name*: it fixes
        two components in the whole corpus and would guess wrong the first time
        someone names a worker after a model instead of an engine.
        """
        detected = component.backend
        if detected.is_known:
            return detected
        if plan_backend:
            return Fact.known(
                plan_backend, "spec.backendFramework", "declared on the deployment"
            )
        return Fact.absent(
            component.name, "no engine on the command line and no spec.backendFramework"
        )

    def discovery_env(self) -> Mapping[str, str]:
        """The variables the operator injects and no manifest declares."""
        return {
            "NATS_SERVER": self.nats_server,
            "ETCD_ENDPOINTS": self.etcd_endpoints,
        }


DYNAMO_DEFAULTS = Defaults()


@dataclass(frozen=True)
class Render:
    """One component, resolved into something runnable — with provenance."""

    role: Role
    name: str
    command: Fact[tuple[str, ...]]
    image: Fact[str]
    ports: Mapping[PortName, Fact[int]]
    env: Mapping[str, Fact[str]]
    replicas: int = 1

    @property
    def is_runnable(self) -> bool:
        return self.command.is_known and self.image.is_known

    def why_not_runnable(self) -> str | None:
        if self.command.is_known and self.image.is_known:
            return None
        missing = []
        if not self.command.is_known:
            missing.append(
                f"command: {self.command.detail or self.command.status.value}"
            )
        if not self.image.is_known:
            missing.append(f"image: {self.image.detail or self.image.status.value}")
        return f"{self.name} is not runnable — " + "; ".join(missing)

    def to_record(self) -> dict:
        return {
            "role": str(self.role),
            "name": self.name,
            "command": list(self.command.or_else(())),
            "command_from": self.command.source,
            "image": self.image.or_else(None),
            "replicas": self.replicas,
            "ports": {
                k.value: {"port": v.or_else(None), "from": v.source}
                for k, v in sorted(self.ports.items(), key=lambda kv: kv[0].value)
            },
            "env": {k: v.or_else(None) for k, v in sorted(self.env.items())},
        }


class Stack:
    """A :class:`Plan` plus the defaults that make it runnable off Kubernetes.

    The plan says what the deployment *is*. The defaults say what the operator
    would have added. Together they are enough to start the thing somewhere that
    has no operator — which is the whole requirement for the compose and
    all-in-one tiers.
    """

    def __init__(
        self,
        plan: Plan,
        defaults: Defaults = DYNAMO_DEFAULTS,
        overrides: Mapping[str, str] | None = None,
    ) -> None:
        self.plan = plan
        self.defaults = defaults
        self.overrides = dict(overrides or {})
        self.plan_backend = (plan._graph.get("spec") or {}).get("backendFramework")

    @classmethod
    def from_file(cls, path, defaults: Defaults = DYNAMO_DEFAULTS, **kw) -> "Stack":
        return cls(Plan.from_file(path), defaults, **kw)

    # ------------------------------------------------------------- ports

    def resolve_port(self, component: Component, kind: PortName) -> Fact[int]:
        """Where a port comes from, in descending order of authority.

        manifest ``ports:`` -> a flag on the command line -> an environment
        variable -> the operator's default -> ``ABSENT``.

        The ladder exists because almost nothing declares a port. Answering with
        a bare integer would hide which rung produced it, and a connection
        failure then gives no clue whether to fix the manifest, the flag, or this
        table.
        """
        container = component._container or {}
        for port in container.get("ports") or []:
            if not isinstance(port, dict) or "containerPort" not in port:
                continue
            name = str(port.get("name", "")).lower()
            if (kind is PortName.SERVICE and name in ("http", "", "service")) or (
                name == kind.value
            ):
                return Fact.known(
                    int(port["containerPort"]),
                    f"{component.name}.ports[{name or 'unnamed'}]",
                    "declared in the manifest",
                )

        flag = {PortName.SERVICE: "--http-port", PortName.SYSTEM: "--system-port"}.get(
            kind
        )
        if flag:
            declared = component.argv.get(flag)
            if declared.is_known:
                try:
                    return Fact.known(
                        int(declared.require()),
                        f"{component.name} {flag}",
                        "on the command line",
                    )
                except ValueError:
                    return Fact.unknown(
                        f"{component.name} {flag}",
                        f"{flag} is {declared.require()!r}, which is not a port number",
                    )

        env = {
            e.get("name"): e.get("value")
            for e in (container.get("env") or [])
            if isinstance(e, dict)
        }
        if kind is PortName.SERVICE and env.get("DYNAMO_PORT"):
            return Fact.known(
                int(env["DYNAMO_PORT"]),
                f"{component.name} env DYNAMO_PORT",
                "consts.go:39",
            )

        if kind is PortName.SERVICE:
            return Fact.known(
                self.defaults.service_port,
                "operator default",
                "consts.go:13 DynamoServicePort",
            )
        if kind is PortName.SYSTEM:
            return Fact.known(
                self.defaults.system_port,
                "operator default",
                "consts.go:20 DynamoSystemPort",
            )
        return Fact.absent(
            component.name, f"no {kind.value} port declared and no default"
        )

    # ------------------------------------------------------------ render

    def render(self, component: Component) -> Render:
        env: dict[str, Fact[str]] = {}
        container = component._container or {}
        for entry in container.get("env") or []:
            if isinstance(entry, dict) and entry.get("name") and "value" in entry:
                env[entry["name"]] = Fact.known(
                    str(entry["value"]),
                    f"{component.name}.env",
                    "declared in the manifest",
                )
        # The operator's injections. Declared values win: a manifest that names
        # its own NATS is making a deliberate choice.
        for key, value in self.defaults.discovery_env().items():
            env.setdefault(
                key,
                Fact.known(
                    self.overrides.get(key, value),
                    "operator default",
                    "graph.go:1570,1577",
                ),
            )
        for key, value in self.overrides.items():
            env[key] = Fact.known(value, "site override", "settings")

        return Render(
            role=component.role,
            name=component.name,
            command=self.defaults.command_for(component, self.plan_backend),
            image=component.image,
            ports={
                PortName.SERVICE: self.resolve_port(component, PortName.SERVICE),
                PortName.SYSTEM: self.resolve_port(component, PortName.SYSTEM),
            },
            env=env,
            replicas=component.replicas,
        )

    def renders(self) -> tuple[Render, ...]:
        return tuple(self.render(c) for c in self.plan)

    def __iter__(self) -> Iterator[Render]:
        return iter(self.renders())

    def __len__(self) -> int:
        return len(self.plan)

    def unrunnable(self) -> tuple[str, ...]:
        """Components that could not be resolved into something startable."""
        return tuple(r.why_not_runnable() for r in self.renders() if not r.is_runnable)

    def to_record(self) -> dict:
        return {
            "plan": self.plan.name,
            "source": self.plan.source,
            "components": [r.to_record() for r in self.renders()],
        }

    def __repr__(self) -> str:
        return f"Stack({self.plan.name!r}, {len(self.plan)} components)"
