# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reading a DynamoGraphDeployment into a :class:`Plan`.

This is the layer that turns a manifest on disk into roles, command lines and
engines — the three things every later tier asks about. It is deliberately
separate from anything that talks to a cluster: a plan can be read, inspected,
edited and printed on a laptop with no credentials, which is what makes
``dynamo-test plan`` possible in a bare CI checkout.

## What it has to cope with

Measured over ``recipes/`` and ``examples/``:

* **Two schemas, both live.** 177 documents use v1beta1 ``spec.components`` (a
  list), 75 use v1alpha1 ``spec.services`` (a mapping). Neither is legacy.
* **Multi-document files.** A manifest often carries a ConfigMap or a Secret
  alongside the deployment, so ``safe_load`` on its own reads the wrong document
  or none at all.
* **Four places a container hides**: ``extraPodSpec.mainContainer``,
  ``container``, ``podTemplate.spec.containers[]``, and the v1alpha1
  ``services[name].extraPodSpec.mainContainer``.
* **32 distinct component names for about nine roles**, including
  ``TrtllmWorker`` and ``TRTLLMWorker`` in the same corpus. Name-matching that is
  not case-insensitive silently resolves one and not the other, which is exactly
  how a log reader ends up returning nothing for a service that is running fine.

## What it does not preserve

Editing a plan and re-emitting it produces semantically equivalent YAML, not
byte-identical YAML: the comments and formatting of the *manifest* are lost,
because the document is re-serialised. The **shell command string inside a
container is preserved exactly**, comments and all, because :class:`ArgV`
splices rather than rebuilds. That is the distinction that matters — a
reformatted manifest still deploys the same thing, whereas a rebuilt command
line does not.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from .argv import ArgV
from .dialect import Dialect, detect, for_backend
from .facts import Fact
from .roles import PortName, Role, RoleBinding, RoleTable

__all__ = [
    "Schema",
    "Component",
    "Plan",
    "role_of",
    "ManifestError",
    "NoGraphDeployment",
]

KIND = "DynamoGraphDeployment"


class ManifestError(ValueError):
    """The manifest could not be read as a deployment plan."""


class NoGraphDeployment(ManifestError):
    """No ``DynamoGraphDeployment`` document in the file."""


class Schema(str, Enum):
    V1ALPHA1 = "v1alpha1"  # spec.services, a mapping
    V1BETA1 = "v1beta1"  # spec.components, a list


# Role inference, ordered. The first pattern that matches wins, which is why
# prefill/decode/encode come before the generic worker rule: `VllmPrefillWorker`
# contains both "prefill" and "worker" and is a prefill.
#
# Matching is case-insensitive because the corpus contains both `TrtllmWorker`
# and `TRTLLMWorker`.
_ROLE_PATTERNS: tuple[tuple[re.Pattern, Role], ...] = (
    (re.compile(r"frontend", re.I), Role.FRONTEND),
    (re.compile(r"prefill", re.I), Role.PREFILL),
    (re.compile(r"decode", re.I), Role.DECODE),
    (re.compile(r"encode", re.I), Role.ENCODE),
    (re.compile(r"planner", re.I), Role.PLANNER),
    (re.compile(r"router|^epp$", re.I), Role.ROUTER),
    (re.compile(r"gateway", re.I), Role.GATEWAY),
    (re.compile(r"kvbm", re.I), Role.KVBM),
    (re.compile(r"etcd", re.I), Role.ETCD),
    (re.compile(r"nats", re.I), Role.NATS),
    # `agg` is an aggregated worker: it prefills and decodes in one process.
    (re.compile(r"worker|^agg$", re.I), Role.WORKER),
)


def role_of(name: str) -> Fact[Role]:
    """The role a component name denotes.

    Returns a :class:`Fact` rather than a default, because guessing here is how
    a selector silently ends up pointed at the wrong service.
    """
    for pattern, role in _ROLE_PATTERNS:
        if pattern.search(name):
            return Fact.known(
                role, f"component name {name!r}", f"matched /{pattern.pattern}/"
            )
    return Fact.absent(
        f"component name {name!r}",
        "matches none of " + ", ".join(p.pattern for p, _ in _ROLE_PATTERNS),
    )


def _containers(component: Mapping[str, Any]) -> list[dict]:
    """Every container spec in a component, in precedence order."""
    found: list[dict] = []
    main = ((component.get("extraPodSpec") or {}).get("mainContainer")) or None
    if isinstance(main, dict):
        found.append(main)
    direct = component.get("container")
    if isinstance(direct, dict):
        found.append(direct)
    pod_spec = (component.get("podTemplate") or {}).get("spec") or {}
    for container in pod_spec.get("containers") or []:
        if isinstance(container, dict):
            found.append(container)
    return found


def _main_container(component: Mapping[str, Any]) -> dict | None:
    """The container that carries the component's command line.

    When a pod declares several, the one with args wins; a sidecar with no
    command is not what a test means by "the worker".
    """
    found = _containers(component)
    for container in found:
        if container.get("args") or container.get("command"):
            return container
    return found[0] if found else None


@dataclass(frozen=True)
class Component:
    """One service in a deployment, with its role and command line resolved."""

    name: str
    role: Role
    replicas: int
    argv: ArgV
    backend: Fact[str]
    _container: dict | None
    _spec: Mapping[str, Any]

    @property
    def dialect(self) -> Dialect | None:
        return for_backend(self.backend.require()) if self.backend.is_known else None

    @property
    def image(self) -> Fact[str]:
        if self._container is None:
            return Fact.unknown(self.name, "component declares no container")
        image = self._container.get("image")
        if image:
            return Fact.known(str(image), f"{self.name}.image")
        return Fact.absent(f"{self.name}.image", "no image field on the main container")

    @property
    def gpus(self) -> Fact[int]:
        """GPU limit, from whichever schema this component uses."""
        limits = (self._spec.get("resources") or {}).get("limits") or {}
        if "gpu" in limits:  # v1alpha1
            return Fact.known(int(limits["gpu"]), f"{self.name}.resources.limits.gpu")
        if self._container is not None:
            climits = (self._container.get("resources") or {}).get("limits") or {}
            if "nvidia.com/gpu" in climits:
                return Fact.known(
                    int(climits["nvidia.com/gpu"]),
                    f"{self.name} container resources.limits['nvidia.com/gpu']",
                )
        return Fact.absent(f"{self.name}", "no GPU limit declared in either schema")

    def read(self, semantic: str) -> Fact[str]:
        """An engine setting, through this component's dialect."""
        dialect = self.dialect
        if dialect is None:
            return Fact.unknown(
                self.name,
                f"no engine dialect: {self.backend.detail or 'backend not identified'}",
            )
        return dialect.read(self.argv, semantic)

    def binding(self) -> RoleBinding:
        """This component's :class:`RoleBinding`.

        ``log_key`` is the component name lowercased — **one** string, used for
        both the bundle directory and the log-scrape key, so the two cannot
        drift into an alias table that resolves some services and not others.
        """
        dialect = self.dialect
        processes = dict(dialect.processes) if dialect else {}
        return RoleBinding(
            role=self.role,
            service=self.name,
            log_key=self.name.lower(),
            metric_labels={"dynamo_component": self.name},
            processes=processes,
            ports=_ports(self._container),
        )


def _ports(container: dict | None) -> dict:
    if not container:
        return {}
    out: dict = {}
    for port in container.get("ports") or []:
        if not isinstance(port, dict) or "containerPort" not in port:
            continue
        name = str(port.get("name", "")).lower()
        if name in ("metrics", "system", "grpc"):
            out[PortName(name)] = int(port["containerPort"])
        elif PortName.SERVICE not in out:
            out[PortName.SERVICE] = int(port["containerPort"])
    return out


class Plan:
    """A deployment, read from a manifest and editable in memory."""

    def __init__(
        self,
        documents: Sequence[Any],
        source: str = "<manifest>",
        select: str | None = None,
    ) -> None:
        self.source = source
        self._documents = list(documents)
        graphs = [
            d for d in self._documents if isinstance(d, dict) and d.get("kind") == KIND
        ]
        if not graphs:
            kinds = sorted(
                {d.get("kind", "?") for d in self._documents if isinstance(d, dict)}
            )
            raise NoGraphDeployment(
                f"{source} declares no {KIND}; it contains: "
                f"{', '.join(kinds) if kinds else '<nothing>'}"
            )
        if len(graphs) > 1 and select is None:
            names = [(g.get("metadata") or {}).get("name", "?") for g in graphs]
            raise ManifestError(
                f"{source} declares {len(graphs)} {KIND} documents "
                f"({', '.join(names)}); a plan describes one deployment. Use "
                "Plan.all_from_yaml()/all_from_file() to get one plan per "
                "deployment, or pass select=<name>."
            )
        if select is not None:
            chosen = [
                g for g in graphs if (g.get("metadata") or {}).get("name") == select
            ]
            if not chosen:
                names = [(g.get("metadata") or {}).get("name", "?") for g in graphs]
                raise ManifestError(
                    f"{source} has no {KIND} named {select!r}; it declares: "
                    f"{', '.join(names)}"
                )
            graphs = chosen
        self._graph = graphs[0]

    # ------------------------------------------------------------- loading

    @staticmethod
    def _documents(text: str, source: str) -> list:
        import yaml

        try:
            return list(yaml.safe_load_all(text))
        except Exception as exc:
            raise ManifestError(f"{source} is not valid YAML: {exc}") from exc

    @classmethod
    def from_yaml(
        cls, text: str, source: str = "<string>", select: str | None = None
    ) -> "Plan":
        return cls(cls._documents(text, source), source=source, select=select)

    @classmethod
    def from_file(cls, path: str | Path, select: str | None = None) -> "Plan":
        path = Path(path)
        return cls.from_yaml(path.read_text(), source=str(path), select=select)

    @classmethod
    def all_from_yaml(cls, text: str, source: str = "<string>") -> tuple["Plan", ...]:
        """One plan per deployment, for files that declare several.

        Four ``global_planner`` examples describe a control plane plus two to
        four model deployments in one file. Treating those as an error would
        make the planner untestable; treating them as one plan would silently
        merge deployments that are meant to be separate.
        """
        documents = cls._documents(text, source)
        names = [
            (d.get("metadata") or {}).get("name")
            for d in documents
            if isinstance(d, dict) and d.get("kind") == KIND
        ]
        if not names:
            return (cls(documents, source=source),)  # raises NoGraphDeployment
        return tuple(
            cls(documents, source=f"{source}#{name}", select=name) for name in names
        )

    @classmethod
    def all_from_file(cls, path: str | Path) -> tuple["Plan", ...]:
        path = Path(path)
        return cls.all_from_yaml(path.read_text(), source=str(path))

    # -------------------------------------------------------------- shape

    @property
    def name(self) -> str:
        return (self._graph.get("metadata") or {}).get("name", "<unnamed>")

    @property
    def schema(self) -> Schema:
        spec = self._graph.get("spec") or {}
        if isinstance(spec.get("components"), list):
            return Schema.V1BETA1
        if isinstance(spec.get("services"), dict):
            return Schema.V1ALPHA1
        raise ManifestError(
            f"{self.source}: spec has neither a components list nor a services mapping"
        )

    def _raw_components(self) -> list[tuple[str, dict]]:
        spec = self._graph.get("spec") or {}
        if self.schema is Schema.V1BETA1:
            return [
                (str(c.get("name", "<unnamed>")), c)
                for c in spec["components"]
                if isinstance(c, dict)
            ]
        return [(str(n), c or {}) for n, c in spec["services"].items()]

    @property
    def components(self) -> tuple[Component, ...]:
        out = []
        for name, raw in self._raw_components():
            container = _main_container(raw)
            argv = ArgV.from_container(container or {}, source=f"{self.source}[{name}]")
            role = role_of(name)
            out.append(
                Component(
                    name=name,
                    role=role.or_else(Role.WORKER),
                    replicas=int(raw.get("replicas", 1) or 1),
                    argv=argv,
                    backend=detect(argv),
                    _container=container,
                    _spec=raw,
                )
            )
        return tuple(out)

    def unresolved_roles(self) -> tuple[str, ...]:
        """Component names whose role could not be inferred.

        Surfaced rather than silently defaulted, so a new component type shows
        up as a question instead of quietly becoming a generic worker.
        """
        return tuple(
            name for name, _ in self._raw_components() if role_of(name).is_absent
        )

    def __getitem__(self, name_or_role: str | Role) -> Component:
        for component in self.components:
            if component.name == name_or_role or component.role == name_or_role:
                return component
        raise KeyError(
            f"{self.source} has no component {name_or_role!r}; it declares: "
            + ", ".join(c.name for c in self.components)
        )

    def __iter__(self) -> Iterator[Component]:
        return iter(self.components)

    def __len__(self) -> int:
        return len(self.components)

    def roles(self) -> RoleTable:
        """The role table for this deployment.

        Built once, here. Components sharing a role — two prefill workers, say —
        would make the table ambiguous, so that is an error rather than a
        last-one-wins.
        """
        bindings: dict[Role, RoleBinding] = {}
        clashes: dict[Role, list[str]] = {}
        for component in self.components:
            if component.role in bindings:
                clashes.setdefault(component.role, [bindings[component.role].service])
                clashes[component.role].append(component.name)
                continue
            bindings[component.role] = component.binding()
        if clashes:
            detail = "; ".join(
                f"{role}: {', '.join(names)}" for role, names in clashes.items()
            )
            raise ManifestError(
                f"{self.source}: several components share a role ({detail}). A role "
                "must name one service, or a selector cannot say which it means."
            )
        return RoleTable(bindings)

    # ------------------------------------------------------------ editing

    def set(self, name_or_role: str | Role, **settings: Any) -> "Plan":
        """Change engine settings on one component, in place.

        Writes through the component's dialect and :class:`ArgV`, so the flag
        spelling is right for the engine and the surrounding shell command is
        untouched.
        """
        component = self[name_or_role]
        dialect = component.dialect
        if dialect is None:
            raise ManifestError(
                f"{component.name}: cannot set {', '.join(settings)} — "
                f"{component.backend.detail or 'no engine identified'}"
            )
        if component._container is None:
            raise ManifestError(f"{component.name} declares no container to configure")
        dialect.apply(component.argv, **settings).apply_to(component._container)
        return self

    def scale(self, name_or_role: str | Role, replicas: int) -> "Plan":
        if replicas < 0:
            raise ValueError(f"replicas must be non-negative, got {replicas}")
        self[name_or_role]._spec["replicas"] = replicas  # type: ignore[index]
        return self

    # ------------------------------------------------------------- output

    def to_yaml(self) -> str:
        """Re-emit every document in the original file.

        Formatting and comments *of the manifest* are not preserved; the shell
        command inside each container is, byte for byte.
        """
        import yaml

        return yaml.safe_dump_all(self._documents, sort_keys=False, width=10**6)

    def to_record(self) -> dict:
        """A JSON-safe summary for the run record."""
        return {
            "name": self.name,
            "schema": self.schema.value,
            "source": self.source,
            "components": [
                {
                    "name": c.name,
                    "role": str(c.role),
                    "replicas": c.replicas,
                    "backend": c.backend.or_else(None),
                    "form": c.argv.form.value,
                    "image": c.image.or_else(None),
                    "gpus": c.gpus.or_else(None),
                    "model": c.read("model").or_else(None),
                }
                for c in self.components
            ],
        }

    def __repr__(self) -> str:
        return (
            f"Plan({self.name!r}, {self.schema.value}, "
            f"{len(self.components)} components)"
        )
