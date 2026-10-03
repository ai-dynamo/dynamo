# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Harness policy specs, their canonical identity, and the replay YAML they produce.

A harness policy spec is a JSON/YAML object::

    {"name": "default@defaults",          # label only; not part of the identity
     "router_mode": "kv_router",          # or "round_robin" (no YAML, no knobs)
     "type": "dynamo-default-cost-fn",    # builtin-catalog worker-selection type
     "parameters": {...},                 # the type's YAML parameters, without "seed"
     "router_config": {...},              # KvRouterConfig knobs (sidecar), e.g. router_temperature
     "seeded": true}                      # optional; default from SEEDED_TYPES

The contract form ``{"worker_selection": {...}, "router_config": {...}}`` is accepted too.

Identity (Amendment A1): ``policy_sha`` is the SHA-256 of the canonical YAML of
``{router_mode, router_config, worker_selection}`` with no seed, written to
``CR/runs/policies/<policy_sha>.yaml``. The router policy parser rejects unknown top-level keys,
so replay consumes a second canonical file holding only ``worker_selection`` with
``parameters.seed = k + 1`` for replicate ``k``: ``CR/runs/policies/replay/<sha256>.yaml``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from learned_routing.canon import canonical_yaml, sha256_text, write_once_text

INSTANCE = "candidate"
ROUTER_MODES = ("kv_router", "round_robin")
# Types whose parameters accept ``seed`` (tie-break and sampling RNG). A spec can override with
# ``"seeded": true|false``.
SEEDED_TYPES = frozenset({"dynamo-default-cost-fn", "learned-choice", "sticky-session"})
RESERVED_ROUTER_CONFIG = frozenset({"router_policy_config"})

BUILTIN_SPECS: dict[str, dict] = {
    "round_robin": {"name": "round_robin", "router_mode": "round_robin"},
    "default": {
        "name": "default@defaults",
        "router_mode": "kv_router",
        "type": "dynamo-default-cost-fn",
        "parameters": {},
        "router_config": {},
    },
}


class PolicySpecError(ValueError):
    pass


@dataclass(frozen=True)
class PolicySpec:
    name: str
    router_mode: str
    type: str | None = None
    parameters: dict = field(default_factory=dict)
    router_config: dict = field(default_factory=dict)
    seeded: bool = False

    @property
    def policy_type(self) -> str:
        return self.type if self.router_mode == "kv_router" else "round_robin"

    def identity(self) -> dict:
        if self.router_mode == "round_robin":
            return {"router_mode": "round_robin"}
        return {
            "router_mode": "kv_router",
            "router_config": dict(self.router_config),
            "worker_selection": self.worker_selection(None),
        }

    def canonical_text(self) -> str:
        return canonical_yaml(self.identity())

    @property
    def sha(self) -> str:
        return sha256_text(self.canonical_text())

    def worker_selection(self, seed: int | None) -> dict:
        parameters = dict(self.parameters)
        if seed is not None and self.seeded:
            parameters["seed"] = int(seed)
        return {
            "aggregated": INSTANCE,
            "instances": [
                {"name": INSTANCE, "type": self.type, "parameters": parameters}
            ],
        }

    def replay_yaml_text(self, seed: int | None) -> str | None:
        if self.router_mode == "round_robin":
            return None
        return canonical_yaml({"worker_selection": self.worker_selection(seed)})

    def effective_seed(self, policy_seed: int) -> int | None:
        return policy_seed if self.router_mode == "kv_router" and self.seeded else None

    def write(self, policies_dir: Path) -> Path:
        return write_once_text(
            Path(policies_dir) / f"{self.sha}.yaml", self.canonical_text()
        )

    def write_replay_yaml(self, policies_dir: Path, seed: int | None) -> Path | None:
        text = self.replay_yaml_text(seed)
        if text is None:
            return None
        return write_once_text(
            Path(policies_dir) / "replay" / f"{sha256_text(text)}.yaml", text
        )

    def to_dict(self) -> dict:
        out: dict[str, Any] = {"name": self.name, "router_mode": self.router_mode}
        if self.router_mode == "kv_router":
            out.update(
                type=self.type,
                parameters=self.parameters,
                router_config=self.router_config,
                seeded=self.seeded,
            )
        return out

    def with_values(
        self,
        parameters: dict | None = None,
        router_config: dict | None = None,
        name: str | None = None,
    ) -> "PolicySpec":
        return PolicySpec(
            name=name or self.name,
            router_mode=self.router_mode,
            type=self.type,
            parameters=dict(self.parameters if parameters is None else parameters),
            router_config=dict(
                self.router_config if router_config is None else router_config
            ),
            seeded=self.seeded,
        )


def _kv_router_config_knobs() -> frozenset[str] | None:
    try:
        from dynamo.llm import KvRouterConfig
    except Exception:  # bindings may be mid-rebuild; the worker validates at run time
        return None
    signature = getattr(KvRouterConfig, "__text_signature__", None) or ""
    names = set()
    for part in signature.strip("()").split(","):
        part = part.strip()
        if not part or part in ("*", "/") or part.startswith("$"):
            continue
        names.add(part.split("=", 1)[0].strip())
    return frozenset(names) or None


def spec_from_dict(raw: dict, *, validate_knobs: bool = True) -> PolicySpec:
    if not isinstance(raw, dict):
        raise PolicySpecError(
            f"policy spec must be a mapping, got {type(raw).__name__}"
        )
    raw = dict(raw)
    if "builtin" in raw:
        base = BUILTIN_SPECS.get(raw.pop("builtin"))
        if base is None:
            raise PolicySpecError(
                f"unknown builtin spec; known: {sorted(BUILTIN_SPECS)}"
            )
        raw = {**base, **raw}
    known = {
        "name",
        "router_mode",
        "type",
        "parameters",
        "router_config",
        "seeded",
        "worker_selection",
        "notes",
    }
    unknown = sorted(set(raw) - known)
    if unknown:
        raise PolicySpecError(
            f"unknown policy spec keys {unknown}; known: {sorted(known)}"
        )
    router_mode = raw.get("router_mode", "kv_router")
    if router_mode not in ROUTER_MODES:
        raise PolicySpecError(
            f"router_mode must be one of {ROUTER_MODES}, got {router_mode!r}"
        )
    router_config = dict(raw.get("router_config") or {})
    if router_mode == "round_robin":
        extra = sorted(set(raw) & {"type", "parameters", "worker_selection"})
        if extra or router_config:
            raise PolicySpecError(
                f"round_robin takes no policy YAML or router_config (got {extra or sorted(router_config)})"
            )
        return PolicySpec(
            name=raw.get("name") or "round_robin", router_mode="round_robin"
        )

    policy_type = raw.get("type")
    parameters = raw.get("parameters")
    if "worker_selection" in raw:
        if policy_type is not None or parameters is not None:
            raise PolicySpecError(
                "give either worker_selection or type/parameters, not both"
            )
        selection = raw["worker_selection"] or {}
        chosen = selection.get("aggregated")
        instances = {i.get("name"): i for i in selection.get("instances") or []}
        if chosen is None or chosen not in instances:
            raise PolicySpecError(
                "worker_selection.aggregated must name one of its instances "
                "(a missing aggregated key silently runs the unseeded builtin default)"
            )
        if set(selection) - {"aggregated", "instances"}:
            raise PolicySpecError("only aggregated worker selection is supported")
        policy_type = instances[chosen].get("type")
        parameters = instances[chosen].get("parameters")
    if not policy_type or not isinstance(policy_type, str):
        raise PolicySpecError(
            "kv_router specs need a policy type; the no-YAML builtin default is unseeded and "
            "nondeterministic, use type dynamo-default-cost-fn"
        )
    parameters = dict(parameters or {})
    if "seed" in parameters:
        raise PolicySpecError(
            "parameters.seed is injected per replicate (policy seed k + 1, Amendment A1); "
            "remove it from the spec"
        )
    reserved = sorted(set(router_config) & RESERVED_ROUTER_CONFIG)
    if reserved:
        raise PolicySpecError(
            f"router_config may not set {reserved}; the harness owns them"
        )
    if validate_knobs:
        knobs = _kv_router_config_knobs()
        if knobs is not None:
            bad = sorted(set(router_config) - knobs)
            if bad:
                raise PolicySpecError(
                    f"unknown KvRouterConfig knobs {bad}; known: {sorted(knobs)}"
                )
    seeded = raw.get("seeded")
    if seeded is None:
        seeded = policy_type in SEEDED_TYPES
    return PolicySpec(
        name=raw.get("name") or policy_type,
        router_mode="kv_router",
        type=policy_type,
        parameters=parameters,
        router_config=router_config,
        seeded=bool(seeded),
    )


def _documents(path: Path) -> list[Any]:
    text = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        loaded = yaml.safe_load(text)
        return loaded if isinstance(loaded, list) else [loaded]
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    loaded = json.loads(text)
    return loaded if isinstance(loaded, list) else [loaded]


def load_specs(sources: Iterable[str]) -> list[PolicySpec]:
    """Load specs from files (JSON object/list, JSONL, YAML) or builtin names."""
    specs: list[PolicySpec] = []
    for source in sources:
        if source in BUILTIN_SPECS:
            specs.append(spec_from_dict(BUILTIN_SPECS[source]))
            continue
        path = Path(source)
        if not path.exists():
            raise PolicySpecError(
                f"{source}: no such file and not a builtin spec ({sorted(BUILTIN_SPECS)})"
            )
        specs.extend(spec_from_dict(doc) for doc in _documents(path))
    seen: dict[str, str] = {}
    unique = []
    for spec in specs:
        if spec.sha in seen:
            continue
        seen[spec.sha] = spec.name
        unique.append(spec)
    return unique


def default_reference() -> PolicySpec:
    """default@defaults: the normalization reference for lr-train (contract objective)."""
    return spec_from_dict(BUILTIN_SPECS["default"])
