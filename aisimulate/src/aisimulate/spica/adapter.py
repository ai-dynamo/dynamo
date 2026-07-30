# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamo-neutral contracts for extending a Spica search.

Adapters own feature-specific search-space validation and candidate
materialization.  The Spica core owns parameter namespacing and replay
orchestration; an adapter only returns data described by the contracts in this
module.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias, runtime_checkable

if TYPE_CHECKING:
    from .replay import BackendDeploymentSpec


API_VERSION = 1

JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]


@dataclass(frozen=True)
class RuntimeHookSpec:
    """One serializable runtime extension requested from a replay runner."""

    provider: str
    kind: str
    api_version: int
    config: dict[str, JSONValue] = field(default_factory=dict)


@dataclass(frozen=True)
class SweepContext:
    """Immutable core context supplied while an adapter prepares its search."""

    core_search_space: Mapping[str, JSONValue]
    workload: Mapping[str, JSONValue]
    goal: Mapping[str, JSONValue]
    show_progress: bool = True


@dataclass(frozen=True)
class CandidateContext:
    """Resolved core candidate available during adapter materialization."""

    sample: Mapping[str, JSONValue]
    backend_deployment: BackendDeploymentSpec
    concurrency: int | None = None


@dataclass(frozen=True)
class SearchSpaceFragment:
    """Adapter-owned search dimensions, grouped by core deployment branch.

    Keys in ``choices_by_branch`` and ``float_ranges_by_branch`` are deployment
    branch names such as ``"agg"`` and ``"disagg"``.  Parameter names inside a
    branch are local to the adapter; the core adds the adapter namespace before
    merging them into a sampler study.
    """

    choices_by_branch: dict[str, dict[str, list[JSONValue]]] = field(
        default_factory=dict
    )
    float_ranges_by_branch: dict[str, dict[str, tuple[float, float]]] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class AdapterSearchPlan:
    """Prepared adapter search state shared by all candidates in one sweep."""

    fragment: SearchSpaceFragment = field(default_factory=SearchSpaceFragment)
    state: JSONValue = None
    diagnostics: dict[str, JSONValue] = field(default_factory=dict)
    potential_runtime_hooks: tuple[RuntimeHookSpec, ...] = ()


@dataclass(frozen=True)
class AdapterReplaySpec:
    """Concrete adapter configuration and hooks for one replay candidate."""

    config: dict[str, JSONValue] = field(default_factory=dict)
    runtime_hooks: tuple[RuntimeHookSpec, ...] = ()


@runtime_checkable
class SimulationAdapter(Protocol):
    """Versioned search-time adapter implemented by optional feature packages."""

    name: str
    api_version: int

    def generate_search_space(
        self,
        search_spec: Mapping[str, JSONValue],
        context: SweepContext,
    ) -> AdapterSearchPlan:
        """Validate and prepare an adapter-owned search space."""

    def materialize_replay(
        self,
        plan: AdapterSearchPlan,
        selection: Mapping[str, JSONValue],
        context: CandidateContext,
    ) -> AdapterReplaySpec:
        """Build the adapter portion of a concrete replay specification."""


def validate_adapter(adapter: Any, *, requested_name: str) -> SimulationAdapter:
    """Validate the structural adapter ABI and return the typed instance."""

    actual_name = getattr(adapter, "name", None)
    if actual_name != requested_name:
        raise TypeError(
            f"adapter {requested_name!r} returned name {actual_name!r}; "
            "the provider name must match the configured adapter name"
        )
    api_version = getattr(adapter, "api_version", None)
    if type(api_version) is not int or api_version != API_VERSION:
        raise ValueError(
            f"adapter {requested_name!r} uses API version {api_version!r}; "
            f"aisimulate requires version {API_VERSION}"
        )
    missing = [
        method
        for method in ("generate_search_space", "materialize_replay")
        if not callable(getattr(adapter, method, None))
    ]
    if missing:
        raise TypeError(
            f"adapter {requested_name!r} does not implement required callable(s): "
            f"{', '.join(missing)}"
        )
    return adapter
