# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Entry-point discovery for optional Spica simulation adapters."""

from __future__ import annotations

import importlib.metadata
from collections.abc import Iterable, Mapping
from typing import Any

from .adapter import SimulationAdapter, validate_adapter

ADAPTER_ENTRY_POINT_GROUP = "aisimulate.adapters"


class AdapterResolutionError(RuntimeError):
    """Base class for actionable adapter discovery failures."""


class AdapterNotFoundError(AdapterResolutionError):
    """No installed or injected provider matches a configured adapter."""


class DuplicateAdapterError(AdapterResolutionError):
    """More than one installed distribution registered the same adapter name."""


class AdapterLoadError(AdapterResolutionError):
    """An entry-point provider could not be imported."""


class AdapterFactoryError(AdapterResolutionError):
    """An entry point did not expose a valid zero-argument adapter factory."""


class AdapterABIError(AdapterResolutionError):
    """A provider returned an object incompatible with the adapter ABI."""


def _installed_entry_points() -> list[importlib.metadata.EntryPoint]:
    return list(
        importlib.metadata.entry_points().select(group=ADAPTER_ENTRY_POINT_GROUP)
    )


def _validate(adapter: Any, *, requested_name: str) -> SimulationAdapter:
    try:
        return validate_adapter(adapter, requested_name=requested_name)
    except (TypeError, ValueError) as exc:
        raise AdapterABIError(str(exc)) from exc


def resolve_adapters(
    configured_names: Iterable[str],
    *,
    injected: Mapping[str, SimulationAdapter] | None = None,
    entry_points: Iterable[importlib.metadata.EntryPoint] | None = None,
) -> dict[str, SimulationAdapter]:
    """Resolve configured adapter names without importing unselected providers.

    Programmatic injection has precedence for the same name.  Installed entry
    points are inspected as metadata first; only the single provider selected
    for a configured name has ``load()`` called.
    """

    names = list(dict.fromkeys(configured_names))
    injected = injected or {}
    installed = (
        list(entry_points) if entry_points is not None else _installed_entry_points()
    )
    resolved: dict[str, SimulationAdapter] = {}

    for name in names:
        if name in injected:
            resolved[name] = _validate(injected[name], requested_name=name)
            continue

        matches = [entry_point for entry_point in installed if entry_point.name == name]
        if not matches:
            available = sorted(
                set(injected).union(entry_point.name for entry_point in installed)
            )
            choices = ", ".join(available) if available else "<none>"
            raise AdapterNotFoundError(
                f"adapter {name!r} is not installed or injected; "
                f"available adapters: {choices}"
            )
        if len(matches) > 1:
            providers = ", ".join(
                sorted(
                    f"{entry_point.value} "
                    f"({getattr(entry_point, 'dist', None) or 'unknown distribution'})"
                    for entry_point in matches
                )
            )
            raise DuplicateAdapterError(
                f"adapter {name!r} has multiple providers in entry-point group "
                f"{ADAPTER_ENTRY_POINT_GROUP!r}: {providers}"
            )

        entry_point = matches[0]
        try:
            factory = entry_point.load()
        except Exception as exc:
            raise AdapterLoadError(
                f"failed to load adapter {name!r} from {entry_point.value!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if not callable(factory):
            raise AdapterFactoryError(
                f"adapter {name!r} entry point {entry_point.value!r} "
                "must resolve to a callable factory"
            )
        try:
            adapter = factory()
        except Exception as exc:
            raise AdapterFactoryError(
                f"adapter {name!r} factory {entry_point.value!r} failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        resolved[name] = _validate(adapter, requested_name=name)

    return resolved
