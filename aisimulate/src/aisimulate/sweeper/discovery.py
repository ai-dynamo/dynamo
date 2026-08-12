# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Entry-point discovery for optional Sweeper configuration providers."""

from __future__ import annotations

import importlib.metadata
from collections.abc import Iterable, Mapping
from typing import Any

from .provider import SweepConfigProvider, validate_provider

SWEEP_CONFIG_PROVIDER_ENTRY_POINT_GROUP = "aisimulate.sweep_config_providers"


class ProviderResolutionError(RuntimeError):
    """Base class for actionable provider discovery failures."""


class ProviderNotFoundError(ProviderResolutionError):
    """No installed or injected provider matches a configured adapter."""


class DuplicateProviderError(ProviderResolutionError):
    """More than one installed distribution registered the same adapter name."""


class ProviderLoadError(ProviderResolutionError):
    """An entry-point provider could not be imported."""


class ProviderFactoryError(ProviderResolutionError):
    """An entry point did not expose a valid zero-argument provider factory."""


class ProviderABIError(ProviderResolutionError):
    """A provider returned an object incompatible with the provider ABI."""


def _installed_entry_points() -> list[importlib.metadata.EntryPoint]:
    return list(
        importlib.metadata.entry_points().select(
            group=SWEEP_CONFIG_PROVIDER_ENTRY_POINT_GROUP
        )
    )


def _validate(provider: Any, *, requested_name: str) -> SweepConfigProvider:
    try:
        return validate_provider(provider, requested_name=requested_name)
    except (TypeError, ValueError) as exc:
        raise ProviderABIError(str(exc)) from exc


def resolve_providers(
    configured_names: Iterable[str],
    *,
    injected: Mapping[str, SweepConfigProvider] | None = None,
    entry_points: Iterable[importlib.metadata.EntryPoint] | None = None,
) -> dict[str, SweepConfigProvider]:
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
    resolved: dict[str, SweepConfigProvider] = {}

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
            raise ProviderNotFoundError(
                f"provider for adapter {name!r} is not installed or injected; "
                f"available providers: {choices}"
            )
        if len(matches) > 1:
            providers = ", ".join(
                sorted(
                    f"{entry_point.value} "
                    f"({getattr(entry_point, 'dist', None) or 'unknown distribution'})"
                    for entry_point in matches
                )
            )
            raise DuplicateProviderError(
                f"adapter {name!r} has multiple providers in entry-point group "
                f"{SWEEP_CONFIG_PROVIDER_ENTRY_POINT_GROUP!r}: {providers}"
            )

        entry_point = matches[0]
        try:
            factory = entry_point.load()
        except Exception as exc:
            raise ProviderLoadError(
                f"failed to load provider {name!r} from {entry_point.value!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if not callable(factory):
            raise ProviderFactoryError(
                f"provider {name!r} entry point {entry_point.value!r} "
                "must resolve to a callable factory"
            )
        try:
            provider = factory()
        except Exception as exc:
            raise ProviderFactoryError(
                f"provider {name!r} factory {entry_point.value!r} failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        resolved[name] = _validate(provider, requested_name=name)

    return resolved
