# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Discover optional replay-runner compositions without importing their packages."""

from __future__ import annotations

import importlib.metadata
from collections.abc import Iterable
from typing import Any

from .runner import EngineReplayRunnerFactory
from .sweeper.replay import RunnerFactory

RUNNER_FACTORY_ENTRY_POINT_GROUP = "aisimulate.runner_factories"


class RunnerFactoryResolutionError(RuntimeError):
    """An optional stack could not provide a usable replay runner."""


class RunnerFactoryNotFoundError(RunnerFactoryResolutionError):
    """No installed distribution registered the requested stack."""


class DuplicateRunnerFactoryError(RunnerFactoryResolutionError):
    """More than one installed distribution registered the requested stack."""


class RunnerFactoryLoadError(RunnerFactoryResolutionError):
    """An installed runner factory could not be imported or constructed."""


class RunnerFactoryABIError(RunnerFactoryResolutionError):
    """A discovered object does not implement the RunnerFactory contract."""


def _installed_entry_points() -> list[importlib.metadata.EntryPoint]:
    return list(
        importlib.metadata.entry_points().select(group=RUNNER_FACTORY_ENTRY_POINT_GROUP)
    )


def _validate(factory: Any, *, stack: str) -> RunnerFactory:
    missing = [
        name
        for name in ("capabilities", "create")
        if not callable(getattr(factory, name, None))
    ]
    if missing:
        raise RunnerFactoryABIError(
            f"runner factory for stack {stack!r} does not implement callable(s): "
            f"{', '.join(missing)}"
        )
    return factory


def resolve_runner_factory(
    stack: str,
    *,
    entry_points: Iterable[importlib.metadata.EntryPoint] | None = None,
) -> RunnerFactory:
    """Resolve the selected stack while keeping Dynamo out of core imports.

    ``engine`` is built into the standalone :mod:`aisimulate` distribution. Other
    stacks are installed plugins registered in :data:`RUNNER_FACTORY_ENTRY_POINT_GROUP`.
    """

    if stack == "engine":
        return EngineReplayRunnerFactory()

    installed = (
        list(entry_points) if entry_points is not None else _installed_entry_points()
    )
    matches = [entry_point for entry_point in installed if entry_point.name == stack]
    if not matches:
        available = sorted({entry_point.name for entry_point in installed})
        choices = ", ".join(["engine", *available])
        install_hint = (
            " Install both standalone wheels with "
            "`uv pip install aisimulate ai-dynamo`."
            if stack == "dynamo"
            else ""
        )
        raise RunnerFactoryNotFoundError(
            f"runner for stack {stack!r} is not installed; available stacks: "
            f"{choices or 'engine'}.{install_hint}"
        )
    if len(matches) > 1:
        providers = ", ".join(
            sorted(
                f"{entry_point.value} "
                f"({getattr(entry_point, 'dist', None) or 'unknown distribution'})"
                for entry_point in matches
            )
        )
        raise DuplicateRunnerFactoryError(
            f"stack {stack!r} has multiple runner factories in entry-point group "
            f"{RUNNER_FACTORY_ENTRY_POINT_GROUP!r}: {providers}"
        )

    entry_point = matches[0]
    try:
        constructor = entry_point.load()
    except Exception as exc:
        raise RunnerFactoryLoadError(
            f"failed to load runner for stack {stack!r} from "
            f"{entry_point.value!r}: {type(exc).__name__}: {exc}"
        ) from exc
    if not callable(constructor):
        raise RunnerFactoryABIError(
            f"runner entry point {entry_point.value!r} for stack {stack!r} "
            "must resolve to a zero-argument callable"
        )
    try:
        factory = constructor()
    except Exception as exc:
        raise RunnerFactoryLoadError(
            f"runner factory {entry_point.value!r} for stack {stack!r} failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    return _validate(factory, stack=stack)
