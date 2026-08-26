# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the backend-neutral Sweeper with Dynamo's replay implementation."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SweepResult:
    """Resolved search input and its final ranked candidates."""

    config: Any
    candidates: list[Any]


RoundCallback = Callable[[int, list[Any]], None]


def _load_sweeper_api() -> tuple[type[Any], type[Any]]:
    """Load AI Simulate lazily so unrelated Dynamo CLIs remain importable."""
    try:
        sweeper_api = importlib.import_module("aisimulate.sweeper")
        return sweeper_api.SmartSearchConfig, sweeper_api.Sweeper
    except (AttributeError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "dynamo.profiler.sweeper requires the aisimulate package installed "
            "by the Dynamo planner image"
        ) from exc


def _load_runner_factory() -> type[Any]:
    """Load Dynamo Replay only when this composition is executed."""
    try:
        simulation = importlib.import_module("dynamo.replay.simulation")
        return simulation.DynamoReplayRunnerFactory
    except (AttributeError, ModuleNotFoundError) as exc:
        raise RuntimeError("Dynamo Replay runner is unavailable") from exc


def load_sweep_config(config_path: str | Path) -> Any:
    """Load and validate one native AI Simulate SmartSearchConfig."""
    SmartSearchConfig, _ = _load_sweeper_api()
    return SmartSearchConfig.from_yaml(str(config_path))


def run_sweep(
    config: Any,
    *,
    show_progress: bool = True,
    on_round: RoundCallback | None = None,
) -> SweepResult:
    """Execute one validated AI Simulate config using Dynamo Replay."""
    _, Sweeper = _load_sweeper_api()

    DynamoReplayRunnerFactory = _load_runner_factory()
    sweeper = Sweeper(
        runner_factory=DynamoReplayRunnerFactory(),
        show_progress=show_progress,
    )
    return SweepResult(
        config=config,
        candidates=list(sweeper.run(config, on_round=on_round)),
    )
