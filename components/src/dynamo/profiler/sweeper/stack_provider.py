# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Registers Dynamo Replay as a `--stack dynamo` implementation for
`aisimulate recommend`/`predict` (DEP #14282). Dynamo Replay is imported
lazily so unrelated Dynamo CLIs remain importable without aisimulate or
Dynamo Replay installed.
"""

from __future__ import annotations

import importlib
from typing import Any


def _load_runner_factory() -> type[Any]:
    """Load Dynamo Replay only when this stack is actually selected, so
    unrelated Dynamo CLIs remain importable without aisimulate or Dynamo
    Replay installed."""
    try:
        simulation = importlib.import_module("dynamo.replay.simulation")
        return simulation.DynamoReplayRunnerFactory
    except (AttributeError, ModuleNotFoundError) as exc:
        raise RuntimeError("Dynamo Replay runner is unavailable") from exc


def create_stack() -> Any:
    """Registered under aisimulate.runner_factories (confirmed group name,
    see module docstring). Returns an instantiated DynamoReplayRunnerFactory
    -- the same object #13765's original run_sweep() constructed and passed
    to Sweeper(runner_factory=...). Whether aisimulate actually wants an
    instance vs. the class itself vs. something else is the one remaining
    unconfirmed detail; see module docstring."""
    factory_cls = _load_runner_factory()
    return factory_cls()
