# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Registers Dynamo Replay as a `--stack dynamo` implementation for
`aisimulate recommend`/`predict` (DEP #14282).

Entry-point group name CONFIRMED via review feedback on this PR:
"aisimulate recommend --stack dynamo resolves stack names from the
aisimulate.runner_factories entry-point group." Not the earlier guess
("aisimulate.stacks") -- pyproject.toml registers this factory under
"aisimulate.runner_factories" accordingly.

Still not independently confirmed: the exact object create_stack() is
expected to return. It currently returns an instantiated
DynamoReplayRunnerFactory (matching what #13765's original run_sweep()
constructed and passed to Sweeper(runner_factory=...)), which is a
reasonable reading of "runner_factories" as a group name, but no real
aisimulate diff analogous to output_adapter.py's has been reviewed to
confirm whether aisimulate expects the factory class itself, an
instantiated factory, or something else the entry point should point at
instead. Worth re-verifying the same way dgd_output_adapter.py's ABI was:
against a real diff, not just the group name in isolation.

This module owns no new simulation logic: it wraps the same
DynamoReplayRunnerFactory that `#13765`'s original runner.py already used
to run Dynamo Replay from the (now removed) standalone CLI. Nothing about
how Dynamo Replay executes a search changes; only how it becomes reachable
does.
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
