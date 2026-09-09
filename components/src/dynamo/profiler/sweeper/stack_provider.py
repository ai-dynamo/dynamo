# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Registers Dynamo Replay as a `--stack dynamo` implementation for
`aisimulate recommend`/`predict` (DEP #14282).

Confirmed real: the review thread on PR #13765 establishes "dynamo is the
stack" as the intended shape -- `aisimulate recommend --stack dynamo
--config input.yaml --set dgd.name=qwen` -- and `aisimulate.main`'s real,
shipped source imports `resolve_runner_factory` from a sibling `.stack`
module (seen in the same diff that confirmed the output-adapter ABI),
confirming a stack-resolution extension point genuinely exists.

SPECULATIVE: the exact entry-point group name and the exact shape
`resolve_runner_factory` expects to find registered under it are not
confirmed -- unlike dgd_output_adapter.py, no diff for aisimulate's own
`stack.py` has been reviewed. `create_stack()` is modeled on the one
CONFIRMED, already-shipped registration pattern in this codebase
(dynamo.planner.simulation:create_provider, dynamo.router.simulation:
create_provider under "aisimulate.sweep_config_providers") since it is the
most credible analog available -- not because it is confirmed correct for
this extension point. Expect this module, and its pyproject.toml entry
point, to need rework once the real interface is confirmed the same way
dgd_output_adapter.py's was.

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
    """Speculative factory -- see module docstring. Returns a
    DynamoReplayRunnerFactory instance, the same object #13765's original
    run_sweep() constructed and passed to Sweeper(runner_factory=...)."""
    factory_cls = _load_runner_factory()
    return factory_cls()
