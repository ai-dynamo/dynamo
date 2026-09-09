# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Dynamo's `--stack dynamo` provider (DEP #14282).

Replaces test_runner.py's coverage of the removed run_sweep/
load_sweep_config -- that logic duplicated what `aisimulate recommend`
owns directly and was removed along with the standalone CLI
(__main__.py). Only _load_runner_factory/create_stack survive, since they
are the real, reusable Dynamo Replay wiring, not CLI-owned logic.
"""

from __future__ import annotations

import sys
import types

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]

try:
    from dynamo.profiler.sweeper.stack_provider import _load_runner_factory, create_stack
except ImportError as exc:
    pytest.skip(f"Skip (missing dependency): {exc}", allow_module_level=True)


def test_load_runner_factory_returns_the_real_dynamo_replay_class(monkeypatch) -> None:
    fake_factory_cls = type("FakeDynamoReplayRunnerFactory", (), {})
    fake_module = types.ModuleType("dynamo.replay.simulation")
    fake_module.DynamoReplayRunnerFactory = fake_factory_cls
    monkeypatch.setitem(sys.modules, "dynamo.replay.simulation", fake_module)

    assert _load_runner_factory() is fake_factory_cls


def test_load_runner_factory_raises_a_clear_error_when_dynamo_replay_is_unavailable(
    monkeypatch,
) -> None:
    monkeypatch.setitem(sys.modules, "dynamo.replay.simulation", None)
    monkeypatch.delitem(sys.modules, "dynamo.replay.simulation", raising=False)

    def _raise_missing(name, *args, **kwargs):
        raise ModuleNotFoundError(name)

    import importlib

    monkeypatch.setattr(importlib, "import_module", _raise_missing)

    with pytest.raises(RuntimeError, match="Dynamo Replay runner is unavailable"):
        _load_runner_factory()


def test_create_stack_instantiates_the_loaded_factory(monkeypatch) -> None:
    instances = []

    class FakeFactory:
        def __init__(self):
            instances.append(self)

    monkeypatch.setattr(
        "dynamo.profiler.sweeper.stack_provider._load_runner_factory",
        lambda: FakeFactory,
    )

    stack = create_stack()

    assert isinstance(stack, FakeFactory)
    assert instances == [stack]
