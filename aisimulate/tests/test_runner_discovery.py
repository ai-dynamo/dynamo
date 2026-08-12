# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional stack discovery contracts."""

from __future__ import annotations

from importlib.metadata import EntryPoint

import pytest

from aisimulate.runner import EngineReplayRunnerFactory
from aisimulate.runner_discovery import (
    RUNNER_FACTORY_ENTRY_POINT_GROUP,
    DuplicateRunnerFactoryError,
    RunnerFactoryABIError,
    RunnerFactoryNotFoundError,
    resolve_runner_factory,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.planner,
    pytest.mark.gpu_0,
]


class _Factory:
    def capabilities(self):
        return object()

    def create(self, worker_id):
        return worker_id


def _entry_point(name: str, value: str) -> EntryPoint:
    return EntryPoint(
        name=name,
        value=value,
        group=RUNNER_FACTORY_ENTRY_POINT_GROUP,
    )


def test_engine_runner_is_built_in_without_entry_point_discovery():
    factory = resolve_runner_factory("engine", entry_points=[])

    assert isinstance(factory, EngineReplayRunnerFactory)


def test_dynamo_runner_loads_only_the_selected_entry_point(monkeypatch):
    dynamo = _entry_point("dynamo", "example:DynamoFactory")
    unused = _entry_point("other", "example:OtherFactory")
    loaded = []

    def load(entry_point):
        loaded.append(entry_point.name)
        return _Factory

    monkeypatch.setattr(EntryPoint, "load", load)

    factory = resolve_runner_factory("dynamo", entry_points=[unused, dynamo])

    assert isinstance(factory, _Factory)
    assert loaded == ["dynamo"]


def test_missing_dynamo_runner_has_standalone_wheel_install_hint():
    with pytest.raises(RunnerFactoryNotFoundError, match="aisimulate ai-dynamo"):
        resolve_runner_factory("dynamo", entry_points=[])


def test_duplicate_runner_registration_is_rejected():
    with pytest.raises(DuplicateRunnerFactoryError, match="multiple runner factories"):
        resolve_runner_factory(
            "dynamo",
            entry_points=[
                _entry_point("dynamo", "first:create"),
                _entry_point("dynamo", "second:create"),
            ],
        )


def test_invalid_runner_factory_abi_is_rejected(monkeypatch):
    entry_point = _entry_point("dynamo", "example:Invalid")
    monkeypatch.setattr(EntryPoint, "load", lambda self: dict)

    with pytest.raises(RunnerFactoryABIError, match="capabilities, create"):
        resolve_runner_factory("dynamo", entry_points=[entry_point])
