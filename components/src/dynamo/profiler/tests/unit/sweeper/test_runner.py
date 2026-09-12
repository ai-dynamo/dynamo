# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from dynamo.profiler.sweeper import runner as runner_module

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.planner,
    pytest.mark.parallel,
]


def test_load_sweep_config_uses_native_smart_search_yaml(monkeypatch, tmp_path) -> None:
    captured = {}

    class FakeConfig:
        @classmethod
        def from_yaml(cls, path):
            captured["config_path"] = path
            return "config"

    monkeypatch.setattr(
        runner_module,
        "_load_sweeper_api",
        lambda: (FakeConfig, object),
    )
    config_path = tmp_path / "sweep.yaml"

    config = runner_module.load_sweep_config(config_path)

    assert config == "config"
    assert captured["config_path"] == str(config_path)


def test_run_sweep_injects_dynamo_runner_and_round_callback(monkeypatch) -> None:
    captured = {}

    class FakeSweeper:
        def __init__(self, *, runner_factory, show_progress):
            captured["runner_factory"] = runner_factory
            captured["show_progress"] = show_progress

        def run(self, config, *, on_round):
            captured["config"] = config
            captured["on_round"] = on_round
            on_round(1, ["candidate"])
            return ["result"]

    class FakeRunnerFactory:
        pass

    rounds = []

    def callback(round_number, candidates):
        rounds.append((round_number, candidates))

    monkeypatch.setattr(
        runner_module,
        "_load_sweeper_api",
        lambda: (object, FakeSweeper),
    )
    monkeypatch.setattr(
        runner_module,
        "_load_runner_factory",
        lambda: FakeRunnerFactory,
    )

    result = runner_module.run_sweep(
        "config",
        show_progress=False,
        on_round=callback,
    )

    assert result.candidates == ["result"]
    assert result.config == "config"
    assert isinstance(captured["runner_factory"], FakeRunnerFactory)
    assert captured["show_progress"] is False
    assert captured["config"] == "config"
    assert captured["on_round"] is callback
    assert rounds == [(1, ["candidate"])]
