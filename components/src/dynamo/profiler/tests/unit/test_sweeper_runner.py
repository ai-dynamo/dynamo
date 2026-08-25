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


def test_run_sweep_injects_dynamo_runner(monkeypatch, tmp_path) -> None:
    captured = {}

    class FakeConfig:
        @classmethod
        def from_yaml(cls, path):
            captured["config_path"] = path
            return "config"

    class FakeSweeper:
        def __init__(self, *, runner_factory, show_progress):
            captured["runner_factory"] = runner_factory
            captured["show_progress"] = show_progress

        def run(self, config):
            captured["config"] = config
            return ["result"]

    class FakeRunnerFactory:
        pass

    monkeypatch.setattr(
        runner_module,
        "_load_sweeper_api",
        lambda: (FakeConfig, FakeSweeper),
    )
    monkeypatch.setattr(
        runner_module,
        "_load_runner_factory",
        lambda: FakeRunnerFactory,
    )
    config_path = tmp_path / "sweep.yaml"

    result = runner_module.run_sweep(config_path, show_progress=False)

    assert result.candidates == ["result"]
    assert result.config == "config"
    assert captured["config_path"] == str(config_path)
    assert isinstance(captured["runner_factory"], FakeRunnerFactory)
    assert captured["show_progress"] is False
    assert captured["config"] == "config"
