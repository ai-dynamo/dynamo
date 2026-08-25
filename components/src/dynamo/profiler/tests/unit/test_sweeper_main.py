# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import pytest
import yaml

from dynamo.profiler.sweeper import __main__ as main_module
from dynamo.profiler.sweeper import runner as runner_module
from dynamo.profiler.sweeper.runner import SweepResult

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.planner,
    pytest.mark.parallel,
]


class _Candidate:
    def __init__(self) -> None:
        self.config = {"backend": "vllm"}
        self.used_gpus = 2
        self.score = 1.5
        self.metrics = {"throughput": 42.0}
        self.objectives = None


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
            return [_Candidate()]

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

    assert len(result.candidates) == 1
    assert result.config == "config"
    assert captured["config_path"] == str(config_path)
    assert isinstance(captured["runner_factory"], FakeRunnerFactory)
    assert captured["show_progress"] is False
    assert captured["config"] == "config"


def test_main_writes_candidate_json_and_dgd_yaml(monkeypatch, tmp_path, capsys) -> None:
    output_dir = tmp_path / "output"
    monkeypatch.setattr(
        main_module,
        "run_sweep",
        lambda *_args, **_kwargs: SweepResult(
            config=SimpleNamespace(workload=SimpleNamespace(isl=4000, osl=1000)),
            candidates=[_Candidate()],
        ),
    )

    def fake_materialize(_candidate, workload, options, *, candidate_index):
        assert workload.isl == 4000
        assert options.backend_version == "0.20.1"
        assert options.dynamo_version == "1.5.0"
        assert candidate_index == 0
        return """apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeployment
metadata:
  name: candidate-000
spec:
  components: []
"""

    monkeypatch.setattr(
        main_module,
        "materialize_candidate_dgd",
        fake_materialize,
    )

    result = main_module.main(
        [
            "--config",
            "sweep.yaml",
            "--backend",
            "vllm",
            "--backend-image",
            "runtime:image",
            "--backend-version",
            "0.20.1",
            "--dynamo-version",
            "1.5.0",
            "--output-dir",
            str(output_dir),
            "--num-gpus-per-node",
            "8",
            "--no-progress",
        ]
    )

    assert result == 0
    assert "wrote 1 candidate DGD(s)" in capsys.readouterr().out
    dgd = yaml.safe_load((output_dir / "candidate-000.dgd.yaml").read_text())
    assert dgd["kind"] == "DynamoGraphDeployment"
    candidate = json.loads((output_dir / "candidate-000.json").read_text())
    assert candidate["config"] == {"backend": "vllm"}
    assert candidate["dgd_file"] == "candidate-000.dgd.yaml"
    index = json.loads((output_dir / "index.json").read_text())
    assert index == {"candidates": [candidate]}
