# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import yaml

from dynamo.profiler.sweeper import __main__ as main_module
from dynamo.profiler.sweeper.runner import SweepResult

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.planner,
    pytest.mark.parallel,
]


class _Candidate:
    def __init__(self, score: float, *, used_gpus: int = 2) -> None:
        self.config = {
            "backend": "vllm",
            "backend_version": "0.20.1",
            "candidate_score": score,
        }
        self.used_gpus = used_gpus
        self.score = score
        self.metrics = {"throughput": score}
        self.objectives = None


def _config(*, pareto: bool = False, backends: tuple[str, ...] = ("vllm",)):
    return SimpleNamespace(
        goal=SimpleNamespace(is_pareto=pareto),
        search_space=SimpleNamespace(backend=backends),
        workload=SimpleNamespace(isl=4000, osl=1000),
    )


def _args(output_dir, *extra: str) -> list[str]:
    return [
        "--config",
        "sweep.yaml",
        "--dgd-runtime-image",
        "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.5.0",
        "--dgd-num-gpus-per-node",
        "8",
        "--output-dir",
        str(output_dir),
        "--no-progress",
        *extra,
    ]


def _rendered_dgd(name: str, score: float) -> str:
    return f"""apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeployment
metadata:
  name: {name}
  annotations:
    test-score: "{score}"
spec:
  components: []
"""


def test_scalar_publishes_only_new_best_candidates(
    monkeypatch, tmp_path, capsys
) -> None:
    output_dir = tmp_path / "output"
    config = _config()
    first = _Candidate(1.5)
    worse = _Candidate(1.0)
    better = _Candidate(2.0, used_gpus=4)
    rendered_scores = []

    monkeypatch.setattr(main_module, "load_sweep_config", lambda _path: config)

    def fake_run_sweep(received_config, *, show_progress, on_round=None):
        assert received_config is config
        assert show_progress is False
        assert on_round is not None
        on_round(1, [first])
        on_round(2, [first, worse])
        on_round(3, [first, worse, better])
        return SweepResult(config=config, candidates=[better, first, worse])

    def fake_render(candidate, workload, options, *, dgd_name, renderer):
        assert workload is config.workload
        assert options.runtime_image.endswith(":1.5.0")
        assert options.dynamo_runtime_version == "1.5.0"
        assert options.runtime_version_override is None
        assert dgd_name == "qwen"
        assert renderer == "direct"
        rendered_scores.append(candidate.score)
        return _rendered_dgd(dgd_name, candidate.score)

    monkeypatch.setattr(main_module, "run_sweep", fake_run_sweep)
    monkeypatch.setattr(main_module, "render_dgd", fake_render)

    result = main_module.main(
        _args(output_dir, "--dgd-name", "qwen", "--renderer", "direct")
    )

    assert result == 0
    assert rendered_scores == [1.5, 2.0]
    dgd = yaml.safe_load((output_dir / "qwen.yaml").read_text())
    assert dgd["metadata"]["annotations"]["test-score"] == "2.0"
    assert list(output_dir.glob(".*.tmp")) == []
    stdout = capsys.readouterr().out
    assert "new best after round 1" in stdout
    assert "new best after round 2" not in stdout
    assert "new best after round 3" in stdout
    assert (
        (output_dir / "index.json").read_text()
        == """{
  "artifacts": [
    {
      "path": "qwen.yaml"
    }
  ],
  "output": "dgd",
  "renderer": "direct"
}
"""
    )


def test_scalar_ctrl_c_preserves_best_known_dgd(monkeypatch, tmp_path, capsys) -> None:
    output_dir = tmp_path / "output"
    config = _config()
    candidate = _Candidate(1.5)

    monkeypatch.setattr(main_module, "load_sweep_config", lambda _path: config)

    def fake_run_sweep(_config, *, show_progress, on_round=None):
        assert on_round is not None
        on_round(1, [candidate])
        raise KeyboardInterrupt

    monkeypatch.setattr(main_module, "run_sweep", fake_run_sweep)
    monkeypatch.setattr(
        main_module,
        "render_dgd",
        lambda *_args, dgd_name, **_kwargs: _rendered_dgd(dgd_name, 1.5),
    )

    result = main_module.main(_args(output_dir, "--dgd-name", "qwen"))

    assert result == 130
    assert yaml.safe_load((output_dir / "qwen.yaml").read_text())["kind"] == (
        "DynamoGraphDeployment"
    )
    assert "best known DGD remains" in capsys.readouterr().err


def test_unrenderable_new_best_retains_previous_dgd(
    monkeypatch, tmp_path, capsys
) -> None:
    output_dir = tmp_path / "output"
    config = _config()
    first = _Candidate(1.5)
    unrenderable = _Candidate(2.0)

    monkeypatch.setattr(main_module, "load_sweep_config", lambda _path: config)

    def fake_run_sweep(_config, *, show_progress, on_round=None):
        assert on_round is not None
        on_round(1, [first])
        on_round(2, [first, unrenderable])
        return SweepResult(config=config, candidates=[unrenderable, first])

    def fake_render(candidate, _workload, _options, *, dgd_name, renderer):
        if candidate is unrenderable:
            raise main_module.CandidateMaterializationError("unsupported strategy")
        return _rendered_dgd(dgd_name, candidate.score)

    monkeypatch.setattr(main_module, "run_sweep", fake_run_sweep)
    monkeypatch.setattr(main_module, "render_dgd", fake_render)

    result = main_module.main(_args(output_dir, "--dgd-name", "qwen"))

    assert result == 2
    dgd = yaml.safe_load((output_dir / "qwen.yaml").read_text())
    assert dgd["metadata"]["annotations"]["test-score"] == "1.5"
    stderr = capsys.readouterr().err
    assert "retaining" in stderr
    assert "best candidate could not be rendered" in stderr


def test_pareto_writes_prefixed_kustomize_sources(monkeypatch, tmp_path) -> None:
    output_dir = tmp_path / "output"
    config = _config(pareto=True)
    candidates = [_Candidate(1.5), _Candidate(1.0)]

    monkeypatch.setattr(main_module, "load_sweep_config", lambda _path: config)

    def fake_run_sweep(received_config, *, show_progress, on_round=None):
        assert received_config is config
        assert on_round is None
        return SweepResult(config=config, candidates=candidates)

    monkeypatch.setattr(main_module, "run_sweep", fake_run_sweep)
    monkeypatch.setattr(
        main_module,
        "render_dgd",
        lambda candidate, _workload, _options, *, dgd_name, renderer: _rendered_dgd(
            dgd_name, candidate.score
        ),
    )

    result = main_module.main(
        _args(
            output_dir,
            "--dgd-name-prefix",
            "qwen-pareto",
            "--output",
            "kustomize",
        )
    )

    assert result == 0
    for index in range(2):
        source = output_dir / f"qwen-pareto-{index:03d}"
        assert yaml.safe_load((source / "deploy.yaml").read_text())["kind"] == (
            "DynamoGraphDeployment"
        )
        assert yaml.safe_load((source / "kustomization.yaml").read_text()) == {
            "apiVersion": "kustomize.config.k8s.io/v1beta1",
            "kind": "Kustomization",
            "resources": ["deploy.yaml"],
        }


@pytest.mark.parametrize(
    ("pareto", "invalid_flag", "valid_flag"),
    [
        (False, "--dgd-name-prefix", "--dgd-name"),
        (True, "--dgd-name", "--dgd-name-prefix"),
    ],
)
def test_dgd_name_form_must_match_goal(
    monkeypatch, tmp_path, pareto, invalid_flag, valid_flag, capsys
) -> None:
    monkeypatch.setattr(
        main_module, "load_sweep_config", lambda _path: _config(pareto=pareto)
    )

    with pytest.raises(SystemExit, match="2"):
        main_module.main(_args(tmp_path, invalid_flag, "qwen"))

    assert valid_flag in capsys.readouterr().err
