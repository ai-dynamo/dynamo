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


class _FakeEventPublisher:
    """Records every emit() call; stands in for SweeperEventPublisher."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []
        self.closed = False

    def emit(self, event_type: str, data: dict) -> None:
        self.events.append((event_type, data))

    def close(self) -> None:
        self.closed = True


class _FakeCandidateRecord:
    def __init__(self, *, feasible: bool, payload=None, reason=None) -> None:
        self.status = SimpleNamespace(value="feasible" if feasible else "infeasible")
        self._payload = payload or {}
        self.reason = reason

    def as_candidate(self):
        payload = self._payload
        return SimpleNamespace(model_dump=lambda mode="json": payload)


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

    def fake_run_sweep(
        received_config, *, show_progress, on_round=None, on_candidate=None
    ):
        assert received_config is config
        assert show_progress is False
        assert on_round is not None
        # No --dgd-namespace: the event plane is never started, so run_sweep
        # must still be called (with on_candidate=None) but nothing consumes it.
        assert on_candidate is None
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
    assert (output_dir / "index.json").read_text() == """{
  "artifacts": [
    {
      "path": "qwen.yaml"
    }
  ],
  "output": "dgd",
  "renderer": "direct"
}
"""


def test_scalar_ctrl_c_preserves_best_known_dgd(monkeypatch, tmp_path, capsys) -> None:
    output_dir = tmp_path / "output"
    config = _config()
    candidate = _Candidate(1.5)

    monkeypatch.setattr(main_module, "load_sweep_config", lambda _path: config)

    def fake_run_sweep(_config, *, show_progress, on_round=None, on_candidate=None):
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

    def fake_run_sweep(_config, *, show_progress, on_round=None, on_candidate=None):
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

    def fake_run_sweep(
        received_config, *, show_progress, on_round=None, on_candidate=None
    ):
        assert received_config is config
        assert on_round is None
        assert on_candidate is None
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


def test_pareto_keeps_renderable_candidates(monkeypatch, tmp_path, capsys) -> None:
    output_dir = tmp_path / "output"
    config = _config(pareto=True)
    candidates = [_Candidate(1.5), _Candidate(1.0), _Candidate(0.5)]

    monkeypatch.setattr(main_module, "load_sweep_config", lambda _path: config)
    monkeypatch.setattr(
        main_module,
        "run_sweep",
        lambda *_args, **_kwargs: SweepResult(config=config, candidates=candidates),
    )

    def fake_render(candidate, _workload, _options, *, dgd_name, renderer):
        if candidate is candidates[1]:
            raise main_module.CandidateMaterializationError("unsupported strategy")
        return _rendered_dgd(dgd_name, candidate.score)

    monkeypatch.setattr(main_module, "render_dgd", fake_render)

    result = main_module.main(_args(output_dir, "--dgd-name-prefix", "qwen-pareto"))

    assert result == 0
    assert sorted(path.name for path in output_dir.glob("*.yaml")) == [
        "qwen-pareto-000.yaml",
        "qwen-pareto-002.yaml",
    ]
    assert "skipping Pareto candidate qwen-pareto-001" in capsys.readouterr().err


def test_pareto_fails_when_no_candidate_can_be_rendered(
    monkeypatch, tmp_path, capsys
) -> None:
    config = _config(pareto=True)
    candidates = [_Candidate(1.5), _Candidate(1.0)]

    monkeypatch.setattr(main_module, "load_sweep_config", lambda _path: config)
    monkeypatch.setattr(
        main_module,
        "run_sweep",
        lambda *_args, **_kwargs: SweepResult(config=config, candidates=candidates),
    )

    def fail_render(*_args, **_kwargs):
        raise main_module.CandidateMaterializationError("unsupported strategy")

    monkeypatch.setattr(
        main_module,
        "render_dgd",
        fail_render,
    )

    result = main_module.main(
        _args(tmp_path / "output", "--dgd-name-prefix", "qwen-pareto")
    )

    assert result == 2
    assert "no Pareto candidate could be rendered" in capsys.readouterr().err


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


def test_event_publisher_is_never_started_without_dgd_namespace(
    monkeypatch, tmp_path
) -> None:
    config = _config(pareto=True)
    started = []

    monkeypatch.setattr(main_module, "load_sweep_config", lambda _path: config)
    monkeypatch.setattr(
        main_module,
        "_start_event_publisher",
        lambda namespace: started.append(namespace),
    )
    monkeypatch.setattr(
        main_module,
        "run_sweep",
        lambda *_args, **_kwargs: SweepResult(
            config=config, candidates=[_Candidate(1.0)]
        ),
    )
    monkeypatch.setattr(
        main_module,
        "render_dgd",
        lambda candidate, _workload, _options, *, dgd_name, renderer: _rendered_dgd(
            dgd_name, candidate.score
        ),
    )

    result = main_module.main(_args(tmp_path / "output", "--dgd-name-prefix", "qwen"))

    assert result == 0
    assert started == []


def test_start_event_publisher_fails_open_and_warns(monkeypatch, capsys) -> None:
    import sys

    # No dynamo.runtime.DistributedRuntime available (e.g. bindings not
    # built) must never abort the sweep -- only skip progress events.
    monkeypatch.setitem(sys.modules, "dynamo.runtime", SimpleNamespace())

    publisher = main_module._start_event_publisher("my-namespace")

    assert publisher is None
    assert "event plane unavailable" in capsys.readouterr().err


def test_candidate_event_payload_maps_feasible_and_non_feasible_outcomes() -> None:
    feasible = _FakeCandidateRecord(feasible=True, payload={"score": 1.5})
    infeasible = _FakeCandidateRecord(feasible=False, reason="over gpu_budget")

    assert main_module._candidate_event_payload(feasible) == {
        "outcome": "materialized",
        "candidate": {"score": 1.5},
    }
    assert main_module._candidate_event_payload(infeasible) == {
        "outcome": "materialization_failed",
        "error": "over gpu_budget",
    }


def test_combine_round_callbacks_calls_every_active_callback() -> None:
    calls = []

    def first(round_number, candidates):
        calls.append(("first", round_number, candidates))

    def second(round_number, candidates):
        calls.append(("second", round_number, candidates))

    combined = main_module._combine_round_callbacks(first, None, second)
    combined(1, ["candidate"])

    assert calls == [("first", 1, ["candidate"]), ("second", 1, ["candidate"])]
    assert main_module._combine_round_callbacks(None, None) is None
    assert main_module._combine_round_callbacks(first, None) is first


def test_scalar_run_emits_round_search_and_run_completed_events(
    monkeypatch, tmp_path
) -> None:
    output_dir = tmp_path / "output"
    config = _config()
    candidate = _Candidate(1.5)
    fake_publisher = _FakeEventPublisher()

    monkeypatch.setattr(main_module, "load_sweep_config", lambda _path: config)
    monkeypatch.setattr(
        main_module, "_start_event_publisher", lambda namespace: fake_publisher
    )
    monkeypatch.setattr(
        main_module,
        "render_dgd",
        lambda c, _workload, _options, *, dgd_name, renderer: _rendered_dgd(
            dgd_name, c.score
        ),
    )

    def fake_run_sweep(_config, *, show_progress, on_round=None, on_candidate=None):
        on_round(1, [candidate])
        on_candidate(_FakeCandidateRecord(feasible=True, payload={"score": 1.5}))
        on_candidate(_FakeCandidateRecord(feasible=False, reason="over gpu_budget"))
        return SweepResult(config=config, candidates=[candidate])

    monkeypatch.setattr(main_module, "run_sweep", fake_run_sweep)

    result = main_module.main(
        _args(output_dir, "--dgd-name", "qwen", "--dgd-namespace", "my-ns")
    )

    assert result == 0
    assert ("round.completed", {"round_no": 1, "cumulative_candidates": 1}) in (
        fake_publisher.events
    )
    assert (
        "search.resolved",
        {"outcome": "materialized", "candidate": {"score": 1.5}},
    ) in fake_publisher.events
    assert (
        "search.resolved",
        {"outcome": "materialization_failed", "error": "over gpu_budget"},
    ) in fake_publisher.events
    assert fake_publisher.events[-1] == (
        "run.completed",
        {"outcome": "succeeded"},
    )
    assert fake_publisher.closed is True


def test_pareto_run_emits_events_with_no_dgd_publisher_involved(
    monkeypatch, tmp_path
) -> None:
    output_dir = tmp_path / "output"
    config = _config(pareto=True)
    candidates = [_Candidate(1.5)]
    fake_publisher = _FakeEventPublisher()

    monkeypatch.setattr(main_module, "load_sweep_config", lambda _path: config)
    monkeypatch.setattr(
        main_module, "_start_event_publisher", lambda namespace: fake_publisher
    )
    monkeypatch.setattr(
        main_module,
        "render_dgd",
        lambda c, _workload, _options, *, dgd_name, renderer: _rendered_dgd(
            dgd_name, c.score
        ),
    )

    def fake_run_sweep(_config, *, show_progress, on_round=None, on_candidate=None):
        assert on_round is not None
        on_round(1, candidates)
        return SweepResult(config=config, candidates=candidates)

    monkeypatch.setattr(main_module, "run_sweep", fake_run_sweep)

    result = main_module.main(
        _args(output_dir, "--dgd-name-prefix", "qwen", "--dgd-namespace", "my-ns")
    )

    assert result == 0
    assert ("round.completed", {"round_no": 1, "cumulative_candidates": 1}) in (
        fake_publisher.events
    )
    assert fake_publisher.events[-1] == ("run.completed", {"outcome": "succeeded"})


def test_run_completed_reports_failure_on_exception(monkeypatch, tmp_path) -> None:
    config = _config(pareto=True)
    fake_publisher = _FakeEventPublisher()

    monkeypatch.setattr(main_module, "load_sweep_config", lambda _path: config)
    monkeypatch.setattr(
        main_module, "_start_event_publisher", lambda namespace: fake_publisher
    )
    monkeypatch.setattr(
        main_module,
        "run_sweep",
        lambda *_args, **_kwargs: SweepResult(config=config, candidates=[]),
    )

    result = main_module.main(
        _args(
            tmp_path / "output", "--dgd-name-prefix", "qwen", "--dgd-namespace", "my-ns"
        )
    )

    assert result == 2
    outcome_events = [
        event for event in fake_publisher.events if event[0] == "run.completed"
    ]
    assert len(outcome_events) == 1
    assert outcome_events[0][1]["outcome"] == "failed"
    assert "no feasible candidate found" in outcome_events[0][1]["error"]
    assert fake_publisher.closed is True


def test_run_completed_reports_interrupted_on_keyboard_interrupt(
    monkeypatch, tmp_path
) -> None:
    config = _config()
    fake_publisher = _FakeEventPublisher()

    monkeypatch.setattr(main_module, "load_sweep_config", lambda _path: config)
    monkeypatch.setattr(
        main_module, "_start_event_publisher", lambda namespace: fake_publisher
    )

    def fake_run_sweep(_config, *, show_progress, on_round=None, on_candidate=None):
        raise KeyboardInterrupt

    monkeypatch.setattr(main_module, "run_sweep", fake_run_sweep)

    result = main_module.main(
        _args(tmp_path / "output", "--dgd-name", "qwen", "--dgd-namespace", "my-ns")
    )

    assert result == 130
    assert fake_publisher.events[-1] == (
        "run.completed",
        {"outcome": "failed", "error": "interrupted"},
    )
