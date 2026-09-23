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


class _FakeCandidateRecord:
    """Stand-in for aisimulate's CandidateRecord: identity plus the plain
    ``Candidate`` (or any sentinel) its ``as_candidate()`` resolves to."""

    def __init__(self, candidate_id, candidate):
        self.candidate_id = candidate_id
        self._candidate = candidate

    def as_candidate(self):
        return self._candidate


class _FakeSweepResult:
    """Stand-in for aisimulate's own SweepResult: a candidate ledger plus
    the active scalar/Pareto selection, mirroring the real
    ``candidates`` / ``selected_candidate_ids`` contract."""

    def __init__(self, records, selected_ids):
        self.candidates = records
        self.selected_candidate_ids = selected_ids


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

        def run(self, config, *, on_round, on_candidate=None):
            captured["config"] = config
            captured["on_round"] = on_round
            captured["on_candidate"] = on_candidate
            on_round(1, ["candidate"])
            record = _FakeCandidateRecord("candidate-000001", "result")
            if on_candidate is not None:
                on_candidate(record)
            return _FakeSweepResult([record], ["candidate-000001"])

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
    # No on_candidate was passed to run_sweep, so run_sweep must not
    # synthesize one -- aisimulate's Sweeper.run() sees None either way.
    assert captured["on_candidate"] is None


def test_run_sweep_threads_on_candidate_through_to_sweeper(monkeypatch) -> None:
    captured = {}

    class FakeSweeper:
        def __init__(self, *, runner_factory, show_progress):
            pass

        def run(self, config, *, on_round, on_candidate=None):
            record = _FakeCandidateRecord("candidate-000001", "result")
            captured["on_candidate"] = on_candidate
            if on_candidate is not None:
                on_candidate(record)
            return _FakeSweepResult([record], ["candidate-000001"])

    class FakeRunnerFactory:
        pass

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

    seen = []
    callback = seen.append
    result = runner_module.run_sweep(
        "config",
        show_progress=False,
        on_candidate=callback,
    )

    assert captured["on_candidate"] is callback
    assert len(seen) == 1
    assert seen[0].candidate_id == "candidate-000001"
    assert result.candidates == ["result"]


def test_run_sweep_returns_only_the_selected_view_in_selection_order(
    monkeypatch,
) -> None:
    """result.candidates must be aisimulate's scalar top-N or Pareto
    selection resolved through the ledger, not the raw retained ledger
    (which may include non-selected records under ALL retention)."""

    class FakeSweeper:
        def __init__(self, *, runner_factory, show_progress):
            pass

        def run(self, config, *, on_round, on_candidate=None):
            records = [
                _FakeCandidateRecord("candidate-000001", "first"),
                _FakeCandidateRecord("candidate-000002", "second"),
                _FakeCandidateRecord("candidate-000003", "third"),
            ]
            # Selection order deliberately differs from ledger order, and
            # candidate-000002 is retained but not selected.
            return _FakeSweepResult(records, ["candidate-000003", "candidate-000001"])

    class FakeRunnerFactory:
        pass

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

    result = runner_module.run_sweep("config", show_progress=False)

    assert result.candidates == ["third", "first"]
