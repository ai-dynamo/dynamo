# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from dynamo.aisimulate.output.dgd.materialization import CandidateMaterializationError
from dynamo.aisimulate.output.dgdr_run import adapter as adapter_module
from dynamo.aisimulate.output.dgdr_run.adapter import (
    DGDRRunOutputAdapter,
    DGDRRunOutputConfig,
    candidate_id_for,
    create_adapter,
)
from dynamo.aisimulate.output.dgdr_run.snapshot import SNAPSHOT_FILE_NAME, RunPhase

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.planner,
    pytest.mark.parallel,
]


@dataclass
class Record:
    candidate_id: str
    score: float | None
    used_gpus: int | None
    config: dict[str, Any]
    status: str = "feasible"
    objectives: dict[str, float] | None = None


def record(tag: str, score: float, gpus: int = 8, **kwargs: Any) -> Record:
    return Record(
        candidate_id=f"sweeper-{tag}",
        score=score,
        used_gpus=gpus,
        config={"backend": "vllm", "backend_version": "0.1", "tag": tag},
        **kwargs,
    )


def make_config(tmp_path: Path, **overrides: Any) -> DGDRRunOutputConfig:
    values: dict[str, Any] = {
        "name": "sweep",
        "snapshot_dir": str(tmp_path),
        "runtime_image": "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.3",
        "num_gpus_per_node": 8,
        "max_candidates": 3,
        "snapshot_interval_seconds": 0,
    }
    values.update(overrides)
    return DGDRRunOutputConfig(**values)


def load(tmp_path: Path) -> dict[str, Any]:
    return yaml.safe_load((tmp_path / SNAPSHOT_FILE_NAME).read_text())


def manifests_by_id(snapshot: dict[str, Any]) -> dict[str, str]:
    return {c["id"]: c["manifest"] for c in snapshot["candidates"]}


def tags(snapshot: dict[str, Any]) -> list[str]:
    return [candidate["parameters"]["tag"] for candidate in snapshot["candidates"]]


def wait_for(predicate: Any, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition not reached before the timeout")


@pytest.fixture
def renders(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replace the shared materializer with a counting fake."""
    calls: list[str] = []

    def fake_render(candidate: Any, workload: Any, options: Any, **kwargs: Any) -> str:
        tag = candidate.config["tag"]
        calls.append(tag)
        if tag.startswith("bad"):
            raise CandidateMaterializationError(f"cannot render {tag}")
        return (
            "apiVersion: nvidia.com/v1beta1\n"
            "kind: DynamoGraphDeployment\n"
            f"metadata:\n  name: {kwargs['dgd_name']}\n"
            f"spec:\n  marker: {tag}\n"
        )

    monkeypatch.setattr(adapter_module, "render_dgd", fake_render)
    return calls


# -- projection ---------------------------------------------------------------


def test_start_writes_the_initial_snapshot_immediately(
    tmp_path: Path, renders: list[str]
) -> None:
    config = make_config(tmp_path, snapshot_interval_seconds=60)
    with DGDRRunOutputAdapter(config, workload=None):
        snapshot = load(tmp_path)
        assert snapshot["run"] == {"phase": "Running", "terminal": False}
        assert snapshot["candidates"] == []
        assert snapshot["progress"] == {"round": 0, "evaluated": 0}


def test_projection_is_bounded_and_best_first_whatever_is_evaluated(
    tmp_path: Path, renders: list[str]
) -> None:
    adapter = DGDRRunOutputAdapter(make_config(tmp_path), workload=None)
    for index in range(500):
        adapter.on_candidate(record(f"p{index:03d}", score=float(index)))
    adapter.close()

    snapshot = load(tmp_path)
    assert tags(snapshot) == ["p499", "p498", "p497"]
    assert snapshot["progress"]["evaluated"] == 500
    # Only retained points are ever materialized, never all 500.
    assert len(renders) == 3


def test_ties_break_on_fewer_gpus_then_id_deterministically(
    tmp_path: Path, renders: list[str]
) -> None:
    adapter = DGDRRunOutputAdapter(make_config(tmp_path, max_candidates=4), None)
    adapter.on_candidate(record("big", score=5.0, gpus=16))
    adapter.on_candidate(record("small", score=5.0, gpus=8))
    adapter.on_candidate(record("best", score=9.0, gpus=64))
    adapter.close()

    assert tags(load(tmp_path)) == ["best", "small", "big"]


def test_infeasible_candidates_are_counted_but_never_retained(
    tmp_path: Path, renders: list[str]
) -> None:
    adapter = DGDRRunOutputAdapter(make_config(tmp_path), workload=None)
    adapter.on_candidate(record("ok", 1.0))
    adapter.on_candidate(record("nope", 99.0, status="infeasible"))
    adapter.close()

    snapshot = load(tmp_path)
    assert tags(snapshot) == ["ok"]
    assert snapshot["progress"]["evaluated"] == 2


def test_snapshot_preserves_the_evaluated_point_not_just_its_manifest(
    tmp_path: Path, renders: list[str]
) -> None:
    adapter = DGDRRunOutputAdapter(make_config(tmp_path), workload=None)
    adapter.on_candidate(record("a", 87.7, gpus=4, objectives={"goodput": 12.5}))
    adapter.close()

    (candidate,) = load(tmp_path)["candidates"]
    assert candidate["outcome"] == "materialized"
    assert candidate["parameters"] == {
        "backend": "vllm",
        "backend_version": "0.1",
        "tag": "a",
    }
    assert candidate["metrics"] == {
        "score": 87.7,
        "usedGpus": 4,
        "objectives": {"goodput": 12.5},
    }
    assert "marker: a" in candidate["manifest"]


def test_round_progress_is_monotonic(tmp_path: Path, renders: list[str]) -> None:
    adapter = DGDRRunOutputAdapter(make_config(tmp_path), workload=None)
    adapter.on_round(3, [])
    adapter.on_round(2, [])
    adapter.close()
    assert load(tmp_path)["progress"]["round"] == 3


# -- identity, immutability, materialization ----------------------------------


def test_identity_and_manifests_do_not_depend_on_rank_or_observation_order(
    tmp_path: Path, renders: list[str]
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    one = DGDRRunOutputAdapter(make_config(first), workload=None)
    one.on_candidate(record("a", 1.0))
    one.on_candidate(record("b", 2.0))
    one.close()
    # Same two points, opposite scores and opposite observation order.
    two = DGDRRunOutputAdapter(make_config(second), workload=None)
    two.on_candidate(record("b", 1.0))
    two.on_candidate(record("a", 2.0))
    two.close()

    one_snapshot, two_snapshot = load(first), load(second)
    assert tags(one_snapshot) == ["b", "a"]
    assert tags(two_snapshot) == ["a", "b"]
    assert manifests_by_id(one_snapshot) == manifests_by_id(two_snapshot)


def test_identity_depends_only_on_the_resolved_parameters() -> None:
    assert candidate_id_for({"a": 1, "b": 2}) == candidate_id_for({"b": 2, "a": 1})
    assert candidate_id_for({"a": 1}) != candidate_id_for({"a": 2})
    assert candidate_id_for({"a": 1}).startswith("evaluated-point-")


def test_a_new_better_point_reorders_but_does_not_rematerialize_retained_ones(
    tmp_path: Path, renders: list[str]
) -> None:
    adapter = DGDRRunOutputAdapter(make_config(tmp_path), workload=None)
    adapter.start()
    adapter.on_candidate(record("a", 1.0))
    adapter.on_candidate(record("b", 2.0))
    wait_for(lambda: tags(load(tmp_path)) == ["b", "a"])
    before = {c["id"]: c["manifest"] for c in load(tmp_path)["candidates"]}
    assert sorted(renders) == ["a", "b"]

    adapter.on_candidate(record("c", 3.0))
    wait_for(lambda: tags(load(tmp_path)) == ["c", "b", "a"])
    adapter.close()

    after = {c["id"]: c["manifest"] for c in load(tmp_path)["candidates"]}
    assert {key: after[key] for key in before} == before
    assert sorted(renders) == ["a", "b", "c"]  # only the new point was rendered


def test_progress_updates_do_not_rematerialize(
    tmp_path: Path, renders: list[str]
) -> None:
    adapter = DGDRRunOutputAdapter(make_config(tmp_path), workload=None)
    adapter.start()
    adapter.on_candidate(record("a", 1.0))
    wait_for(lambda: tags(load(tmp_path)) == ["a"])
    for round_no in range(1, 20):
        adapter.on_round(round_no, [])
    wait_for(lambda: load(tmp_path)["progress"]["round"] == 19)
    adapter.close()
    assert renders == ["a"]


def test_a_point_is_immutable_once_observed(tmp_path: Path, renders: list[str]) -> None:
    adapter = DGDRRunOutputAdapter(make_config(tmp_path), workload=None)
    adapter.on_candidate(record("a", 1.0))
    adapter.on_candidate(record("a", 50.0))  # same parameters, noisy re-evaluation
    adapter.close()

    (candidate,) = load(tmp_path)["candidates"]
    assert candidate["metrics"]["score"] == 1.0
    assert load(tmp_path)["progress"]["evaluated"] == 2


def test_a_persistent_materialization_failure_is_reported_and_retried_a_bounded_number_of_times(
    tmp_path: Path, renders: list[str]
) -> None:
    adapter = DGDRRunOutputAdapter(make_config(tmp_path), workload=None)
    adapter.start()
    adapter.on_candidate(record("bad1", 5.0))
    wait_for(lambda: len(load(tmp_path)["candidates"]) == 1)
    for round_no in range(1, 10):
        adapter.on_round(round_no, [])
    wait_for(lambda: load(tmp_path)["progress"]["round"] == 9)
    adapter.close()

    (candidate,) = load(tmp_path)["candidates"]
    assert candidate["outcome"] == "materialization_failed"
    assert "cannot render bad1" in candidate["error"]
    assert "manifest" not in candidate
    assert renders == ["bad1"] * adapter_module._MAX_RENDER_ATTEMPTS


# -- rate limiting and coalescing ---------------------------------------------


def test_many_callbacks_in_one_interval_produce_one_periodic_write(
    tmp_path: Path, renders: list[str]
) -> None:
    config = make_config(tmp_path, snapshot_interval_seconds=0.5)
    adapter = DGDRRunOutputAdapter(config, workload=None)
    adapter.start()
    assert adapter.snapshot_writes == 1  # initial

    for index in range(300):
        adapter.on_candidate(record(f"p{index:03d}", score=float(index)))
    time.sleep(0.15)
    assert adapter.snapshot_writes == 1  # still inside the interval

    wait_for(lambda: adapter.snapshot_writes == 2, timeout=5)
    snapshot = load(tmp_path)
    # The one periodic write carries the latest state, not an arbitrary
    # intermediate one.
    assert snapshot["progress"]["evaluated"] == 300
    assert tags(snapshot) == ["p299", "p298", "p297"]

    time.sleep(0.2)
    assert adapter.snapshot_writes == 2
    adapter.close()


def test_terminal_state_bypasses_the_rate_limit_and_is_always_flushed(
    tmp_path: Path, renders: list[str]
) -> None:
    config = make_config(tmp_path, snapshot_interval_seconds=3600)
    adapter = DGDRRunOutputAdapter(config, workload=None)
    adapter.start()
    for index in range(100):
        adapter.on_candidate(record(f"p{index:03d}", score=float(index)))

    started = time.monotonic()
    adapter.close(phase=RunPhase.FAILED, message="boom", error="OOM")
    assert time.monotonic() - started < 5

    snapshot = load(tmp_path)
    assert snapshot["run"] == {
        "phase": "Failed",
        "terminal": True,
        "message": "boom",
        "error": "OOM",
    }
    assert tags(snapshot) == ["p099", "p098", "p097"]
    assert snapshot["progress"]["evaluated"] == 100
    assert adapter.snapshot_writes == 2  # initial + terminal, nothing in between


def test_the_terminal_snapshot_contains_every_update_accepted_before_close(
    tmp_path: Path, renders: list[str]
) -> None:
    config = make_config(tmp_path, snapshot_interval_seconds=3600)
    with DGDRRunOutputAdapter(config, workload=None) as adapter:
        for index in range(50):
            adapter.on_candidate(record(f"w{index}", score=1.0 + index))
        adapter.on_candidate(record("winner", score=1000.0))  # right before close

    snapshot = load(tmp_path)
    assert snapshot["run"]["terminal"] is True
    assert tags(snapshot)[0] == "winner"


def test_callbacks_never_wait_for_a_slow_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = threading.Event()

    def slow_render(candidate: Any, *args: Any, **kwargs: Any) -> str:
        release.wait(5)
        return "kind: DynamoGraphDeployment\n"

    monkeypatch.setattr(adapter_module, "render_dgd", slow_render)
    adapter = DGDRRunOutputAdapter(make_config(tmp_path), workload=None)
    adapter.start()
    adapter.on_candidate(record("a", 1.0))
    time.sleep(0.1)  # the writer is now stuck rendering

    started = time.monotonic()
    for index in range(1000):
        adapter.on_candidate(record(f"q{index}", score=float(index)))
    assert time.monotonic() - started < 2
    release.set()
    adapter.close()
    # Nothing was dropped while the writer was stuck.
    assert load(tmp_path)["progress"]["evaluated"] == 1001


def test_updates_after_close_are_ignored_and_close_is_idempotent(
    tmp_path: Path, renders: list[str]
) -> None:
    adapter = DGDRRunOutputAdapter(make_config(tmp_path), workload=None)
    adapter.on_candidate(record("a", 1.0))
    adapter.close(phase=RunPhase.SUCCEEDED)
    written = (tmp_path / SNAPSHOT_FILE_NAME).read_text()
    writes = adapter.snapshot_writes

    adapter.on_candidate(record("late", 99.0))
    adapter.on_round(9, [])
    adapter.close(phase=RunPhase.FAILED, error="ignored")

    assert (tmp_path / SNAPSHOT_FILE_NAME).read_text() == written
    assert adapter.snapshot_writes == writes


def test_close_works_when_start_was_never_called(
    tmp_path: Path, renders: list[str]
) -> None:
    adapter = DGDRRunOutputAdapter(make_config(tmp_path), workload=None)
    adapter.close(phase=RunPhase.FAILED, error="setup failed")
    assert load(tmp_path)["run"]["phase"] == "Failed"


def test_close_rejects_a_non_terminal_phase(tmp_path: Path) -> None:
    adapter = DGDRRunOutputAdapter(make_config(tmp_path), workload=None)
    with pytest.raises(ValueError, match="terminal"):
        adapter.close(phase=RunPhase.RUNNING)


# -- failure handling ---------------------------------------------------------


def test_context_manager_records_success_and_failure(
    tmp_path: Path, renders: list[str]
) -> None:
    ok = tmp_path / "ok"
    with DGDRRunOutputAdapter(make_config(ok), workload=None):
        pass
    assert load(ok)["run"]["phase"] == "Succeeded"

    failed = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="search crashed"):
        with DGDRRunOutputAdapter(make_config(failed), workload=None):
            raise RuntimeError("search crashed")
    snapshot = load(failed)
    assert snapshot["run"]["phase"] == "Failed"
    assert snapshot["run"]["error"] == "RuntimeError"


def test_a_terminal_write_failure_is_raised_not_swallowed(
    tmp_path: Path, renders: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = DGDRRunOutputAdapter(make_config(tmp_path), workload=None)
    adapter.start()

    def broken(*_: Any, **__: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(adapter_module, "write_snapshot", broken)
    with pytest.raises(OSError, match="disk full"):
        adapter.close()


def test_a_failed_periodic_write_is_retried_with_the_latest_state(
    tmp_path: Path, renders: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = DGDRRunOutputAdapter(make_config(tmp_path), workload=None)
    adapter.start()
    real = adapter_module.write_snapshot
    attempts = {"count": 0}

    def flaky(directory: Path, snapshot: Any) -> Path:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise OSError("transient")
        return real(directory, snapshot)

    monkeypatch.setattr(adapter_module, "write_snapshot", flaky)
    adapter.on_candidate(record("a", 1.0))
    wait_for(lambda: tags(load(tmp_path)) == ["a"])
    adapter.close()
    assert attempts["count"] >= 2


def test_a_candidate_that_cannot_be_read_does_not_break_the_search(
    tmp_path: Path, renders: list[str]
) -> None:
    adapter = DGDRRunOutputAdapter(make_config(tmp_path), workload=None)
    adapter.on_candidate(SimpleNamespace(status="feasible"))  # type: ignore[arg-type]
    adapter.on_candidate(record("ok", 1.0))
    adapter.close()
    assert tags(load(tmp_path)) == ["ok"]


# -- configuration and plugin -------------------------------------------------


def test_pareto_objectives_are_rejected_explicitly(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError, match="scalar"):
        DGDRRunOutputAdapter(make_config(tmp_path), workload=None, is_pareto=True)


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_candidates": 0},
        {"snapshot_interval_seconds": -1},
        {"name": "Not_DNS"},
        {"num_gpus_per_node": 0},
        {"unknown_field": 1},
    ],
)
def test_invalid_configuration_is_rejected(
    tmp_path: Path, overrides: dict[str, Any]
) -> None:
    with pytest.raises(ValueError):
        make_config(tmp_path, **overrides)


def test_there_is_no_hardcoded_candidate_ceiling(
    tmp_path: Path, renders: list[str]
) -> None:
    adapter = DGDRRunOutputAdapter(make_config(tmp_path, max_candidates=200), None)
    for index in range(150):
        adapter.on_candidate(record(f"p{index:03d}", score=float(index)))
    adapter.close()
    assert len(load(tmp_path)["candidates"]) == 150


def test_a_snapshot_directory_is_required() -> None:
    config = DGDRRunOutputConfig(
        name="sweep",
        runtime_image="nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.3",
        num_gpus_per_node=8,
    )
    with pytest.raises(ValueError, match="snapshot directory"):
        DGDRRunOutputAdapter(config, workload=None)


def test_the_plugin_is_discoverable_and_builds_a_live_adapter(tmp_path: Path) -> None:
    plugin = create_adapter()
    assert plugin.name == "dgdr_run"
    live = plugin.live(
        {
            "name": "sweep",
            "snapshot_dir": str(tmp_path),
            "runtime_image": "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.3",
            "num_gpus_per_node": 8,
        },
        workload=None,
    )
    assert isinstance(live, DGDRRunOutputAdapter)


def test_plugin_subscribe_validates_config_before_search() -> None:
    plugin = adapter_module.DGDRRunOutputPlugin()
    good = {
        "name": "run",
        "runtime_image": "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.3",
        "num_gpus_per_node": 8,
    }

    assert plugin.subscribe(good) is None
    with pytest.raises(ValueError, match="num_gpus_per_node"):
        plugin.subscribe({**good, "num_gpus_per_node": 0})
    with pytest.raises(ValueError, match="canonical"):
        plugin.subscribe(
            {**good, "runtime_image": "nvcr.io/nvidia/ai-dynamo/vllm-runtime:latest"}
        )


def test_plugin_write_publishes_the_final_selection_as_a_terminal_snapshot(
    tmp_path: Path, renders: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeSearchConfig:
        @staticmethod
        def model_validate(_: Any) -> Any:
            return SimpleNamespace(workload="workload")

    monkeypatch.setattr(adapter_module, "SmartSearchConfig", FakeSearchConfig)
    result = SimpleNamespace(
        views=SimpleNamespace(pareto_front=[]),
        provenance=SimpleNamespace(config={}),
        selected_candidates=[
            record("a", 9.0),
            record("b", 5.0),
        ],
    )
    written = create_adapter().write(
        {
            "name": "sweep",
            "runtime_image": "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.3",
            "num_gpus_per_node": 8,
        },
        result=result,  # type: ignore[arg-type]
        output_dir=tmp_path,
    )

    assert written == [Path(SNAPSHOT_FILE_NAME)]
    snapshot = load(tmp_path)
    assert snapshot["run"]["terminal"] is True
    assert tags(snapshot) == ["a", "b"]


def test_plugin_write_refuses_pareto_and_empty_results(tmp_path: Path) -> None:
    config = {
        "name": "sweep",
        "runtime_image": "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.3",
        "num_gpus_per_node": 8,
    }
    pareto = SimpleNamespace(views=SimpleNamespace(pareto_front=[object()]))
    with pytest.raises(NotImplementedError):
        create_adapter().write(config, result=pareto, output_dir=tmp_path)  # type: ignore[arg-type]
    empty = SimpleNamespace(
        views=SimpleNamespace(pareto_front=[]), selected_candidates=[]
    )
    with pytest.raises(CandidateMaterializationError):
        create_adapter().write(config, result=empty, output_dir=tmp_path)  # type: ignore[arg-type]


PLUGIN_CONFIG = {
    "name": "sweep",
    "runtime_image": "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.3",
    "num_gpus_per_node": 8,
    "snapshot_interval_seconds": 0,
}


def test_plugin_subscribe_with_a_context_publishes_live(
    tmp_path: Path, renders: list[str]
) -> None:
    config = {**PLUGIN_CONFIG, "snapshot_dir": str(tmp_path)}
    callbacks = create_adapter().subscribe(
        config, context=SimpleNamespace(workload="workload")
    )

    assert callbacks is not None
    assert callbacks.on_candidate is not None
    assert callbacks.on_round is not None
    # The initial snapshot exists before the first candidate is evaluated.
    assert load(tmp_path)["run"] == {"phase": "Running", "terminal": False}

    callbacks.on_candidate(record("a", 9.0))
    wait_for(lambda: tags(load(tmp_path)) == ["a"])


def test_plugin_subscribe_without_context_or_snapshot_dir_only_validates(
    tmp_path: Path,
) -> None:
    plugin = create_adapter()
    with_dir = {**PLUGIN_CONFIG, "snapshot_dir": str(tmp_path)}
    assert plugin.subscribe(with_dir) is None
    assert plugin.subscribe(with_dir, context=SimpleNamespace(workload=None)) is None
    assert (
        plugin.subscribe(PLUGIN_CONFIG, context=SimpleNamespace(workload="workload"))
        is None
    )
    assert not (tmp_path / SNAPSHOT_FILE_NAME).exists()


def _final_result(evaluated: int = 42) -> Any:
    return SimpleNamespace(
        views=SimpleNamespace(pareto_front=[]),
        provenance=SimpleNamespace(config={}),
        selected_candidates=[record("a", 9.0)],
        counts=SimpleNamespace(evaluated=evaluated),
    )


@pytest.fixture
def fake_search_config(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeSearchConfig:
        @staticmethod
        def model_validate(_: Any) -> Any:
            return SimpleNamespace(workload="workload")

    monkeypatch.setattr(adapter_module, "SmartSearchConfig", FakeSearchConfig)


def test_plugin_write_records_progress_from_the_result_counts(
    tmp_path: Path, renders: list[str], fake_search_config: None
) -> None:
    create_adapter().write(PLUGIN_CONFIG, result=_final_result(42), output_dir=tmp_path)
    assert load(tmp_path)["progress"]["evaluated"] == 42


def test_plugin_write_reports_no_artifact_for_a_snapshot_outside_output_dir(
    tmp_path: Path, renders: list[str], fake_search_config: None
) -> None:
    shared = tmp_path / "shared"
    shared.mkdir()
    out = tmp_path / "out"
    out.mkdir()

    written = create_adapter().write(
        {**PLUGIN_CONFIG, "snapshot_dir": str(shared)},
        result=_final_result(),
        output_dir=out,
    )

    assert written == []
    assert load(shared)["run"]["terminal"] is True
    assert not (out / SNAPSHOT_FILE_NAME).exists()


def test_a_transient_materialization_failure_is_retried_and_recovers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempts: list[int] = []

    def flaky_render(candidate: Any, workload: Any, options: Any, **kwargs: Any) -> str:
        attempts.append(1)
        if len(attempts) == 1:
            raise CandidateMaterializationError("cold start")
        return "apiVersion: nvidia.com/v1beta1\nkind: DynamoGraphDeployment\n"

    monkeypatch.setattr(adapter_module, "render_dgd", flaky_render)
    adapter = DGDRRunOutputAdapter(make_config(tmp_path), workload=None)
    adapter.start()
    adapter.on_candidate(record("a", 5.0))
    wait_for(lambda: len(load(tmp_path)["candidates"]) == 1)
    assert load(tmp_path)["candidates"][0]["outcome"] == "materialization_failed"

    adapter.on_round(1, [])  # any later write retries the failed render
    wait_for(lambda: load(tmp_path)["candidates"][0]["outcome"] == "materialized")
    adapter.close()

    (candidate,) = load(tmp_path)["candidates"]
    assert candidate["outcome"] == "materialized"
    assert "error" not in candidate
    assert len(attempts) == 2


def test_a_materialized_candidate_is_never_rendered_again(
    tmp_path: Path, renders: list[str]
) -> None:
    adapter = DGDRRunOutputAdapter(make_config(tmp_path), workload=None)
    adapter.start()
    adapter.on_candidate(record("a", 5.0))
    wait_for(lambda: len(load(tmp_path)["candidates"]) == 1)
    for round_no in range(1, 6):
        adapter.on_round(round_no, [])
    wait_for(lambda: load(tmp_path)["progress"]["round"] == 5)
    adapter.close()
    assert renders == ["a"]


def test_plugin_write_keeps_the_round_recorded_by_the_live_snapshot(
    tmp_path: Path, renders: list[str], fake_search_config: None
) -> None:
    live = create_adapter().subscribe(
        {**PLUGIN_CONFIG, "snapshot_dir": str(tmp_path)},
        context=SimpleNamespace(workload="workload"),
    )
    assert live is not None and live.on_round is not None
    live.on_round(3, [])
    wait_for(lambda: load(tmp_path)["progress"]["round"] == 3)

    create_adapter().write(
        {**PLUGIN_CONFIG, "snapshot_dir": str(tmp_path)},
        result=_final_result(4),
        output_dir=tmp_path,
    )

    snapshot = load(tmp_path)
    assert snapshot["run"]["terminal"] is True
    assert snapshot["progress"] == {"round": 3, "evaluated": 4}


def test_plugin_write_without_a_live_snapshot_starts_at_round_zero(
    tmp_path: Path, renders: list[str], fake_search_config: None
) -> None:
    create_adapter().write(PLUGIN_CONFIG, result=_final_result(2), output_dir=tmp_path)
    assert load(tmp_path)["progress"] == {"round": 0, "evaluated": 2}
