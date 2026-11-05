# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from dynamo.aisimulate.output.dgd.renderers import CandidateMaterializationError
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
    pytest.mark.timeout(60),
]


@dataclass
class Record:
    candidate_id: str
    score: float | None
    used_gpus: int | None
    config: dict[str, Any]
    status: str = "feasible"
    objectives: dict[str, float] | None = None
    metrics: dict[str, float | None] = field(default_factory=dict)


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
    poll = threading.Event()
    while time.monotonic() < deadline:
        if predicate():
            return
        poll.wait(min(0.01, deadline - time.monotonic()))
    raise AssertionError("condition not reached before the timeout")


class ControlledWait:
    """Let a test release one writer delay without waiting for wall-clock time."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._advanced = False
        self._stopped = False
        self.waiting = threading.Event()
        self.timeouts: list[float] = []

    def is_set(self) -> bool:
        with self._condition:
            return self._stopped

    def wait(self, timeout: float | None = None) -> bool:
        with self._condition:
            if timeout is None:
                self._condition.wait_for(lambda: self._stopped)
            else:
                self.timeouts.append(timeout)
                self.waiting.set()
                self._condition.wait_for(lambda: self._advanced or self._stopped)
                self._advanced = False
            return self._stopped

    def advance(self) -> None:
        with self._condition:
            self._advanced = True
            self._condition.notify_all()

    def set(self) -> None:
        with self._condition:
            self._stopped = True
            self._condition.notify_all()


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


@pytest.mark.parametrize(
    ("records", "max_candidates", "expected"),
    [
        pytest.param(
            tuple(record(f"p{index:03d}", float(index)) for index in range(500)),
            3,
            ["p499", "p498", "p497"],
            id="bounded-best-first",
        ),
        pytest.param(
            (
                record("big", 5.0, gpus=16),
                record("small", 5.0, gpus=8),
                record("best", 9.0, gpus=64),
            ),
            4,
            ["best", "small", "big"],
            id="ties-use-fewer-gpus",
        ),
        pytest.param(
            (record("ok", 1.0), record("nope", 99.0, status="infeasible")),
            3,
            ["ok"],
            id="infeasible-counted-not-retained",
        ),
        pytest.param(
            (SimpleNamespace(status="feasible"), record("ok", 1.0)),
            3,
            ["ok"],
            id="malformed-callback-isolated",
        ),
    ],
)
def test_projection_is_bounded_ranked_and_feasible(
    tmp_path: Path,
    renders: list[str],
    records: tuple[Any, ...],
    max_candidates: int,
    expected: list[str],
) -> None:
    adapter = DGDRRunOutputAdapter(
        make_config(tmp_path, max_candidates=max_candidates), workload=None
    )
    for candidate in records:
        adapter.on_candidate(candidate)
    adapter.close()

    snapshot = load(tmp_path)
    assert tags(snapshot) == expected
    assert snapshot["progress"]["evaluated"] == len(records)
    assert len(renders) == len(expected)


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


@pytest.mark.parametrize(
    ("runtime_image", "derive", "backends", "expected"),
    [
        pytest.param(
            "nvcr.io/nvidia/ai-dynamo/dynamo-planner:1.2.3",
            True,
            ("vllm", "sglang", "trtllm"),
            {
                "vllm": "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.3",
                "sglang": "nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.2.3",
                "trtllm": "nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:1.2.3",
            },
            id="derive-per-backend",
        ),
        pytest.param(
            "example.com/my-runtime:1.2.3",
            False,
            ("sglang",),
            {"sglang": "example.com/my-runtime:1.2.3"},
            id="explicit-image",
        ),
    ],
)
def test_runtime_image_selection_reaches_the_renderer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runtime_image: str,
    derive: bool,
    backends: tuple[str, ...],
    expected: dict[str, str],
) -> None:
    images: dict[str, str] = {}

    def capture_image(
        candidate: Any, workload: Any, options: Any, **kwargs: Any
    ) -> str:
        del workload, kwargs
        images[candidate.config["backend"]] = options.runtime_image
        return "apiVersion: nvidia.com/v1beta1\nkind: DynamoGraphDeployment\n"

    monkeypatch.setattr(adapter_module, "render_dgd", capture_image)
    adapter = DGDRRunOutputAdapter(
        make_config(
            tmp_path,
            runtime_image=runtime_image,
            derive_runtime_images=derive,
        ),
        workload=None,
    )
    for index, backend in enumerate(backends):
        candidate = record(backend, float(index))
        candidate.config["backend"] = backend
        adapter.on_candidate(candidate)
    adapter.close()

    assert images == expected


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


@pytest.mark.parametrize("run_name", ["a" * 63, "team.model"])
def test_aic_name_prefix_is_stable_dns_safe_and_candidate_specific(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, run_name: str
) -> None:
    names: list[str] = []

    def capture_name(candidate: Any, workload: Any, options: Any, **kwargs: Any) -> str:
        del candidate, workload, options
        assert kwargs["renderer"] == "aic"
        names.append(kwargs["dgd_name"])
        return "apiVersion: nvidia.com/v1beta1\nkind: DynamoGraphDeployment\n"

    monkeypatch.setattr(adapter_module, "render_dgd", capture_name)
    adapter = DGDRRunOutputAdapter(
        make_config(tmp_path, name=run_name, max_candidates=2), workload=None
    )
    adapter.on_candidate(record("first", 2.0))
    adapter.on_candidate(record("second", 1.0))
    adapter.close()

    assert len(names) == 2
    assert len(set(names)) == 2
    assert all(len(name) <= 28 for name in names)
    assert all(name[0].isalnum() and name[-1].isalnum() for name in names)
    assert all("." not in name for name in names)
    assert names[0] == adapter_module._dgd_name(
        run_name, candidate_id_for(record("first", 2.0).config)
    )


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

    for round_no in range(1, 20):
        adapter.on_round(round_no, [])
    adapter.on_round(18, [])
    wait_for(lambda: load(tmp_path)["progress"]["round"] == 19)
    assert sorted(renders) == ["a", "b"]

    adapter.on_candidate(record("c", 3.0))
    wait_for(lambda: tags(load(tmp_path)) == ["c", "b", "a"])
    adapter.close()

    after = {c["id"]: c["manifest"] for c in load(tmp_path)["candidates"]}
    assert {key: after[key] for key in before} == before
    assert sorted(renders) == ["a", "b", "c"]


def test_a_point_is_immutable_once_observed(tmp_path: Path, renders: list[str]) -> None:
    adapter = DGDRRunOutputAdapter(make_config(tmp_path), workload=None)
    adapter.on_candidate(record("a", 1.0))
    adapter.on_candidate(record("a", 50.0))  # same parameters, noisy re-evaluation
    adapter.close()

    (candidate,) = load(tmp_path)["candidates"]
    assert candidate["metrics"]["score"] == 1.0
    assert load(tmp_path)["progress"]["evaluated"] == 2


# -- lifecycle and concurrency ------------------------------------------------


def test_callbacks_within_one_interval_coalesce_into_the_latest_snapshot(
    tmp_path: Path, renders: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    writes: list[Any] = []
    periodic_write = threading.Event()
    real_write = adapter_module.write_snapshot

    def observe_write(directory: Path, snapshot: Any) -> Path:
        path = real_write(directory, snapshot)
        writes.append(snapshot)
        if len(writes) == 2:
            periodic_write.set()
        return path

    monkeypatch.setattr(adapter_module, "write_snapshot", observe_write)
    adapter = DGDRRunOutputAdapter(
        make_config(tmp_path, snapshot_interval_seconds=60), workload=None
    )
    adapter.start()
    interval = ControlledWait()
    adapter._stop = interval  # type: ignore[assignment]

    adapter.on_candidate(record("p000", 0.0))
    assert interval.waiting.wait(2), "writer did not enter the rate-limit delay"
    for index in range(1, 300):
        adapter.on_candidate(record(f"p{index:03d}", float(index)))
    interval.advance()

    assert periodic_write.wait(2), "periodic snapshot was not written"
    snapshot = load(tmp_path)
    assert len(writes) == 2  # initial + one coalesced periodic write
    assert snapshot["progress"]["evaluated"] == 300
    assert tags(snapshot) == ["p299", "p298", "p297"]
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


@pytest.mark.parametrize(
    ("error", "phase", "error_name"),
    [
        pytest.param(None, "Succeeded", None, id="success"),
        pytest.param(
            RuntimeError("search crashed"), "Failed", "RuntimeError", id="error"
        ),
    ],
)
def test_context_manager_records_terminal_outcome(
    tmp_path: Path,
    renders: list[str],
    error: RuntimeError | None,
    phase: str,
    error_name: str | None,
) -> None:
    adapter = DGDRRunOutputAdapter(make_config(tmp_path), workload=None)
    if error is None:
        with adapter:
            pass
    else:
        with pytest.raises(RuntimeError, match="search crashed"), adapter:
            raise error

    run = load(tmp_path)["run"]
    assert run["phase"] == phase
    assert run.get("error") == error_name


def test_callbacks_never_wait_for_a_slow_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rendering = threading.Event()
    release = threading.Event()

    def slow_render(candidate: Any, *args: Any, **kwargs: Any) -> str:
        rendering.set()
        assert release.wait(5), "test did not release the renderer"
        return "kind: DynamoGraphDeployment\n"

    monkeypatch.setattr(adapter_module, "render_dgd", slow_render)
    adapter = DGDRRunOutputAdapter(make_config(tmp_path), workload=None)
    adapter.start()
    adapter.on_candidate(record("a", 1.0))
    assert rendering.wait(5), "writer did not enter the renderer"

    callbacks_done = threading.Event()

    def send_callbacks() -> None:
        for index in range(1000):
            adapter.on_candidate(record(f"q{index}", score=float(index)))
        callbacks_done.set()

    sender = threading.Thread(target=send_callbacks)
    sender.start()
    assert callbacks_done.wait(2), "callbacks blocked behind the writer"
    sender.join()
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


def test_close_rejects_a_non_terminal_phase(tmp_path: Path) -> None:
    adapter = DGDRRunOutputAdapter(make_config(tmp_path), workload=None)
    with pytest.raises(ValueError, match="terminal"):
        adapter.close(phase=RunPhase.RUNNING)


# -- failure handling ---------------------------------------------------------


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
    monkeypatch.setattr(adapter_module, "_WRITE_RETRY_BACKOFF", 0.02)
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


def test_a_persistent_write_failure_waits_for_backoff_instead_of_spinning(
    tmp_path: Path, renders: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(adapter_module, "_WRITE_RETRY_BACKOFF", 30)
    adapter = DGDRRunOutputAdapter(make_config(tmp_path), workload=None)
    adapter.start()
    backoff = ControlledWait()
    adapter._stop = backoff  # type: ignore[assignment]
    real_write = adapter_module.write_snapshot
    attempts: list[int] = []
    failed = threading.Event()

    def failing(directory: Path, snapshot: Any) -> Path:
        del directory, snapshot
        attempts.append(1)
        failed.set()
        raise OSError("disk full")

    monkeypatch.setattr(adapter_module, "write_snapshot", failing)
    adapter.on_candidate(record("a", 1.0))
    assert failed.wait(2), "writer did not attempt the snapshot"
    assert backoff.waiting.wait(2), "writer did not enter retry backoff"
    assert len(attempts) == 1
    assert backoff.timeouts and backoff.timeouts[0] > 0

    monkeypatch.setattr(adapter_module, "write_snapshot", real_write)
    adapter.close()


# -- configuration and plugin -------------------------------------------------


def test_pareto_objectives_are_rejected_explicitly(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError, match="scalar"):
        DGDRRunOutputAdapter(make_config(tmp_path), workload=None, is_pareto=True)


@pytest.mark.parametrize(
    ("overrides", "valid"),
    [
        pytest.param({"max_candidates": 1}, True, id="min-candidates"),
        pytest.param({"max_candidates": 100}, True, id="max-candidates"),
        pytest.param({"max_candidates": 0}, False, id="too-few-candidates"),
        pytest.param({"max_candidates": 101}, False, id="too-many-candidates"),
        pytest.param({"snapshot_interval_seconds": -1}, False, id="negative-interval"),
        pytest.param({"name": "Not_DNS"}, False, id="invalid-name"),
        pytest.param({"num_gpus_per_node": 0}, False, id="zero-gpus"),
        pytest.param({"unknown_field": 1}, False, id="unknown-field"),
    ],
)
def test_configuration_boundaries(
    tmp_path: Path, overrides: dict[str, Any], valid: bool
) -> None:
    if valid:
        config = make_config(tmp_path, **overrides)
        assert config.max_candidates == overrides["max_candidates"]
        return
    with pytest.raises(ValueError):
        make_config(tmp_path, **overrides)


@pytest.mark.parametrize(
    "candidate_parameters",
    [
        pytest.param(
            {
                "hardware_sku": "concrete_candidate_sku",
                "systems_paths": ["/external/systems"],
            },
            id="operator-explicit-width",
        ),
        pytest.param({}, id="standalone-backwards-compatible"),
    ],
)
def test_explicit_node_width_does_not_load_a_system_spec(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    candidate_parameters: dict[str, Any],
) -> None:
    def unexpected_load(*_: Any, **__: Any) -> dict[str, Any]:
        pytest.fail("an explicit num_gpus_per_node must bypass system-spec loading")

    monkeypatch.setattr(adapter_module, "load_system_spec", unexpected_load)
    config = make_config(tmp_path, num_gpus_per_node=4)

    assert (
        config.generation_options("vllm", candidate_parameters).num_gpus_per_node == 4
    )


@pytest.mark.parametrize(
    ("hardware_sku", "num_gpus_per_node"),
    [
        pytest.param("h200_sxm", 8, id="auto-resolved-concrete-catalog-sku"),
        pytest.param("external_future_gpu", 16, id="external-systems-path-sku"),
    ],
)
def test_concrete_candidate_sku_uses_aisimulates_system_spec_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hardware_sku: str,
    num_gpus_per_node: int,
) -> None:
    systems_paths = (
        ["/external/systems"] if hardware_sku.startswith("external") else None
    )
    loaded: list[tuple[str, list[str] | None]] = []

    def load(sku: str, systems_paths: list[str] | None = None) -> dict[str, Any]:
        loaded.append((sku, systems_paths))
        return {"node": {"num_gpus_per_node": num_gpus_per_node}}

    monkeypatch.setattr(adapter_module, "load_system_spec", load)
    config = make_config(tmp_path, num_gpus_per_node=None)
    # AISimulate unrolls engine.hardware:auto and per-candidate choices before
    # publishing the CandidateRecord, so its config always carries a concrete SKU.
    candidate_parameters: dict[str, Any] = {"hardware_sku": hardware_sku}
    if systems_paths is not None:
        candidate_parameters["systems_paths"] = systems_paths

    assert (
        config.generation_options("vllm", candidate_parameters).num_gpus_per_node
        == num_gpus_per_node
    )
    assert loaded == [(hardware_sku, systems_paths)]


@pytest.mark.parametrize(
    ("candidate_parameters", "message"),
    [
        pytest.param(None, "candidate parameters are required", id="missing-candidate"),
        pytest.param({}, "nonempty hardware_sku", id="missing-hardware-sku"),
        pytest.param(
            {"hardware_sku": ""}, "nonempty hardware_sku", id="empty-hardware-sku"
        ),
        pytest.param(
            {"hardware_sku": "h200_sxm", "systems_paths": []},
            "candidate systems_paths",
            id="empty-systems-paths",
        ),
    ],
)
def test_candidate_topology_source_is_required_at_materialization(
    tmp_path: Path,
    candidate_parameters: dict[str, Any] | None,
    message: str,
) -> None:
    config = make_config(tmp_path, num_gpus_per_node=None)

    with pytest.raises(CandidateMaterializationError, match=message):
        config.generation_options("vllm", candidate_parameters)


@pytest.mark.parametrize(
    "system_spec",
    [
        pytest.param([], id="non-object-system-spec"),
        pytest.param({}, id="missing-node"),
        pytest.param({"node": None}, id="non-object-node"),
        pytest.param({"node": {}}, id="missing-node-width"),
        pytest.param({"node": {"num_gpus_per_node": 0}}, id="non-positive-node-width"),
        pytest.param({"node": {"num_gpus_per_node": True}}, id="boolean-node-width"),
        pytest.param({"node": {"num_gpus_per_node": "8"}}, id="string-node-width"),
    ],
)
def test_malformed_system_spec_topology_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    system_spec: Any,
) -> None:
    monkeypatch.setattr(
        adapter_module,
        "load_system_spec",
        lambda _, systems_paths=None: system_spec,
    )
    config = make_config(tmp_path, num_gpus_per_node=None)

    with pytest.raises(
        CandidateMaterializationError,
        match="AISimulate system spec.*(object|positive integer)",
    ):
        config.generation_options("vllm", {"hardware_sku": "malformed_system"})


def test_system_spec_topology_errors_are_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        adapter_module,
        "load_system_spec",
        lambda _, systems_paths=None: {},
    )
    config = make_config(tmp_path, num_gpus_per_node=None)

    with pytest.raises(CandidateMaterializationError) as raised:
        config.generation_options("vllm", {"hardware_sku": "x" * 1000})

    assert len(str(raised.value)) == adapter_module._MAX_ERROR_MESSAGE


def test_custom_candidate_system_root_reaches_candidate_materialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    materialized_node_widths: list[int] = []
    systems_root = tmp_path / "external-systems"
    systems_root.mkdir()
    (systems_root / "custom_twelve_gpu_node.yaml").write_text(
        "node:\n  num_gpus_per_node: 12\n",
        encoding="utf-8",
    )

    def capture_node_width(
        candidate: Any, workload: Any, options: Any, **kwargs: Any
    ) -> str:
        del candidate, workload, kwargs
        materialized_node_widths.append(options.num_gpus_per_node)
        return "apiVersion: nvidia.com/v1beta1\nkind: DynamoGraphDeployment\n"

    monkeypatch.setattr(adapter_module, "render_dgd", capture_node_width)
    adapter = DGDRRunOutputAdapter(
        make_config(tmp_path, num_gpus_per_node=None),
        workload=None,
    )
    candidate = record("a", 1.0)
    candidate.config.update(
        {
            "hardware_sku": "custom_twelve_gpu_node",
            "systems_paths": [str(systems_root)],
        }
    )
    adapter.on_candidate(candidate)
    adapter.close()

    assert materialized_node_widths == [12]
    assert load(tmp_path)["candidates"][0]["outcome"] == "materialized"


@pytest.mark.parametrize(
    "invalid_part", ["topology", "unknown_backend", "empty_backend"]
)
def test_invalid_candidate_does_not_abort_other_candidates(
    tmp_path: Path,
    renders: list[str],
    monkeypatch: pytest.MonkeyPatch,
    invalid_part: str,
) -> None:
    def load_candidate_system(
        sku: str, systems_paths: list[str] | None = None
    ) -> dict[str, Any]:
        del systems_paths
        if sku == "broken_system":
            return {"node": {}}
        return {"node": {"num_gpus_per_node": 8}}

    monkeypatch.setattr(adapter_module, "load_system_spec", load_candidate_system)
    adapter = DGDRRunOutputAdapter(
        make_config(
            tmp_path,
            num_gpus_per_node=None,
            max_candidates=2,
            derive_runtime_images=True,
        ),
        workload=None,
    )
    broken = record("broken-topology", 2.0)
    broken.config["hardware_sku"] = (
        "broken_system" if invalid_part == "topology" else "h200_sxm"
    )
    if invalid_part != "topology":
        broken.config["backend"] = (
            "unsupported" if invalid_part == "unknown_backend" else ""
        )
    valid = record("valid-topology", 1.0)
    valid.config["hardware_sku"] = "h200_sxm"

    adapter.on_candidate(broken)
    adapter.on_candidate(valid)
    adapter.close()

    snapshot = load(tmp_path)
    assert snapshot["run"] == {"phase": "Succeeded", "terminal": True}
    assert [candidate["outcome"] for candidate in snapshot["candidates"]] == [
        "materialization_failed",
        "materialized",
    ]
    assert renders == ["valid-topology"]


def test_max_candidates_has_no_default_of_its_own(tmp_path: Path) -> None:
    # The operator supplies the already-defaulted API value; a missing one must fail.
    with pytest.raises(ValueError, match="max_candidates"):
        DGDRRunOutputConfig(
            name="sweep",
            snapshot_dir=str(tmp_path),
            runtime_image="nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.3",
            num_gpus_per_node=8,
        )


def test_a_snapshot_directory_is_required() -> None:
    config = DGDRRunOutputConfig(
        name="sweep",
        runtime_image="nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.3",
        num_gpus_per_node=8,
        max_candidates=1,
    )
    with pytest.raises(ValueError, match="snapshot directory"):
        DGDRRunOutputAdapter(config, workload=None)


def test_plugin_discovery_validation_and_optional_live_subscription(
    tmp_path: Path,
) -> None:
    plugin = create_adapter()
    assert plugin.name == "dgdr_run"
    with_dir = {
        "name": "run",
        "snapshot_dir": str(tmp_path),
        "runtime_image": "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.3",
        "num_gpus_per_node": 8,
        "max_candidates": 5,
    }
    assert isinstance(plugin.live(with_dir, workload=None), DGDRRunOutputAdapter)
    assert plugin.subscribe(with_dir) is None
    assert plugin.subscribe(with_dir, context=SimpleNamespace(workload=None)) is None
    without_dir = {
        key: value for key, value in with_dir.items() if key != "snapshot_dir"
    }
    assert (
        plugin.subscribe(without_dir, context=SimpleNamespace(workload="workload"))
        is None
    )
    assert not (tmp_path / SNAPSHOT_FILE_NAME).exists()
    with pytest.raises(ValueError, match="num_gpus_per_node"):
        plugin.subscribe({**with_dir, "num_gpus_per_node": 0})
    with pytest.raises(ValueError, match="canonical"):
        plugin.subscribe(
            {
                **with_dir,
                "runtime_image": "nvcr.io/nvidia/ai-dynamo/vllm-runtime:latest",
            }
        )


def test_plugin_write_keeps_sweepers_selection_order(
    tmp_path: Path, renders: list[str], fake_search_config: None
) -> None:
    # Sweeper's selection can be goal-dependent (e.g. fewest GPUs first), so the
    # adapter must not re-rank it by score.
    result = SimpleNamespace(
        views=SimpleNamespace(pareto_front=[]),
        provenance=SimpleNamespace(config={}),
        selected_candidates=[record("low", 1.0), record("high", 9.0)],
        counts=SimpleNamespace(evaluated=2),
    )
    written = create_adapter().write(
        PLUGIN_CONFIG,
        result=result,
        output_dir=tmp_path,  # type: ignore[arg-type]
    )
    snapshot = load(tmp_path)
    assert written == [Path(SNAPSHOT_FILE_NAME)]
    assert snapshot["run"] == {"phase": "Succeeded", "terminal": True}
    assert snapshot["progress"] == {"round": 0, "evaluated": 2}
    assert tags(snapshot) == ["low", "high"]


def test_plugin_write_refuses_pareto_and_empty_results(tmp_path: Path) -> None:
    config = {
        "name": "sweep",
        "runtime_image": "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.3",
        "num_gpus_per_node": 8,
        "max_candidates": 5,
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
    "max_candidates": 5,
    "snapshot_interval_seconds": 0,
}


@pytest.mark.parametrize("search_fails", [False, True], ids=["complete", "failure"])
def test_pinned_aisimulate_dispatches_the_complete_lifecycle_to_dgdr_run(
    tmp_path: Path,
    renders: list[str],
    fake_search_config: None,
    monkeypatch: pytest.MonkeyPatch,
    search_fails: bool,
) -> None:
    import aisimulate.recommend as recommendation
    from aisimulate.config import CoreRecommendationConfig
    from aisimulate.sweeper import search

    config = CoreRecommendationConfig.model_validate(
        {
            "engine": {
                "model": "openai/gpt-oss-120b",
                "hardware": "gb300",
                "backend": "vllm",
                "mode": "aggregated",
                "context_length": 16384,
                "workers": {"aggregated": {}},
            },
            "traffic": {
                "source": {
                    "type": "synthetic",
                    "input_tokens": 10240,
                    "output_tokens": 1024,
                },
                "load": {
                    "type": "concurrency",
                    "concurrency": {"choices": [8]},
                },
                "stop": {"requests_per_load_unit": 1},
            },
            "optimization": {
                "target": "throughput_per_gpu",
                "constraints": {
                    "min_candidate_gpus": 1,
                    "max_candidate_gpus": 8,
                },
            },
            "optimizer": {"parallelism": 1, "max_trials": 1},
        }
    )
    output_config = {
        **PLUGIN_CONFIG,
        "snapshot_dir": str(tmp_path),
        # Keep candidate and round updates buffered so their presence after a
        # successful search proves that on_complete forced the final live flush.
        "snapshot_interval_seconds": 3600,
    }
    result = _final_result(1)
    search_error = RuntimeError("search crashed")

    # Use the real pinned AISimulate lifecycle dispatcher around a deterministic search.
    monkeypatch.setattr(
        recommendation, "resolve_budget", lambda *args, **kwargs: {"cpu_limit": 1}
    )

    class LifecycleSweeper:
        def __init__(self, **kwargs: Any) -> None:
            del kwargs

        def run(
            self,
            smart: Any,
            *,
            top_n: int | None,
            on_candidate: Any = None,
            on_round: Any = None,
        ) -> Any:
            del smart, top_n
            assert on_candidate is not None
            assert on_round is not None
            on_candidate(record("a", 9.0))
            on_round(3, [])
            if search_fails:
                raise search_error
            return result

    monkeypatch.setattr(search, "Sweeper", LifecycleSweeper)

    # A failed search must dispatch on_failure and leave a terminal failed snapshot.
    if search_fails:
        with pytest.raises(RuntimeError, match="search crashed") as raised:
            recommendation._run_recommendation(
                config,
                stack="engine",
                runner_factory=object(),
                output_configs={"dgdr_run": output_config},
                output_adapters={"dgdr_run": create_adapter()},
            )
        assert raised.value is search_error
        snapshot = load(tmp_path)
        assert snapshot["run"] == {
            "phase": "Failed",
            "terminal": True,
            "message": "search crashed",
            "error": "RuntimeError",
        }
        assert snapshot["progress"] == {"round": 3, "evaluated": 1}
        assert tags(snapshot) == ["a"]
        return

    # A successful search must dispatch on_complete before the final writer takes over.
    returned = recommendation._run_recommendation(
        config,
        stack="engine",
        runner_factory=object(),
        output_configs={"dgdr_run": output_config},
        output_adapters={"dgdr_run": create_adapter()},
    )
    assert returned is result
    completed = load(tmp_path)
    assert completed["run"] == {"phase": "Running", "terminal": False}
    assert completed["progress"] == {"round": 3, "evaluated": 1}
    assert tags(completed) == ["a"]

    # The separate final writer publishes the authoritative terminal selection.
    create_adapter().write(output_config, result=returned, output_dir=tmp_path)
    terminal = load(tmp_path)
    assert terminal["run"] == {"phase": "Succeeded", "terminal": True}
    assert terminal["progress"] == {"round": 3, "evaluated": 1}
    assert tags(terminal) == ["a"]


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


# -- lifecycle: how the search ended --------------------------------------------


def _subscribed(tmp_path: Path) -> Any:
    callbacks = create_adapter().subscribe(
        {**PLUGIN_CONFIG, "snapshot_dir": str(tmp_path)},
        context=SimpleNamespace(workload="workload"),
    )
    assert callbacks is not None
    return callbacks


def test_a_long_failure_message_is_truncated(
    tmp_path: Path, renders: list[str]
) -> None:
    callbacks = _subscribed(tmp_path)
    callbacks.on_failure(RuntimeError("x" * 5000))
    assert len(load(tmp_path)["run"]["message"]) == adapter_module._MAX_ERROR_MESSAGE


def test_a_search_that_finds_nothing_is_recorded_as_failed(
    tmp_path: Path, renders: list[str]
) -> None:
    callbacks = _subscribed(tmp_path)
    callbacks.on_candidate(record("infeasible", 1.0, status="infeasible"))

    callbacks.on_complete()

    run = load(tmp_path)["run"]
    assert run["phase"] == "Failed"
    assert run["terminal"] is True
    assert run["message"] == "no feasible candidate found"


# -- robustness -----------------------------------------------------------------


def test_a_broken_renderer_leaves_a_terminal_failed_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*_: Any, **__: Any) -> str:
        raise RuntimeError("renderer not installed")

    monkeypatch.setattr(adapter_module, "render_dgd", broken)
    adapter = DGDRRunOutputAdapter(make_config(tmp_path), workload=None)
    adapter.on_candidate(record("a", 9.0))

    with pytest.raises(RuntimeError, match="renderer not installed"):
        adapter.close()

    snapshot = load(tmp_path)
    assert snapshot["run"]["phase"] == "Failed"
    assert snapshot["run"]["terminal"] is True
    assert snapshot["run"]["error"] == "RuntimeError"
    assert all(c["outcome"] != "materialization_failed" for c in snapshot["candidates"])


def test_a_candidate_error_message_is_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing(*_: Any, **__: Any) -> str:
        raise CandidateMaterializationError("x" * 5000)

    monkeypatch.setattr(adapter_module, "render_dgd", failing)
    adapter = DGDRRunOutputAdapter(make_config(tmp_path), workload=None)
    adapter.on_candidate(record("a", 9.0))
    adapter.close()

    (candidate,) = load(tmp_path)["candidates"]
    assert 0 < len(candidate["error"]) <= 500


def test_the_renderer_receives_the_candidates_gpu_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[Any] = []

    def capture(candidate: Any, *_: Any, **kwargs: Any) -> str:
        seen.append(candidate)
        return "kind: DynamoGraphDeployment\n"

    monkeypatch.setattr(adapter_module, "render_dgd", capture)
    adapter = DGDRRunOutputAdapter(make_config(tmp_path), workload=None)
    adapter.on_candidate(record("a", 9.0, gpus=4))
    adapter.close()

    (candidate,) = seen
    assert candidate.used_gpus == 4
    assert candidate.config["tag"] == "a"


# -- evaluated metrics survive, and a run without a deployable candidate fails ----


SOURCE_METRICS: dict[str, float | None] = {
    "throughput": 70.4,
    "ttft": 18.2,
    "itl": 7.1,
    "e2e_latency": 300.5,
    "goodput": 66.0,
    "power_w": 410.0,
    "normalized_power": 0.8,
    "broken": float("nan"),
}


def test_every_finite_source_metric_survives_in_live_and_terminal_snapshots(
    tmp_path: Path, renders: list[str], fake_search_config: None
) -> None:
    expected = {
        **{key: value for key, value in SOURCE_METRICS.items() if key != "broken"},
        "broken": None,
        "score": 87.7,
        "usedGpus": 4,
        "objectives": {"goodput": 12.5, "invalid": None},
    }
    source = {
        **SOURCE_METRICS,
        "score": 999.0,
        "usedGpus": 99,
        "objectives": 123.0,
    }

    live = tmp_path / "live"
    live.mkdir()
    adapter = DGDRRunOutputAdapter(make_config(live), workload=None)
    adapter.start()
    adapter.on_candidate(
        record(
            "a",
            87.7,
            gpus=4,
            metrics=source,
            objectives={"goodput": 12.5, "invalid": float("nan")},
        )
    )
    wait_for(lambda: tags(load(live)) == ["a"])
    assert load(live)["candidates"][0]["metrics"] == expected
    adapter.close()

    final = tmp_path / "final"
    final.mkdir()
    result = _final_result(1)
    result.selected_candidates = [
        record(
            "a",
            87.7,
            gpus=4,
            metrics=source,
            objectives={"goodput": 12.5, "invalid": float("nan")},
        )
    ]
    create_adapter().write(
        {**PLUGIN_CONFIG, "snapshot_dir": str(final)}, result=result, output_dir=final
    )
    assert load(final)["candidates"][0]["metrics"] == expected
    for directory in (live, final):
        text = (directory / SNAPSHOT_FILE_NAME).read_text().lower()
        assert "nan" not in text and "inf" not in text.replace("information", "")


@pytest.mark.parametrize(
    ("failure_limits", "phase", "expected_outcomes", "expected_attempts"),
    [
        pytest.param(
            (99, 99),
            "Failed",
            ("materialization_failed", "materialization_failed"),
            (3, 3),
            id="all-fail-bounded",
        ),
        pytest.param(
            (99, 0),
            "Succeeded",
            ("materialization_failed", "materialized"),
            (3, 1),
            id="partial-success",
        ),
        pytest.param(
            (1,),
            "Succeeded",
            ("materialized",),
            (2,),
            id="transient-recovers",
        ),
    ],
)
def test_materialization_retry_and_terminal_policy(
    tmp_path: Path,
    fake_search_config: None,
    monkeypatch: pytest.MonkeyPatch,
    failure_limits: tuple[int, ...],
    phase: str,
    expected_outcomes: tuple[str, ...],
    expected_attempts: tuple[int, ...],
) -> None:
    monkeypatch.setattr(adapter_module, "_WRITE_RETRY_BACKOFF", 0)
    attempts = [0] * len(failure_limits)

    def flaky_render(candidate: Any, *_: Any, **__: Any) -> str:
        index = int(candidate.config["tag"].removeprefix("candidate-"))
        attempts[index] += 1
        if attempts[index] <= failure_limits[index]:
            raise CandidateMaterializationError(f"cannot render candidate-{index}")
        return "apiVersion: nvidia.com/v1beta1\nkind: DynamoGraphDeployment\n"

    monkeypatch.setattr(adapter_module, "render_dgd", flaky_render)
    result = _final_result(len(failure_limits))
    result.selected_candidates = [
        record(f"candidate-{index}", float(len(failure_limits) - index))
        for index in range(len(failure_limits))
    ]

    if phase == "Failed":
        with pytest.raises(
            CandidateMaterializationError, match="no selected candidate"
        ):
            create_adapter().write(
                {**PLUGIN_CONFIG, "max_candidates": len(failure_limits)},
                result=result,
                output_dir=tmp_path,
            )
    else:
        create_adapter().write(
            {**PLUGIN_CONFIG, "max_candidates": len(failure_limits)},
            result=result,
            output_dir=tmp_path,
        )

    snapshot = load(tmp_path)
    assert snapshot["run"]["phase"] == phase
    assert tuple(c["outcome"] for c in snapshot["candidates"]) == expected_outcomes
    assert tuple(attempts) == expected_attempts
    if phase == "Failed":
        assert snapshot["run"]["error"] == "CandidateMaterializationError"
        assert "cannot render candidate-0" in snapshot["run"]["message"]
        assert len(snapshot["run"]["message"]) <= adapter_module._MAX_ERROR_MESSAGE
