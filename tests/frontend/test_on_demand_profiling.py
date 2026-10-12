# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Control plane tests without GPU engines or a privileged perf installation."""

import asyncio
import gzip
import hashlib
import json
import os
import sqlite3
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import aiohttp
import msgspec
import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from dynamo.frontend.profiling import hosting
from dynamo.frontend.profiling.hosting import host, profiling_coordinator
from dynamo.frontend.profiling.http import application
from dynamo.profiling import (
    Coordinator,
    adapters,
    agent,
    bundle,
    collectors,
    errors,
    nsys,
)
from dynamo.profiling.adapters import CollectorRegistry
from dynamo.profiling.artifacts import SharedFilesystemValidator
from dynamo.profiling.clock import estimate_offset
from dynamo.profiling.collectors import CollectionError, PerfCollector, TorchCollector
from dynamo.profiling.errors import Unavailable
from dynamo.profiling.models import Config, Plan, RuntimeSnapshot, Target, Worker

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.timeout(20),
]


@pytest.fixture
def collector():
    started = asyncio.Event()
    return SimpleNamespace(
        started=started,
        evidence={"target_id": "frontend", "collector": "linux_perf"},
        start=AsyncMock(side_effect=started.set),
        stop=AsyncMock(),
    )


@pytest.fixture
def controller(tmp_path, monkeypatch, collector):
    monkeypatch.setattr(adapters, "PerfCollector", lambda *args: collector)
    return Coordinator(
        Config(str(tmp_path / "admin.sock"), str(tmp_path / "artifacts")),
        RuntimeSnapshot(123, "frontend-epoch", "test-host"),
    )


@pytest.fixture
def provider_controller(tmp_path, collector):
    registry = CollectorRegistry()
    registry.register("nsys", lambda *args: [collector])
    return Coordinator(
        Config("unused", str(tmp_path), finalize_timeout_s=0.01),
        RuntimeSnapshot(123, "epoch", "host"),
        registry=registry,
    )


def plan(request_id="request-1", duration_ms=50):
    return Plan(request_id, [Target("frontend", "frontend-epoch")], duration_ms)


@pytest.mark.asyncio
async def test_deadline_and_artifact_manifest(controller, collector):
    session = controller.submit(plan())
    await session.task
    assert session.state == "completed"
    collector.start.assert_awaited_once()
    collector.stop.assert_awaited_once()
    manifest = json.loads((session.directory / "manifest.json").read_text())
    assert manifest["profiles"][0]["collector"] == "linux_perf"
    assert manifest["lifecycle_otel"]["coverage"] == "unknown"
    assert manifest["lifecycle_otel"]["instrumentation_mode"] == "unchanged"
    assert controller.active is None


@pytest.mark.asyncio
async def test_retry_reuses_session_and_conflicting_plan_rejected(controller):
    session = controller.submit(plan())
    deadline = session.deadline_monotonic
    assert controller.submit(plan()) is session
    assert session.deadline_monotonic == deadline
    with pytest.raises(errors.Conflict):
        controller.submit(plan(duration_ms=100))
    with pytest.raises(errors.Conflict):
        controller.submit(plan("another-request"))
    await controller.close()
    assert controller.submit(plan()) is session


@pytest.mark.parametrize(
    "bad_plan,exception",
    [
        (Plan("id", [Target("frontend", "stale")], 50), errors.Conflict),
        (Plan("id", [Target("unknown", "epoch")], 50), errors.BadRequest),
        (Plan("id", [], 50), errors.BadRequest),
        (Plan("id", [Target("frontend", "frontend-epoch")], 0), errors.BadRequest),
        (Plan("id", [Target("frontend", "frontend-epoch")], 60001), errors.BadRequest),
        (
            Plan("id", [Target("frontend", "frontend-epoch")], 50, mode="nsys"),
            errors.Unsupported,
        ),
    ],
)
def test_invalid_plans(controller, bad_plan, exception):
    with pytest.raises(exception):
        controller.validate(bad_plan)
    assert not controller.sessions


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["error", "timeout"])
async def test_uncertain_start_is_bounded_and_cleaned_up(
    controller, collector, failure
):
    async def hang():
        await asyncio.Event().wait()

    collector.start.side_effect = (
        CollectionError("permission denied") if failure == "error" else hang
    )
    session = controller.submit(plan())
    await asyncio.wait_for(session.task, 1)
    assert session.state == "failed"
    collector.stop.assert_awaited_once()
    assert (
        "permission denied" if failure == "error" else "TimeoutError"
    ) in session.errors


@pytest.mark.asyncio
async def test_worker_only_plan_still_profiles_frontend_and_targets_run_concurrently(
    controller, collector, monkeypatch
):
    controller.workers["worker"] = Worker(
        "worker", "worker-epoch", "vllm", "http://configured"
    )
    stop_entered = asyncio.Event()
    finish_export = asyncio.Event()
    entered = []

    async def start():
        # Sequential startup would never let the second worker enter.
        entered.append(None)
        while len(entered) < 2:
            await asyncio.sleep(0)

    async def stop():
        stop_entered.set()
        await finish_export.wait()

    worker_collector = SimpleNamespace(
        evidence={"collector": "torch_profiler"}, start=start, stop=stop
    )
    controller.workers["second"] = Worker(
        "second", "epoch", "vllm", "http://configured"
    )
    fast_worker = SimpleNamespace(
        evidence={"collector": "torch_profiler"}, start=start, stop=AsyncMock()
    )
    monkeypatch.setattr(
        adapters,
        "TorchCollector",
        lambda worker, config: (
            worker_collector if worker.target_id == "worker" else fast_worker
        ),
    )
    session = controller.submit(
        Plan("id", [Target("worker", "worker-epoch"), Target("second", "epoch")], 50)
    )
    await asyncio.wait_for(stop_entered.wait(), 1)
    try:
        assert session.state == "finalizing"
        collector.stop.assert_awaited_once()
        fast_worker.stop.assert_awaited_once()
        assert controller.active is session
    finally:
        finish_export.set()
        await session.task


@pytest.mark.asyncio
async def test_stop_before_recording_reports_missing_coverage(tmp_path, collector):
    probing = asyncio.Event()
    release = asyncio.Event()

    async def measure(worker, config):
        probing.set()
        await release.wait()
        return {"alignment": "unknown"}

    controller = Coordinator(
        Config("unused", str(tmp_path), frontend_perf_enabled=False),
        RuntimeSnapshot(
            123, "epoch", "host", [Worker("decode", "epoch", "vllm", "http://w")]
        ),
        registry=CollectorRegistry(),
        clock_sampler=SimpleNamespace(measure=measure),
    )
    controller.registry.register("torch_profile", lambda *args: [collector])
    session = controller.submit(Plan("early", [Target("decode", "epoch")], 5000))
    await asyncio.wait_for(probing.wait(), 1)
    controller.stop(session.session_id)
    release.set()
    await asyncio.wait_for(session.task, 1)
    assert session.state == "failed"
    assert session.errors == ["Capture stopped before recording started"]
    collector.start.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("frontend_perf", [True, False])
async def test_rest_validation_stop_and_unknown_fields(
    controller, collector, monkeypatch, frontend_perf
):
    controller.config = msgspec.structs.replace(
        controller.config, frontend_perf_enabled=frontend_perf
    )
    if frontend_perf:
        capture_plan = plan(duration_ms=5000)
    else:

        def forbidden_perf(*args):
            raise AssertionError("Disabled frontend perf must never be instantiated")

        monkeypatch.setattr(adapters, "PerfCollector", forbidden_perf)
        monkeypatch.setattr(adapters, "TorchCollector", lambda *args: collector)
        controller.workers["worker"] = Worker(
            "worker", "epoch", "vllm", "http://configured"
        )
        collector.evidence.update(target_id="worker", collector="torch_profiler")
        capture_plan = Plan("id", [Target("worker", "epoch")], 5000)
        with pytest.raises(errors.Unsupported):
            controller.validate(plan())
    async with TestClient(TestServer(application(controller))) as client:
        targets = await client.get("/admin/profiling/v1/targets")
        frontend = (await targets.json())["targets"][0]
        assert frontend["target_id"] == "frontend"
        assert frontend["collectors"] == (["linux_perf"] if frontend_perf else [])
        payload = msgspec.to_builtins(capture_plan)
        validation = await client.post(
            "/admin/profiling/v1/plans/validate", json=payload
        )
        assert validation.status == 200
        assert (await validation.json())["frontend_collector"] == (
            "linux_perf" if frontend_perf else None
        )
        bad = await client.post(
            "/admin/profiling/v1/sessions", json={**payload, "pid": 1}
        )
        assert bad.status == 400
        created = await client.post("/admin/profiling/v1/sessions", json=payload)
        assert created.status == 202
        session_id = (await created.json())["session_id"]
        stopped = await client.post(f"/admin/profiling/v1/sessions/{session_id}/stop")
        assert stopped.status == 202
        await controller.close()
        artifacts = await client.get(
            f"/admin/profiling/v1/sessions/{session_id}/artifacts"
        )
        manifest = await artifacts.json()
        assert manifest["state"] == "completed"
        assert manifest["frontend"]["profiling_enabled"] is frontend_perf
        assert manifest["frontend"]["omission_reason"] == (
            None if frontend_perf else "disabled_by_operator_config"
        )
        assert [profile["collector"] for profile in manifest["profiles"]] == (
            ["linux_perf"] if frontend_perf else ["torch_profiler"]
        )
        missing = await client.get("/admin/profiling/v1/sessions/missing")
        assert missing.status == 404
    collector.start.assert_awaited_once()
    collector.stop.assert_awaited_once()
    assert controller.active is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "backend,content_type", [("vllm", "text/plain"), ("sglang", "application/json")]
)
async def test_backend_http200_error_is_failure_and_options_forwarded(
    tmp_path, backend, content_type
):
    bodies = []

    async def hook(request):
        bodies.append(await request.json())
        return web.Response(
            text=json.dumps({"status": "error", "message": "engine rejected capture"}),
            content_type=content_type,
        )

    app = web.Application()
    app.router.add_post("/engine/control/start_profile", hook)
    async with TestServer(app) as server:
        worker = Worker(
            "worker",
            "epoch",
            backend,
            str(server.make_url("/")),
            {"profile_prefix": "trusted"},
        )
        adapter = TorchCollector(worker, Config("socket", str(tmp_path)))
        with pytest.raises(CollectionError, match="engine rejected capture"):
            await adapter.start()
    assert bodies == [{"profile_prefix": "trusted"}]


@pytest.mark.asyncio
async def test_backend_text_plain_json_start_and_stop(tmp_path):
    calls = []

    async def hook(request):
        calls.append((request.path, await request.json()))
        return web.Response(text='{"status":"ok"}', content_type="text/plain")

    app = web.Application()
    app.router.add_post("/engine/control/start_profile", hook)
    app.router.add_post("/engine/control/stop_profile", hook)
    async with TestServer(app) as server:
        worker = Worker("worker", "epoch", "vllm", str(server.make_url("/")), {})
        adapter = TorchCollector(worker, Config("socket", str(tmp_path)))
        await adapter.start()
        await adapter.stop()
        assert adapter.evidence["coverage"] == "export_unverified"
        assert (
            adapter.evidence["timestamp_scope"]
            == "control_responses_not_native_recording_bounds"
        )
        assert (
            adapter.evidence["start_requested_at_ns"]
            <= adapter.evidence["started_at_ns"]
        )
        assert (
            adapter.evidence["stop_requested_at_ns"]
            <= adapter.evidence["stopped_at_ns"]
        )
    assert calls == [
        ("/engine/control/start_profile", {}),
        ("/engine/control/stop_profile", {}),
    ]


@pytest.mark.asyncio
async def test_perf_attach_ack_and_signal_only_recorder(tmp_path, monkeypatch):
    commands = []
    read_ends = []

    class Recorder:
        returncode = None

        def __init__(self):
            self.signals = []

        def send_signal(self, signum):
            self.signals.append(signum)

        async def wait(self):
            self.returncode = 0
            return 0

    recorder = Recorder()
    monkeypatch.setattr(collectors, "process_epoch", lambda pid: "epoch")

    async def spawn(*args, **kwargs):
        commands.append(args)
        output = Path(args[args.index("--output") + 1])
        output.write_bytes(b"perf-data")
        control_read, ack_write = kwargs["pass_fds"]
        # Duplicating mimics inherited subprocess descriptors.
        read_ends.extend([os.dup(control_read), os.dup(ack_write)])
        os.write(read_ends[1], b"ack\n")
        return recorder

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    adapter = PerfCollector(123, "epoch", tmp_path)
    try:
        await adapter.start()
        await adapter.stop()
    finally:
        for descriptor in read_ends:
            os.close(descriptor)
    assert commands[0][commands[0].index("--pid") + 1] == "123"
    assert "--delay=-1" in commands[0]
    assert recorder.signals == [collectors.signal.SIGINT]
    assert adapter.evidence["coverage"] == "captured_unverified"


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["error", "timeout"])
async def test_finalization_failure_blocks_new_capture(controller, collector, failure):
    async def hang():
        await asyncio.Event().wait()

    controller.config = msgspec.structs.replace(
        controller.config, finalize_timeout_s=0.01
    )
    collector.stop.side_effect = (
        CollectionError("flush failed") if failure == "error" else hang
    )
    session = controller.submit(plan())
    await asyncio.wait_for(session.task, 1)
    assert session.state == "failed"
    assert collector.evidence["coverage"] == "partial_or_unknown"
    assert ("flush failed" if failure == "error" else "TimeoutError") in session.errors
    assert controller.active is None
    with pytest.raises(errors.Conflict, match="cleanup is uncertain"):
        controller.submit(plan("retry-after-stop-failure"))


@pytest.mark.asyncio
async def test_frontend_epoch_checked_before_perf_spawn(tmp_path, monkeypatch):
    monkeypatch.setattr(collectors, "process_epoch", lambda pid: "replacement-epoch")
    adapter = PerfCollector(123, "old-epoch", tmp_path)
    with pytest.raises(CollectionError, match="epoch changed"):
        await adapter.start()
    assert adapter.process is None
    assert adapter.control == -1
    await adapter.stop()


@pytest.mark.asyncio
async def test_manifest_write_failure_cannot_prevent_collector_cleanup(
    controller, collector, monkeypatch
):
    session = controller.submit(plan())
    original_persist = controller.persist

    def full_disk(session):
        if session.collectors:
            raise OSError("disk full")
        original_persist(session)

    monkeypatch.setattr(controller, "persist", full_disk)
    await session.task
    assert session.state == "failed"
    collector.stop.assert_awaited_once()
    assert controller.active is None
    assert any("disk full" in error for error in session.errors)


def test_process_epochs_distinguish_same_tick_replicas_and_pid_reuse(monkeypatch):
    starts = iter((42, 42, 43))

    def read_proc(path):
        if path.name == "boot_id":
            return "host-boot\n"
        # The comm field may itself contain spaces and parentheses.
        return "123 (worker (replica)) " + " ".join(
            ["S", *(["0"] * 18), str(next(starts))]
        )

    monkeypatch.setattr(Path, "read_text", read_proc)
    first = collectors.process_epoch(123)
    sibling = collectors.process_epoch(124)
    replacement = collectors.process_epoch(123)
    assert first != sibling
    assert first != replacement


RANKS = ("dp0_pp0_tp0_dcp0_ep0_rank0", "dp1_pp0_tp0_dcp0_ep1_rank0")


@pytest.fixture
def socket_directory():
    # pytest's nested temp path can exceed AF_UNIX's platform length limit.
    with tempfile.TemporaryDirectory(prefix="dyn-prof-") as directory:
        yield Path(directory)


def trace(path):
    payload = {
        "traceEvents": [
            {"name": "test_kernel", "cat": "kernel", "ph": "X", "ts": 1, "dur": 2}
        ]
    }
    path.write_bytes(gzip.compress(json.dumps(payload).encode()))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case", ["complete", "missing", "corrupt", "stale_epoch", "unknown_membership"]
)
async def test_worker_capture_identity_clock_and_rank_validation(
    tmp_path, monkeypatch, case
):
    monkeypatch.setattr(
        agent,
        "process_epoch",
        lambda pid: "new-epoch" if case == "stale_epoch" else "epoch",
    )
    worker_agent = agent.ProfilingAgent("vllm")
    calls = []
    unselected_calls = []
    options = {}
    outputs = tmp_path / "native"
    outputs.mkdir()
    # A valid old trace from the missing rank must never satisfy this capture.
    trace(outputs / f"old_{RANKS[1]}.1.pt.trace.json.gz")

    async def start(request):
        options.update(await request.json())
        calls.append("start")
        return web.Response(text='{"status":"ok"}', content_type="text/plain")

    async def stop(request):
        calls.append("stop")
        for rank in RANKS[:1] if case == "missing" else RANKS:
            path = outputs / f"{options['profile_prefix']}_{rank}.2.pt.trace.json.gz"
            trace(path)
            if case == "corrupt" and rank == RANKS[1]:
                path.write_bytes(path.read_bytes()[:-5])
        return web.Response(text='{"status":"ok"}', content_type="text/plain")

    async def info(request):
        return web.json_response(await worker_agent.info(await request.json()))

    async def unselected(request):
        unselected_calls.append(request.path)
        return web.json_response({"status": "ok"})

    app = web.Application()
    app.router.add_post("/engine/control/start_profile", start)
    app.router.add_post("/engine/control/stop_profile", stop)
    app.router.add_post("/engine/control/profiling_info", info)
    app.router.add_post("/unselected/engine/control/{action}", unselected)
    async with TestServer(app) as server:
        worker = Worker(
            "decode",
            "epoch",
            "vllm",
            str(server.make_url("/")),
            artifact_location=str(outputs),
            verify_runtime=True,
            expected_rank_ids=[] if case == "unknown_membership" else list(RANKS),
        )
        config = Config(
            str(tmp_path / "admin.sock"),
            str(tmp_path / "sessions"),
            frontend_perf_enabled=False,
            lifecycle_evidence_location="otel-backend://existing",
        )
        controller = Coordinator(
            config,
            RuntimeSnapshot(
                123,
                "frontend-epoch",
                "frontend-host",
                [
                    worker,
                    Worker(
                        "unselected",
                        "other-epoch",
                        "vllm",
                        str(server.make_url("/unselected")),
                    ),
                ],
            ),
        )
        session = controller.submit(
            Plan(
                "capture",
                [Target("decode", "epoch")],
                5000,
                trace_ids=["trigger-trace"],
                request_ids=["trigger-request"],
            )
        )

        async def wait_recording():
            while session.state in ("accepted", "starting"):
                await asyncio.sleep(0)
            assert session.state == "recording"

        if case != "stale_epoch":
            await asyncio.wait_for(wait_recording(), 5)
            controller.stop(session.session_id)
        await asyncio.wait_for(session.task, 5)
    manifest = controller.manifest(session)
    assert controller.active is None
    assert unselected_calls == []
    assert manifest["lifecycle_otel"]["trace_ids"] == ["trigger-trace"]
    assert manifest["lifecycle_otel"]["coverage"] == "unknown"
    if case == "stale_epoch":
        assert session.state == "failed"
        assert calls == []
        assert "identity changed" in session.errors[0]
        return
    assert calls == ["start", "stop"]
    assert options["profile_prefix"] == f"dynamo-{session.session_id}"
    clocks = manifest["clock"]["workers"]["decode"]
    assert clocks["start"]["alignment"] == clocks["end"]["alignment"] == "estimated"
    assert clocks["start"]["uncertainty_ns"] >= 0
    assert set(manifest["clock"]["workers"]) == {"decode"}
    assert [item["target_id"] for item in manifest["profiles"]] == ["decode"]
    assert manifest["profiles"][0]["identity_verification"] == "control_process_epoch"
    validation = manifest["artifact_validation"][0]
    assert all("old_" not in item["location"] for item in validation["artifacts"])
    if case == "complete":
        assert session.state == "completed"
        assert manifest["artifact_status"] == "verified"
        assert validation["coverage"] == "complete"
        assert len(validation["artifacts"]) == 2
        for item in validation["artifacts"]:
            assert item["events"] == item["cuda_kernels"] == 1
            raw = gzip.decompress(Path(item["location"]).read_bytes())
            assert item["sha256_uncompressed"] == hashlib.sha256(raw).hexdigest()
    elif case == "unknown_membership":
        assert session.state == "completed"
        assert manifest["artifact_status"] == "unverified"
        assert validation["coverage"] == "unknown"
    else:
        assert session.state == "failed"
        assert manifest["artifact_status"] == "partial"
        if case == "missing":
            assert validation["missing_rank_ids"] == [RANKS[1]]
        else:
            assert validation["artifacts"][1]["validation"] == "invalid"


def test_clock_offset_bound_and_clock_step_rejection():
    t1 = 1_000_000_000
    offset = 5_000_000
    sample = estimate_offset(
        t1,
        t1 + offset + 800_000,
        t1 + offset + 1_000_000,
        t1 + 2_000_000,
        2_000_000,
        200_000,
    )
    assert abs(sample["offset_ns"] - offset) <= sample["uncertainty_ns"]
    assert sample["uncertainty_ns"] == 900_000
    with pytest.raises(CollectionError, match="Clock changed"):
        estimate_offset(
            t1, t1 + offset, t1 + offset + 200_000, t1 + 2_000_000, 20_000_000, 200_000
        )


@pytest.mark.asyncio
async def test_unmounted_store_remains_unverified(tmp_path):
    result = await SharedFilesystemValidator().validate(
        {
            "collector": "torch_profiler",
            "artifact_prefix": "capture",
            "artifact_location": str(tmp_path / "unmounted"),
        },
        Config("socket", str(tmp_path)),
    )
    assert result["status"] == "unverified"
    assert result["reason"] == "artifact_store_not_mounted"


@pytest.mark.asyncio
async def test_trace_validation_limits_and_duplicate_rank_files(tmp_path):
    for suffix in ("1", "2"):
        trace(tmp_path / f"capture_{RANKS[0]}.{suffix}.pt.trace.json.gz")
    evidence = {
        "collector": "torch_profiler",
        "artifact_location": str(tmp_path),
        "artifact_prefix": "capture",
        "expected_rank_ids": [RANKS[0]],
    }
    validator = SharedFilesystemValidator()
    result = await validator.validate(evidence, Config("socket", str(tmp_path)))
    assert result["status"] == "invalid"
    assert result["duplicate_rank_files"] is True
    result = await validator.validate(
        evidence, Config("socket", str(tmp_path), max_trace_bytes=10)
    )
    assert all(item["validation"] == "invalid" for item in result["artifacts"])
    assert "byte limit" in result["artifacts"][0]["error"]


@pytest.mark.asyncio
async def test_embedded_host_uses_pluggable_coordinator_and_drains(
    socket_directory, provider_controller, collector
):
    controller = provider_controller
    config = msgspec.structs.replace(
        controller.config,
        socket_path=str(socket_directory / "admin.sock"),
        finalize_timeout_s=10.0,
    )
    controller.config = config
    async with host(controller) as embedded:
        assert embedded is controller
        assert Path(config.socket_path).stat().st_mode & 0o777 == 0o600
        async with aiohttp.ClientSession(
            connector=aiohttp.UnixConnector(path=config.socket_path)
        ) as client:
            response = await client.post(
                "http://localhost/admin/profiling/v1/sessions",
                json={
                    "client_request_id": "plugin",
                    "mode": "nsys",
                    "targets": [{"target_id": "frontend", "process_epoch": "epoch"}],
                    "duration_ms": 5000,
                },
            )
            assert response.status == 202
            session_id = (await response.json())["session_id"]
            conflict = await client.post(
                "http://localhost/admin/profiling/v1/sessions",
                json={
                    "client_request_id": "other",
                    "mode": "nsys",
                    "targets": [{"target_id": "frontend", "process_epoch": "epoch"}],
                    "duration_ms": 5000,
                },
            )
            assert conflict.status == 409
        await asyncio.wait_for(collector.started.wait(), 1)
    collector.stop.assert_awaited_once()
    assert controller.lookup(session_id).state == "completed"
    assert not Path(config.socket_path).exists()
    with pytest.raises(Unavailable):
        controller.submit(Plan("new", [Target("frontend", "epoch")], 50, mode="nsys"))


@pytest.mark.asyncio
async def test_frontend_passes_runtime_snapshot_without_spawning_helper(
    tmp_path, socket_directory, monkeypatch
):
    monkeypatch.setattr(hosting, "process_epoch", lambda pid: "frontend-epoch")

    async def forbidden_spawn(*args, **kwargs):
        raise AssertionError("Embedded hosting must not spawn a helper")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", forbidden_spawn)
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "socket_path": str(socket_directory / "admin.sock"),
                "output_dir": str(tmp_path / "sessions"),
                "frontend_perf_enabled": False,
            }
        )
    )
    context = {"input_type": "http"}
    async with profiling_coordinator(str(path), runtime_context=context) as coordinator:
        assert coordinator.runtime.frontend_pid == os.getpid()
        assert coordinator.runtime.frontend_epoch == "frontend-epoch"
        assert coordinator.runtime.context == context
        assert coordinator.targets()[0]["collectors"] == []
    assert coordinator.closed


@pytest.mark.asyncio
async def test_callable_api_owns_runtime_and_manifest_snapshots(
    tmp_path, provider_controller, collector
):
    controller = provider_controller
    config = Config("unused", str(tmp_path))
    worker = Worker(
        "decode",
        "epoch",
        "vllm",
        "http://configured",
        {"option": [1]},
        expected_rank_ids=[RANKS[0]],
    )
    runtime = RuntimeSnapshot(123, "epoch", "host", [worker], {"labels": ["accepted"]})
    controller = Coordinator(config, runtime, registry=provider_controller.registry)
    runtime.workers.clear()
    runtime.context["labels"].append("caller-mutated")
    worker.start_options["option"].append(2)
    inventory = controller.targets()
    inventory[1]["expected_rank_ids"].clear()
    assert controller.workers["decode"].expected_rank_ids == [RANKS[0]]
    assert controller.workers["decode"].start_options == {"option": [1]}
    original_plan = Plan(
        "snapshot",
        [Target("frontend", "epoch")],
        5000,
        mode="nsys",
        trace_ids=["trace"],
    )
    session = controller.submit(original_plan)
    original_plan.trace_ids.append("caller-mutated")
    controller.runtime.context["labels"].append("later-context")
    await asyncio.wait_for(collector.started.wait(), 1)
    manifest = controller.manifest(session)
    manifest["profiles"][0]["target_id"] = "caller-mutated"
    manifest["runtime"]["context"]["labels"].clear()
    manifest["lifecycle_otel"]["trace_ids"].clear()
    await controller.close()
    final = controller.manifest(session)
    assert final["profiles"][0]["target_id"] == "frontend"
    assert final["runtime"]["context"] == {"labels": ["accepted"]}
    assert final["lifecycle_otel"]["trace_ids"] == ["trace"]
    assert session.plan.trace_ids == ["trace"]
    assert (
        final["timing_contract"]["duration_semantics"]
        == "best_effort_lease_from_acceptance"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case", ["observed", "no_marker", "duplicate_marker", "no_base"]
)
async def test_native_recording_bounds_preserve_nanosecond_precision(tmp_path, case):
    marker = '{"ph":"X","cat":"Trace","name":"PyTorch Profiler (0)","ts":714173694365.002,"dur":9800295.631}'
    events = ['{"ph":"X","cat":"kernel","name":"test_kernel","ts":1,"dur":2}']
    if case != "no_marker":
        events.append(marker)
    if case == "duplicate_marker":
        events.append(marker)
    base = '"baseTimeNanoseconds":1790857026000000000,' if case != "no_base" else ""
    (tmp_path / f"capture_{RANKS[0]}.1.pt.trace.json").write_text(
        "{" + base + '"traceEvents":[' + ",".join(events) + "]}"
    )
    result = await SharedFilesystemValidator().validate(
        {
            "collector": "torch_profiler",
            "artifact_prefix": "capture",
            "artifact_location": str(tmp_path),
            "expected_rank_ids": [RANKS[0]],
        },
        Config("unused", str(tmp_path)),
    )
    assert result["status"] == "verified"
    bounds = result["artifacts"][0]["recording_bounds"]
    if case == "observed":
        assert bounds == {
            "status": "observed",
            "clock": "worker_trace_unix_ns_unattested",
            "start_unix_ns": 1791571199694365002,
            "end_unix_ns": 1791571209494660633,
            "duration_ns": 9800295631,
        }
    else:
        assert bounds["status"] == "unknown"


def nsys_report(path, activity="cpu", processes=1):
    path.write_bytes(b"native-report-fixture")
    with sqlite3.connect(path.with_suffix(".sqlite")) as conn:
        conn.execute("CREATE TABLE META_DATA_EXPORT (name TEXT, value TEXT)")
        conn.executemany(
            "INSERT INTO META_DATA_EXPORT VALUES (?, ?)",
            [("EXPORT_TIME", "0.01 sec"), ("EXPORT_PARAM_INPUT_FILE", path.name)],
        )
        if activity == "cpu":
            conn.execute(
                "CREATE TABLE COMPOSITE_EVENTS (start INTEGER, cpuCycles INTEGER, globalTid INTEGER)"
            )
            conn.execute(
                "INSERT INTO COMPOSITE_EVENTS VALUES (1, 1, ?)", (os.getpid() << 24,)
            )
        else:
            conn.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (globalPid INTEGER)")
            conn.executemany(
                "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?)",
                [(pid,) for pid in range(processes)],
            )


@pytest.mark.asyncio
async def test_nsys_repeated_client_capture_without_subprocess(tmp_path, monkeypatch):
    pending = []
    handles = []
    prefix = str(tmp_path / "frontend")

    def begin(**kwargs):
        assert kwargs == {"message": "ROI", "domain": "app"}
        handle = object()
        handles.append(handle)
        return handle

    def end(handle):
        assert handle is handles[-1]

        async def export():
            await asyncio.sleep(0.02)
            nsys_report(tmp_path / f"frontend.{len(handles)}.nsys-rep")
            if len(handles) == 1:
                nsys_report(tmp_path / "frontend-other.1.nsys-rep")

        pending.append(asyncio.create_task(export()))

    async def forbidden(*args, **kwargs):
        raise AssertionError("Nsight runtime control cannot invoke subprocesses")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", forbidden)
    monkeypatch.setattr(nsys, "nvtx", SimpleNamespace(start_range=begin, end_range=end))
    monkeypatch.setattr(nsys, "process_epoch", lambda pid: "epoch")
    # Existing output from the same persistent launch cannot satisfy a new capture.
    nsys_report(tmp_path / "frontend.0.nsys-rep")
    coordinator = Coordinator(
        Config(
            "unused",
            str(tmp_path / "sessions"),
            frontend_perf_enabled=False,
            nsys_frontend_output_prefix=prefix,
            artifact_validation_timeout_s=2,
        ),
        RuntimeSnapshot(os.getpid(), "epoch", "host"),
    )
    async with TestClient(TestServer(application(coordinator))) as client:
        for window in range(3):
            response = await client.post(
                "/admin/profiling/v1/sessions",
                json={
                    "client_request_id": str(window),
                    "mode": "nsys",
                    "duration_ms": 5000,
                    "targets": [{"target_id": "frontend", "process_epoch": "epoch"}],
                },
            )
            assert response.status == 202
            session = coordinator.lookup((await response.json())["session_id"])
            while session.state != "recording":
                await asyncio.sleep(0)
            await client.post(f"/admin/profiling/v1/sessions/{session.session_id}/stop")
            await session.task
            manifest = coordinator.manifest(session)
            assert session.state == "completed", manifest
            assert manifest["frontend"]["profiling_enabled"] is True
            result = manifest["artifact_validation"][0]
            assert result["status"] == "verified"
            assert result["artifacts"][0]["location"].endswith(
                f"frontend.{window+1}.nsys-rep"
            )
            assert (
                result["artifacts"][0]["sha256"]
                == hashlib.sha256(b"native-report-fixture").hexdigest()
            )
    await asyncio.gather(*pending)
    assert len(handles) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["vllm", "sglang"])
async def test_nsys_worker_uses_backend_controls(tmp_path, monkeypatch, backend):
    requests = AsyncMock()
    monkeypatch.setattr(TorchCollector, "_request", requests)
    worker = Worker(
        "decode",
        "epoch",
        backend,
        "http://worker",
        nsys_output_prefix=str(tmp_path / "worker"),
    )
    config = Config("unused", str(tmp_path), frontend_perf_enabled=True)
    runtime = RuntimeSnapshot(123, "frontend", "host", [worker])
    selected = nsys.nsys_collectors(
        config,
        runtime,
        Plan("id", [Target("decode", "epoch")], 50, mode="nsys"),
        tmp_path,
    )
    assert len(selected) == 1  # no implicit frontend/perf capture in Nsight mode
    await selected[0].start()
    await selected[0].stop()
    assert requests.await_args_list[0].args == (
        "start",
        {"activities": ["CUDA_PROFILER"]} if backend == "sglang" else {},
        config.start_timeout_s,
    )
    assert requests.await_args_list[1].args == ("stop", {}, config.finalize_timeout_s)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "defect",
    [
        "missing_gpu",
        "missing_rank",
        "ambiguous",
        "corrupt",
        "incomplete",
        "missing_cpu",
    ],
)
async def test_nsys_rejects_incomplete_or_ambiguous_exports(tmp_path, defect):
    path = tmp_path / "worker.1.nsys-rep"
    nsys_report(path, "cpu" if defect == "missing_gpu" else "gpu")
    if defect == "ambiguous":
        nsys_report(tmp_path / "worker.2.nsys-rep", "gpu")
    if defect == "incomplete":
        with sqlite3.connect(path.with_suffix(".sqlite")) as conn:
            conn.execute("DELETE FROM META_DATA_EXPORT WHERE name = 'EXPORT_TIME'")
    if defect == "corrupt":
        path.with_suffix(".sqlite").write_bytes(b"not a database")
    result = await SharedFilesystemValidator().validate(
        {
            "collector": "nsys",
            "output_prefix": str(tmp_path / "worker"),
            "required_activity": (
                "cpu_events" if defect == "missing_cpu" else "cuda_events"
            ),
            "expected_gpu_processes": 2 if defect == "missing_rank" else 1,
            "previous_reports": [],
        },
        Config("unused", str(tmp_path), artifact_validation_timeout_s=0.03),
    )
    assert result["status"] in ("partial", "invalid")
    assert result["error"]


def test_vllm_nsight_launch_rejects_torch_plan(tmp_path):
    worker = Worker(
        "decode",
        "epoch",
        "vllm",
        "http://worker",
        nsys_output_prefix=str(tmp_path / "worker"),
    )
    coordinator = Coordinator(
        Config("unused", str(tmp_path)),
        RuntimeSnapshot(123, "frontend", "host", [worker]),
    )
    assert coordinator.targets()[1]["collectors"] == ["nsys"]
    with pytest.raises(errors.Unsupported, match="separate launch"):
        coordinator.submit(Plan("id", [Target("decode", "epoch")], 50))
    assert not coordinator.sessions


@pytest.mark.asyncio
@pytest.mark.parametrize("with_kernels", [False, True])
async def test_nsys_accepts_cuda_memory_activity(tmp_path, with_kernels):
    # A decoder waiting for KV transfer can perform copies without launching
    # kernels. A process appearing in both tables must still count as one rank.
    path = tmp_path / "worker.1.nsys-rep"
    nsys_report(path, "gpu" if with_kernels else "cpu")
    with sqlite3.connect(path.with_suffix(".sqlite")) as conn:
        conn.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(globalPid INTEGER)")
        conn.execute("INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (0)")
    result = await SharedFilesystemValidator().validate(
        {
            "collector": "nsys",
            "output_prefix": str(tmp_path / "worker"),
            "previous_reports": [],
            "required_activity": "cuda_events",
            "expected_gpu_processes": 1,
        },
        Config("unused", str(tmp_path), artifact_validation_timeout_s=0.05),
    )
    assert result["status"] == "verified"
    assert result["events"]["CUPTI_ACTIVITY_KIND_MEMCPY"] == 1
    assert result["gpu_processes"] == [0]


def test_bundle_writes_per_request_deployment_timelines(tmp_path):
    base = 1_700_000_000_000_000_000
    trace = tmp_path / "native" / "capture_dp0.1.pt.trace.json.gz"
    trace.parent.mkdir()
    kernels = [
        {
            "ph": "X",
            "cat": "kernel",
            "name": name,
            "pid": 7,
            "tid": 1,
            "ts": ts,
            "dur": 2,
        }
        for name, ts in (("before_operation", 1000), ("inside_operation", 2500.5))
    ]
    metadata = [
        {
            "ph": "M",
            "name": "process_name",
            "pid": 7,
            "tid": 0,
            "args": {"name": "EngineCore"},
        },
        {
            "ph": "M",
            "name": "process_labels",
            "pid": 7,
            "tid": 0,
            "args": {"labels": "CPU"},
        },
    ]
    native = {"baseTimeNanoseconds": base, "traceEvents": metadata + kernels}
    trace.write_bytes(gzip.compress(json.dumps(native).encode()))
    session = tmp_path / "session"
    session.mkdir()
    bounds = {
        "status": "observed",
        "start_unix_ns": base + 1_000_000,
        "end_unix_ns": base + 4_000_000,
    }
    clock = {"alignment": "estimated", "offset_ns": 1000, "hostname": "h", "pid": 42}
    manifest = {
        "session_id": "s",
        "state": "completed",
        "artifact_status": "verified",
        "plan": {"trace_ids": ["trigger"]},
        "profiles": [
            {
                "target_id": "decode-0",
                "collector": "torch_profiler",
                "started_at_ns": base + 900_000,
            }
        ],
        "clock": {"workers": {"decode-0": {"start": clock}}},
        "artifact_validation": [
            {
                "target_id": "decode-0",
                "status": "verified",
                "artifacts": [
                    {
                        "location": str(trace),
                        "rank_id": "dp0",
                        "recording_bounds": bounds,
                    }
                ],
            }
        ],
    }
    (session / "manifest.json").write_text(json.dumps(manifest))

    def span(trace_id, span_id, parent, name, start, **attributes):
        return {
            "traceId": trace_id,
            "spanId": span_id,
            "parentSpanId": parent,
            "name": name,
            "startTimeUnixNano": str(base + start),
            "endTimeUnixNano": str(base + start + 3_000_000),
            "attributes": [
                {"key": k, "value": {"stringValue": str(v)}}
                for k, v in attributes.items()
            ],
        }

    def phase(trace_id, prefix, name, worker, pid, start):
        # Span host attributes are not worker host identities; only the PID joins.
        epoch = {"dynamo.process.epoch": f"{pid}:x", "dynamo.instance.id": "other"}
        return [
            span(
                trace_id,
                prefix + "r",
                "a",
                "kv_router.route_request",
                start,
                phase=name.title(),
                worker_id=worker,
            ),
            span(
                trace_id,
                prefix + "h",
                prefix + "r",
                "handle_payload",
                start,
                instance_id=worker,
            ),
            span(
                trace_id,
                prefix + "o",
                prefix + "h",
                f"worker.operation.{name}",
                start,
                **epoch,
            ),
        ]

    def request(trace_id, decode_worker, decode_pid, start):
        return (
            [span(trace_id, "a", "", "http-request", start)]
            + phase(trace_id, "p", "prefill", "W3", 99, start)
            + phase(trace_id, "d", "decode", decode_worker, decode_pid, start)
        )

    otel = tmp_path / "otel.jsonl"
    spans = (
        request("captured", "W1", 42, 2_000_000)
        + request("uncaptured", "W2", 43, 2_000_000)
        + request("after", "W1", 42, 9_000_000)
    )
    otel.write_text(
        json.dumps({"resourceSpans": [{"scopeSpans": [{"spans": spans}]}]}) + "\n"
    )
    summary = bundle.build(session, [otel], tmp_path / "out")
    index = json.loads((tmp_path / "out/lifecycle/trace-index.json").read_text())
    assert index["router_targets"] == {"W1": "decode-0"}
    assert [entry["trace_id"] for entry in index["requests"]] == ["captured"]
    decode = index["requests"][0]["phases"][1]
    assert (decode["target_id"], decode["recording_overlap_ns"]) == (
        "decode-0",
        2_000_000,
    )
    exported = (tmp_path / "out/lifecycle/traces.otlp.jsonl").read_text()
    assert "captured" in exported and "uncaptured" not in exported
    assert summary["coverage"]["lifecycle"]["trigger_traces_found"] == "0/1"
    assert summary["coverage"]["native_timeline"]["decode-0"]["events_written"] == 1
    timeline = tmp_path / "out" / index["requests"][0]["timeline"]
    events = json.loads(gzip.decompress(timeline.read_bytes()))["traceEvents"]
    order = {
        e["pid"]: e["args"]["sort_index"]
        for e in events
        if e.get("name") == "process_sort_index"
    }
    names = {
        e["pid"]: e["args"]["name"] for e in events if e.get("name") == "process_name"
    }
    assert [names[pid] for pid in sorted(order, key=order.get)] == [
        "Request · OTel spans",
        "Recording windows",
        "decode-0 · EngineCore (CPU)",
        "Control",
    ]
    lanes = {e["name"]: e["args"]["process"] for e in events if e.get("cat") == "otel"}
    assert lanes["worker.operation.prefill"] == "prefill worker W3 (not captured)"
    assert lanes["worker.operation.decode"] == "decode-0"
    assert lanes["kv_router.route_request"] == "frontend"
    assert [e["name"] for e in events if e.get("cat") == "kernel"] == [
        "inside_operation"
    ]
    kernel = next(e for e in events if e.get("cat") == "kernel")
    # Raw base+2.5005 ms on the worker clock, minus its 1 us offset, from the
    # earliest control event at base+0.9 ms.
    assert kernel["ts"] == 1599.5
    assert all(isinstance(e.get("tid", 0), int) for e in events)


def test_bundle_reads_nsight_frontend_and_worker_exports(tmp_path):
    base, clock = 1_700_000_000_000_000_000, 5_000_000_000
    # Nsight event timestamps are relative to the session start (utcEpochNs).

    def export(name, rows):
        path = tmp_path / f"{name}.sqlite"
        with sqlite3.connect(path) as db:
            db.execute(
                "CREATE TABLE TARGET_INFO_SESSION_START_TIME (utcEpochNs, systemClockNs)"
            )
            db.execute(
                "INSERT INTO TARGET_INFO_SESSION_START_TIME VALUES (?, ?)",
                (base, clock),
            )
            db.execute("CREATE TABLE StringIds (id, value)")
            db.executemany(
                "INSERT INTO StringIds VALUES (?, ?)", [(1, "epoll_wait"), (2, "gemm")]
            )
            db.execute("CREATE TABLE PROCESSES (globalPid, pid, name)")
            db.execute("INSERT INTO PROCESSES VALUES (0, 42, 'VLLM::EngineCore')")
            for table, columns, values in rows:
                db.execute(f"CREATE TABLE {table} ({columns})")
                db.executemany(
                    f"INSERT INTO {table} VALUES ({','.join('?' * len(values[0]))})",
                    values,
                )
        (tmp_path / f"{name}.nsys-rep").write_bytes(b"report")
        return path

    tid = (42 << 24) | 7
    frontend = export(
        "frontend",
        [
            (
                "OSRT_API",
                "start, end, nameId, globalTid",
                [(2_500_000, 2_600_000, 1, tid)],
            )
        ],
    )
    worker = export(
        "worker",
        [
            (
                "NVTX_EVENTS",
                "start, end, eventType, text, textId, globalTid, domainId",
                [
                    (2_400_000, 2_700_000, 59, "forward", None, tid, 1),
                    (0, None, 75, "PyTorch", None, 0, 1),
                ],
            ),
            (
                "CUPTI_ACTIVITY_KIND_KERNEL",
                "start, end, demangledName, deviceId, streamId",
                [
                    (1_000_000, 1_100_000, 2, 0, 7),
                    (2_500_000, 2_600_000, 2, 0, 7),
                ],
            ),
        ],
    )
    bounds = {
        "status": "observed",
        "start_unix_ns": base + 1_000_000,
        "end_unix_ns": base + 4_000_000,
    }
    session = tmp_path / "session"
    session.mkdir()
    (session / "manifest.json").write_text(
        json.dumps(
            {
                "session_id": "s",
                "state": "completed",
                "artifact_status": "verified",
                "plan": {"trace_ids": []},
                "profiles": [
                    {"target_id": t, "collector": "nsys"}
                    for t in ("frontend", "decode-0")
                ],
                "clock": {
                    "workers": {
                        "decode-0": {
                            "start": {
                                "alignment": "estimated",
                                "offset_ns": 0,
                                "pid": 42,
                            }
                        }
                    }
                },
                "artifact_validation": [
                    {
                        "target_id": t,
                        "status": "verified",
                        "recording_bounds": bounds,
                        "artifacts": [
                            {"location": str(path.with_suffix(".nsys-rep"))},
                            {"location": str(path)},
                        ],
                    }
                    for t, path in (("frontend", frontend), ("decode-0", worker))
                ],
            }
        )
    )

    def span(span_id, parent, name, **attributes):
        return {
            "traceId": "t",
            "spanId": span_id,
            "parentSpanId": parent,
            "name": name,
            "startTimeUnixNano": str(base + 2_000_000),
            "endTimeUnixNano": str(base + 3_000_000),
            "attributes": [
                {"key": k, "value": {"stringValue": str(v)}}
                for k, v in attributes.items()
            ],
        }

    spans = [
        span("a", "", "http-request"),
        span("r", "a", "kv_router.route_request", phase="Decode", worker_id="W"),
        span("h", "r", "handle_payload", instance_id="W"),
        span("o", "h", "worker.operation.decode", **{"dynamo.process.epoch": "42:x"}),
    ]
    otel = tmp_path / "otel.jsonl"
    otel.write_text(
        json.dumps({"resourceSpans": [{"scopeSpans": [{"spans": spans}]}]}) + "\n"
    )
    bundle.build(session, [otel], tmp_path / "out")
    index = json.loads((tmp_path / "out/lifecycle/trace-index.json").read_text())
    path = tmp_path / "out" / index["requests"][0]["timeline"]
    events = json.loads(gzip.decompress(path.read_bytes()))["traceEvents"]
    order = {
        e["pid"]: e["args"]["sort_index"]
        for e in events
        if e.get("name") == "process_sort_index"
    }
    names = {
        e["pid"]: e["args"]["name"] for e in events if e.get("name") == "process_name"
    }
    assert [names[pid] for pid in sorted(order, key=order.get)][:5] == [
        "Request · OTel spans",
        "Recording windows",
        "frontend · VLLM::EngineCore (CPU)",
        "decode-0 · VLLM::EngineCore (CPU)",
        "decode-0 · CUDA device 0 (GPU 0)",
    ]
    native = sorted(
        (e["cat"], e["name"])
        for e in events
        if e.get("cat") in ("osrt", "pytorch", "kernel")
    )
    # The kernel before the decode operation is excluded; the frontend
    # OS-runtime call is included within the request's lifetime.
    assert native == [
        ("kernel", "gemm"),
        ("osrt", "epoll_wait"),
        ("pytorch", "forward"),
    ]
