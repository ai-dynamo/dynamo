# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Control plane tests without GPU engines or a privileged perf installation."""

import asyncio
import json
import os
from pathlib import Path

import msgspec
import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from dynamo.frontend.profiling.http import application
from dynamo.profiling import adapters, collectors
from dynamo.profiling import coordinator as service
from dynamo.profiling import errors
from dynamo.profiling.collectors import CollectionError, PerfCollector, TorchCollector
from dynamo.profiling.models import Config, Plan, RuntimeSnapshot, Target, Worker

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


@pytest.fixture
def controller(tmp_path, monkeypatch):
    class FakePerf:
        def __init__(self, pid, epoch, directory):
            self.evidence = {"target_id": "frontend", "collector": "linux_perf"}

        async def start(self):
            self.evidence["started"] = True

        async def stop(self):
            self.evidence["stopped"] = True

    monkeypatch.setattr(adapters, "PerfCollector", FakePerf)
    return service.Coordinator(
        Config(str(tmp_path / "admin.sock"), str(tmp_path / "artifacts")),
        RuntimeSnapshot(123, "frontend-epoch", "test-host"),
    )


def plan(request_id="request-1", duration_ms=50):
    return Plan(request_id, [Target("frontend", "frontend-epoch")], duration_ms)


@pytest.mark.asyncio
async def test_deadline_and_artifact_manifest(controller):
    session = controller.submit(plan())
    await session.task
    assert session.state == "completed"
    assert session.collectors[0].evidence["stopped"] is True
    manifest = json.loads((session.directory / "manifest.json").read_text())
    assert manifest["profiles"][0]["collector"] == "linux_perf"
    assert manifest["lifecycle_otel"]["coverage"] == "unknown"
    assert manifest["lifecycle_otel"]["instrumentation_mode"] == "unchanged"
    assert controller.active is None


@pytest.mark.asyncio
async def test_retry_reuses_session_and_conflicting_plan_rejected(controller):
    session = controller.submit(plan())
    assert controller.submit(plan()) is session
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
async def test_uncertain_start_rolls_back_and_reports_failure(controller, monkeypatch):
    class FailingPerf:
        def __init__(self, *args):
            self.evidence = {"collector": "linux_perf"}

        async def start(self):
            raise CollectionError("permission denied")

        async def stop(self):
            self.evidence["stopped"] = True

    monkeypatch.setattr(adapters, "PerfCollector", FailingPerf)
    session = controller.submit(plan())
    await session.task
    assert session.state == "failed"
    assert session.collectors[0].evidence["stopped"]
    assert "permission denied" in session.errors


@pytest.mark.asyncio
async def test_worker_only_plan_still_profiles_frontend_and_stop_is_concurrent(
    controller, monkeypatch
):
    controller.workers["worker"] = Worker(
        "worker", "worker-epoch", "vllm", "http://configured"
    )
    stop_entered = asyncio.Event()
    finish_export = asyncio.Event()

    class SlowTorch:
        def __init__(self, worker, config):
            self.evidence = {"collector": "torch_profiler"}

        async def start(self):
            pass

        async def stop(self):
            stop_entered.set()
            await finish_export.wait()

    monkeypatch.setattr(adapters, "TorchCollector", SlowTorch)
    session = controller.submit(Plan("id", [Target("worker", "worker-epoch")], 50))
    await asyncio.wait_for(stop_entered.wait(), 1)
    assert session.state == "finalizing"
    assert session.collectors[0].evidence["stopped"]
    assert controller.active is session
    finish_export.set()
    await session.task


@pytest.mark.asyncio
async def test_rest_validation_stop_and_unknown_fields(controller):
    async with TestClient(TestServer(application(controller))) as client:
        targets = await client.get("/admin/profiling/v1/targets")
        assert (await targets.json())["targets"][0]["target_id"] == "frontend"
        payload = msgspec.to_builtins(plan(duration_ms=5000))
        validation = await client.post(
            "/admin/profiling/v1/plans/validate", json=payload
        )
        assert (await validation.json())["frontend_collector"] == "linux_perf"
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
        assert (await artifacts.json())["state"] == "completed"
        missing = await client.get("/admin/profiling/v1/sessions/missing")
        assert missing.status == 404


@pytest.mark.asyncio
async def test_explicit_worker_only_override_skips_perf_and_records_omission(
    tmp_path, monkeypatch
):
    def forbidden_perf(*args):
        raise AssertionError("Disabled frontend perf must never be instantiated")

    calls = []

    class WorkerCollector:
        def __init__(self, worker, config):
            self.evidence = {
                "target_id": worker.target_id,
                "collector": "torch_profiler",
            }

        async def start(self):
            calls.append("start")

        async def stop(self):
            calls.append("stop")

    monkeypatch.setattr(adapters, "PerfCollector", forbidden_perf)
    monkeypatch.setattr(adapters, "TorchCollector", WorkerCollector)
    config = Config(
        str(tmp_path / "admin.sock"),
        str(tmp_path / "artifacts"),
        workers=[Worker("worker", "epoch", "vllm", "http://configured")],
        frontend_perf_enabled=False,
    )
    controller = service.Coordinator(
        config, RuntimeSnapshot(123, "frontend-epoch", "test-host", config.workers)
    )
    assert controller.targets()[0]["collectors"] == []
    with pytest.raises(errors.Unsupported):
        controller.validate(plan())
    payload = msgspec.to_builtins(Plan("id", [Target("worker", "epoch")], 50))
    async with TestClient(TestServer(application(controller))) as client:
        validation = await client.post(
            "/admin/profiling/v1/plans/validate", json=payload
        )
        assert validation.status == 200
        assert (await validation.json())["frontend_collector"] is None
        created = await client.post("/admin/profiling/v1/sessions", json=payload)
        assert created.status == 202
        session_id = (await created.json())["session_id"]
        await controller.sessions[session_id].task
        artifacts = await client.get(
            f"/admin/profiling/v1/sessions/{session_id}/artifacts"
        )
        manifest = await artifacts.json()
    assert manifest["state"] == "completed"
    assert manifest["frontend"]["profiling_enabled"] is False
    assert manifest["frontend"]["omission_reason"] == "disabled_by_operator_config"
    assert [profile["collector"] for profile in manifest["profiles"]] == [
        "torch_profiler"
    ]
    assert calls == ["start", "stop"]
    assert controller.active is None


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["vllm", "sglang"])
@pytest.mark.parametrize("content_type", ["application/json", "text/plain"])
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
async def test_startup_timeout_is_bounded_and_cleaned_up(controller, monkeypatch):
    class HangingPerf:
        def __init__(self, *args):
            self.evidence = {"collector": "linux_perf"}

        async def start(self):
            await asyncio.Event().wait()

        async def stop(self):
            self.evidence["stopped"] = True

    monkeypatch.setattr(adapters, "PerfCollector", HangingPerf)
    session = controller.submit(plan())
    await asyncio.wait_for(session.task, 1)
    assert session.state == "failed"
    assert session.collectors[0].evidence["stopped"]
    assert "TimeoutError" in session.errors


@pytest.mark.asyncio
async def test_finalization_failure_preserves_partial_coverage(controller, monkeypatch):
    class BrokenPerf:
        def __init__(self, *args):
            self.evidence = {"collector": "linux_perf"}

        async def start(self):
            pass

        async def stop(self):
            raise CollectionError("flush failed")

    monkeypatch.setattr(adapters, "PerfCollector", BrokenPerf)
    session = controller.submit(plan())
    await session.task
    assert session.state == "failed"
    assert session.collectors[0].evidence["coverage"] == "partial_or_unknown"
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
    controller, monkeypatch
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
    assert session.collectors[0].evidence["stopped"]
    assert controller.active is None
    assert any("disk full" in error for error in session.errors)
