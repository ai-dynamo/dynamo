# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Control plane tests without GPU engines or a privileged perf installation."""

import asyncio
import functools
import gzip
import hashlib
import json
import os
import socket
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
import textwrap
import threading
import time
import zlib
from contextlib import closing
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
    inspection,
    native,
    nsys,
    nvtxt,
)
from dynamo.profiling.artifacts import STAMP_ALLOWANCE_NS, SharedFilesystemValidator
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

BASE = 1_700_000_000_000_000_000


async def hang(*args):
    await asyncio.Event().wait()


async def until(predicate, timeout=5):
    async def poll():
        while not predicate():
            await asyncio.sleep(0)

    await asyncio.wait_for(poll(), timeout)


def kev(name, ts, dur=1, cat="kernel", pid=1, tid=1):
    """A Kineto complete event, in the key order Kineto writes."""
    return dict(ph="X", cat=cat, name=name, pid=pid, tid=tid, ts=ts, dur=dur)


def profiler_range(ts, dur):
    return kev("PyTorch Profiler (0)", ts, dur, "Trace", "Spans", 0)


def meta(kind, pid, **args):
    return {"ph": "M", "name": kind, "pid": pid, "tid": 0, "args": args}


def otel_span(trace_id, span_id, parent, name, start=2_000_000, dur=1_000_000, **kv):
    """An OTLP JSON span starting `start` ns after BASE."""
    return {
        "traceId": trace_id,
        "spanId": span_id,
        "parentSpanId": parent,
        "name": name,
        "startTimeUnixNano": str(BASE + start),
        "endTimeUnixNano": str(BASE + start + dur),
        "attributes": [
            {"key": k, "value": {"stringValue": str(v)}} for k, v in kv.items()
        ],
    }


def write_otel(path, spans):
    path.write_text(
        json.dumps({"resourceSpans": [{"scopeSpans": [{"spans": spans}]}]}) + "\n"
    )
    return path


def torch_target(path, **profile):
    artifacts = [{"location": str(path), "rank_id": "dp0"}]
    return {"collector": "torch_profiler", **profile}, artifacts


def write_session(tmp_path, targets, clocks=None, trace_ids=(), bounds=None, **extra):
    """A finished session; targets maps target_id -> (profile fields, artifacts)."""
    session = tmp_path / "session"
    session.mkdir()
    manifest = {
        "session_id": "s",
        "state": "completed",
        "artifact_status": "present",
        "plan": {"trace_ids": list(trace_ids)},
        "profiles": [{"target_id": t, **p} for t, (p, _) in targets.items()],
        "clock": {"workers": clocks or {}},
        "artifact_validation": [
            {"target_id": t, "status": "present", "artifacts": artifacts}
            | ({"recording_bounds": bounds} if bounds else {})
            for t, (_, artifacts) in targets.items()
        ],
        **extra,
    }
    (session / "manifest.json").write_text(json.dumps(manifest))
    return session


def read_timeline(path):
    return json.loads(gzip.decompress(path.read_bytes()))["traceEvents"]


def lane_order(events):
    """Process names in their timeline sort order."""
    order = {
        e["pid"]: e["args"]["sort_index"]
        for e in events
        if e.get("name") == "process_sort_index"
    }
    names = {
        e["pid"]: e["args"]["name"] for e in events if e.get("name") == "process_name"
    }
    return [names[pid] for pid in sorted(order, key=order.get)]


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
    return Coordinator(
        Config("unused", str(tmp_path), finalize_timeout_s=0.01),
        RuntimeSnapshot(123, "epoch", "host"),
        registry={"nsys": lambda *args: [collector]},
    )


def plan(request_id="request-1", duration_ms=50):
    return Plan(request_id, [Target("frontend", "frontend-epoch")], duration_ms)


def nsys_plan(request_id, *target_ids, duration_ms=5000):
    targets = [Target(t, "epoch") for t in target_ids or ["frontend"]]
    return Plan(request_id, targets, duration_ms, mode="nsys")


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
@pytest.mark.parametrize(
    "phase,failure,error",
    [
        ("start", CollectionError("permission denied"), "permission denied"),
        ("start", hang, "TimeoutError"),
        ("stop", CollectionError("flush failed"), "flush failed"),
        ("stop", hang, "TimeoutError"),
        ("stop", RuntimeError("unexpected stop failure"), "unexpected stop failure"),
    ],
)
async def test_start_or_stop_failure_is_bounded_and_releases_lease(
    controller, collector, phase, failure, error
):
    controller.config = msgspec.structs.replace(
        controller.config, finalize_timeout_s=0.01
    )
    getattr(collector, phase).side_effect = failure
    session = controller.submit(plan())
    await asyncio.wait_for(session.task, 1)
    assert session.state == "failed"
    assert error in session.errors
    assert controller.active is None
    collector.stop.assert_awaited_once()
    assert controller.cleanup_uncertain is (phase == "stop")
    if phase == "stop":
        assert collector.evidence["coverage"] == "partial_or_unknown"
        # A new capture is refused until the stuck collector is reconciled.
        with pytest.raises(errors.Conflict, match="cleanup is uncertain"):
            controller.submit(plan("retry-after-stop-failure"))


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
    # Sequential startup would have timed out the spinning start.
    assert session.errors == []


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
        registry={"torch_profile": lambda *args: [collector]},
        clock_sampler=SimpleNamespace(measure=measure),
    )
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
        assert await validation.json() == {"valid": True}
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
        status = await client.get(f"/admin/profiling/v1/sessions/{session_id}")
        manifest = await status.json()
        assert manifest["state"] == "completed"
        assert manifest["frontend"]["profiling_enabled"] is frontend_perf
        view = await client.get(f"/admin/profiling/v1/sessions/{session_id}/artifacts")
        keys = ("session_id", "state", "artifact_status", "artifact_validation")
        assert await view.json() == {key: manifest[key] for key in keys} | {
            "directory": str(controller.lookup(session_id).directory),
            "otel": manifest["lifecycle_otel"],
        }
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
async def test_perf_attach_ack_and_signal_only_recorder(tmp_path, monkeypatch):
    commands = []
    read_ends = []
    signals = []
    recorder = SimpleNamespace(
        returncode=None, send_signal=signals.append, wait=AsyncMock(return_value=0)
    )
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
    assert signals == [collectors.signal.SIGINT]
    assert adapter.evidence["coverage"] == "captured_unverified"


@pytest.mark.asyncio
async def test_frontend_epoch_checked_before_perf_spawn(tmp_path, monkeypatch):
    spawn = AsyncMock()
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(collectors, "process_epoch", lambda pid: "replacement-epoch")
    adapter = PerfCollector(123, "old-epoch", tmp_path)
    with pytest.raises(CollectionError, match="epoch changed"):
        await adapter.start()
    spawn.assert_not_called()
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
    payload = {"traceEvents": [kev("test_kernel", 1, 2)]}
    path.write_bytes(gzip.compress(json.dumps(payload).encode()))


def torch_evidence(location, ranks=RANKS[:1]):
    return {
        "collector": "torch_profiler",
        "artifact_location": str(location),
        "artifact_prefix": "capture",
        "expected_rank_ids": list(ranks),
    }


async def validate_torch(location, ranks=RANKS[:1]):
    config = Config("unused", str(location))
    return await SharedFilesystemValidator().validate(
        torch_evidence(location, ranks), config
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case", ["complete", "missing", "stale_epoch", "unknown_membership"]
)
async def test_worker_capture_identity_clock_and_rank_validation(
    tmp_path, monkeypatch, case
):
    monkeypatch.setattr(
        agent,
        "process_epoch",
        lambda pid: "new-epoch" if case == "stale_epoch" else "epoch",
    )
    worker_agent = agent.ProfilingAgent("vllm", 4242)
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
            trace(outputs / f"{options['profile_prefix']}_{rank}.2.pt.trace.json.gz")
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
        unselected = Worker(
            "unselected", "other-epoch", "vllm", str(server.make_url("/unselected"))
        )
        controller = Coordinator(
            config, RuntimeSnapshot(123, "epoch", "host", [worker, unselected])
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
        if case != "stale_epoch":
            await until(lambda: session.state not in ("accepted", "starting"))
            assert session.state == "recording"
            controller.stop(session.session_id)
        await asyncio.wait_for(session.task, 5)
    manifest = controller.build_manifest(session)
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
    assert clocks["start"]["worker_instance_id"] == 4242
    assert set(manifest["clock"]["workers"]) == {"decode"}
    assert [item["target_id"] for item in manifest["profiles"]] == ["decode"]
    assert manifest["profiles"][0]["identity_verification"] == "control_process_epoch"
    validation = manifest["artifact_validation"][0]
    assert all("old_" not in item["location"] for item in validation["artifacts"])
    if case == "complete":
        # Only presence is checked on the capture path; full inspection (and
        # truncation detection) runs offline in the bundle job.
        assert session.state == "completed"
        assert manifest["artifact_status"] == "present"
        assert validation["coverage"] == "present"
        assert len(validation["artifacts"]) == 2
        inspected = inspection.inspect_kineto(validation, config)
        assert inspected["status"] == "verified"
        for item in inspected["artifacts"]:
            assert item["events"] == item["cuda_kernels"] == 1
            raw = gzip.decompress(Path(item["location"]).read_bytes())
            assert item["sha256_uncompressed"] == hashlib.sha256(raw).hexdigest()
        # text/plain {"status":"ok"} control responses bracket the timestamps.
        profile = manifest["profiles"][0]
        assert profile["coverage"] == "export_unverified"
        assert profile["start_requested_at_ns"] <= profile["started_at_ns"]
        assert profile["stop_requested_at_ns"] <= profile["stopped_at_ns"]
    elif case == "unknown_membership":
        assert session.state == "completed"
        assert manifest["artifact_status"] == "unverified"
        assert validation["coverage"] == "unknown"
    else:
        assert session.state == "failed"
        assert manifest["artifact_status"] == "partial"
        assert validation["missing_rank_ids"] == [RANKS[1]]


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
    result = await validate_torch(tmp_path / "unmounted")
    assert result["status"] == "unverified"
    assert result["reason"] == "artifact_store_not_mounted"


@pytest.mark.asyncio
async def test_trace_validation_limits_and_duplicate_rank_files(tmp_path):
    for suffix in ("1", "2"):
        trace(tmp_path / f"capture_{RANKS[0]}.{suffix}.pt.trace.json.gz")
    result = await validate_torch(tmp_path)
    assert result["status"] == "invalid"
    assert result["duplicate_rank_files"] is True
    inspected = inspection.inspect_kineto(
        result, Config("socket", str(tmp_path), max_trace_bytes=10)
    )
    assert all(item["validation"] == "invalid" for item in inspected["artifacts"])
    assert "byte limit" in inspected["artifacts"][0]["error"]


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["adopted", "before_start", "after_stop", "ambiguous"])
async def test_repeat_capture_adopts_trace_under_earlier_prefix(tmp_path, case):
    """vLLM keeps a worker's first trace name; later sessions write under it."""
    start, stop = BASE + 10, BASE + 20
    later = stop + STAMP_ALLOWANCE_NS + 1
    # Neither the first session's own trace nor a later session's ever counts.
    trace(tmp_path / f"first_{RANKS[0]}.{BASE}.pt.trace.json.gz")
    stamps = {
        "adopted": [start + 1, later],
        "before_start": [],
        "after_stop": [later],
        "ambiguous": [start + 1, start + 2],
    }
    for stamp in stamps[case]:
        trace(tmp_path / f"first_{RANKS[0]}.{stamp}.pt.trace.json.gz")
    evidence = torch_evidence(tmp_path) | {
        "start_requested_at_ns": start,
        "stopped_at_ns": stop,
    }
    result = await SharedFilesystemValidator().validate(
        evidence, Config("unused", str(tmp_path))
    )
    if case == "adopted":
        assert result["status"] == "present"
        assert result["association"] == "unique_session_prefix+session_window"
        assert result["adopted"][0]["renamed_from"] == (
            f"first_{RANKS[0]}.{start + 1}.pt.trace.json.gz"
        )
        assert (tmp_path / f"capture_{RANKS[0]}.{start + 1}.pt.trace.json.gz").exists()
    else:
        assert result["status"] == "partial"
        assert "adopted" not in result
    assert (tmp_path / f"first_{RANKS[0]}.{BASE}.pt.trace.json.gz").exists()


@pytest.mark.asyncio
async def test_uncompressed_rank_traces_are_present_and_inspected(tmp_path):
    """Workers may write plain JSON (torch_profiler_use_gzip=false)."""
    payload = {"traceEvents": [kev("k", 1, 2)]}
    for rank in RANKS:
        path = tmp_path / f"capture_{rank}.1.pt.trace.json"
        path.write_text(json.dumps(payload))
    path.write_text(json.dumps(payload)[:-5])
    result = await validate_torch(tmp_path, RANKS)
    evidence = torch_evidence(tmp_path, RANKS)
    # Presence does not parse, so the truncated rank is only caught offline.
    assert result["status"] == "present"
    inspected = inspection.inspect_target(result, evidence)
    assert inspected["status"] == "invalid"
    assert [a["validation"] for a in inspected["artifacts"]] == ["valid", "invalid"]
    path.write_text(json.dumps(payload))
    assert inspection.inspect_target(result, evidence)["status"] == "verified"
    # Corrupt deflate data (zlib.error) fails the rank, not the bundle.
    path = tmp_path / "corrupt.pt.trace.json.gz"
    path.write_bytes(gzip.compress(b"")[:10] + b"\xff" * 16)
    assert inspection.inspect_rank({"location": str(path)})["validation"] == "invalid"


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
    sessions = "http://localhost/admin/profiling/v1/sessions"

    def body(request_id):
        return msgspec.to_builtins(nsys_plan(request_id))

    async with host(controller) as embedded:
        assert embedded is controller
        assert Path(config.socket_path).stat().st_mode & 0o777 == 0o600
        async with aiohttp.ClientSession(
            connector=aiohttp.UnixConnector(path=config.socket_path)
        ) as client:
            response = await client.post(sessions, json=body("plugin"))
            assert response.status == 202
            session_id = (await response.json())["session_id"]
            conflict = await client.post(sessions, json=body("other"))
            assert conflict.status == 409
        await asyncio.wait_for(collector.started.wait(), 1)
    collector.stop.assert_awaited_once()
    session = controller.lookup(session_id)
    assert session.state == "completed"
    # Draining on shutdown leaves export verification to the bundle job.
    assert session.artifact_validation == [
        {
            "target_id": "frontend",
            "status": "unverified",
            "coverage": "unknown",
            "reason": "host_shutdown",
        }
    ]
    assert not Path(config.socket_path).exists()
    with pytest.raises(Unavailable):
        controller.submit(nsys_plan("new"))


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
async def test_close_stops_collectors_without_waiting_for_artifacts(
    provider_controller, collector
):
    controller = provider_controller
    validate = AsyncMock(side_effect=hang)  # an export that never arrives
    controller.artifact_validator = SimpleNamespace(validate=validate)
    session = controller.submit(nsys_plan("close"))
    await asyncio.wait_for(collector.started.wait(), 1)
    await asyncio.wait_for(controller.close(), 1)
    collector.stop.assert_awaited_once()
    validate.assert_not_awaited()
    assert session.state == "completed"
    assert session.artifact_validation[0]["reason"] == "host_shutdown"
    assert controller.build_manifest(session)["artifact_status"] == "unverified"


@pytest.mark.asyncio
async def test_unstarted_nsight_capture_does_not_block_the_next(tmp_path, monkeypatch):
    monkeypatch.setattr(nsys, "nvtx", None)
    controller = Coordinator(
        Config(
            "unused",
            str(tmp_path / "sessions"),
            nsys_frontend_output_prefix=str(tmp_path / "frontend"),
        ),
        RuntimeSnapshot(os.getpid(), "epoch", "host"),
    )
    # The plan is refused before anything starts, so nothing can be left behind.
    with pytest.raises(errors.Unsupported, match=r"ai-dynamo\[profiling\]"):
        controller.submit(nsys_plan("first"))
    assert not controller.sessions
    assert controller.cleanup_uncertain is False


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
    result = await validate_torch(tmp_path)
    assert result["status"] == "present"
    inspected = inspection.inspect_kineto(result, Config("unused", str(tmp_path)))
    assert inspected["status"] == "verified"
    bounds = inspected["artifacts"][0]["recording_bounds"]
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


def nsys_report(path, **tables):
    """A report and its finished sqlite export; tables: name=(columns, rows)."""
    path.write_bytes(b"report")
    exported = [("EXPORT_TIME", "0.01 sec"), ("EXPORT_PARAM_INPUT_FILE", path.name)]
    tables = {"META_DATA_EXPORT": ("name, value", exported), **tables}
    with sqlite3.connect(path.with_suffix(".sqlite")) as db:
        for name, (columns, rows) in tables.items():
            db.execute(f"CREATE TABLE {name} ({columns})")
            marks = ",".join("?" * (columns.count(",") + 1))
            db.executemany(f"INSERT INTO {name} VALUES ({marks})", rows)
    return path.with_suffix(".sqlite")


CPU = {"COMPOSITE_EVENTS": ("start, cpuCycles, globalTid", [(1, 1, os.getpid() << 24)])}
GPU = {"CUPTI_ACTIVITY_KIND_KERNEL": ("globalPid", [(0,)])}


@pytest.mark.asyncio
async def test_nsys_repeated_client_capture_without_subprocess(tmp_path, monkeypatch):
    pending = []
    handles = []
    loop = asyncio.get_running_loop()
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
            # Nsight exports from its own process. Writing on the loop would block
            # it on the SQLite lock while the validator reads the same file.
            report = tmp_path / f"frontend.{len(handles)}.nsys-rep"
            await asyncio.to_thread(nsys_report, report)
            if len(handles) == 1:
                await asyncio.to_thread(
                    nsys_report, tmp_path / "frontend-other.1.nsys-rep"
                )

        # Range transitions run in a worker thread; schedule on the loop.
        loop.call_soon_threadsafe(lambda: pending.append(asyncio.create_task(export())))

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
            # Generous for slow runner filesystems; validation returns on arrival.
            artifact_validation_timeout_s=10,
        ),
        RuntimeSnapshot(os.getpid(), "epoch", "host"),
    )
    async with TestClient(TestServer(application(coordinator))) as client:
        for window in range(3):
            body = msgspec.to_builtins(nsys_plan(str(window)))
            response = await client.post("/admin/profiling/v1/sessions", json=body)
            assert response.status == 202
            session = coordinator.lookup((await response.json())["session_id"])
            await until(lambda: session.state == "recording")
            await client.post(f"/admin/profiling/v1/sessions/{session.session_id}/stop")
            await session.task
            manifest = coordinator.build_manifest(session)
            assert session.state == "completed", manifest
            assert manifest["frontend"]["profiling_enabled"] is True
            result = manifest["artifact_validation"][0]
            assert result["status"] == "present"
            assert result["artifacts"][0]["location"].endswith(
                f"frontend.{window+1}.nsys-rep"
            )
            assert result["artifacts"][0]["bytes"] == len(b"report")
    await asyncio.gather(*pending)
    assert len(handles) == 3


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
def test_nsys_worker_uses_backend_controls(tmp_path, backend):
    prefix = str(tmp_path / "worker")
    worker = Worker("decode", "epoch", backend, "http://w", nsys_output_prefix=prefix)
    config = Config("unused", str(tmp_path), frontend_perf_enabled=True)
    runtime = RuntimeSnapshot(123, "frontend", "host", [worker])
    # No implicit frontend/perf capture in Nsight mode, even with perf enabled.
    (selected,) = nsys.nsys_collectors(
        config, runtime, nsys_plan("id", "decode"), tmp_path
    )
    assert isinstance(selected, nsys.NsysWorkerCollector)
    sglang = backend == "sglang"
    assert selected.worker.start_options == (
        {"activities": ["CUDA_PROFILER"]} if sglang else {}
    )
    # vLLM's CUDA-profiler launch cannot also serve Torch profiling.
    coordinator = Coordinator(config, runtime)
    assert coordinator.targets()[1]["collectors"] == (
        ["torch_profiler", "nsys"] if sglang else ["nsys"]
    )
    if not sglang:
        with pytest.raises(errors.Unsupported, match="separate launch"):
            coordinator.submit(Plan("id", [Target("decode", "epoch")], 50))
        assert not coordinator.sessions


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
    nsys_report(path, **(CPU if defect == "missing_gpu" else GPU))
    if defect == "ambiguous":
        nsys_report(tmp_path / "worker.2.nsys-rep", **GPU)
    if defect == "incomplete":
        with sqlite3.connect(path.with_suffix(".sqlite")) as conn:
            conn.execute("DELETE FROM META_DATA_EXPORT WHERE name = 'EXPORT_TIME'")
    if defect == "corrupt":
        path.with_suffix(".sqlite").write_bytes(b"not a database")
    evidence = {
        "collector": "nsys",
        "output_prefix": str(tmp_path / "worker"),
        "required_activity": "cpu_events" if defect == "missing_cpu" else "cuda_events",
        "expected_gpu_processes": 2 if defect == "missing_rank" else 1,
        "previous_reports": [],
    }
    # The first poll always runs; the short deadline only ends exports that
    # never become ready (corrupt, incomplete).
    config = Config("unused", str(tmp_path), artifact_validation_timeout_s=0.03)
    result = await SharedFilesystemValidator().validate(evidence, config)
    if defect in ("ambiguous", "corrupt", "incomplete"):
        # Readiness and association are checked on the capture path.
        assert result["status"] in ("partial", "invalid")
    else:
        # Activity and rank checks run offline in the bundle job.
        assert result["status"] == "present"
        result = inspection.inspect_target(
            {"artifacts": [{"location": str(path)}]}, evidence
        )
        assert result["status"] == "invalid"
    assert result["error"]


@pytest.mark.parametrize("with_kernels", [False, True])
def test_nsys_accepts_cuda_memory_activity(tmp_path, with_kernels):
    # A decoder waiting for KV transfer can perform copies without launching
    # kernels. A process appearing in both tables must still count as one rank.
    path = tmp_path / "worker.1.nsys-rep"
    tables = GPU if with_kernels else CPU
    nsys_report(path, **tables, CUPTI_ACTIVITY_KIND_MEMCPY=("globalPid", [(0,)]))
    result = inspection.inspect_nsys(
        {"artifacts": [{"location": str(path)}]},
        {"required_activity": "cuda_events", "expected_gpu_processes": 1},
        inspection.OFFLINE,
    )
    assert result["status"] == "verified"
    assert result["events"]["CUPTI_ACTIVITY_KIND_MEMCPY"] == 1
    assert result["gpu_processes"] == [0]


def test_bundle_writes_per_request_deployment_timelines(tmp_path):
    trace = tmp_path / "native" / "capture_dp0.1.pt.trace.json.gz"
    trace.parent.mkdir()
    events = [
        meta("process_name", 7, name="EngineCore"),
        meta("process_labels", 7, labels="CPU"),
        kev("before_recording", 100, 2, pid=7),
        kev("before_operation", 1000, 2, pid=7),
        kev("inside_operation", 2500.5, 2, pid=7),
        profiler_range(1000, 3000),
    ]
    native = {"baseTimeNanoseconds": BASE, "traceEvents": events}
    trace.write_bytes(gzip.compress(json.dumps(native).encode()))
    clock = {"alignment": "estimated", "offset_ns": 1000, "hostname": "h", "pid": 42}
    session = write_session(
        tmp_path,
        {"decode-0": torch_target(trace, started_at_ns=BASE + 900_000)},
        {"decode-0": {"start": clock}},
        ["trigger"],
    )
    span = functools.partial(otel_span, dur=3_000_000)

    def phase(trace_id, prefix, name, worker, pid, start):
        # Span host attributes are not worker host identities; only the PID joins.
        epoch = {"dynamo.process.epoch": f"{pid}:x", "dynamo.instance.id": "other"}
        decision = {"phase": name.title(), "worker_id": worker}
        route, handle, op = prefix + "r", prefix + "h", prefix + "o"
        return [
            span(trace_id, route, "a", "kv_router.route_request", start, **decision),
            span(trace_id, handle, route, "handle_payload", start, instance_id=worker),
            span(trace_id, op, handle, f"worker.operation.{name}", start, **epoch),
        ]

    def request(trace_id, decode_worker, decode_pid, start):
        return (
            [span(trace_id, "a", "", "http-request", start)]
            + phase(trace_id, "p", "prefill", "W3", 99, start)
            + phase(trace_id, "d", "decode", decode_worker, decode_pid, start)
        )

    otel = write_otel(
        tmp_path / "otel.jsonl",
        request("captured", "W1", 42, 2_000_000)
        + request("uncaptured", "W2", 43, 2_000_000)
        + request("after", "W1", 42, 9_000_000),
    )
    summary = bundle.build(session, [otel], tmp_path / "out", per_request=True)
    # The command runs the module as __main__; parallel verification must work there.
    subprocess.run(
        [
            sys.executable,
            "-m",
            "dynamo.profiling.bundle",
            str(session),
            "--otel-jsonl",
            str(otel),
            "--output",
            str(tmp_path / "cli"),
        ],
        check=True,
        capture_output=True,
    )
    cli = json.loads((tmp_path / "cli/bundle.json").read_text())
    assert cli["artifact_verification"]["decode-0"]["status"] == "verified"
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
    assert summary["artifact_verification"]["decode-0"]["status"] == "verified"
    # Window: both kernels and the profiler range; per request: the kernel
    # inside its operation and the overlapping profiler range.
    assert summary["coverage"]["native_timeline"]["decode-0"]["events_written"] == 5
    events = read_timeline(tmp_path / "out" / index["requests"][0]["timeline"])
    assert lane_order(events) == [
        "Requests · OTel spans",
        "Recording windows",
        "decode-0 · EngineCore (CPU)",
        "decode-0 · Kineto profiler (Spans)",
        "Control",
    ]
    lanes = {e["name"]: e["args"]["process"] for e in events if e.get("cat") == "otel"}
    assert lanes["worker.operation.prefill"] == "prefill worker W3 (not captured)"
    assert lanes["worker.operation.decode"] == "decode-0"
    assert lanes["kv_router.route_request"] == "frontend"
    # The default window timeline shows every request and each captured
    # worker for its whole recording, including work outside any request.
    window = read_timeline(tmp_path / "out/timeline.json.gz")
    threads = {e["args"]["name"] for e in window if e.get("name") == "thread_name"}
    assert "req captured · prefill (not captured) → decode-0" in threads
    assert sorted(e["name"] for e in window if e.get("cat") == "kernel") == [
        "before_operation",
        "inside_operation",
    ]
    assert lane_order(window)[:3] == lane_order(events)[:3]
    # Raw base+2.5005 ms on the worker clock, minus its 1 us offset, from the
    # earliest control event at base+0.9 ms.
    kernels = [(e["name"], e["ts"]) for e in events if e.get("cat") == "kernel"]
    assert kernels == [("inside_operation", 1599.5)]
    assert all(isinstance(e.get("tid", 0), int) for e in events)


@pytest.mark.asyncio
async def test_timed_out_frontend_range_start_is_closed_on_stop(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    closed, handle = [], object()

    def start_range(**kwargs):
        entered.set()
        release.wait(5)  # Nsight blocking the range transition past the budget
        return handle

    monkeypatch.setattr(
        nsys, "nvtx", SimpleNamespace(start_range=start_range, end_range=closed.append)
    )
    monkeypatch.setattr(nsys, "process_epoch", lambda pid: "epoch")
    collector = nsys.NsysFrontendCollector(
        RuntimeSnapshot(os.getpid(), "epoch", "host"), str(tmp_path / "frontend")
    )
    # The start budget expires exactly while the range transition is blocked.
    start = asyncio.ensure_future(collector.start())
    await asyncio.to_thread(entered.wait, 5)
    start.cancel()
    with pytest.raises(asyncio.CancelledError):
        await start
    release.set()
    await collector.stop()
    assert closed == [handle]


@pytest.mark.asyncio
async def test_host_replaces_stale_socket_but_not_a_live_one(
    socket_directory, provider_controller
):
    path = socket_directory / "admin.sock"
    with socket.socket(socket.AF_UNIX) as stale:
        stale.bind(str(path))  # left behind by a crashed host
    controller = provider_controller
    controller.config = msgspec.structs.replace(
        controller.config, socket_path=str(path)
    )
    async with host(controller):
        with pytest.raises(ValueError, match="running host"):
            async with host(controller):
                pass
        # The refused host must leave the live socket in place.
        with socket.socket(socket.AF_UNIX) as client:
            client.connect(str(path))
    assert not path.exists()


def test_bundle_without_observed_bounds_or_requests(tmp_path):
    trace = tmp_path / "capture_dp0.1.pt.trace.json"
    trace.write_text(
        json.dumps({"baseTimeNanoseconds": 10**18, "traceEvents": [kev("k", 1)]})
    )
    session = write_session(
        tmp_path, {"decode-0": torch_target(trace)}, accepted_at_ns=5
    )
    otel = write_otel(tmp_path / "otel.jsonl", [])
    out = tmp_path / "out"
    bundle.build(session, [otel], out, per_request=True)
    (out / "reports").mkdir(exist_ok=True)
    (out / "reports/stale.nsys-rep").write_bytes(b"stale")
    # A rebuild starts from an empty output: no stale reports or timelines.
    summary = bundle.build(session, [otel], out)
    assert not (out / "timelines").exists()
    assert not (out / "reports/stale.nsys-rep").exists()
    assert not any(f.startswith(("timelines/", "reports/")) for f in summary["files"])
    # No profiler marker: verified, but without bounds its events are not drawn.
    assert summary["artifact_verification"]["decode-0"]["status"] == "verified"
    assert summary["coverage"]["native_timeline"]["decode-0"]["status"] == "omitted"
    assert summary["timeline_origin_unix_ns"] == 5
    # An output that is, or holds, the session or an input is never cleared.
    held = tmp_path / "held"
    held.mkdir()
    inner = write_otel(held / "otel.jsonl", [])
    for output, otel_path in ((session, otel), (tmp_path, otel), (held, inner)):
        with pytest.raises(SystemExit, match="contains the session or its inputs"):
            bundle.build(session, [otel_path], output)
    assert (session / "manifest.json").exists() and inner.exists()


def kineto_layout(base: int, events: list[dict]) -> str:
    """A trace laid out the way Kineto's writer emits it."""
    blocks = ",\n".join(textwrap.indent(json.dumps(e, indent=2), "  ") for e in events)
    return (
        f'{{\n  "schemaVersion": 1,\n  "baseTimeNanoseconds": {base},\n'
        f'  "traceEvents": [\n{blocks}\n  ],"traceName": "t"\n}}\n'
    )


@pytest.mark.parametrize("suffix", [".json", ".json.gz"])
def test_kineto_block_reader_matches_token_parser(tmp_path, suffix):
    base = 1_790_857_026_000_000_000
    events = [
        meta("process_name", 7, name="x"),
        kev("k", 5.5, 2, pid=7),
        profiler_range(1075990243460.021, 10000.123),
    ]
    path = tmp_path / f"rank{suffix}"
    encode = gzip.compress if suffix.endswith(".gz") else bytes
    path.write_bytes(encode(kineto_layout(base, events).encode()))
    assert list(native.kineto_events(path)) == list(
        native.kineto_events(path, fast=False)
    )
    config = Config("unused", str(tmp_path))
    deadline = time.monotonic() + 60
    fast = inspection.inspect_trace(path, config, deadline)
    assert fast == inspection.scan_trace(path, config, deadline, inspection.scan_tokens)
    assert (fast["events"], fast["cuda_kernels"]) == (3, 1)
    assert fast["recording_bounds"]["start_unix_ns"] == base + 1075990243460021
    # Any other layout falls back to the token parser instead of failing.
    native_layout = {"baseTimeNanoseconds": base, "traceEvents": events}
    path.write_bytes(encode(json.dumps(native_layout).encode()))
    assert inspection.inspect_trace(path, config, deadline)["events"] == 3
    assert len(list(native.kineto_events(path, fast=False))) == 3


def test_parallel_gzip_is_one_standard_member(tmp_path):
    data = os.urandom(5000) + b"timeline " * 3000
    path = tmp_path / "out.gz"
    stream = bundle.ParallelGzip(path)
    stream.BLOCK = 1024  # many blocks, compressed on threads
    for start in range(0, len(data), 777):
        stream.write(data[start : start + 777])
    stream.close()
    assert gzip.decompress(path.read_bytes()) == data
    # One member: the stream ends exactly at its trailer.
    reader = zlib.decompressobj(16 + zlib.MAX_WBITS)
    assert reader.decompress(path.read_bytes()) == data and not reader.unused_data


def test_bundle_fails_only_the_target_with_a_malformed_trace(tmp_path):
    base = 10**18
    good = tmp_path / "good_dp0.1.pt.trace.json"
    good.write_text(kineto_layout(base, [profiler_range(0, 100), kev("k", 1)]))
    bad = tmp_path / "bad_dp0.1.pt.trace.json"
    # Structurally intact (verification passes) but one event is not JSON.
    layout = kineto_layout(base, [profiler_range(0, 100), kev("bad-k", 1)])
    bad.write_text(layout.replace('"bad-k"', '"bad-k",,'))
    session = write_session(
        tmp_path,
        {"good": torch_target(good), "bad": torch_target(bad)},
        accepted_at_ns=base,
    )
    otel = write_otel(tmp_path / "otel.jsonl", [])
    summary = bundle.build(session, [otel], tmp_path / "out")
    native_timeline = summary["coverage"]["native_timeline"]
    assert native_timeline["good"]["status"] == "included"
    assert native_timeline["bad"]["status"] == "failed"
    window = read_timeline(tmp_path / "out/timeline.json.gz")
    assert [e["name"] for e in window if e.get("cat") == "kernel"] == ["k"]
    names = [e["args"]["name"] for e in window if e.get("name") == "process_name"]
    assert not any(name.startswith("bad ·") for name in names)
    assert summary["artifact_verification"]["bad"]["status"] == "verified"


def nsight_session(tmp_path):
    # Nsight event timestamps are relative to the session start (utcEpochNs).
    common = {
        "TARGET_INFO_SESSION_START_TIME": (
            "utcEpochNs, systemClockNs",
            [(BASE, 5_000_000_000)],
        ),
        "StringIds": ("id, value", [(1, "epoll_wait"), (2, "gemm")]),
        "PROCESSES": ("globalPid, pid, name", [(0, 42, "VLLM::EngineCore")]),
    }
    tid = (42 << 24) | 7
    frontend = nsys_report(
        tmp_path / "frontend.nsys-rep",
        **common,
        OSRT_API=("start, end, nameId, globalTid", [(2_500_000, 2_600_000, 1, tid)]),
    )
    worker = nsys_report(
        tmp_path / "worker.nsys-rep",
        **common,
        NVTX_EVENTS=(
            "start, end, eventType, text, textId, globalTid, domainId",
            [
                (2_400_000, 2_700_000, 59, "forward", None, tid, 1),
                (0, None, 75, "PyTorch", None, 0, 1),
            ],
        ),
        CUPTI_ACTIVITY_KIND_KERNEL=(
            "start, end, demangledName, deviceId, streamId, globalPid",
            [(1_000_000, 1_100_000, 2, 0, 7, 99), (2_500_000, 2_600_000, 2, 0, 7, 99)],
        ),
    )
    bounds = {
        "status": "observed",
        "start_unix_ns": BASE + 1_000_000,
        "end_unix_ns": BASE + 4_000_000,
    }

    def target(path, **profile):
        report = {"location": str(path.with_suffix(".nsys-rep"))}
        return {"collector": "nsys", **profile}, [report, {"location": str(path)}]

    session = write_session(
        tmp_path,
        {
            "frontend": target(frontend, required_activity="cpu_events", pid=42),
            "decode-0": target(
                worker, required_activity="cuda_events", expected_gpu_processes=1
            ),
        },
        {"decode-0": {"start": {"alignment": "estimated", "offset_ns": 0, "pid": 42}}},
        bounds=bounds,
    )
    spans = [
        otel_span("t", "a", "", "http-request"),
        otel_span(
            "t", "r", "a", "kv_router.route_request", phase="Decode", worker_id="W"
        ),
        otel_span("t", "h", "r", "handle_payload", instance_id="W"),
        otel_span(
            "t", "o", "h", "worker.operation.decode", **{"dynamo.process.epoch": "42:x"}
        ),
    ]
    return session, write_otel(tmp_path / "otel.jsonl", spans)


def test_nvtx_request_ranges_follow_the_span_tree():
    ms = 1_000_000  # realistic times: the jitter tolerance is 1 ms
    spans = {
        # Listed child-first with equal starts: order must come from the tree.
        "op": (10 * ms, 50 * ms, "worker.operation.decode", 1, "dispatch"),
        "dispatch": (10 * ms, 60 * ms, "request.dispatch", 1, "root"),
        "root": (10 * ms, 100 * ms, "http-request", 1, None),
        # A worker span overrunning its parent by clock-offset error.
        "stream": (55 * ms, 70 * ms, "response.streaming", 1, "dispatch"),
        # A sibling of "op" that crosses its end: a genuine overlap.
        "admit": (20 * ms, 55 * ms, "worker.admission", 1, "dispatch"),
    }
    threads = nvtxt.tree_threads(spans)
    assert [message for _, _, message, _ in threads[0]] == [
        "http-request",
        "request.dispatch",
        "worker.operation.decode",
        "response.streaming (clamped to parent)",
    ]
    assert threads[0][3][:2] == (55 * ms, 60 * ms - 1)
    assert [message for _, _, message, _ in threads[1]] == ["worker.admission"]
    # Equal starts become strictly nested (1 ns in), so the order is unambiguous.
    pushes = [r for r in nvtxt.events(threads[0]) if r[1] == "push"]
    assert [r[0] for r in pushes] == [10 * ms, 10 * ms + 1, 10 * ms + 2, 55 * ms]
    # A sibling inside an earlier sibling (ending within the jitter tolerance)
    # nests under it, as in Perfetto, instead of opening an overlap row.
    siblings = {
        "http": (0, 1000 * ms, "http-request", 1, None),
        "lifecycle": (0, 900 * ms, "request.lifecycle", 1, "http"),
        "stream": (100 * ms, 900 * ms + ms // 2, "response.streaming", 1, "http"),
    }
    threads = nvtxt.tree_threads(siblings)
    assert len(threads) == 1
    assert [m for _, _, m, _ in threads[0]] == [
        "http-request",
        "request.lifecycle",
        "response.streaming (clamped to parent)",
    ]
    # The real shape, clipped to a report window: equal intervals, and the
    # sibling that adopts the streaming span already has its own child.
    clipped = {
        "lifecycle": (0, 9 * ms, "request.lifecycle", 1, None),
        "route": (0, 9 * ms, "kv_router.route_request", 1, "lifecycle"),
        "payload": (0, 9 * ms, "handle_payload", 1, "route"),
        "stream": (0, 9 * ms, "response.streaming", 1, "lifecycle"),
    }
    (row,) = nvtxt.tree_threads(clipped)
    starts = [start for start, _, _, _ in row]
    ends = [end for _, end, _, _ in row]
    assert starts == sorted(set(starts)) and ends == sorted(set(ends), reverse=True)


def test_bundle_reads_nsight_worker_exports(tmp_path):
    session, otel = nsight_session(tmp_path)
    bundle.build(session, [otel], tmp_path / "out", per_request=True)
    index = json.loads((tmp_path / "out/lifecycle/trace-index.json").read_text())
    events = read_timeline(tmp_path / "out" / index["requests"][0]["timeline"])
    assert lane_order(events) == [
        "Requests · OTel spans",
        "Recording windows",
        "decode-0 · VLLM::EngineCore (CPU)",
        "decode-0 · CUDA device 0 (GPU 0)",
    ]
    native = sorted(
        (e["cat"], e["name"])
        for e in events
        if e.get("cat") in ("osrt", "pytorch", "kernel")
    )
    # The kernel before the decode operation is excluded, and the frontend
    # host trace is listed but not drawn.
    assert native == [("kernel", "gemm"), ("pytorch", "forward")]
    summary = json.loads((tmp_path / "out/bundle.json").read_text())
    assert summary["coverage"]["native_timeline"]["frontend"]["status"] == "excluded"
    assert {t: v["status"] for t, v in summary["artifact_verification"].items()} == {
        "frontend": "verified",
        "decode-0": "verified",
    }
    recordings = {e["name"] for e in events if e.get("cat") == "capture"}
    assert recordings == {"frontend recording", "decode-0 recording"}
    archived, digest = bundle.archive(tmp_path / "out", summary)
    with tarfile.open(archived) as tar:
        names = set(tar.getnames())
    assert {
        "out/bundle.json",
        "out/" + index["requests"][0]["timeline"],
        "out/profiles/decode-0/worker.sqlite",
        "out/profiles/decode-0/worker.nsys-rep",
    } <= names
    assert digest == hashlib.sha256(archived.read_bytes()).hexdigest()
    assert (tmp_path / "out.tar.gz.sha256").read_text() == f"{digest}  out.tar.gz\n"


def test_nsight_bundle_imports_request_spans_as_nvtx(tmp_path, monkeypatch):
    session, otel = nsight_session(tmp_path)
    # An install with only ImportNvtxt (no nsys_writer package or backend).
    install = tmp_path / "nsys"
    (install / "host-linux-x64").mkdir(parents=True)
    importer = install / "host-linux-x64" / "ImportNvtxt"
    importer.write_text(
        f"#!{sys.executable}\n"
        "import shutil, sys\n"
        "args = dict(zip(sys.argv[1::2], sys.argv[2::2]))\n"
        "shutil.copy(args['-i'], args['-o'])\n"
        "lines = open(args['-n']).read().splitlines()\n"
        "count = sum(l.startswith(('RangePush,', 'Marker,')) for l in lines)\n"
        "print(f'{count} events imported')\n"
    )
    importer.chmod(0o755)
    summary = bundle.build(session, [otel], tmp_path / "out", nsys_root=str(install))
    # Nsight default: native reports with request spans, no Perfetto conversion.
    assert summary["timeline"] is None
    assert not (tmp_path / "out/timeline.json.gz").exists()
    reports = summary["nsight_reports"]
    # As in the Perfetto view, requests go to one report (the frontend's);
    # worker reports are packaged unchanged.
    assert {t: r["status"] for t, r in reports.items()} == {
        "frontend": "merged",
        "decode-0": "native",
    }
    text = (tmp_path / "out/reports/frontend.nvtxt").read_text()
    assert "TimeBase = Manual" in text
    assert 'NameOsThread, 0, 1, "Capture control"' in text
    assert 'NameOsThread, 0, 2, "Recording · decode-0"' in text
    assert 'NameOsThread, 0, 4, "req t · decode-0"' in text
    # The span tree nests on its request thread; base+2..3 ms is 5.002..5.003 s
    # on this report's clock (session start 5 s).
    # Each range is coloured by span kind.
    lines = text.splitlines()
    pushes = [
        (fields[1], fields[3], line.split('"')[1].split(" · ")[0])
        for line in lines
        if line.startswith("RangePush, 500200000")
        and (fields := line.split(", "))[2] == "4"
    ]
    assert pushes == [
        ("5002000000", "0xFF4E79A7", "http-request"),
        ("5002000001", "0xFFF28E2B", "kv_router.route_request"),
        ("5002000002", "0xFFBAB0AC", "handle_payload"),
        ("5002000003", "0xFFB07AA1", "worker.operation.decode"),
    ]
    assert reports["frontend"]["method"] == "ImportNvtxt"
    assert reports["frontend"]["events_imported"] == sum(
        line.startswith(("RangePush,", "Marker,")) for line in lines
    )
    assert (tmp_path / "out/reports/frontend.nsys-rep").read_bytes() == b"report"
    assert (tmp_path / "out/reports/decode-0.nsys-rep").read_bytes() == b"report"
    archived, _ = bundle.archive(tmp_path / "out", summary)
    with tarfile.open(archived) as tar:
        names = set(tar.getnames())
    assert {"out/reports/decode-0.nsys-rep", "out/reports/frontend.nvtxt"} <= names
    # Without ImportNvtxt the NVTXT is still written and Perfetto is the fallback.
    monkeypatch.delenv("NSYS_HOME", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path / "missing"))  # no nsys or ImportNvtxt
    summary = bundle.build(
        session, [otel], tmp_path / "plain", nsys_root=str(tmp_path / "missing")
    )
    assert summary["nsight_reports"]["frontend"]["status"] == "nvtxt_only"
    assert summary["timeline"]["events"]
    assert (tmp_path / "plain/reports/frontend.nvtxt").exists()


@pytest.mark.asyncio
async def test_slow_frontend_range_end_does_not_delay_worker_stop(
    tmp_path, monkeypatch
):
    ended, stopped = [], []

    def end_range(handle):
        time.sleep(0.3)  # Nsight blocking while it ends the capture range
        ended.append(time.monotonic())

    monkeypatch.setattr(
        nsys,
        "nvtx",
        SimpleNamespace(start_range=lambda **kw: object(), end_range=end_range),
    )
    monkeypatch.setattr(nsys, "process_epoch", lambda pid: "epoch")
    worker = SimpleNamespace(
        evidence={"target_id": "decode-0", "collector": "torch_profiler"},
        start=AsyncMock(),
        stop=AsyncMock(side_effect=lambda: stopped.append(time.monotonic())),
    )
    runtime = RuntimeSnapshot(
        os.getpid(), "epoch", "host", [Worker("decode-0", "epoch", "vllm", "http://w")]
    )
    controller = Coordinator(
        Config("unused", str(tmp_path / "sessions")),
        runtime,
        registry={
            "nsys": lambda config, runtime, plan, directory: [
                nsys.NsysFrontendCollector(runtime, str(tmp_path / "frontend")),
                worker,
            ]
        },
        artifact_validator=SimpleNamespace(
            validate=AsyncMock(return_value={"status": "verified"})
        ),
    )
    session = controller.submit(
        nsys_plan("slow-end", "frontend", "decode-0", duration_ms=50)
    )
    await asyncio.wait_for(session.task, 5)
    assert session.state == "completed", session.errors
    assert stopped and ended and stopped[0] < ended[0]


@pytest.mark.asyncio
async def test_rejected_start_does_not_latch_and_reconcile_clears(
    controller, collector
):
    controller.config = msgspec.structs.replace(
        controller.config, finalize_timeout_s=0.01
    )
    # The start was refused, so a failed stop leaves nothing running.
    collector.start.side_effect = CollectionError("not ready")
    collector.stop.side_effect = CollectionError("not profiling")
    session = controller.submit(plan())
    await session.task
    assert collector.evidence["start_outcome"] == "rejected"
    assert controller.cleanup_uncertain is False
    # After an acknowledged start, a failed stop latches until reconciled.
    collector.start.side_effect = None
    await controller.submit(plan("second")).task
    assert controller.cleanup_uncertain is True
    with pytest.raises(errors.Conflict, match="POST reconcile"):
        controller.submit(plan("third"))
    assert controller.reconcile() == ["frontend: stop failed: not profiling"]
    await controller.submit(plan("third")).task


@pytest.mark.asyncio
async def test_finished_sessions_are_evicted_not_refused(controller):
    controller.config = msgspec.structs.replace(controller.config, max_sessions=2)
    sessions = []
    for number in range(3):
        session = controller.submit(plan(f"request-{number}"))
        await session.task
        sessions.append(session)
    assert list(controller.sessions) == [s.session_id for s in sessions[1:]]
    with pytest.raises(errors.NotFound):
        controller.lookup(sessions[0].session_id)
    assert (sessions[0].directory / "manifest.json").is_file()
    with pytest.raises(errors.Conflict, match="evicted"):
        controller.submit(plan("request-0"))


@pytest.mark.asyncio
async def test_empty_profiling_config_disables_profiling():
    async with profiling_coordinator("") as coordinator:
        assert coordinator is None


@pytest.mark.asyncio
async def test_one_noisy_clock_sample_is_discarded(monkeypatch, tmp_path):
    from dynamo.profiling import clock

    monkeypatch.setattr(agent, "process_epoch", lambda pid: "epoch")
    worker_agent = agent.ProfilingAgent("vllm", 4242)
    real, calls = clock.estimate_offset, []

    def flaky(*args):
        calls.append(None)
        if len(calls) == 1:
            raise CollectionError("Clock changed during sampling")
        return real(*args)

    monkeypatch.setattr(clock, "estimate_offset", flaky)

    async def info(request):
        return web.json_response(await worker_agent.info(await request.json()))

    app = web.Application()
    app.router.add_post("/engine/control/profiling_info", info)
    async with TestServer(app) as server:
        worker = Worker(
            "decode", "epoch", "vllm", str(server.make_url("/")), verify_runtime=True
        )
        config = Config("unused", str(tmp_path), clock_probe_samples=2)
        sample = await clock.WorkerClockSampler().measure(worker, config)
    assert len(calls) == 2
    assert sample["alignment"] == "estimated"


@pytest.mark.asyncio
async def test_sglang_torch_traces_use_the_session_prefix(tmp_path):
    outputs = tmp_path / "native"
    outputs.mkdir()
    worker = Worker(
        "decode",
        "epoch",
        "sglang",
        "http://w",
        artifact_location=str(outputs),
        expected_rank_ids=["TP-0", "TP-1"],
    )
    config = Config("unused", str(tmp_path), frontend_perf_enabled=False)
    runtime = RuntimeSnapshot(123, "frontend", "host", [worker])
    (selected,) = adapters.torch_collectors(
        config, runtime, Plan("id", [Target("decode", "epoch")], 50), tmp_path / "s1"
    )
    prefix = selected.worker.start_options["profile_prefix"]
    assert prefix == selected.evidence["artifact_prefix"] == "dynamo-s1"
    # SGLang names <prefix>-<profile id>-TP-<n>; another session's file never counts.
    for name in ("1761.5-TP-0", "1761.5-TP-1"):
        trace(outputs / f"{prefix}-{name}.trace.json.gz")
    trace(outputs / "dynamo-s0-1760.1-TP-0.trace.json.gz")
    result = await SharedFilesystemValidator().validate(selected.evidence, config)
    assert result["status"] == "present"
    assert result["observed_rank_ids"] == ["TP-0", "TP-1"]


@pytest.mark.asyncio
async def test_frontend_perf_recording_is_listed_for_the_archive(tmp_path):
    data = tmp_path / "frontend.perf.data"
    data.write_bytes(b"PERFILE2")
    evidence = {"collector": "linux_perf", "artifact": str(data)}
    result = await SharedFilesystemValidator().validate(
        evidence, Config("unused", str(tmp_path))
    )
    assert result["status"] == "present"
    assert result["artifacts"] == [
        {"location": str(data), "bytes": 8, "validation": "present"}
    ]


def test_router_targets_join_on_worker_instance_id():
    # Two pods with one PID: only the runtime connection ID tells them apart.
    clocks = {
        "decode-0": {"start": {"pid": 42, "worker_instance_id": 7001}},
        "decode-1": {"start": {"pid": 42, "worker_instance_id": 7002}},
    }
    spans = {}
    for worker in ("7001", "7002", "7003"):
        for span in (
            otel_span("t", worker, "", "handle_payload", instance_id=worker),
            otel_span(
                "t",
                worker + "o",
                worker,
                "worker.operation.decode",
                **{"dynamo.process.epoch": "42:x"},
            ),
        ):
            spans[span["spanId"]] = span
    traces = {"t": spans}
    mapping = bundle.router_targets({"clock": {"workers": clocks}}, traces)
    assert mapping == {"7001": "decode-0", "7002": "decode-1"}
    # Without instance IDs the shared PID stays unmapped rather than guessed.
    legacy = {target: {"start": {"pid": 42}} for target in clocks}
    assert bundle.router_targets({"clock": {"workers": legacy}}, traces) == {}


def test_nsight_bounds_are_optional_without_system_clock(tmp_path):
    path = tmp_path / "report.sqlite"
    with closing(sqlite3.connect(path)) as db:
        db.execute(
            "CREATE TABLE TARGET_INFO_SESSION_START_TIME"
            " (utcEpochNs INT, utcTime TEXT, localTime TEXT)"
        )
        db.execute("INSERT INTO TARGET_INFO_SESSION_START_TIME VALUES (1, 'u', 'l')")
        db.execute("CREATE TABLE ANALYSIS_DETAILS (startTime INT, stopTime INT)")
        db.execute("INSERT INTO ANALYSIS_DETAILS VALUES (5, 9)")
        db.commit()
        tables = {"ANALYSIS_DETAILS", "TARGET_INFO_SESSION_START_TIME"}
        assert nsys.report_bounds(db, tables) == {"status": "unknown"}
    assert nvtxt.report_clock(path) == (1, 5, 9)


def test_bundle_refuses_an_unrelated_output_directory(tmp_path):
    session = write_session(tmp_path, {})
    otel = write_otel(tmp_path / "otel.jsonl", [])
    out = tmp_path / "results"
    out.mkdir()
    (out / "keep.txt").write_text("unrelated")
    with pytest.raises(SystemExit, match="not an earlier bundle"):
        bundle.build(session, [otel], out)
    assert (out / "keep.txt").exists()


def test_archive_skips_raw_files_gone_since_capture(tmp_path):
    out = tmp_path / "bundle"
    out.mkdir()
    (out / "bundle.json").write_text("{}")
    kept = tmp_path / "frontend.perf.data"
    kept.write_bytes(b"perf")
    summary = {
        "native_profiles": [
            {"target_id": "frontend", "location": str(kept), "archived": True},
            {
                "target_id": "decode-0",
                "location": str(tmp_path / "gone"),
                "archived": False,
            },
        ]
    }
    path, _ = bundle.archive(out, summary)
    with tarfile.open(path) as tar:
        names = tar.getnames()
    assert "bundle/profiles/frontend/frontend.perf.data" in names
    assert not any(name.endswith("/gone") for name in names)


def test_aggregated_worker_operations_are_attributed():
    # Aggregated workers emit `worker.operation` without a phase suffix.
    spans = {}
    for span in (
        otel_span("t", "r", "", "kv_router.route_request", worker_id="7001"),
        otel_span("t", "h", "r", "handle_payload", instance_id="7001"),
        otel_span("t", "o", "h", "worker.operation", start=2_000_000, dur=4_000_000),
    ):
        spans[span["spanId"]] = span
    manifest = {
        "plan": {"trace_ids": []},
        "clock": {"workers": {"agg-0": {"start": {"worker_instance_id": 7001}}}},
    }
    traces = {"t": spans}
    workers = bundle.router_targets(manifest, traces)
    assert workers == {"7001": "agg-0"}
    window = {"agg-0": (BASE + 3_000_000, BASE + 5_000_000)}
    (entry,) = bundle.index_requests(manifest, traces, workers, window)
    (phase,) = entry["phases"]
    assert (phase["phase"], phase["target_id"]) == ("aggregated", "agg-0")
    assert phase["recording_overlap_ns"] == 2_000_000
