# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Coordinator contracts exercised through real HTTP and native trace files."""

import asyncio
import gzip
import json
import os
import tempfile
from pathlib import Path

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from dynamo.frontend.profiling import hosting
from dynamo.frontend.profiling.hosting import host, profiling_coordinator
from dynamo.profiling import (
    Config,
    Coordinator,
    Plan,
    RuntimeSnapshot,
    Target,
    Worker,
    agent,
)
from dynamo.profiling.adapters import CollectorRegistry
from dynamo.profiling.artifacts import SharedFilesystemValidator
from dynamo.profiling.clock import estimate_offset
from dynamo.profiling.collectors import CollectionError
from dynamo.profiling.errors import Unavailable

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]

RANKS = ["dp0_pp0_tp0_dcp0_ep0_rank0", "dp1_pp0_tp0_dcp0_ep1_rank0"]


@pytest.fixture
def socket_directory():
    # pytest's nested temp path can exceed AF_UNIX's platform length limit.
    with tempfile.TemporaryDirectory(prefix="dyn-prof-") as directory:
        yield Path(directory)


def trace(path):
    with gzip.open(path, "wb") as stream:
        stream.write(
            json.dumps(
                {
                    "traceEvents": [
                        {
                            "name": "test_kernel",
                            "cat": "kernel",
                            "ph": "X",
                            "ts": 1,
                            "dur": 2,
                        }
                    ]
                }
            ).encode()
        )


@pytest.mark.asyncio
@pytest.mark.timeout(20)
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
    started = asyncio.Event()
    calls = []
    options = {}
    outputs = tmp_path / "native"
    outputs.mkdir()
    # A valid old trace from the missing rank must never satisfy this capture.
    trace(outputs / f"old_{RANKS[1]}.1.pt.trace.json.gz")

    async def start(request):
        options.update(await request.json())
        calls.append("start")
        started.set()
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

    app = web.Application()
    app.router.add_post("/engine/control/start_profile", start)
    app.router.add_post("/engine/control/stop_profile", stop)
    app.router.add_post("/engine/control/profiling_info", info)
    async with TestServer(app) as server:
        worker = Worker(
            "decode",
            "epoch",
            "vllm",
            str(server.make_url("/")),
            artifact_location=str(outputs),
            verify_runtime=True,
            expected_rank_ids=[] if case == "unknown_membership" else RANKS,
        )
        config = Config(
            str(tmp_path / "admin.sock"),
            str(tmp_path / "sessions"),
            frontend_perf_enabled=False,
            lifecycle_evidence_location="otel-backend://existing",
        )
        controller = Coordinator(
            config, RuntimeSnapshot(123, "frontend-epoch", "frontend-host", [worker])
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
            await asyncio.wait_for(started.wait(), 5)
            controller.stop(session.session_id)
        await asyncio.wait_for(session.task, 5)
    manifest = controller.manifest(session)
    assert controller.active is None
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
    assert manifest["profiles"][0]["identity_verification"] == "control_process_epoch"
    validation = manifest["artifact_validation"][0]
    assert all("old_" not in item["location"] for item in validation["artifacts"])
    if case == "complete":
        assert session.state == "completed"
        assert manifest["artifact_status"] == "verified"
        assert validation["coverage"] == "complete"
        assert len(validation["artifacts"]) == 2
        assert all(item["cuda_kernels"] == 1 for item in validation["artifacts"])
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
@pytest.mark.timeout(20)
async def test_embedded_host_uses_pluggable_coordinator_and_drains(
    tmp_path, socket_directory
):
    started = asyncio.Event()
    stopped = asyncio.Event()

    class Provider:
        def __init__(self):
            self.evidence = {"target_id": "frontend", "collector": "test_provider"}

        async def start(self):
            started.set()

        async def stop(self):
            stopped.set()

    provider = Provider()
    registry = CollectorRegistry()
    registry.register("nsys", lambda config, runtime, plan, directory: [provider])
    config = Config(str(socket_directory / "admin.sock"), str(tmp_path / "sessions"))
    controller = Coordinator(
        config, RuntimeSnapshot(123, "epoch", "host"), registry=registry
    )
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
        await asyncio.wait_for(started.wait(), 5)
    assert stopped.is_set()
    assert controller.lookup(session_id).state == "completed"
    assert not Path(config.socket_path).exists()
    with pytest.raises(Unavailable):
        controller.submit(Plan("new", [Target("frontend", "epoch")], 50, mode="nsys"))


@pytest.mark.asyncio
@pytest.mark.timeout(20)
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
async def test_finalization_timeout_blocks_new_capture_and_plan_is_copied(tmp_path):
    started = asyncio.Event()

    class Provider:
        def __init__(self):
            self.evidence = {"target_id": "frontend", "collector": "test_provider"}

        async def start(self):
            started.set()

        async def stop(self):
            await asyncio.Event().wait()

    registry = CollectorRegistry()
    registry.register("nsys", lambda config, runtime, plan, directory: [Provider()])
    controller = Coordinator(
        Config("unused", str(tmp_path), finalize_timeout_s=0.01),
        RuntimeSnapshot(123, "epoch", "host"),
        registry=registry,
    )
    plan = Plan(
        "capture",
        [Target("frontend", "epoch")],
        5000,
        mode="nsys",
        trace_ids=["original"],
    )
    session = controller.submit(plan)
    plan.trace_ids.append("caller-mutated")
    await asyncio.wait_for(started.wait(), 1)
    controller.stop(session.session_id)
    await asyncio.wait_for(session.task, 1)
    assert session.state == "failed"
    assert session.plan.trace_ids == ["original"]
    assert "TimeoutError" in session.errors
    assert controller.cleanup_uncertain
