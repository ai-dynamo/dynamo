# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared request handling with the real Duru admission scheduler."""

import asyncio
import copy
import struct
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import pytest_asyncio

from dynamo.duru_router import __main__ as main_module
from dynamo.duru_router.__main__ import AccountingUnavailable, DuruRouterHandler
from dynamo.duru_router.events import CompactFeed
from dynamo.duru_router.router import ThunderAgentConfig
from dynamo.duru_router.scoring import Ingress, RouterAccounting
from dynamo.duru_router.tests.helpers import configuration
from dynamo.duru_router.tests.helpers import memory_mode as memory_mode
from dynamo.duru_router.tests.helpers import work_step
from dynamo.thunderagent_router.tests.test_main import FakeKvRouter

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


class AccountingKvRouter(FakeKvRouter):
    def __init__(self, feed, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feed = feed

    async def generate_from_request(self, request, **kwargs):
        stream = await super().generate_from_request(request, **kwargs)
        extra = request["extra_args"]["sampling_options"]["vllm_xargs"]
        identity = dict(
            client_request_id=extra["duru_request_id"],
            root=extra["duru_root_id"],
            agent=extra["duru_agent_id"],
        )
        now = time.time()
        self.feed.capture(
            "accepted_output",
            dict(steps=[work_step(p=8, scheduled_at=now, end=now, **identity)]),
        )
        self.feed.emit("worker_request_finished", requests=[identity])
        return stream


@pytest_asyncio.fixture
async def handler(monkeypatch, tmp_path):
    path, cfg = configuration(tmp_path)
    monkeypatch.setenv("DURU_DIAGNOSTICS", "1")
    monkeypatch.delenv("DURU_ADMISSION_ALPHA", raising=False)
    monkeypatch.setenv("DURU_SCORES", cfg["scores"])
    ingress = Ingress(Path(cfg["ingress"]).parent, in_process=True)
    feed = CompactFeed(Path(cfg["ledgers"][0]).parent)
    owner = RouterAccounting(path, ingress=ingress).start()
    capacities = {(7, 2): 10, (7, 3): 10}
    capacity = SimpleNamespace(snapshot=lambda: capacities, live_worker_ids=lambda: {7})
    result = DuruRouterHandler(
        runtime=None,
        config=SimpleNamespace(namespace="test", endpoint="test.vllm.generate"),
        accounting=owner,
        ingress=ingress,
    )
    result._scheduler = result.scheduler_cls(
        capacity,
        ThunderAgentConfig(
            buffer_per_program=0,
            pause_threshold=1,
            resume_hysteresis=0,
            scheduler_interval_seconds=0.01,
        ),
    )
    result._scheduler.start()
    result._kv_router = AccountingKvRouter(feed)
    monitoring = asyncio.create_task(result.monitor())
    result.test_monitor = monitoring
    result.test_feed = feed
    try:
        yield result
    finally:
        monitoring.cancel()
        await asyncio.gather(monitoring, return_exceptions=True)
        await result.shutdown()
        results = await asyncio.gather(
            asyncio.to_thread(ingress.close),
            asyncio.to_thread(feed.close),
            return_exceptions=True,
        )
        # Closing joins the accounting thread before inspecting its error. The
        # monitor can notice a closed event channel while that thread is still
        # finishing its cleanup and has not published the underlying failure.
        (owner_result,) = await asyncio.gather(
            asyncio.to_thread(owner.close), return_exceptions=True
        )
        if owner._error is not None:
            # Fault injection disconnects the producers; they may observe it
            # either on their next heartbeat or during their final flush.
            for result in results:
                assert result is None or isinstance(result, RuntimeError)
            assert isinstance(owner_result, RuntimeError)
            assert "accounting failed" in str(owner_result)
        else:
            assert results == [None, None]
            assert owner_result is None


async def snapshot_when(handler, predicate):
    async with asyncio.timeout(5):
        while True:
            snapshot = handler.accounting.snapshot()
            if snapshot and predicate(snapshot):
                return snapshot
            await asyncio.sleep(0.01)


async def generate(handler, pid, tokens=8, **kwargs):
    request = {
        "token_ids": struct.pack(f"{tokens}I", *range(tokens)),
        "agent_context": {"session_id": pid},
        "elapsed_since_arrival_ms": 12,
    }
    return [chunk async for chunk in handler.generate(request, **kwargs)]


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["cancel", "session_final"])
async def test_paused_request_never_dispatches_after_termination(
    handler, monkeypatch, ending
):
    scheduler = handler._scheduler
    entered = asyncio.Event()
    original = scheduler._wait_for_resume

    async def wait_for_resume(*args):
        entered.set()
        await original(*args)

    monkeypatch.setattr(scheduler, "_wait_for_resume", wait_for_resume)
    request = asyncio.create_task(generate(handler, "paused", 11))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        assert handler._kv_router.sent is None
        if ending == "cancel":
            request.cancel()
        else:
            final = {"agent_context": {"session_id": "paused", "session_final": True}}
            assert [chunk async for chunk in handler.generate(final)] == []
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(request, 1)
        assert handler._kv_router.sent is None
        assert not scheduler._table.programs
        assert not scheduler._table.paused
        assert not scheduler._admission_gates
    finally:
        request.cancel()
        await asyncio.gather(request, return_exceptions=True)


def test_main_forwards_router_arguments(monkeypatch, tmp_path):
    monkeypatch.setenv("DURU_SCORES", str(tmp_path / "scores.json"))
    monkeypatch.setenv("DURU_ACCOUNTING_CONFIG", str(tmp_path / "scoring.json"))
    monkeypatch.delenv("DURU_ADMISSION_ALPHA", raising=False)
    monkeypatch.setattr(sys, "argv", ["duru", "--endpoint", "test.vllm.generate"])
    invocation = object()
    started = []
    monkeypatch.setattr(main_module, "worker", lambda: invocation)
    monkeypatch.setattr(main_module, "uvloop", SimpleNamespace(run=started.append))

    main_module.main()

    assert started == [invocation]
    assert sys.argv == ["duru", "--endpoint", "test.vllm.generate"]


@pytest.mark.asyncio
async def test_public_requests_join_worker_accounting_and_cleanup_family(handler):
    request = {
        "token_ids": [1, 2, 3],
        "agent_context": {"session_id": "parent"},
        "extra_args": {"sampling_options": {"vllm_xargs": {"custom": 1}}},
    }
    original = copy.deepcopy(request)
    assert [chunk async for chunk in handler.generate(request)] == []
    parent = handler._kv_router.sent["extra_args"]["sampling_options"]["vllm_xargs"]
    assert request == original
    assert parent["custom"] == 1
    assert [chunk async for chunk in handler.generate(request)] == []
    again = handler._kv_router.sent["extra_args"]["sampling_options"]["vllm_xargs"]
    assert again["duru_root_id"] == parent["duru_root_id"]
    assert again["duru_agent_id"] == parent["duru_agent_id"]
    assert again["duru_request_id"] != parent["duru_request_id"]
    child_request = dict(
        request, agent_context={"session_id": "child", "parent_session_id": "parent"}
    )
    assert [chunk async for chunk in handler.generate(child_request)] == []
    child = handler._kv_router.sent["extra_args"]["sampling_options"]["vllm_xargs"]
    assert child["duru_root_id"] == parent["duru_root_id"]
    assert child["duru_agent_id"] != parent["duru_agent_id"]
    assert child["duru_request_id"] != parent["duru_request_id"]
    snapshot = await snapshot_when(
        handler, lambda s: s["accepted_steps"] == 3 and not s["requests"]
    )
    root = parent["duru_root_id"]
    assert snapshot["roots"][root]["W"] == pytest.approx(0.03)
    assert snapshot["roots"][root]["U"] > 0
    await handler.end_session("parent")
    snapshot = await snapshot_when(handler, lambda s: "parent" not in s["programs"])
    assert root in snapshot["roots"]
    await handler.end_session("child")
    snapshot = await snapshot_when(handler, lambda s: not s["roots"])
    assert not any(snapshot["accounting_state"].values())
    assert not handler.sessions and not handler.roots


@pytest.mark.asyncio
async def test_anonymous_request_releases_accounting_without_session_final(handler):
    assert [chunk async for chunk in handler.generate({"token_ids": [1, 2]})] == []
    snapshot = await snapshot_when(
        handler, lambda s: s["accepted_steps"] == 1 and not s["roots"]
    )
    assert not any(snapshot["accounting_state"].values())
    assert not handler.sessions


@pytest.mark.asyncio
async def test_accounting_failure_returns_503_and_keeps_endpoints_available(
    handler, monkeypatch
):
    serving = asyncio.Event()
    stopped = asyncio.Event()

    async def serve_router(*args):
        serving.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    # Replace only the endpoint server; the actual accounting monitor supervises it.
    handler.test_monitor.cancel()
    await asyncio.gather(handler.test_monitor, return_exceptions=True)
    monkeypatch.setattr(main_module, "serve_router", serve_router)
    service = asyncio.create_task(main_module.serve(None, None, handler))
    request = asyncio.create_task(generate(handler, "queued", 11))
    try:
        await asyncio.wait_for(serving.wait(), 2)
        await snapshot_when(handler, lambda s: "queued" in s["programs"])
        fail_accounting(handler)
        with pytest.raises(
            AccountingUnavailable, match="accounting is unavailable"
        ) as error:
            await asyncio.wait_for(request, 3)
        assert error.value.code == 503
        assert not stopped.is_set()
        assert not service.done()
        assert handler._kv_router.sent is None
        assert [s async for s in handler.status()][0]["status"] == "unhealthy"
        with pytest.raises(AccountingUnavailable) as error:
            await generate(handler, "new")
        assert error.value.code == 503
        assert not handler.requests
        assert not handler._scheduler._admission_gates
        # Session cleanup remains available even though generation is unavailable.
        final = {"agent_context": {"session_id": "queued", "session_final": True}}
        assert [chunk async for chunk in handler.generate(final)] == []
        assert not handler.sessions and not handler.roots
        assert not handler._scheduler._table.programs
    finally:
        service.cancel()
        request.cancel()
        await asyncio.gather(service, request, return_exceptions=True)


@pytest_asyncio.fixture
async def streaming(handler, request):
    entered, finish = asyncio.Event(), asyncio.Event()

    class StreamingRouter(AccountingKvRouter):
        async def generate_from_request(self, request, **kwargs):
            await super().generate_from_request(request, **kwargs)

            async def chunks():
                yield {"token_ids": [1]}
                entered.set()
                await finish.wait()
                yield {"token_ids": [2], "finish_reason": "stop"}

            return chunks()

    handler._kv_router = StreamingRouter(handler.test_feed)
    payload = {"token_ids": [1]}
    if getattr(request, "param", None):
        payload["agent_context"] = {"session_id": request.param}
    stream = handler.generate(payload)
    pending = []

    async def start(pulling):
        assert await asyncio.wait_for(asyncio.create_task(anext(stream)), 2) == {
            "token_ids": [1]
        }
        if pulling:
            pending.append(asyncio.create_task(anext(stream)))
            await asyncio.wait_for(entered.wait(), 1)
            return pending[-1]

    try:
        yield stream, finish, start
    finally:
        finish.set()
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        await stream.aclose()


def fail_accounting(handler):
    handler.test_feed.emit(
        "accepted_output",
        steps=[
            dict(client_request_id="unjoined", root="bad", agent="bad", end=time.time())
        ],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("pulling", [False, True])
async def test_accounting_failure_does_not_cancel_dispatched_stream(
    handler, streaming, pulling
):
    stream, finish, start = streaming
    pending = await start(pulling)
    fail_accounting(handler)
    with pytest.raises(AccountingUnavailable):
        await asyncio.wait_for(handler.test_monitor, 3)
    finish.set()
    assert await asyncio.wait_for(
        pending if pending is not None else anext(stream), 1
    ) == {"token_ids": [2], "finish_reason": "stop"}
    with pytest.raises(StopAsyncIteration):
        await anext(stream)
    assert not handler.requests
    assert not handler.sessions and not handler.roots
    assert not handler._scheduler._table.programs


@pytest.mark.asyncio
async def test_ingress_failure_returns_503_before_dispatch(handler, monkeypatch):
    original = handler.ingress.emit

    def emit(event, *args, **kwargs):
        if event == "request_arrived":
            raise RuntimeError("Duru accounting queue overflow")
        return original(event, *args, **kwargs)

    monkeypatch.setattr(handler.ingress, "emit", emit)
    with pytest.raises(AccountingUnavailable) as error:
        await generate(handler, "overflow")
    assert error.value.code == 503
    assert handler.failure is not None
    assert handler._kv_router.sent is None
    assert not handler.requests
    with pytest.raises(AccountingUnavailable):
        await generate(handler, "next")


@pytest.mark.asyncio
async def test_intentional_shutdown_stops_monitor_without_failure(handler):
    await handler.shutdown()
    await asyncio.wait_for(handler.test_monitor, 1)
    assert handler.failure is None


@pytest.mark.asyncio
@pytest.mark.parametrize("pulling", [False, True])
@pytest.mark.parametrize("streaming", ["stream"], indirect=True)
async def test_session_final_closes_stream_across_per_chunk_tasks(
    handler, streaming, pulling
):
    stream, _, start = streaming
    pending = await start(pulling)
    await asyncio.wait_for(handler.end_session("stream"), 2)
    if pending is not None:
        with pytest.raises(asyncio.CancelledError):
            await pending
    with pytest.raises(StopAsyncIteration):
        await anext(stream)
    assert not handler.sessions
    await snapshot_when(handler, lambda s: not s["roots"])
