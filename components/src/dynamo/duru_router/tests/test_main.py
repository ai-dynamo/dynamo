# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared request handling with the real Duru admission scheduler."""

import asyncio
import copy
import json
import os
import struct
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import pytest_asyncio

from dynamo.duru_router import __main__ as main_module
from dynamo.duru_router.__main__ import AccountingUnavailable, DuruRouterHandler
from dynamo.duru_router.events import CompactFeed
from dynamo.duru_router.router import ThunderAgentConfig
from dynamo.duru_router.scoring import (
    AccountingError,
    DispatchAckTimeout,
    Ingress,
    RouterAccounting,
)
from dynamo.duru_router.tests.helpers import configuration
from dynamo.duru_router.tests.helpers import memory_mode as memory_mode
from dynamo.duru_router.tests.helpers import work_step
from dynamo.thunderagent_router.program_state import ProgramLifecycle, ProgramStatus
from dynamo.thunderagent_router.tests.test_main import FakeKvRouter

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.integration,
    pytest.mark.gpu_0,
    pytest.mark.timeout(10),
]


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
        now = time.monotonic()
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
    thread_failures = []
    original_thread_hook = threading.excepthook

    def collect_feed_failure(args):
        if args.thread in (feed.thread, feed.journal.thread):
            thread_failures.append(args.exc_value)
        else:
            original_thread_hook(args)

    monkeypatch.setattr(threading, "excepthook", collect_feed_failure)
    owner = RouterAccounting(path, ingress=ingress)
    snapshot_changed = asyncio.Event()
    loop = asyncio.get_running_loop()
    publish = owner._router_publish

    def notify_snapshot(snapshot):
        publish(snapshot)
        loop.call_soon_threadsafe(snapshot_changed.set)

    monkeypatch.setattr(owner, "_router_publish", notify_snapshot)
    owner.start()
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
    result.test_snapshot_changed = snapshot_changed
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
        if owner._result.done() and owner._result.exception() is not None:
            # Fault injection disconnects the producers; they may observe it
            # either on their next heartbeat or during their final flush.
            for result in results:
                assert result is None or isinstance(result, RuntimeError)
            assert isinstance(owner_result, RuntimeError)
            assert "accounting failed" in str(owner_result)
            assert all(
                error is feed.error or error is feed.journal.error
                for error in thread_failures
            )
        else:
            assert not thread_failures
            assert results == [None, None]
            assert owner_result is None


async def snapshot_when(handler, predicate, *, admissions=True):
    async with asyncio.timeout(5):
        while True:
            handler.test_snapshot_changed.clear()
            if admissions:
                snapshot = handler.accounting.snapshot()
            else:
                handler.accounting.check(admissions=False)
                snapshot = handler.accounting._snapshot
            if snapshot and predicate(snapshot):
                return snapshot
            await handler.test_snapshot_changed.wait()


@pytest.fixture
def timeout_snapshot(monkeypatch):
    """Inject the scorer's expired-deadline result without waiting five minutes."""
    control = {}
    publish = RouterAccounting._router_publish

    def inject(self, snapshot):
        if snapshot and control:
            snapshot = dict(snapshot, dispatch_timeout=dict(control))
        publish(self, snapshot)

    monkeypatch.setattr(RouterAccounting, "_router_publish", inject)
    return control


async def generate(handler, pid, tokens=8, **kwargs):
    request = {
        "token_ids": struct.pack(f"{tokens}I", *range(tokens)),
        "agent_context": {"session_id": pid},
        "elapsed_since_arrival_ms": 12,
    }
    return [chunk async for chunk in handler.generate(request, **kwargs)]


@pytest.mark.asyncio
async def test_status_propagates_unexpected_snapshot_errors(handler, monkeypatch):
    failure = ValueError("malformed snapshot")

    def fail():
        raise failure

    monkeypatch.setattr(handler.accounting.reader, "refresh", fail)
    with pytest.raises(ValueError) as caught:
        _ = [status async for status in handler.status()]
    assert caught.value is failure
    assert handler.failure is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "phase",
    [
        "health",
        "arrival",
        "dispatch_health",
        "dispatch",
        "conversion",
        "completion",
        "agent_end",
        "agent_group_end",
        "monitor",
    ],
)
async def test_accounting_failure_logs_request_context(
    handler, monkeypatch, caplog, phase
):
    handler.test_monitor.cancel()
    await asyncio.gather(handler.test_monitor, return_exceptions=True)
    failure = AccountingError("injected accounting error")
    request_id = "external-context-" + phase
    context = SimpleNamespace(id=lambda: request_id)
    if phase in ("agent_end", "agent_group_end"):
        await generate(handler, "session")
    with monkeypatch.context() as patch:
        if phase in ("health", "monitor", "dispatch_health"):

            def fail_check(**kwargs):
                raise failure

            if phase == "dispatch_health":
                dispatch = handler._before_dispatch

                async def fail_before_dispatch(request):
                    with monkeypatch.context() as nested:
                        nested.setattr(handler.accounting, "check", fail_check)
                        await dispatch(request)

                patch.setattr(handler, "_before_dispatch", fail_before_dispatch)
            else:
                patch.setattr(handler.accounting, "check", fail_check)
        else:
            failing_event = {
                "arrival": "request_arrived",
                "dispatch": "request_dispatched",
                "conversion": "request_dispatch_rejected",
                "completion": "request_completed",
                "agent_end": "agent_end",
                "agent_group_end": "agent_group_end",
            }[phase]
            emit = handler.ingress.emit

            def fail_emit(event, *args, **fields):
                if event == failing_event:
                    raise failure
                return emit(event, *args, **fields)

            patch.setattr(handler.ingress, "emit", fail_emit)
            if phase == "conversion":

                def reject(*args, **kwargs):
                    raise ValueError("conversion rejected")

                patch.setattr(handler._kv_router, "generate_from_request", reject)
        if phase == "monitor":
            with pytest.raises(AccountingUnavailable):
                await handler.monitor()
        elif phase == "completion":
            assert await generate(handler, "session", context=context) == []
        elif phase in ("agent_end", "agent_group_end"):
            final = {"agent_context": {"session_id": "session", "session_final": True}}
            with pytest.raises(AccountingUnavailable):
                _ = [chunk async for chunk in handler.generate(final, context=context)]
        else:
            with pytest.raises(AccountingUnavailable):
                await generate(handler, "session", context=context)
    records = [
        record
        for record in caplog.records
        if record.name == main_module.logger.name and record.levelname == "ERROR"
    ]
    assert len(records) == 1
    expected_id = None if phase == "monitor" else request_id
    assert records[0].args == (expected_id,)
    assert records[0].exc_info[1] is failure


@pytest.mark.asyncio
async def test_interleaved_requests_keep_their_own_error_context(
    handler, monkeypatch, caplog
):
    handler._kv_router = AccountingKvRouter(
        handler.test_feed, chunks=[{"token_ids": [1]}]
    )
    first_context = SimpleNamespace(id=lambda: "first-external-request")
    second_context = SimpleNamespace(id=lambda: "second-external-request")
    stream = handler.generate(
        {"token_ids": [1], "agent_context": {"session_id": "first"}},
        context=first_context,
    )
    try:
        assert await anext(stream) == {"token_ids": [1]}
        assert await generate(handler, "second", tokens=1, context=second_context) == [
            {"token_ids": [1]}
        ]
        emit = handler.ingress.emit

        def fail_completion(event, *args, **fields):
            if event == "request_completed":
                raise RuntimeError("completion recording failed")
            return emit(event, *args, **fields)

        monkeypatch.setattr(handler.ingress, "emit", fail_completion)
        with pytest.raises(StopAsyncIteration):
            await anext(stream)
        records = [
            record
            for record in caplog.records
            if record.name == main_module.logger.name and record.levelname == "ERROR"
        ]
        assert len(records) == 1
        assert records[0].args == ("first-external-request",)
    finally:
        await stream.aclose()


@pytest.mark.asyncio
async def test_invalid_dispatch_timeout_closes_ingress_on_startup(
    tmp_path, monkeypatch
):
    config = tmp_path / "accounting.json"
    config.write_text(
        json.dumps(
            dict(
                ingress=str(tmp_path / "duru-ingress.jsonl"),
                scores=str(tmp_path / "scores"),
            )
        )
    )
    monkeypatch.setenv("DURU_ACCOUNTING_CONFIG", str(config))
    monkeypatch.setenv("DURU_DISPATCH_ACK_TIMEOUT_SECONDS", "nan")
    monkeypatch.setenv("DURU_SCORES", "unused")
    ingress = SimpleNamespace(journal=object(), closed=False)

    def close():
        ingress.closed = True

    ingress.close = close
    monkeypatch.setattr(main_module, "parse_args", lambda: SimpleNamespace())
    monkeypatch.setattr(main_module, "Ingress", lambda *args, **kwargs: ingress)
    with pytest.raises(ValueError, match="DURU_DISPATCH_ACK_TIMEOUT_SECONDS"):
        await main_module.worker.__wrapped__(None)
    assert ingress.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("wall_jump", [-3600, 3600])
async def test_live_accounting_survives_wall_clock_adjustments(
    handler, monkeypatch, wall_jump
):
    await generate(handler, "clock-test")
    first = await snapshot_when(handler, lambda s: s["accepted_steps"] == 1)
    adjusted_wall = time.time() + wall_jump
    monkeypatch.setattr(time, "time", lambda: adjusted_wall)

    await generate(handler, "clock-test")
    second = await snapshot_when(handler, lambda s: s["accepted_steps"] == 2)

    assert second["observation_time"] >= first["observation_time"]
    assert 0 <= time.monotonic() - second["time"] < 5
    assert handler.accounting.reader.refresh()
    assert handler.accounting.reader.reason == "fresh"
    assert handler.failure is None


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


@pytest.mark.parametrize(
    "alpha_args,expected",
    [([], "1.0"), (["--alpha", "0.7"], "0.7")],
    ids=["default", "explicit"],
)
def test_main_parses_alpha_and_forwards_router_arguments(
    monkeypatch, tmp_path, alpha_args, expected
):
    monkeypatch.setenv("DURU_SCORES", str(tmp_path / "scores.json"))
    monkeypatch.setenv("DURU_ACCOUNTING_CONFIG", str(tmp_path / "scoring.json"))
    monkeypatch.delenv("DURU_ADMISSION_ALPHA", raising=False)
    monkeypatch.setattr(
        sys, "argv", ["duru", *alpha_args, "--endpoint", "test.vllm.generate"]
    )
    invocation = object()
    started = []
    monkeypatch.setattr(main_module, "worker", lambda: invocation)
    monkeypatch.setattr(main_module, "uvloop", SimpleNamespace(run=started.append))

    main_module.main()

    assert started == [invocation]
    assert os.environ["DURU_ADMISSION_ALPHA"] == expected
    assert sys.argv == ["duru", "--endpoint", "test.vllm.generate"]


@pytest.mark.asyncio
@pytest.mark.timeout(10)
@pytest.mark.parametrize("waiter", ["none", "retry", "shutdown"])
async def test_session_final_survives_caller_cancellation(handler, waiter):
    started = asyncio.Event()
    closing = asyncio.Event()
    release = asyncio.Event()
    closed = asyncio.Event()

    class SlowClosingRouter(AccountingKvRouter):
        async def generate_from_request(self, request, **kwargs):
            await super().generate_from_request(request, **kwargs)

            async def chunks():
                started.set()
                try:
                    await asyncio.Event().wait()
                    yield {"token_ids": [1]}
                finally:
                    closing.set()
                    await release.wait()
                    closed.set()

            return chunks()

    handler._kv_router = SlowClosingRouter(handler.test_feed)
    final = {"agent_context": {"session_id": "session", "session_final": True}}

    async def finish():
        return [chunk async for chunk in handler.generate(final)]

    tasks = []
    try:
        request = asyncio.create_task(generate(handler, "session"))
        tasks.append(request)
        await asyncio.wait_for(started.wait(), 2)
        caller = asyncio.create_task(finish())
        tasks.append(caller)
        await asyncio.wait_for(closing.wait(), 2)
        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller
        assert not closed.is_set()

        if waiter != "none":
            entered = asyncio.Event()

            async def wait_for_cleanup():
                entered.set()
                if waiter == "retry":
                    return await finish()
                return await handler.shutdown()

            pending = asyncio.create_task(wait_for_cleanup())
            tasks.append(pending)
            await asyncio.wait_for(entered.wait(), 1)
            assert not pending.done()

        release.set()
        await asyncio.wait_for(closed.wait(), 2)
        with pytest.raises(asyncio.CancelledError):
            await request
        if waiter != "none":
            await asyncio.wait_for(pending, 2)
        await snapshot_when(handler, lambda s: not s["roots"])
        assert not handler.sessions
        assert not handler.roots
        assert not handler.requests
        assert not handler._scheduler._table.programs
        assert not handler._scheduler._admission_gates
    finally:
        release.set()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


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
@pytest.mark.timeout(10)
@pytest.mark.parametrize("phase", ["conversion", "await", "stream"])
@pytest.mark.parametrize("anonymous", [False, True])
async def test_dispatch_rejection_only_releases_confirmed_unsent_work(
    handler, phase, anonymous
):
    failure = RuntimeError("Invalid sampling option")
    sent = []

    class RejectingRouter:
        def generate_from_request(self, request, **kwargs):
            sent.append(request)
            if phase == "conversion":
                # PyO3 request conversion raises before returning an awaitable.
                raise failure

            async def dispatch():
                if phase == "await":
                    raise failure

                async def chunks():
                    raise failure
                    yield

                return chunks()

            return dispatch()

    handler._kv_router = RejectingRouter()
    payload = {
        "token_ids": [1, 2],
        "sampling_options": {"temperature": "invalid"},
    }
    if not anonymous:
        payload["agent_context"] = {"session_id": "rejected"}
    with pytest.raises(RuntimeError) as error:
        _ = [chunk async for chunk in handler.generate(payload)]
    assert error.value is failure
    if not anonymous:
        await handler.end_session("rejected")
    cutoff = time.monotonic()
    handler.emit("u_clock")
    snapshot = await snapshot_when(handler, lambda s: s["observation_time"] >= cutoff)
    assert not handler.sessions
    assert not handler.requests
    assert not handler._scheduler._table.programs
    assert not snapshot["requests"]
    if phase == "conversion":
        assert not snapshot["roots"]
        assert not any(snapshot["accounting_state"].values())
    else:
        assert snapshot["accounting_state"]["worker_requests"] == 1
        assert snapshot["accounting_state"]["request_identities"] == 1
        assert snapshot["accounting_state"]["closing_roots"] == 1
        identity = sent[0]["extra_args"]["sampling_options"]["vllm_xargs"]
        handler.test_feed.emit(
            "worker_request_finished",
            requests=[
                dict(
                    client_request_id=identity["duru_request_id"],
                    root=identity["duru_root_id"],
                    agent=identity["duru_agent_id"],
                )
            ],
        )
        snapshot = await snapshot_when(handler, lambda s: not s["roots"])
        assert not any(snapshot["accounting_state"].values())


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
            dict(
                client_request_id="unjoined",
                root="bad",
                agent="bad",
                end=time.monotonic(),
            )
        ],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["completed", "backend_error", "cancelled"])
async def test_dispatch_ack_timeout_stops_admission_but_drains_dispatched_work(
    timeout_snapshot, handler, monkeypatch, outcome
):
    missing_identity = {}
    dispatch_error = RuntimeError("delivery uncertain")

    class MissingAckRouter:
        async def generate_from_request(self, request, **kwargs):
            extra = request["extra_args"]["sampling_options"]["vllm_xargs"]
            missing_identity.update(
                client_request_id=extra["duru_request_id"],
                root=extra["duru_root_id"],
                agent=extra["duru_agent_id"],
            )
            raise dispatch_error

    handler._kv_router = MissingAckRouter()
    with pytest.raises(RuntimeError) as caught:
        await generate(handler, "lost", tokens=1)
    assert caught.value is dispatch_error
    await handler.end_session("lost")
    await snapshot_when(
        handler,
        lambda s: s["accounting_state"]["worker_requests"] == 1
        and s["accounting_state"]["closing_roots"] == 1,
    )

    entered, release = asyncio.Event(), asyncio.Event()
    backend_error = RuntimeError("backend stream failed after timeout")

    class ActiveRouter(AccountingKvRouter):
        async def generate_from_request(self, request, **kwargs):
            await super().generate_from_request(request, **kwargs)

            async def chunks():
                yield {"token_ids": [1]}
                entered.set()
                await release.wait()
                if outcome == "backend_error":
                    raise backend_error
                yield {"token_ids": [2], "finish_reason": "stop"}

            return chunks()

    handler._kv_router = ActiveRouter(handler.test_feed)
    stream = handler.generate(
        {"token_ids": [1], "agent_context": {"session_id": "active"}}
    )
    pending = None
    queued = None
    try:
        assert await asyncio.wait_for(anext(stream), 2) == {"token_ids": [1]}
        active = handler._scheduler._table.programs["active"]
        pending = asyncio.create_task(anext(stream))
        await asyncio.wait_for(entered.wait(), 1)
        queued = asyncio.create_task(generate(handler, "queued", tokens=11))
        await snapshot_when(handler, lambda s: "queued" in s["programs"])
        heartbeat = asyncio.Event()
        emit = handler.emit

        def observe_heartbeat(event, **fields):
            emit(event, **fields)
            if event == "u_clock" and isinstance(handler.failure, DispatchAckTimeout):
                heartbeat.set()

        monkeypatch.setattr(handler, "emit", observe_heartbeat)
        timeout_snapshot.update(
            request_id=missing_identity["client_request_id"],
            timeout_seconds=300.0,
            failed_at=time.monotonic() - 300.0,
        )
        await asyncio.wait_for(heartbeat.wait(), 2)
        assert isinstance(handler.failure, DispatchAckTimeout)
        assert not handler.test_monitor.done()
        assert not pending.done()
        handler.accounting.check(admissions=False)
        with pytest.raises(AccountingUnavailable) as rejected:
            await asyncio.wait_for(queued, 2)
        assert rejected.value.code == 503
        assert missing_identity["client_request_id"] in str(rejected.value)
        with pytest.raises(AccountingUnavailable) as rejected:
            await generate(handler, "new")
        assert rejected.value.code == 503
        status = [row async for row in handler.status()][0]
        assert status["status"] == "unhealthy"
        assert missing_identity["client_request_id"] in status["error"]

        if outcome == "cancelled":
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
        else:
            release.set()
            if outcome == "backend_error":
                with pytest.raises(RuntimeError) as caught:
                    await pending
                assert caught.value is backend_error
            else:
                assert await pending == {"token_ids": [2], "finish_reason": "stop"}
                with pytest.raises(StopAsyncIteration):
                    await anext(stream)
        assert active.status == ProgramStatus.ACTING
        assert active.token_total == (3 if outcome == "completed" else 2)
        assert not handler.requests
        await handler.end_session("active")
        await handler.end_session("queued")
        snapshot = await snapshot_when(
            handler,
            lambda s: not s["requests"]
            and set(s["roots"]) == {missing_identity["root"]},
            admissions=False,
        )
        assert snapshot["accounting_state"]["worker_requests"] == 1
        handler.test_feed.emit("worker_request_finished", requests=[missing_identity])
        snapshot = await snapshot_when(
            handler, lambda s: not s["roots"], admissions=False
        )
        assert not any(snapshot["accounting_state"].values())
        assert not handler.accounting._result.done()
        assert not handler.test_monitor.done()
        with pytest.raises(AccountingUnavailable):
            await generate(handler, "still-blocked-after-late-ack")

        # An admission-only timeout must not mask a later accounting failure.
        fail_accounting(handler)
        with pytest.raises(AccountingUnavailable):
            await asyncio.wait_for(handler.test_monitor, 2)
        assert not isinstance(handler.failure, DispatchAckTimeout)
        assert not handler.can_record()
    finally:
        release.set()
        for task in (pending, queued):
            if task is not None:
                task.cancel()
        await asyncio.gather(
            *(task for task in (pending, queued) if task is not None),
            return_exceptions=True,
        )
        await stream.aclose()


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
@pytest.mark.timeout(10)
@pytest.mark.parametrize("backend_fails", [False, True])
@pytest.mark.parametrize("monitor_observes", [False, True])
async def test_failed_accounting_does_not_replace_stream_outcome_with_paused_program(
    handler, backend_fails, monitor_observes
):
    backend_error = RuntimeError("Backend stream failed")

    class CompletingRouter(AccountingKvRouter):
        async def generate_from_request(self, request, **kwargs):
            await super().generate_from_request(request, **kwargs)

            async def chunks():
                yield {"token_ids": [1]}
                if backend_fails:
                    raise backend_error
                yield {"token_ids": [2], "finish_reason": "stop"}

            return chunks()

    handler._kv_router = CompletingRouter(handler.test_feed)
    stream = handler.generate(
        {"token_ids": [1], "agent_context": {"session_id": "active"}}
    )
    scheduler = handler._scheduler
    try:
        assert await asyncio.wait_for(anext(stream), 2) == {"token_ids": [1]}
        active = scheduler._table.programs["active"]
        scheduler._table.begin_request("paused", 20)
        paused = scheduler._table.end_request("paused", 20, 0)
        assert await scheduler._pause_acting("paused")
        resumed = scheduler._stat_resumes
        if not monitor_observes:
            handler.test_monitor.cancel()
            await asyncio.gather(handler.test_monitor, return_exceptions=True)
        fail_accounting(handler)
        handler.emit("u_clock")
        if monitor_observes:
            with pytest.raises(AccountingUnavailable):
                await asyncio.wait_for(handler.test_monitor, 3)
        else:
            await asyncio.to_thread(handler.accounting._thread.join, 3)
            assert not handler.accounting._thread.is_alive()
            assert handler.accounting._result.done()
            assert handler.accounting._result.exception() is not None
            assert handler.failure is None

        if backend_fails:
            with pytest.raises(RuntimeError) as error:
                _ = [chunk async for chunk in stream]
            assert error.value is backend_error
        else:
            assert [chunk async for chunk in stream] == [
                {"token_ids": [2], "finish_reason": "stop"}
            ]
        assert active.status == ProgramStatus.ACTING
        assert active.token_total == (2 if backend_fails else 3)
        assert paused.lifecycle == ProgramLifecycle.PAUSED
        assert scheduler._stat_resumes == resumed
        assert not handler.requests
        with pytest.raises(AccountingUnavailable):
            await generate(handler, "new")
    finally:
        await stream.aclose()
        await handler.end_session("active")
        await scheduler.end_program("paused")


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
