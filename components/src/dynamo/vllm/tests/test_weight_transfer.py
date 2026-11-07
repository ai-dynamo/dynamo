# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from unittest.mock import AsyncMock

import pytest

from dynamo.vllm.weight_transfer import WeightTransferSession

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.timeout(10),
]


@pytest.fixture
def session():
    return WeightTransferSession()


async def advance(session, through="flush"):
    await session.pause(AsyncMock(), "abort")
    if through == "pause":
        return
    await session.initialize({"init_info": {"packed": True}}, AsyncMock())
    session.begin({"session_id": "round-1", "weight_version": "policy-1"})
    if through == "begin":
        return
    await session.start({"session_id": "round-1", "seq": 0}, AsyncMock())
    if through == "start":
        return
    await session.update(
        {"session_id": "round-1", "seq": 1, "update_info": {}}, AsyncMock(), AsyncMock()
    )
    if through == "update":
        return
    await session.finish({"session_id": "round-1", "seq": 2}, AsyncMock(), AsyncMock())
    if through == "finish":
        return
    await session.flush(AsyncMock(return_value=True))


@pytest.mark.asyncio
async def test_install_requires_finish_and_explicit_cache_reset(session):
    with pytest.raises(ValueError, match="installed"):
        session.require_generation()
    await advance(session, "begin")
    await session.start({"session_id": "round-1", "seq": 0}, AsyncMock())
    with pytest.raises(ValueError):
        await session.resume(AsyncMock())
    apply = AsyncMock()
    sync = AsyncMock()
    await session.update(
        {"session_id": "round-1", "seq": 1, "update_info": {"names": ["w"]}},
        apply,
        sync,
    )
    apply.assert_awaited_once_with({"names": ["w"]})
    sync.assert_awaited_once_with()
    finish = AsyncMock()
    await session.finish({"session_id": "round-1", "seq": 2}, finish, sync)
    finish.assert_awaited_once_with("policy-1")
    assert session.snapshot().install_complete is True
    assert session.snapshot().session_id == "round-1"
    with pytest.raises(ValueError, match="cache"):
        await session.resume(AsyncMock())
    await session.flush(AsyncMock(return_value=True))
    await session.resume(AsyncMock())
    session.require_generation()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "phase,previous",
    [
        ("initialize", "pause"),
        ("start", "begin"),
        ("update", "start"),
        ("finish", "update"),
        ("flush", "finish"),
    ],
)
@pytest.mark.parametrize("error_type", [RuntimeError, asyncio.CancelledError])
async def test_dispatched_failure_permanently_latches(
    session, phase, previous, error_type
):
    bad = AsyncMock(side_effect=error_type())
    await advance(session, previous)
    operations = {
        "initialize": lambda: session.initialize({"init_info": {}}, bad),
        "start": lambda: session.start({"session_id": "round-1", "seq": 0}, bad),
        "update": lambda: session.update(
            {"session_id": "round-1", "seq": 1, "update_info": {}}, bad, AsyncMock()
        ),
        "finish": lambda: session.finish(
            {"session_id": "round-1", "seq": 2}, bad, AsyncMock()
        ),
        "flush": lambda: session.flush(bad),
    }
    with pytest.raises(error_type):
        await operations[phase]()
    assert session.snapshot().failed is True
    with pytest.raises(ValueError, match="failed"):
        await session.resume(AsyncMock())
    with pytest.raises(ValueError, match="failed"):
        session.require_generation()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        {"session_id": "wrong", "seq": 0},
        {"session_id": "round-1", "seq": 1},
        {"session_id": "round-1", "seq": True},
        {"session_id": "round-1", "seq": 0, "engine_rpc": "unsafe"},
    ],
)
async def test_start_rejects_invalid_sequence_without_dispatch(session, body):
    await advance(session, "begin")
    rpc = AsyncMock()
    with pytest.raises(ValueError):
        await session.start(body, rpc)
    rpc.assert_not_awaited()
    assert session.snapshot().failed is False


@pytest.mark.asyncio
async def test_init_and_cache_require_pause(session):
    rpc = AsyncMock()
    with pytest.raises(ValueError, match="paused"):
        await session.initialize({"init_info": {}}, rpc)
    with pytest.raises(ValueError, match="paused"):
        await session.flush(rpc)
    rpc.assert_not_awaited()


@pytest.mark.asyncio
async def test_empty_install_and_duplicate_session_rejected(session):
    await advance(session, "begin")
    await session.start({"session_id": "round-1", "seq": 0}, AsyncMock())
    with pytest.raises(ValueError, match="chunk"):
        await session.finish(
            {"session_id": "round-1", "seq": 1}, AsyncMock(), AsyncMock()
        )
    with pytest.raises(ValueError):
        session.begin({"session_id": "round-2", "weight_version": "policy-2"})


@pytest.mark.asyncio
async def test_false_cache_reset_is_failure(session):
    await advance(session, "finish")
    with pytest.raises(ValueError, match="cache"):
        await session.flush(AsyncMock(return_value=False))
    assert session.snapshot().failed is True


@pytest.mark.asyncio
async def test_device_sync_failure_latches_before_update_ack(session):
    await advance(session, "start")
    with pytest.raises(RuntimeError):
        await session.update(
            {"session_id": "round-1", "seq": 1, "update_info": {}},
            AsyncMock(),
            AsyncMock(side_effect=RuntimeError()),
        )
    assert session.snapshot().failed is True


@pytest.mark.asyncio
async def test_operation_timeout_latches():
    session = WeightTransferSession(timeout_s=0.001)
    await session.pause(AsyncMock(), "wait")
    with pytest.raises(TimeoutError):
        await session.initialize({"init_info": {}}, lambda _: asyncio.Event().wait())
    assert session.snapshot().failed is True


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_timeout_must_be_finite_and_positive(timeout):
    with pytest.raises(ValueError):
        WeightTransferSession(timeout_s=timeout)


@pytest.mark.asyncio
async def test_second_complete_round_requires_unique_identity(session):
    await advance(session)
    for payload in (
        {"session_id": "round-1", "weight_version": "policy-2"},
        {"session_id": "round-2", "weight_version": "policy-1"},
    ):
        with pytest.raises(ValueError, match="unique"):
            session.begin(payload)
    session.begin({"session_id": "round-2", "weight_version": "policy-2"})
    assert session.snapshot().pending_weight_version == "policy-2"
    with pytest.raises(ValueError):
        session.require_generation()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body", [None, [], {"init_info": []}, {"init_info": {}, "extra": True}]
)
async def test_invalid_init_never_dispatches(session, body):
    await session.pause(AsyncMock(), "abort")
    rpc = AsyncMock()
    with pytest.raises(ValueError):
        await session.initialize(body, rpc)
    rpc.assert_not_awaited()


@pytest.mark.asyncio
async def test_duplicate_initialization_rejected(session):
    await advance(session, "begin")
    with pytest.raises(ValueError, match="already"):
        await session.initialize({"init_info": {}}, AsyncMock())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field,value",
    [
        ("session_id", ""),
        ("session_id", 1),
        ("weight_version", None),
        ("weight_version", "x" * 257),
    ],
)
async def test_invalid_session_identity_rejected(session, field, value):
    await session.pause(AsyncMock(), "abort")
    await session.initialize({"init_info": {}}, AsyncMock())
    with pytest.raises(ValueError):
        session.begin(
            {"session_id": "round-1", "weight_version": "policy-1", field: value},
        )


@pytest.mark.asyncio
async def test_invalid_update_payload_and_premature_flush_rejected(session):
    await advance(session, "begin")
    with pytest.raises(ValueError):
        await session.flush(AsyncMock(return_value=True))
    await session.start({"session_id": "round-1", "seq": 0}, AsyncMock())
    rpc = AsyncMock()
    with pytest.raises(ValueError, match="object"):
        await session.update(
            {"session_id": "round-1", "seq": 1, "update_info": []},
            rpc,
            AsyncMock(),
        )
    rpc.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["keep", None, "unknown"])
async def test_pause_rejects_unsafe_mode_without_dispatch(session, mode):
    rpc = AsyncMock()
    with pytest.raises(ValueError, match="wait/abort"):
        await session.pause(rpc, mode)
    rpc.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [RuntimeError, asyncio.CancelledError])
async def test_pause_failure_latches(session, error):
    with pytest.raises(error):
        await session.pause(AsyncMock(side_effect=error()), "abort")
    assert session.snapshot().failed is True


@pytest.mark.asyncio
async def test_failed_session_can_pause_to_fence_traffic_but_never_reopen(session):
    session.fail()
    rpc = AsyncMock()
    await session.pause(rpc, "abort")
    rpc.assert_awaited_once_with(mode="abort", clear_cache=False)
    with pytest.raises(ValueError, match="failed"):
        session.require_generation()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["pause", "resume"])
async def test_pause_resume_share_the_session_timeout(phase):
    session = WeightTransferSession(timeout_s=0.001)
    if phase == "resume":
        await advance(session)
        operation = session.resume(lambda: asyncio.Event().wait())
    else:
        operation = session.pause(lambda **_: asyncio.Event().wait(), "wait")
    with pytest.raises(TimeoutError):
        await operation
    assert session.snapshot().failed is True


@pytest.mark.asyncio
async def test_installed_policy_stays_closed_until_successful_resume(session):
    await advance(session)
    with pytest.raises(ValueError, match="paused"):
        session.require_generation()
    with pytest.raises(RuntimeError):
        await session.resume(AsyncMock(side_effect=RuntimeError()))
    assert session.snapshot().failed is True
