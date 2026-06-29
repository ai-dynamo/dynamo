# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from dynamo.common.gms_failover import (
    acquire_gms_failover_lock_before_init,
    prepare_gms_failover,
    release_attached_gms_failover_lock,
    run_gms_failover_post_lock_fence,
    run_gms_failover_promotion_warmup,
)

try:
    from gpu_memory_service.failover_lock.interface import FailoverLockError
except ImportError:
    FailoverLockError = RuntimeError

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.none,
    pytest.mark.gpu_0,
]


class _Controller:
    def __init__(self):
        self.quiesce_calls = []
        self.resume_calls = []
        self.mark_resumed_calls = 0

    async def quiesce(self, tags):
        self.quiesce_calls.append(tags)
        return True

    async def resume(self, tags):
        self.resume_calls.append(tags)
        return True

    def mark_resumed(self):
        self.mark_resumed_calls += 1


class _Owner:
    def __init__(self):
        self._quiesce_controller = _Controller()


class _Runtime:
    def __init__(self):
        self.health = []

    def set_health_status(self, ready):
        self.health.append(ready)


class _Lock:
    def __init__(self, path):
        self.path = path
        self.acquired = []
        self.released = 0

    async def acquire(self, engine_id, timeout=None):
        self.acquired.append(engine_id)

    async def release(self):
        self.released += 1


class _BusyOnTryLock(_Lock):
    async def acquire(self, engine_id, timeout=None):
        if timeout == 0.0:
            raise FailoverLockError("lock already held")
        await super().acquire(engine_id, timeout=timeout)


@pytest.mark.asyncio
async def test_gms_failover_disabled_keeps_vanilla_path(monkeypatch):
    monkeypatch.delenv("DYN_GMS_FAILOVER_SHADOW_MODE", raising=False)
    owner = _Owner()
    runtime = _Runtime()

    activation = await prepare_gms_failover(
        owner,
        runtime,
        backend_name="test",
        lock_factory=_Lock,
    )

    assert activation.enabled is False
    assert owner._quiesce_controller.quiesce_calls == []
    assert owner._quiesce_controller.resume_calls == []
    assert runtime.health == []


@pytest.mark.asyncio
async def test_gms_failover_primary_acquires_without_quiesce(monkeypatch):
    monkeypatch.setenv("DYN_GMS_FAILOVER_SHADOW_MODE", "true")
    monkeypatch.setenv("ENGINE_ID", "0")
    monkeypatch.setenv("FAILOVER_LOCK_PATH", "/locks/failover.lock")
    owner = _Owner()

    activation = await prepare_gms_failover(
        owner,
        _Runtime(),
        backend_name="test",
        lock_factory=_Lock,
    )

    assert activation.enabled is True
    assert activation.lock.path == "/locks/failover.lock"
    assert activation.lock.acquired == ["engine-0"]
    assert owner._quiesce_controller.quiesce_calls == []


@pytest.mark.asyncio
async def test_gms_failover_shadow_waits_quiesced_then_resumes(monkeypatch):
    monkeypatch.setenv("DYN_GMS_FAILOVER_SHADOW_MODE", "true")
    monkeypatch.setenv("ENGINE_ID", "1")
    owner = _Owner()
    runtime = _Runtime()

    activation = await prepare_gms_failover(
        owner,
        runtime,
        backend_name="test",
        tags=["kv_cache"],
        lock_factory=_BusyOnTryLock,
    )

    assert activation.enabled is True
    assert activation.lock.acquired == ["engine-1"]
    assert owner._quiesce_controller.quiesce_calls == [["kv_cache"]]
    assert owner._quiesce_controller.resume_calls == [["kv_cache"]]
    assert owner._quiesce_controller.mark_resumed_calls == 1
    assert runtime.health == [True]

    class _Handler:
        pass

    handler = _Handler()
    activation.attach_to(handler)
    assert getattr(handler, "_gms_failover_lock") is activation.lock


@pytest.mark.asyncio
async def test_gms_failover_can_mark_waiting_shadow_unready_when_configured(
    monkeypatch,
):
    monkeypatch.setenv("DYN_GMS_FAILOVER_SHADOW_MODE", "true")
    monkeypatch.setenv("DYN_GMS_FAILOVER_KEEP_SHADOW_READY", "false")
    monkeypatch.setenv("ENGINE_ID", "1")
    owner = _Owner()
    runtime = _Runtime()

    activation = await prepare_gms_failover(
        owner,
        runtime,
        backend_name="test",
        tags=["kv_cache"],
        lock_factory=_BusyOnTryLock,
    )

    assert activation.enabled is True
    assert runtime.health == [False, True]


@pytest.mark.asyncio
async def test_gms_failover_replacement_primary_index_becomes_shadow_when_lock_busy(
    monkeypatch,
):
    monkeypatch.setenv("DYN_GMS_FAILOVER_SHADOW_MODE", "true")
    # Replacement pods can reuse index 0 after a failover while index 1 is the
    # current active holder. The lock, not the static index, must decide role.
    monkeypatch.setenv("ENGINE_ID", "0")
    owner = _Owner()
    runtime = _Runtime()

    activation = await prepare_gms_failover(
        owner,
        runtime,
        backend_name="test",
        tags=["kv_cache"],
        lock_factory=_BusyOnTryLock,
    )

    assert activation.enabled is True
    assert activation.lock.acquired == ["engine-0"]
    assert owner._quiesce_controller.quiesce_calls == [["kv_cache"]]
    assert owner._quiesce_controller.resume_calls == [["kv_cache"]]
    assert owner._quiesce_controller.mark_resumed_calls == 1
    assert runtime.health == [True]


@pytest.mark.asyncio
async def test_gms_failover_private_bootstrap_primary_uses_active_path(monkeypatch):
    monkeypatch.setenv("DYN_GMS_FAILOVER_SHADOW_MODE", "true")
    monkeypatch.setenv("DYN_TEST_GMS_PRIVATE_BOOTSTRAP_KV", "true")
    monkeypatch.setenv("DYN_GMS_FAILOVER_PRIMARY_ENGINE_ID", "0")
    monkeypatch.setenv("ENGINE_ID", "0")
    owner = _Owner()
    runtime = _Runtime()

    activation = await prepare_gms_failover(
        owner,
        runtime,
        backend_name="test",
        tags=["kv_cache"],
        lock_factory=_Lock,
    )

    assert activation.enabled is True
    assert activation.lock.acquired == ["engine-0"]
    assert owner._quiesce_controller.quiesce_calls == []
    assert owner._quiesce_controller.resume_calls == []
    assert owner._quiesce_controller.mark_resumed_calls == 0
    assert runtime.health == []


@pytest.mark.asyncio
async def test_gms_failover_pre_init_lock_acquires_without_quiesce(monkeypatch):
    monkeypatch.setenv("DYN_GMS_FAILOVER_SHADOW_MODE", "true")
    monkeypatch.setenv("ENGINE_ID", "1")
    monkeypatch.setenv("FAILOVER_LOCK_PATH", "/locks/failover.lock")

    activation = await acquire_gms_failover_lock_before_init(
        backend_name="test",
        lock_factory=_Lock,
    )

    assert activation.enabled is True
    assert activation.lock.path == "/locks/failover.lock"
    assert activation.lock.acquired == ["engine-1"]


@pytest.mark.asyncio
async def test_gms_failover_promotion_warmup_drains_non_error_stream(monkeypatch):
    monkeypatch.delenv("DYN_GMS_FAILOVER_PROMOTION_WARMUP", raising=False)
    seen = []

    async def generate(request, context):
        seen.append((request, context.id(), context.trace_headers()))
        yield {"token_ids": [1], "finish_reason": None}
        yield {"token_ids": [], "finish_reason": "stop"}

    await run_gms_failover_promotion_warmup(
        generate,
        {"token_ids": [1], "stop_conditions": {"max_tokens": 1}},
        backend_name="test",
    )

    assert len(seen) == 1
    assert seen[0][0]["token_ids"] == [1]
    assert seen[0][1].startswith("gms-failover-promotion-warmup-")
    assert seen[0][2] == {}


@pytest.mark.asyncio
async def test_gms_failover_promotion_warmup_rejects_error_chunk(monkeypatch):
    monkeypatch.setenv("DYN_GMS_FAILOVER_PROMOTION_WARMUP_ATTEMPTS", "1")

    async def generate(_request, _context):
        yield {"status": "error", "message": "not ready"}

    with pytest.raises(RuntimeError, match="not ready"):
        await run_gms_failover_promotion_warmup(
            generate, {"token_ids": [1]}, backend_name="test"
        )


@pytest.mark.asyncio
async def test_gms_failover_promotion_warmup_backend_override_enables(monkeypatch):
    monkeypatch.setenv("DYN_GMS_FAILOVER_PROMOTION_WARMUP", "0")
    monkeypatch.setenv("DYN_TEST_GMS_FAILOVER_PROMOTION_WARMUP", "1")
    seen = []

    async def generate(request, _context):
        seen.append(request)
        yield {"token_ids": [1], "finish_reason": "stop"}

    await run_gms_failover_promotion_warmup(
        generate, {"token_ids": [7]}, backend_name="test"
    )

    assert seen == [{"token_ids": [7]}]


@pytest.mark.asyncio
async def test_gms_failover_promotion_warmup_backend_override_disables(monkeypatch):
    monkeypatch.setenv("DYN_GMS_FAILOVER_PROMOTION_WARMUP", "1")
    monkeypatch.setenv("DYN_TEST_GMS_FAILOVER_PROMOTION_WARMUP", "0")
    seen = []

    async def generate(request, _context):
        seen.append(request)
        yield {"token_ids": [1], "finish_reason": "stop"}

    await run_gms_failover_promotion_warmup(
        generate, {"token_ids": [7]}, backend_name="test"
    )

    assert seen == []


@pytest.mark.asyncio
async def test_gms_failover_post_lock_fence_honors_backend_override(monkeypatch):
    monkeypatch.setenv("DYN_GMS_FAILOVER_POST_LOCK_FENCE_MS", "100")
    monkeypatch.setenv("DYN_TEST_GMS_FAILOVER_POST_LOCK_FENCE_MS", "25")
    monkeypatch.delenv("GMS_KV_LEASES", raising=False)
    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr("dynamo.common.gms_failover.asyncio.sleep", fake_sleep)

    await run_gms_failover_post_lock_fence(backend_name="test", role="shadow")

    assert sleeps == [0.025]


@pytest.mark.asyncio
async def test_gms_failover_shadow_runs_warmup_before_ready(monkeypatch):
    monkeypatch.setenv("DYN_GMS_FAILOVER_SHADOW_MODE", "true")
    monkeypatch.setenv("DYN_GMS_FAILOVER_KEEP_SHADOW_READY", "false")
    monkeypatch.setenv("ENGINE_ID", "1")
    order = []

    class _OrderedController:
        async def quiesce(self, tags):
            order.append(("quiesce", list(tags)))

        async def resume(self, tags):
            order.append(("resume", list(tags)))

        def mark_resumed(self):
            order.append(("mark_resumed", None))

    class _OrderedOwner:
        _quiesce_controller = _OrderedController()

    async def promotion_warmup():
        order.append(("warmup", None))

    runtime = _Runtime()
    activation = await prepare_gms_failover(
        _OrderedOwner(),
        runtime,
        backend_name="test",
        tags=["kv_cache"],
        lock_factory=_BusyOnTryLock,
        promotion_warmup=promotion_warmup,
    )

    assert activation.enabled is True
    assert order == [
        ("quiesce", ["kv_cache"]),
        ("resume", ["kv_cache"]),
        ("mark_resumed", None),
        ("warmup", None),
    ]
    assert runtime.health == [False, True]


@pytest.mark.asyncio
async def test_release_attached_gms_failover_lock_releases_and_detaches():
    class _Handler:
        pass

    handler = _Handler()
    lock = _Lock("/locks/failover.lock")
    setattr(handler, "_gms_failover_lock", lock)

    released = await release_attached_gms_failover_lock(handler, backend_name="test")

    assert released is True
    assert lock.released == 1
    assert getattr(handler, "_gms_failover_lock") is None


@pytest.mark.asyncio
async def test_release_attached_gms_failover_lock_without_lock_is_noop():
    class _Handler:
        pass

    handler = _Handler()

    released = await release_attached_gms_failover_lock(handler, backend_name="test")

    assert released is False
