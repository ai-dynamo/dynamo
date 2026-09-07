# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from dynamo.common.gms_failover import (
    acquire_gms_failover_lock_before_init,
    prepare_gms_failover,
    release_attached_gms_failover_lock,
    release_attached_gms_failover_lock_nowait,
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
async def test_activation_barrier_runs_before_shadow_resume(monkeypatch):
    monkeypatch.setenv("DYN_GMS_FAILOVER_SHADOW_MODE", "true")
    monkeypatch.setenv("ENGINE_ID", "1")
    owner = _Owner()
    events = []

    async def barrier():
        assert owner._quiesce_controller.resume_calls == []
        events.append("all-ranks-fenced")

    await prepare_gms_failover(
        owner,
        _Runtime(),
        backend_name="test",
        tags=["kv_cache"],
        lock_factory=_BusyOnTryLock,
        activation_barrier=barrier,
    )

    assert events == ["all-ranks-fenced"]
    assert owner._quiesce_controller.resume_calls == [["kv_cache"]]


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
async def test_gms_failover_can_warm_shadow_before_quiesce(monkeypatch):
    monkeypatch.setenv("DYN_GMS_FAILOVER_SHADOW_MODE", "true")
    events = []

    class Controller(_Controller):
        async def quiesce(self, tags):
            events.append("quiesce")
            return await super().quiesce(tags)

        async def resume(self, tags):
            events.append("resume")
            return await super().resume(tags)

    owner = _Owner()
    owner._quiesce_controller = Controller()

    async def warmup():
        events.append("warmup")

    activation = await prepare_gms_failover(
        owner,
        _Runtime(),
        backend_name="test",
        tags=["kv_cache"],
        lock_factory=_BusyOnTryLock,
        promotion_warmup=warmup,
        warm_standby_before_quiesce=True,
    )

    assert activation.enabled is True
    assert events == ["warmup", "quiesce", "resume"]


@pytest.mark.asyncio
async def test_prequiesce_warmup_requires_callback(monkeypatch):
    monkeypatch.setenv("DYN_GMS_FAILOVER_SHADOW_MODE", "true")
    with pytest.raises(RuntimeError, match="requires promotion_warmup"):
        await prepare_gms_failover(
            _Owner(),
            _Runtime(),
            backend_name="test",
            lock_factory=_BusyOnTryLock,
            warm_standby_before_quiesce=True,
        )


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
@pytest.mark.parametrize(
    "name",
    (
        "DYN_GMS_FAILOVER_PRIVATE_BOOTSTRAP_KV",
        "DYN_TEST_GMS_PRIVATE_BOOTSTRAP_KV",
        "GMS_TEST_PRIVATE_BOOTSTRAP_KV",
    ),
)
async def test_gms_failover_private_bootstrap_fails_closed(monkeypatch, name):
    monkeypatch.setenv("DYN_GMS_FAILOVER_SHADOW_MODE", "true")
    monkeypatch.setenv(name, "true")

    with pytest.raises(
        RuntimeError, match="private-bootstrap KV is no longer supported"
    ):
        await prepare_gms_failover(
            _Owner(),
            _Runtime(),
            backend_name="test",
            tags=["kv_cache"],
            lock_factory=_Lock,
        )


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


def test_explicit_directory_manifest_is_complete_shared_identity(monkeypatch):
    from gms_kv_ring.common.content_directory import resolve_manifest_id

    monkeypatch.setenv("GMS_KV_DIRECTORY_MANIFEST", "model-layout-v7")

    assert (
        resolve_manifest_id("vllm", 16, keyspace="vllm-native-hbm-v1")
        == "model-layout-v7"
    )
    assert resolve_manifest_id("vllm", 0) == "model-layout-v7"


def test_authoritative_failover_requires_explicit_directory_manifest(monkeypatch):
    from dynamo.common.gms_failover import _promote_content_directory_after_fence

    monkeypatch.setenv("GMS_KV_DIRECTORY_MODE", "authoritative")
    monkeypatch.setenv("GMS_KV_DIRECTORY_SOCKET", "/tmp/not-contacted.sock")
    monkeypatch.delenv("GMS_KV_DIRECTORY_MANIFEST", raising=False)

    with pytest.raises(
        RuntimeError,
        match="requires GMS_KV_DIRECTORY_MANIFEST",
    ):
        _promote_content_directory_after_fence("vllm", "shadow")


@pytest.mark.asyncio
async def test_gms_failover_promotes_directory_before_lease_reclaim(monkeypatch):
    monkeypatch.setenv("GMS_KV_DIRECTORY_MODE", "shadow")
    monkeypatch.setenv("DYN_GMS_FAILOVER_POST_LOCK_FENCE_MS", "0")
    order = []

    def promote(backend_name, role):
        order.append(("promote", backend_name, role))

        return {7, 9}

    async def to_thread(fn, *args):
        return fn(*args)

    def reclaim(backend_name, role, protected_blocks=None):
        order.append(("reclaim", backend_name, role, protected_blocks))

    monkeypatch.setattr(
        "dynamo.common.gms_failover._promote_content_directory_after_fence",
        promote,
    )
    monkeypatch.setattr("dynamo.common.gms_failover.asyncio.to_thread", to_thread)
    monkeypatch.setattr(
        "dynamo.common.gms_failover._reclaim_foreign_kv_leases_after_fence",
        reclaim,
    )

    await run_gms_failover_post_lock_fence(backend_name="vllm", role="shadow")

    assert order == [
        ("promote", "vllm", "shadow"),
        ("reclaim", "vllm", "shadow", {7, 9}),
    ]


def test_post_fence_reclaim_uses_allocator_namespace(monkeypatch):
    from types import SimpleNamespace

    from gpu_memory_service.integrations.common import kv_lease_client

    from dynamo.common import gms_failover

    monkeypatch.setenv("GMS_KV_LEASES", "1")
    calls = []

    monkeypatch.setattr(kv_lease_client, "resolve_lease_device", lambda _env: 0)
    monkeypatch.setattr(
        kv_lease_client,
        "reclaim_foreign_kv_leases_in_shm_dir",
        lambda engine, device, **kwargs: (
            calls.append((engine, device, kwargs))
            or SimpleNamespace(files=1, reclaimed_blocks=2, errors=0)
        ),
    )

    gms_failover._reclaim_foreign_kv_leases_after_fence(
        "sglang", "shadow", protected_blocks={7}
    )

    assert calls[0][0:2] == ("sglang", 0)
    assert calls[0][2]["namespace_suffix"] == "page-pool"
    assert calls[0][2]["protected_blocks"] == {7}


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


def test_release_attached_gms_failover_lock_nowait_releases_and_detaches():
    class _NowaitLock:
        def __init__(self):
            self.released = 0

        def release_nowait(self):
            self.released += 1
            return True

    handler = _Owner()
    lock = _NowaitLock()
    handler._gms_failover_lock = lock

    assert release_attached_gms_failover_lock_nowait(handler, backend_name="test")
    assert lock.released == 1
    assert handler._gms_failover_lock is None
