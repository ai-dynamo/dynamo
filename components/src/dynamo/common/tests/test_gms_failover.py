# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import os
import signal
import time
from types import SimpleNamespace

import pytest

from dynamo.common import gms_failover
from dynamo.common.gms_failover import (
    _requiesce_after_activation_error,
    acquire_gms_failover_lock_before_init,
    configure_failover_nccl_environment,
    prepare_gms_failover,
    quiesce_local_gpu_cohort_after_rank_loss,
    release_attached_gms_failover_lock,
    release_attached_gms_failover_lock_nowait,
    run_gms_failover_post_lock_fence,
    run_gms_failover_promotion_warmup,
)

try:
    from gpu_memory_service.failover_lock.interface import (
        FailoverLockContended,
        FailoverLockError,
    )
except ImportError:
    FailoverLockContended = FailoverLockError = RuntimeError

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
            raise FailoverLockContended("lock already held")
        await super().acquire(engine_id, timeout=timeout)


class _BrokenOnTryLock(_Lock):
    async def acquire(self, engine_id, timeout=None):
        if timeout == 0.0:
            raise FailoverLockError("permission denied")
        await super().acquire(engine_id, timeout=timeout)


def _status(tmp_path, backend="vllm", role="shadow", engine_id="0"):
    target = tmp_path / f"{backend}-{role}-engine{engine_id}-{os.getpid()}.json"
    return json.loads(target.read_text())


def test_frozen_reclaim_status_reports_quarantined_capacity(monkeypatch, tmp_path):
    monkeypatch.setenv("DYN_GMS_FAILOVER_RECLAIM_STATUS_DIR", str(tmp_path))
    monkeypatch.setenv("ENGINE_ID", "3")
    gms_failover._record_frozen_reclaim_status(
        "vllm", "shadow", "quarantined", "MPS timed out", quarantined_blocks=17
    )
    target = tmp_path / f"vllm-shadow-engine3-{os.getpid()}.json"
    payload = json.loads(target.read_text())
    assert payload["engine_id"] == "3"
    assert payload["status"] == "quarantined"
    assert payload["quarantined_blocks"] == 17
    assert payload["updated_at_unix_ms"] > 0

    gms_failover._record_frozen_reclaim_status(
        "vllm", "shadow", "reclaimed", quarantined_blocks=0
    )
    assert json.loads(target.read_text())["quarantined_blocks"] == 0
    assert not list(tmp_path.glob("*.pending"))


def test_removed_whole_pool_recovery_mode_fails_closed(monkeypatch):
    monkeypatch.delenv("GMS_VLLM_KV_RECOVERY_MODE", raising=False)
    monkeypatch.delenv("GMS_KV_RECOVERY_MODE", raising=False)
    gms_failover._validate_kv_recovery_mode("vllm")

    monkeypatch.setenv("GMS_VLLM_KV_RECOVERY_MODE", "whole_pool")
    with pytest.raises(RuntimeError, match="only the lease-backed"):
        gms_failover._validate_kv_recovery_mode("vllm")

    monkeypatch.setenv("GMS_VLLM_KV_RECOVERY_MODE", "invalid")
    with pytest.raises(RuntimeError, match="unsupported vllm"):
        gms_failover._validate_kv_recovery_mode("vllm")


def test_peer_rank_loss_quiesces_live_local_mps_client(monkeypatch):
    from gpu_memory_service.integrations.common import gpu_quiescence

    monkeypatch.setenv("GMS_VLLM_KV_RECOVERY_MODE", "granular")
    monkeypatch.setattr(gpu_quiescence, "gms_mps_provider_enabled", lambda _: True)
    calls = []
    monkeypatch.setattr(
        gpu_quiescence,
        "terminate_current_gpu_cohort_sync",
        lambda **kwargs: calls.append(kwargs)
        or SimpleNamespace(
            quiesced=True,
            provider="gms-mps",
            detail="terminated",
            elapsed_ms=3.0,
        ),
    )

    assert quiesce_local_gpu_cohort_after_rank_loss("vllm")
    assert quiesce_local_gpu_cohort_after_rank_loss("vllm", require_cuda_success=True)
    assert calls == [{"backend_name": "vllm"}] * 2


def test_peer_rank_loss_inventory_retirement_is_not_strict_cuda_proof(monkeypatch):
    from gpu_memory_service.integrations.common import gpu_quiescence

    monkeypatch.setattr(gpu_quiescence, "gms_mps_provider_enabled", lambda _: True)
    monkeypatch.setattr(
        gpu_quiescence,
        "terminate_current_gpu_cohort_sync",
        lambda **_kwargs: SimpleNamespace(
            quiesced=True,
            provider="gms-mps-inventory",
            detail="absent from inventory",
            elapsed_ms=3.0,
        ),
    )

    assert quiesce_local_gpu_cohort_after_rank_loss("sglang")
    assert not quiesce_local_gpu_cohort_after_rank_loss(
        "sglang", require_cuda_success=True
    )


def test_peer_rank_loss_without_mps_keeps_process_fencing_path(monkeypatch):
    from gpu_memory_service.integrations.common import gpu_quiescence

    monkeypatch.setenv("GMS_VLLM_KV_RECOVERY_MODE", "granular")
    monkeypatch.setattr(gpu_quiescence, "gms_mps_provider_enabled", lambda _: False)
    assert not quiesce_local_gpu_cohort_after_rank_loss("vllm")


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
async def test_shadow_preconnects_proof_before_waiting_for_lock(monkeypatch):
    from gpu_memory_service.integrations.common import gpu_quiescence

    monkeypatch.setenv("DYN_GMS_FAILOVER_SHADOW_MODE", "true")
    monkeypatch.setenv("ENGINE_ID", "1")
    calls = []
    monkeypatch.setattr(
        gpu_quiescence,
        "preconnect_gpu_quiescence_session",
        lambda **kwargs: calls.append(kwargs),
    )

    activation = await prepare_gms_failover(
        _Owner(), _Runtime(), backend_name="test", lock_factory=_BusyOnTryLock
    )

    assert activation.enabled
    assert calls == [{"backend_name": "test"}]


@pytest.mark.asyncio
async def test_post_lock_fence_runs_before_shadow_admission(monkeypatch):
    monkeypatch.setenv("DYN_GMS_FAILOVER_SHADOW_MODE", "true")
    monkeypatch.setenv("ENGINE_ID", "1")
    events = []

    class OrderedController(_Controller):
        async def quiesce(self, tags):
            events.append("quiesce")
            return await super().quiesce(tags)

        async def resume(self, tags):
            events.append("resume")
            return await super().resume(tags)

    owner = _Owner()
    owner._quiesce_controller = OrderedController()

    async def barrier():
        events.append("barrier")

    async def classify(*, backend_name, role):
        events.append("classify")

    monkeypatch.setattr(
        "dynamo.common.gms_failover.run_gms_failover_post_lock_fence",
        classify,
    )

    await prepare_gms_failover(
        owner,
        _Runtime(),
        backend_name="test",
        tags=["kv_cache"],
        lock_factory=_BusyOnTryLock,
        activation_barrier=barrier,
    )

    assert events == ["quiesce", "classify", "barrier", "resume"]


@pytest.mark.asyncio
async def test_post_lock_fence_failure_blocks_admission(monkeypatch):
    monkeypatch.setenv("DYN_GMS_FAILOVER_SHADOW_MODE", "true")
    monkeypatch.setenv("ENGINE_ID", "1")
    runtime = _Runtime()
    killed = []

    async def fail_classification(*, backend_name, role):
        raise RuntimeError("classification failed")

    monkeypatch.setattr(
        "dynamo.common.gms_failover.run_gms_failover_post_lock_fence",
        fail_classification,
    )
    monkeypatch.setattr(
        "dynamo.common.gms_failover.os.kill",
        lambda pid, sig: killed.append((pid, sig)),
    )

    with pytest.raises(RuntimeError, match="classification failed"):
        await prepare_gms_failover(
            _Owner(),
            runtime,
            backend_name="test",
            tags=["kv_cache"],
            lock_factory=_BusyOnTryLock,
        )

    assert runtime.health[-1] is False
    assert killed == []


@pytest.mark.asyncio
async def test_gms_failover_propagates_operational_lock_error(monkeypatch):
    monkeypatch.setenv("DYN_GMS_FAILOVER_SHADOW_MODE", "true")
    owner = _Owner()

    with pytest.raises(FailoverLockError, match="permission denied"):
        await prepare_gms_failover(
            owner,
            _Runtime(),
            backend_name="test",
            tags=["kv_cache"],
            lock_factory=_BrokenOnTryLock,
        )

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
async def test_gms_failover_replacement_primary_index_becomes_shadow_when_lock_busy(
    monkeypatch,
):
    monkeypatch.setenv("DYN_GMS_FAILOVER_SHADOW_MODE", "true")
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
    assert os.environ["DYN_GMS_FAILOVER_ACTIVE_LOCK_HELD"] == "1"


@pytest.mark.asyncio
async def test_pre_init_releases_lock_when_post_lock_fence_fails(monkeypatch):
    monkeypatch.setenv("DYN_GMS_FAILOVER_SHADOW_MODE", "true")
    monkeypatch.setenv("DYN_GMS_FAILOVER_POST_LOCK_FENCE_MS", "0")
    locks = []

    def lock_factory(path):
        lock = _Lock(path)
        locks.append(lock)
        return lock

    async def fail_fence(*, backend_name, role):
        raise RuntimeError(f"{backend_name} {role} fence failed")

    monkeypatch.setattr(
        "dynamo.common.gms_failover.run_gms_failover_post_lock_fence",
        fail_fence,
    )

    with pytest.raises(RuntimeError, match="fence failed"):
        await acquire_gms_failover_lock_before_init(
            backend_name="test",
            lock_factory=lock_factory,
        )

    assert locks[0].released == 1
    assert "DYN_GMS_FAILOVER_ACTIVE_LOCK_HELD" not in os.environ


@pytest.mark.asyncio
async def test_immediate_active_releases_lock_when_activation_barrier_fails(
    monkeypatch,
):
    monkeypatch.setenv("DYN_GMS_FAILOVER_SHADOW_MODE", "true")
    monkeypatch.setenv("DYN_GMS_FAILOVER_POST_LOCK_FENCE_MS", "0")
    locks = []

    def lock_factory(path):
        lock = _Lock(path)
        locks.append(lock)
        return lock

    async def fail_barrier():
        raise RuntimeError("activation barrier failed")

    with pytest.raises(RuntimeError, match="activation barrier failed"):
        await prepare_gms_failover(
            _Owner(),
            _Runtime(),
            backend_name="test",
            lock_factory=lock_factory,
            activation_barrier=fail_barrier,
        )

    assert locks[0].released == 1


@pytest.mark.asyncio
async def test_gms_failover_promotion_warmup_drains_non_error_stream(monkeypatch):
    monkeypatch.delenv("DYN_GMS_FAILOVER_PROMOTION_WARMUP", raising=False)
    seen = []

    async def generate(request, context):
        seen.append((request, context.id(), context.trace_headers()))
        context.notify_first_token()
        yield {"token_ids": [1], "finish_reason": None}
        seen.append("after-first-chunk")
        yield {"token_ids": [], "finish_reason": "stop"}
        seen.append("stream-drained")

    await run_gms_failover_promotion_warmup(
        generate,
        {"token_ids": [1], "stop_conditions": {"max_tokens": 1}},
        backend_name="test",
    )

    assert seen[0][0]["token_ids"] == [1]
    assert seen[0][1].startswith("gms-failover-promotion-warmup-")
    assert seen[0][2] is None
    assert seen[1:] == ["after-first-chunk", "stream-drained"]


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
async def test_gms_failover_promotion_warmup_honors_backend_concurrency(monkeypatch):
    monkeypatch.setenv("DYN_TEST_GMS_FAILOVER_PROMOTION_WARMUP_CONCURRENCY", "4")
    seen = []

    async def generate(request, _context):
        seen.append(request)
        yield {"token_ids": [1], "finish_reason": "stop"}

    await run_gms_failover_promotion_warmup(
        generate, {"token_ids": [7]}, backend_name="test"
    )

    assert seen == [{"token_ids": [7]}] * 4


@pytest.mark.asyncio
async def test_gms_failover_promotion_warmup_runs_exact_request_count(monkeypatch):
    monkeypatch.setenv("DYN_GMS_FAILOVER_PROMOTION_WARMUP_REQUESTS", "5")
    monkeypatch.setenv("DYN_TEST_GMS_FAILOVER_PROMOTION_WARMUP_CONCURRENCY", "2")
    seen = []

    async def generate(request, _context):
        seen.append(request)
        yield {"token_ids": [1], "finish_reason": "stop"}

    await run_gms_failover_promotion_warmup(
        generate, {"token_ids": [7]}, backend_name="test"
    )

    assert seen == [{"token_ids": [7]}] * 5


@pytest.mark.asyncio
async def test_gms_failover_promotion_warmup_covers_isolated_token_shapes(monkeypatch):
    monkeypatch.setenv("DYN_VLLM_GMS_FAILOVER_PROMOTION_WARMUP_TOKEN_COUNTS", "1,8,16")
    monkeypatch.setenv("DYN_VLLM_GMS_FAILOVER_PROMOTION_WARMUP_CONCURRENCY", "2")
    seen = []

    async def generate(request, _context):
        seen.append(request)
        yield {"token_ids": [1], "finish_reason": "stop"}

    await run_gms_failover_promotion_warmup(
        generate, {"token_ids": [7]}, backend_name="vllm"
    )

    assert [len(request["token_ids"]) for request in seen] == [1, 1, 8, 8, 16, 16]
    salts = [request["nvext"]["cache_salt"] for request in seen]
    assert len(set(salts)) == len(salts)


@pytest.mark.asyncio
async def test_gms_failover_promotion_warmup_covers_decode_tokens(monkeypatch):
    monkeypatch.setenv("DYN_VLLM_GMS_FAILOVER_PROMOTION_WARMUP_TOKEN_COUNTS", "1,704")
    monkeypatch.setenv("DYN_VLLM_GMS_FAILOVER_PROMOTION_WARMUP_OUTPUT_TOKENS", "8")
    seen = []

    async def generate(request, _context):
        seen.append(request)
        yield {"token_ids": [1], "finish_reason": "stop"}

    await run_gms_failover_promotion_warmup(
        generate, {"token_ids": [7], "max_tokens": 1}, backend_name="vllm"
    )

    assert [len(request["token_ids"]) for request in seen] == [1, 704]
    assert [request["max_tokens"] for request in seen] == [8, 8]
    assert all(request["ignore_eos"] for request in seen)


@pytest.mark.asyncio
async def test_gms_failover_promotion_warmup_rejects_token_shapes_for_text(monkeypatch):
    monkeypatch.setenv("DYN_VLLM_GMS_FAILOVER_PROMOTION_WARMUP_TOKEN_COUNTS", "8")

    async def generate(_request, _context):
        yield {"token_ids": [1], "finish_reason": "stop"}

    with pytest.raises(ValueError, match="non-empty token_ids"):
        await run_gms_failover_promotion_warmup(
            generate, {"prompt": "Test"}, backend_name="vllm"
        )


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
    assert os.environ["DYN_GMS_FAILOVER_ACTIVE_LOCK_HELD"] == "1"


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


def test_post_lock_directory_promotion_forces_fresh_epoch(monkeypatch):
    from gms_kv_ring.common import content_directory

    from dynamo.common.gms_failover import _promote_content_directory_after_fence

    calls = []

    class FakeDirectory:
        def __init__(self, socket_path, **kwargs):
            calls.append(("init", socket_path, kwargs))

        def promote(self, **kwargs):
            calls.append(("promote", kwargs))
            return 9

        def hbm_lease_inventory(self):
            return {"rank": [3, 5]}, {"rank": [(3, 13), (5, 15)]}

        def close(self):
            calls.append(("close",))

    monkeypatch.setenv("GMS_KV_DIRECTORY_MODE", "authoritative")
    monkeypatch.setenv("GMS_KV_DIRECTORY_SOCKET", "/tmp/directory.sock")
    monkeypatch.setenv("GMS_KV_DIRECTORY_MANIFEST", "model-layout-v7")
    monkeypatch.setenv("ENGINE_ID", "primary")
    monkeypatch.setattr(content_directory, "ContentDirectory", FakeDirectory)

    protected, leases = _promote_content_directory_after_fence("vllm", "shadow")

    assert protected == {3, 5}
    assert leases == {(3, 13), (5, 15)}
    assert ("promote", {"force_new_epoch": True}) in calls
    assert calls[-1] == ("close",)


def test_post_lock_directory_promotion_preserves_legacy_block_fallback(monkeypatch):
    from gms_kv_ring.common import content_directory

    from dynamo.common.gms_failover import _promote_content_directory_after_fence

    class FakeDirectory:
        def __init__(self, _socket_path, **_kwargs):
            pass

        def promote(self, **_kwargs):
            return 9

        def hbm_lease_inventory(self):
            return {"rank": [3, 5]}, None

        def close(self):
            pass

    monkeypatch.setenv("GMS_KV_DIRECTORY_MODE", "authoritative")
    monkeypatch.setenv("GMS_KV_DIRECTORY_SOCKET", "/tmp/directory.sock")
    monkeypatch.setenv("GMS_KV_DIRECTORY_MANIFEST", "model-layout-v7")
    monkeypatch.setattr(content_directory, "ContentDirectory", FakeDirectory)

    protected, leases = _promote_content_directory_after_fence("vllm", "shadow")

    assert protected == {3, 5}
    assert leases is None


@pytest.mark.asyncio
async def test_gms_failover_promotes_directory_before_lease_reclaim(monkeypatch):
    from gpu_memory_service.integrations.common import gpu_quiescence
    from gpu_memory_service.integrations.vllm import writer_lifecycle

    monkeypatch.setenv("GMS_KV_DIRECTORY_MODE", "authoritative")
    monkeypatch.setenv("DYN_GMS_FAILOVER_POST_LOCK_FENCE_MS", "0")

    async def fenced_predecessor():
        return "/shared/old-cohort"

    monkeypatch.setattr(
        writer_lifecycle, "fence_predecessor_writers", fenced_predecessor
    )
    order = []

    def promote(backend_name, role):
        order.append(("promote", backend_name, role))

        return {7, 9}, {(7, 17), (9, 19)}

    async def to_thread(fn, *args):
        return fn(*args)

    def reclaim(
        backend_name,
        role,
        *,
        gpu_quiesced,
        recovery_owner_id=None,
        protected_blocks=None,
        protected_leases=None,
        inherit_quarantine=False,
    ):
        order.append(
            (
                "reclaim",
                backend_name,
                role,
                gpu_quiesced,
                recovery_owner_id,
                protected_blocks,
                protected_leases,
            )
        )

    async def prove(**_kwargs):
        order.append(("prove", "vllm", "shadow"))
        return SimpleNamespace(
            quiesced=True, provider="test", detail="", elapsed_ms=0.1
        )

    monkeypatch.setattr(
        "dynamo.common.gms_failover._promote_content_directory_after_fence",
        promote,
    )
    monkeypatch.setattr("dynamo.common.gms_failover.asyncio.to_thread", to_thread)
    monkeypatch.setattr(
        "dynamo.common.gms_failover._recover_foreign_kv_leases_after_fence",
        reclaim,
    )
    monkeypatch.setattr(
        gpu_quiescence, "gpu_quiescence_provider_configured", lambda _name: True
    )
    monkeypatch.setattr(gpu_quiescence, "prove_predecessor_gpu_quiescence", prove)

    await run_gms_failover_post_lock_fence(backend_name="vllm", role="shadow")

    assert order[0] == ("promote", "vllm", "shadow")
    assert order[1][:4] == ("reclaim", "vllm", "shadow", False)
    assert order[1][4]
    assert order[1][5:] == ({7, 9}, {(7, 17), (9, 19)})
    assert order[2] == ("prove", "vllm", "shadow")
    assert order[3][:5] == ("reclaim", "vllm", "shadow", True, order[1][4])


@pytest.mark.asyncio
async def test_gpu_proof_blocks_lease_reclaim_and_shadow_admission(monkeypatch):
    from types import SimpleNamespace

    from gpu_memory_service.integrations.common import gpu_quiescence
    from gpu_memory_service.integrations.vllm import writer_lifecycle

    from dynamo.common import gms_failover

    monkeypatch.setenv("GMS_KV_DIRECTORY_MODE", "off")
    proof_allowed = asyncio.Event()
    proof_started = asyncio.Event()
    calls = []

    async def prove(**_kwargs):
        proof_started.set()
        await proof_allowed.wait()
        return SimpleNamespace(
            quiesced=True, provider="test", detail="", elapsed_ms=0.1
        )

    def recover(_backend, _role, *, gpu_quiesced, recovery_owner_id=None, **_kwargs):
        calls.append((gpu_quiesced, recovery_owner_id))

    monkeypatch.setattr(
        gpu_quiescence,
        "prove_predecessor_gpu_quiescence",
        prove,
    )
    monkeypatch.setattr(
        gpu_quiescence, "gpu_quiescence_provider_configured", lambda _name: True
    )
    monkeypatch.setattr(
        writer_lifecycle,
        "fence_predecessor_writers",
        lambda: asyncio.sleep(0, result="old-cohort"),
    )
    monkeypatch.setattr(
        gms_failover,
        "_recover_foreign_kv_leases_after_fence",
        recover,
    )

    fence = asyncio.create_task(
        run_gms_failover_post_lock_fence(backend_name="vllm", role="shadow")
    )
    await proof_started.wait()
    assert not fence.done()
    assert len(calls) == 1
    assert calls[0][0] is False
    proof_allowed.set()
    await fence

    assert len(calls) == 2
    assert calls[1] == (True, calls[0][1])


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["vllm", "sglang"])
async def test_frozen_predecessor_serves_before_gpu_proof(monkeypatch, backend):
    from gpu_memory_service.integrations.common import kv_lease_client
    from gpu_memory_service.integrations.sglang import (
        writer_lifecycle as sglang_writers,
    )
    from gpu_memory_service.integrations.vllm import writer_lifecycle as vllm_writers

    monkeypatch.setenv(gms_failover.FROZEN_PREDECESSOR_ENV, "1")
    monkeypatch.setenv("DYN_GMS_FAILOVER_SHADOW_MODE", "1")
    monkeypatch.setenv("GMS_KV_LEASES", "1")
    monkeypatch.setenv("GMS_KV_DIRECTORY_MODE", "authoritative")
    monkeypatch.setenv("GMS_KV_DIRECTORY_STANDBY", "1")
    monkeypatch.setenv("GMS_KV_DIRECTORY_MANIFEST", "frozen-test")
    monkeypatch.setenv("GMS_KV_DIRECTORY_SOCKET", "/tmp/frozen-test.sock")
    monkeypatch.setenv("DYN_GMS_GPU_QUIESCENCE_PROVIDER", "gms-mps")
    monkeypatch.setenv("ENGINE_ID", "1")
    monkeypatch.setenv("DYN_GMS_FAILOVER_PRIMARY_ENGINE_ID", "0")
    writers = sglang_writers if backend == "sglang" else vllm_writers
    monkeypatch.setattr(
        writers,
        "fence_predecessor_writers",
        lambda: asyncio.sleep(0, result="old-cohort"),
    )
    monkeypatch.setattr(kv_lease_client, "resolve_lease_device", lambda _env: 0)
    monkeypatch.setattr(
        kv_lease_client, "current_kv_lease_owner_id", lambda *_args: "new-cohort"
    )
    monkeypatch.setattr(
        gms_failover,
        "_promote_content_directory_after_fence",
        lambda *_args: ({7}, {(7, 17)}),
    )
    monkeypatch.setattr(
        gms_failover, "_release_frozen_shadow_headroom", lambda *_args: None
    )
    calls = []
    monkeypatch.setattr(
        gms_failover,
        "_recover_foreign_kv_leases_after_fence",
        lambda *_args, gpu_quiesced, **kwargs: calls.append(
            (gpu_quiesced, kwargs.get("protected_leases"))
        ),
    )
    if backend == "sglang":
        monkeypatch.setattr(
            sglang_writers,
            "mark_gms_recovery_ready",
            lambda: calls.append(("recovery-ready", None)),
        )
        monkeypatch.setattr(
            sglang_writers,
            "mark_gpu_quiescence_ready",
            lambda: calls.append(("gpu-ready", None)),
        )
        monkeypatch.setattr(
            sglang_writers,
            "mark_gms_reclaim_ready",
            lambda: calls.append(("reclaim-ready", None)),
        )

    proof_started = asyncio.Event()
    release_proof = asyncio.Event()

    async def prove(**_kwargs):
        proof_started.set()
        await release_proof.wait()
        return SimpleNamespace(
            quiesced=True, provider="test", detail="", elapsed_ms=1.0
        )

    from gpu_memory_service.integrations.common import gpu_quiescence

    monkeypatch.setattr(gpu_quiescence, "prove_predecessor_gpu_quiescence", prove)
    old_tasks = set(gms_failover._gpu_quiescence_tasks)
    await run_gms_failover_post_lock_fence(backend_name=backend, role="shadow")
    tasks = gms_failover._gpu_quiescence_tasks - old_tasks
    assert len(tasks) == 1
    await asyncio.wait_for(proof_started.wait(), 1)
    assert calls[0] == (False, {(7, 17)})
    assert not any(result[0] is True for result in calls)
    assert os.environ[f"DYN_{backend.upper()}_GMS_POOL_FENCED"] == "1"

    release_proof.set()
    await asyncio.wait_for(asyncio.gather(*tasks), 1)
    assert (True, None) in calls
    if backend == "sglang":
        assert calls.index(("recovery-ready", None)) < calls.index((True, None))
        assert calls.index((True, None)) < calls.index(("reclaim-ready", None))
        assert calls[-1] == ("gpu-ready", None)


@pytest.mark.asyncio
async def test_frozen_reclaim_bounds_stalled_gpu_proof(monkeypatch):
    from gpu_memory_service.integrations.common import gpu_quiescence

    monkeypatch.setenv("DYN_GMS_FAILOVER_BACKGROUND_GPU_PROOF_SECS", "0.02")
    cancelled = asyncio.Event()

    async def stalled_proof(**_kwargs):
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    monkeypatch.setattr(
        gpu_quiescence, "prove_predecessor_gpu_quiescence", stalled_proof
    )
    await asyncio.wait_for(
        gms_failover._reclaim_frozen_predecessor(
            backend_name="vllm",
            role="shadow",
            predecessor_cohort="old",
            recovery_owner_id="new",
            lease_device=0,
        ),
        1,
    )
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_frozen_reclaim_stops_retrying_cuda_201(monkeypatch):
    from gpu_memory_service.integrations.common import gpu_quiescence

    calls = 0

    async def invalid_context(**_kwargs):
        nonlocal calls
        calls += 1
        return SimpleNamespace(
            quiesced=False,
            provider="gms-mps",
            detail="MPS did not certify termination: cuda_result=201",
            elapsed_ms=1.0,
        )

    monkeypatch.setattr(
        gpu_quiescence, "prove_predecessor_gpu_quiescence", invalid_context
    )
    await gms_failover._reclaim_frozen_predecessor(
        backend_name="vllm",
        role="shadow",
        predecessor_cohort="old",
        recovery_owner_id="new",
        lease_device=0,
    )
    assert calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["vllm", "sglang"])
async def test_process_death_grace_reclaims_without_claiming_gpu_proof(
    monkeypatch, tmp_path, backend
):
    from gpu_memory_service.integrations.common import gpu_quiescence

    cohort = tmp_path / "retired-cohort"
    cohort.write_bytes(b"R")
    if backend == "sglang":
        monkeypatch.setenv("GMS_SGLANG_WRITER_COHORT_PATH", str(tmp_path / "successor"))
    monkeypatch.setenv("DYN_GMS_FAILOVER_RECLAIM_POLICY", "process-death-timeout")
    monkeypatch.setenv("DYN_GMS_FAILOVER_PROCESS_DEATH_GRACE_SECS", "0.01")
    monkeypatch.setenv("DYN_GMS_FAILOVER_RECLAIM_STATUS_DIR", str(tmp_path))

    async def failed_proof(**_kwargs):
        return SimpleNamespace(
            quiesced=False, provider="gms-mps", detail="cuda_result=201", elapsed_ms=1
        )

    monkeypatch.setattr(
        gpu_quiescence, "prove_predecessor_gpu_quiescence", failed_proof
    )
    reclaimed = []
    monkeypatch.setattr(
        gms_failover,
        "_recover_foreign_kv_leases_after_fence",
        lambda *args, **kwargs: reclaimed.append(kwargs),
    )
    start = time.monotonic()
    await gms_failover._reclaim_frozen_predecessor(
        backend_name=backend,
        role="shadow",
        predecessor_cohort=str(cohort),
        recovery_owner_id="current-owner",
        lease_device=0,
    )
    assert time.monotonic() - start >= 0.01
    assert reclaimed == [
        {
            "gpu_quiesced": False,
            "recovery_owner_id": "current-owner",
            "process_death_timeout_elapsed": True,
        }
    ]
    status = _status(tmp_path, backend)
    assert status["status"] == "reclaimed-best-effort"
    assert "CUDA unproven" in status["detail"]
    if backend == "sglang":
        from gpu_memory_service.integrations.sglang import writer_lifecycle

        assert writer_lifecycle.gms_reclaim_ready()
        assert not writer_lifecycle.gpu_quiescence_ready()


@pytest.mark.asyncio
@pytest.mark.parametrize("state", [b"", b"R"])
async def test_process_death_grace_rejects_live_or_unretired_cohort(
    monkeypatch, tmp_path, state
):
    import fcntl

    cohort = tmp_path / "cohort"
    cohort.write_bytes(state)
    monkeypatch.setenv("DYN_GMS_FAILOVER_PROCESS_DEATH_GRACE_SECS", "0.01")
    fd = os.open(cohort, os.O_RDWR)
    waits = []
    try:
        fcntl.flock(fd, fcntl.LOCK_SH)
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                gms_failover._wait_process_death_reclaim_grace(
                    str(cohort), waits.append
                ),
                0.3,
            )
    finally:
        os.close(fd)
    assert waits
    if not state:
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                gms_failover._wait_process_death_reclaim_grace(str(cohort)), 0.3
            )


@pytest.mark.asyncio
async def test_process_death_grace_rechecks_and_is_cancellable(monkeypatch, tmp_path):
    from gpu_memory_service.integrations.common import process_lifecycle

    cohort = tmp_path / "cohort"
    cohort.write_bytes(b"R")
    monkeypatch.setenv("DYN_GMS_FAILOVER_PROCESS_DEATH_GRACE_SECS", "0.01")
    answers = iter([True])
    monkeypatch.setattr(
        process_lifecycle,
        "retired_writer_cohort_has_no_processes",
        lambda _path: next(answers, False),
    )
    # A failed recheck after the grace never authorizes; it keeps waiting.
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(
            gms_failover._wait_process_death_reclaim_grace(str(cohort)), 0.3
        )
    monkeypatch.setattr(
        process_lifecycle, "retired_writer_cohort_has_no_processes", lambda _path: True
    )
    monkeypatch.setenv("DYN_GMS_FAILOVER_PROCESS_DEATH_GRACE_SECS", "3600")
    task = asyncio.create_task(
        gms_failover._wait_process_death_reclaim_grace(str(cohort))
    )
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_process_death_grace_retries_transient_probe_contention(
    monkeypatch, tmp_path
):
    from gpu_memory_service.integrations.common import process_lifecycle

    cohort = tmp_path / "cohort"
    cohort.write_bytes(b"R")
    monkeypatch.setenv("DYN_GMS_FAILOVER_PROCESS_DEATH_GRACE_SECS", "0.05")
    answers = iter([False, True, False, True])
    monkeypatch.setattr(
        process_lifecycle,
        "retired_writer_cohort_has_no_processes",
        lambda _path: next(answers),
    )
    await asyncio.wait_for(
        gms_failover._wait_process_death_reclaim_grace(str(cohort)), 1
    )


@pytest.mark.asyncio
async def test_process_death_grace_retries_until_predecessor_dies(
    monkeypatch, tmp_path
):
    from gpu_memory_service.integrations.common import process_lifecycle

    cohort = tmp_path / "cohort"
    cohort.write_bytes(b"R")
    monkeypatch.setenv("DYN_GMS_FAILOVER_PROCESS_DEATH_GRACE_SECS", "0.01")
    dead_after = time.monotonic() + 1.5
    monkeypatch.setattr(
        process_lifecycle,
        "retired_writer_cohort_has_no_processes",
        lambda _path: time.monotonic() >= dead_after,
    )
    waits = []
    await asyncio.wait_for(
        gms_failover._wait_process_death_reclaim_grace(str(cohort), waits.append), 10
    )
    assert waits
    assert time.monotonic() >= dead_after


@pytest.mark.asyncio
async def test_process_death_grace_parallel_rank_probes(monkeypatch, tmp_path):
    cohort = tmp_path / "cohort"
    cohort.write_bytes(b"R")
    monkeypatch.setenv("DYN_GMS_FAILOVER_PROCESS_DEATH_GRACE_SECS", "0.1")
    await asyncio.wait_for(
        asyncio.gather(
            *[
                gms_failover._wait_process_death_reclaim_grace(str(cohort))
                for _ in range(16)
            ]
        ),
        10,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("grace", ["0", "-1", "nan", "inf", "bad"])
async def test_process_death_grace_rejects_invalid_interval(monkeypatch, grace):
    monkeypatch.setenv("DYN_GMS_FAILOVER_PROCESS_DEATH_GRACE_SECS", grace)
    with pytest.raises(ValueError):
        await gms_failover._wait_process_death_reclaim_grace("/tmp/cohort")


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
@pytest.mark.parametrize("shadow_mode", [None, "false"])
def test_frozen_predecessor_flag_does_not_affect_vanilla_workers(
    monkeypatch, backend, shadow_mode
):
    monkeypatch.setenv(gms_failover.FROZEN_PREDECESSOR_ENV, "1")
    if shadow_mode is None:
        monkeypatch.delenv("DYN_GMS_FAILOVER_SHADOW_MODE", raising=False)
    else:
        monkeypatch.setenv("DYN_GMS_FAILOVER_SHADOW_MODE", shadow_mode)
    monkeypatch.delenv("GMS_KV_LEASES", raising=False)

    assert gms_failover.frozen_predecessor_enabled(backend) is False


def test_frozen_headroom_is_anonymous_until_takeover(monkeypatch):
    from gpu_memory_service.integrations.common import kv_lease_client

    monkeypatch.setenv("DYN_GMS_FAILOVER_FROZEN_HEADROOM_BLOCKS", "24")
    monkeypatch.setattr(
        gms_failover, "frozen_predecessor_enabled", lambda _backend: True
    )
    monkeypatch.setattr(kv_lease_client, "resolve_lease_device", lambda _env: 0)
    reservation = SimpleNamespace(reserved_blocks=0, reserved_for_owner=None)
    monkeypatch.setattr(
        kv_lease_client,
        "read_kv_lease_reservation",
        lambda *_args, **_kwargs: ("test", reservation),
    )
    writes = []
    monkeypatch.setattr(
        kv_lease_client,
        "set_kv_lease_reservation",
        lambda *_args, **kwargs: writes.append(kwargs),
    )
    gms_failover.arm_frozen_shadow_headroom("vllm")
    assert writes == [{"reserved_blocks": 24, "namespace_suffix": "block-pool"}]
    reservation.reserved_blocks = 24
    gms_failover.arm_frozen_shadow_headroom("vllm")
    assert len(writes) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("quiesced", "provider", "detail"),
    [
        (False, "gms-mps", "timeout"),
        (True, "gms-mps-inventory", "host absent, GPU work unproven"),
    ],
)
async def test_frozen_predecessor_keeps_quarantine_after_failed_gpu_proof(
    monkeypatch, tmp_path, quiesced, provider, detail
):
    from gpu_memory_service.integrations.common import gpu_quiescence, kv_lease_client
    from gpu_memory_service.integrations.vllm import writer_lifecycle

    monkeypatch.setenv(gms_failover.FROZEN_PREDECESSOR_ENV, "1")
    monkeypatch.setenv("DYN_GMS_FAILOVER_SHADOW_MODE", "1")
    monkeypatch.setenv("DYN_GMS_FAILOVER_BACKGROUND_GPU_PROOF_SECS", "0")
    monkeypatch.setenv("DYN_GMS_FAILOVER_RECLAIM_STATUS_DIR", str(tmp_path))
    monkeypatch.setenv("GMS_KV_LEASES", "1")
    monkeypatch.setenv("GMS_KV_DIRECTORY_MODE", "authoritative")
    monkeypatch.setenv("GMS_KV_DIRECTORY_STANDBY", "1")
    monkeypatch.setenv("GMS_KV_DIRECTORY_MANIFEST", "frozen-test")
    monkeypatch.setenv("GMS_KV_DIRECTORY_SOCKET", "/tmp/frozen-test.sock")
    monkeypatch.setenv("DYN_GMS_GPU_QUIESCENCE_PROVIDER", "gms-mps")
    monkeypatch.setenv("ENGINE_ID", "1")
    monkeypatch.setattr(
        writer_lifecycle,
        "fence_predecessor_writers",
        lambda: asyncio.sleep(0, result="old-cohort"),
    )
    monkeypatch.setattr(kv_lease_client, "resolve_lease_device", lambda _env: 0)
    monkeypatch.setattr(
        kv_lease_client, "current_kv_lease_owner_id", lambda *_args: "new-cohort"
    )
    monkeypatch.setattr(
        gms_failover,
        "_promote_content_directory_after_fence",
        lambda *_args: ({7}, {(7, 17)}),
    )
    monkeypatch.setattr(
        gms_failover, "_release_frozen_shadow_headroom", lambda *_args: None
    )
    calls = []

    def recover(*_args, gpu_quiesced, **_kwargs):
        calls.append(gpu_quiesced)
        return SimpleNamespace(quarantined_blocks=4)

    monkeypatch.setattr(gms_failover, "_recover_foreign_kv_leases_after_fence", recover)

    async def failed_proof(**_kwargs):
        return SimpleNamespace(
            quiesced=quiesced, provider=provider, detail=detail, elapsed_ms=8.0
        )

    monkeypatch.setattr(
        gpu_quiescence, "prove_predecessor_gpu_quiescence", failed_proof
    )
    old_tasks = set(gms_failover._gpu_quiescence_tasks)
    await run_gms_failover_post_lock_fence(backend_name="vllm", role="shadow")
    tasks = gms_failover._gpu_quiescence_tasks - old_tasks
    await asyncio.wait_for(asyncio.gather(*tasks), 1)
    assert calls == [False]
    payload = _status(tmp_path, engine_id="1")
    assert payload["status"] == "quarantined"
    assert payload["quarantined_blocks"] == 4


@pytest.mark.asyncio
async def test_frozen_reclaim_retries_transient_gpu_proof(monkeypatch):
    from gpu_memory_service.integrations.common import gpu_quiescence

    attempts = []
    reclaimed = []

    async def prove(**_kwargs):
        attempts.append(1)
        return SimpleNamespace(
            quiesced=len(attempts) == 2,
            provider="test",
            detail="still retiring",
            elapsed_ms=1.0,
        )

    monkeypatch.setattr(gpu_quiescence, "prove_predecessor_gpu_quiescence", prove)
    monkeypatch.setattr(
        gms_failover,
        "_recover_foreign_kv_leases_after_fence",
        lambda *_args, **kwargs: reclaimed.append(kwargs["gpu_quiesced"]),
    )
    await gms_failover._reclaim_frozen_predecessor(
        backend_name="vllm",
        role="shadow",
        predecessor_cohort="old",
        recovery_owner_id="new",
        lease_device=0,
    )
    assert len(attempts) == 2
    assert reclaimed == [True]


@pytest.mark.asyncio
async def test_busy_lease_reclamation_stays_pending_and_retries(monkeypatch, tmp_path):
    from gpu_memory_service.integrations.common import gpu_quiescence

    monkeypatch.setenv("DYN_GMS_FAILOVER_RECLAIM_STATUS_DIR", str(tmp_path))
    monkeypatch.setenv("DYN_GMS_FAILOVER_BACKGROUND_GPU_PROOF_SECS", "0.1")

    async def prove(**_kwargs):
        return SimpleNamespace(
            quiesced=True, provider="gms-mps", detail="certified", elapsed_ms=1.0
        )

    attempts = []
    observed = []

    def busy_then_reclaim(*_args, **_kwargs):
        attempts.append(1)
        if len(attempts) < 4:
            if len(attempts) > 1:
                observed.append(_status(tmp_path)["status"])
            raise RuntimeError(
                "KV lease recovery is busy with live successor mutations"
            )
        return SimpleNamespace(reclaimed_blocks=4)

    monkeypatch.setattr(gpu_quiescence, "prove_predecessor_gpu_quiescence", prove)
    monkeypatch.setattr(
        gms_failover, "_recover_foreign_kv_leases_after_fence", busy_then_reclaim
    )
    await asyncio.wait_for(
        gms_failover._reclaim_frozen_predecessor(
            backend_name="vllm",
            role="shadow",
            predecessor_cohort="old",
            recovery_owner_id="new",
            lease_device=0,
            quarantined_blocks=4,
        ),
        10,
    )
    assert len(attempts) == 4
    assert observed == ["pending", "pending"]
    payload = _status(tmp_path)
    assert payload["status"] == "reclaimed"
    assert payload["quarantined_blocks"] == 0
    assert payload["phase_one_quarantined_blocks"] == 4
    assert payload["reclaimed_blocks"] == 4


@pytest.mark.asyncio
async def test_process_death_policy_stays_pending_while_proof_fails(
    monkeypatch, tmp_path
):
    from gpu_memory_service.integrations.common import gpu_quiescence

    monkeypatch.setenv("DYN_GMS_FAILOVER_RECLAIM_POLICY", "process-death-timeout")
    monkeypatch.setenv("DYN_GMS_FAILOVER_RECLAIM_STATUS_DIR", str(tmp_path))

    async def failed_proof(**_kwargs):
        return SimpleNamespace(
            quiesced=False, provider="gms-mps", detail="cuda_result=201", elapsed_ms=1
        )

    statuses = []

    async def wait_grace(_cohort, _on_waiting=None):
        statuses.append(_status(tmp_path)["status"])

    monkeypatch.setattr(
        gpu_quiescence, "prove_predecessor_gpu_quiescence", failed_proof
    )
    monkeypatch.setattr(gms_failover, "_wait_process_death_reclaim_grace", wait_grace)
    monkeypatch.setattr(
        gms_failover,
        "_recover_foreign_kv_leases_after_fence",
        lambda *_args, **_kwargs: None,
    )
    await gms_failover._reclaim_frozen_predecessor(
        backend_name="vllm",
        role="shadow",
        predecessor_cohort="old",
        recovery_owner_id="new",
        lease_device=0,
    )
    assert statuses == ["pending"]
    assert _status(tmp_path)["status"] == "reclaimed-best-effort"


@pytest.mark.asyncio
async def test_strict_sglang_refusal_publishes_terminal_marker(monkeypatch, tmp_path):
    from gpu_memory_service.integrations.common import gpu_quiescence
    from gpu_memory_service.integrations.sglang import writer_lifecycle

    monkeypatch.setenv("GMS_SGLANG_WRITER_COHORT_PATH", str(tmp_path / "successor"))
    monkeypatch.setenv("DYN_GMS_FAILOVER_RECLAIM_STATUS_DIR", str(tmp_path))
    monkeypatch.delenv("DYN_GMS_FAILOVER_RECLAIM_POLICY", raising=False)

    async def failed_proof(**_kwargs):
        return SimpleNamespace(
            quiesced=False, provider="gms-mps", detail="cuda_result=201", elapsed_ms=1
        )

    monkeypatch.setattr(
        gpu_quiescence, "prove_predecessor_gpu_quiescence", failed_proof
    )
    assert not writer_lifecycle.gms_reclaim_refused()
    await gms_failover._reclaim_frozen_predecessor(
        backend_name="sglang",
        role="shadow",
        predecessor_cohort="old",
        recovery_owner_id="new",
        lease_device=0,
    )
    assert writer_lifecycle.gms_reclaim_refused()
    assert not writer_lifecycle.gms_reclaim_ready()
    assert _status(tmp_path, "sglang")["status"] == "quarantined"


def _frozen_env(monkeypatch):
    monkeypatch.setenv(gms_failover.FROZEN_PREDECESSOR_ENV, "1")
    monkeypatch.setenv("DYN_GMS_FAILOVER_SHADOW_MODE", "1")
    monkeypatch.setenv("GMS_KV_LEASES", "1")
    monkeypatch.setenv("GMS_KV_DIRECTORY_MODE", "authoritative")
    monkeypatch.setenv("GMS_KV_DIRECTORY_STANDBY", "1")
    monkeypatch.setenv("GMS_KV_DIRECTORY_MANIFEST", "frozen-test")
    monkeypatch.setenv("GMS_KV_DIRECTORY_SOCKET", "/tmp/frozen-test.sock")
    monkeypatch.setenv("DYN_GMS_GPU_QUIESCENCE_PROVIDER", "gms-mps")
    monkeypatch.setenv("ENGINE_ID", "1")


@pytest.mark.parametrize(
    ("policy", "grace"),
    [("process-death", "2"), ("process-death-timeout", "0"), ("gpu_proof", "2")],
)
def test_frozen_predecessor_rejects_invalid_reclaim_policy_at_boot(
    monkeypatch, policy, grace
):
    _frozen_env(monkeypatch)
    monkeypatch.setenv("DYN_GMS_FAILOVER_RECLAIM_POLICY", policy)
    monkeypatch.setenv("DYN_GMS_FAILOVER_PROCESS_DEATH_GRACE_SECS", grace)
    with pytest.raises(RuntimeError, match="DYN_GMS_FAILOVER_"):
        gms_failover.frozen_predecessor_enabled("vllm")


@pytest.mark.parametrize(
    ("policy", "inherit"), [("gpu-proof", False), ("process-death-timeout", True)]
)
def test_only_process_death_policy_inherits_stranded_quarantine(
    monkeypatch, policy, inherit
):
    _frozen_env(monkeypatch)
    monkeypatch.setenv("DYN_GMS_FAILOVER_RECLAIM_POLICY", policy)
    assert gms_failover._inherit_predecessor_quarantine("vllm") is inherit
    monkeypatch.delenv(gms_failover.FROZEN_PREDECESSOR_ENV)
    assert gms_failover._inherit_predecessor_quarantine("vllm") is False


def test_frozen_remote_vllm_rank_classifies_before_background_proof(monkeypatch):
    import threading

    from gpu_memory_service.integrations.common import kv_lease_client

    monkeypatch.setattr(
        gms_failover, "frozen_predecessor_enabled", lambda *_args, **_kwargs: True
    )
    monkeypatch.setattr(
        gms_failover,
        "_promote_content_directory_after_fence",
        lambda *_args: ({7}, {(7, 17)}),
    )
    monkeypatch.setattr(kv_lease_client, "resolve_lease_device", lambda _env: 0)
    monkeypatch.setattr(
        kv_lease_client, "current_kv_lease_owner_id", lambda *_args: "new-rank"
    )
    order = []
    monkeypatch.setattr(
        gms_failover,
        "_recover_foreign_kv_leases_after_fence",
        lambda *_args, **kwargs: order.append(
            ("classify", kwargs["protected_leases"], kwargs["gpu_quiesced"])
        ),
    )
    monkeypatch.setattr(
        gms_failover,
        "_release_frozen_shadow_headroom",
        lambda *_args: order.append(("release",)),
    )

    class DeferredThread:
        def __init__(self, *, target, **_kwargs):
            self.target = target

        def start(self):
            order.append(("proof-start",))

    monkeypatch.setattr(threading, "Thread", DeferredThread)
    gms_failover.classify_frozen_vllm_worker_rank()
    assert order == [
        ("classify", {(7, 17)}, False),
        ("release",),
        ("proof-start",),
    ]


@pytest.mark.asyncio
async def test_failed_gpu_proof_keeps_foreign_leases_quarantined(monkeypatch):
    from gpu_memory_service.integrations.common import gpu_quiescence
    from gpu_memory_service.integrations.sglang import writer_lifecycle

    calls = []
    monkeypatch.setenv("GMS_KV_DIRECTORY_MODE", "off")
    monkeypatch.setattr(
        writer_lifecycle,
        "fence_predecessor_writers",
        lambda: asyncio.sleep(0, result="old-cohort"),
    )
    monkeypatch.setattr(
        gpu_quiescence, "gpu_quiescence_provider_configured", lambda _name: True
    )

    async def failed_proof(**_kwargs):
        return SimpleNamespace(
            quiesced=False, provider="gms-mps", detail="CUDA 806", elapsed_ms=4.0
        )

    monkeypatch.setattr(
        gpu_quiescence, "prove_predecessor_gpu_quiescence", failed_proof
    )
    monkeypatch.setattr(
        "dynamo.common.gms_failover._recover_foreign_kv_leases_after_fence",
        lambda *_args, gpu_quiesced, **_kwargs: calls.append(gpu_quiesced),
    )
    monkeypatch.setattr(
        writer_lifecycle,
        "mark_gms_recovery_ready",
        lambda: calls.append("recovery_ready"),
    )
    monkeypatch.setattr(
        writer_lifecycle,
        "mark_gpu_quiescence_ready",
        lambda: calls.append("gpu_ready"),
    )

    with pytest.raises(
        RuntimeError, match="could not prove predecessor GPU quiescence"
    ):
        await run_gms_failover_post_lock_fence(backend_name="sglang", role="shadow")

    assert calls == [False]


@pytest.mark.asyncio
async def test_sglang_marks_gpu_ready_after_proven_reclaim(monkeypatch):
    from types import SimpleNamespace

    from gpu_memory_service.integrations.common import gpu_quiescence
    from gpu_memory_service.integrations.sglang import writer_lifecycle

    from dynamo.common import gms_failover

    order = []

    async def prove(**_kwargs):
        return SimpleNamespace(
            quiesced=True, provider="test", detail="", elapsed_ms=0.1
        )

    monkeypatch.setattr(gpu_quiescence, "prove_predecessor_gpu_quiescence", prove)
    monkeypatch.setattr(
        gpu_quiescence, "gpu_quiescence_provider_configured", lambda _name: True
    )
    monkeypatch.setattr(
        writer_lifecycle,
        "fence_predecessor_writers",
        lambda: asyncio.sleep(0, result="old-cohort"),
    )
    monkeypatch.setattr(
        gms_failover,
        "_recover_foreign_kv_leases_after_fence",
        lambda *_args, gpu_quiesced, **_kwargs: order.append(
            "reclaim" if gpu_quiesced else "classify"
        ),
    )
    monkeypatch.setattr(
        writer_lifecycle,
        "mark_gms_recovery_ready",
        lambda: order.append("recovery_ready"),
    )
    monkeypatch.setattr(
        writer_lifecycle,
        "mark_gpu_quiescence_ready",
        lambda: order.append("ready"),
    )

    await run_gms_failover_post_lock_fence(backend_name="sglang", role="shadow")

    assert order == ["classify", "reclaim", "recovery_ready", "ready"]


@pytest.mark.asyncio
async def test_cancelled_fence_drains_directory_promotion(monkeypatch):
    monkeypatch.setenv("GMS_KV_DIRECTORY_MODE", "shadow")
    monkeypatch.setenv("DYN_GMS_FAILOVER_POST_LOCK_FENCE_MS", "0")
    promotion_started = asyncio.Event()
    allow_promotion = asyncio.Event()
    reclaimed = []

    async def to_thread(fn, *args):
        promotion_started.set()
        await allow_promotion.wait()
        return {7}, {(7, 17)}

    monkeypatch.setattr("dynamo.common.gms_failover.asyncio.to_thread", to_thread)
    monkeypatch.setattr(
        "dynamo.common.gms_failover._recover_foreign_kv_leases_after_fence",
        lambda *args, **kwargs: reclaimed.append((args, kwargs)),
    )

    fence = asyncio.create_task(
        run_gms_failover_post_lock_fence(backend_name="vllm", role="shadow")
    )
    await promotion_started.wait()
    fence.cancel()
    await asyncio.sleep(0)
    assert not fence.done()
    fence.cancel()
    await asyncio.sleep(0)
    assert not fence.done()

    allow_promotion.set()
    with pytest.raises(asyncio.CancelledError):
        await fence
    assert reclaimed == []


def test_post_fence_reclaim_uses_allocator_namespace(monkeypatch):
    from types import SimpleNamespace

    from gpu_memory_service.integrations.common import kv_lease_client

    from dynamo.common import gms_failover

    monkeypatch.setenv("GMS_KV_LEASES", "on")
    calls = []

    monkeypatch.setattr(kv_lease_client, "resolve_lease_device", lambda _env: 0)
    monkeypatch.setattr(
        kv_lease_client,
        "recover_foreign_kv_leases_in_shm_dir",
        lambda engine, device, **kwargs: (
            calls.append((engine, device, kwargs))
            or SimpleNamespace(
                files=1,
                released_idle_blocks=3,
                quarantined_blocks=4,
                reclaimed_blocks=0,
                errors=0,
            )
        ),
    )

    gms_failover._recover_foreign_kv_leases_after_fence(
        "sglang",
        "shadow",
        gpu_quiesced=False,
        protected_blocks={7},
        protected_leases={(7, 17)},
    )

    assert calls[0][0:2] == ("sglang", 0)
    assert calls[0][2]["namespace_suffix"] == "page-pool"
    assert calls[0][2]["protected_blocks"] == {7}
    assert calls[0][2]["protected_leases"] == {(7, 17)}


def test_post_fence_reclaim_honors_disabled_engine_override(monkeypatch):
    from gpu_memory_service.integrations.common import kv_lease_client

    from dynamo.common import gms_failover

    monkeypatch.setenv("GMS_KV_LEASES", "1")
    monkeypatch.setenv("GMS_SGLANG_KV_LEASES", "0")
    calls = []
    monkeypatch.setattr(
        kv_lease_client,
        "recover_foreign_kv_leases_in_shm_dir",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    gms_failover._recover_foreign_kv_leases_after_fence(
        "sglang", "shadow", gpu_quiesced=False
    )

    assert calls == []


def test_post_fence_reclaim_rejects_disabled_classification(monkeypatch):
    monkeypatch.setenv("GMS_KV_LEASES", "on")
    monkeypatch.setenv("DYN_GMS_FAILOVER_RECLAIM_FOREIGN_LEASES", "0")

    with pytest.raises(RuntimeError, match="cannot disable foreign lease"):
        gms_failover._recover_foreign_kv_leases_after_fence(
            "vllm", "shadow", gpu_quiesced=False
        )


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

    class _OrderedRuntime(_Runtime):
        def set_health_status(self, ready):
            order.append(("health", ready))
            super().set_health_status(ready)

    runtime = _OrderedRuntime()
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
        ("health", False),
        ("resume", ["kv_cache"]),
        ("mark_resumed", None),
        ("warmup", None),
        ("health", True),
    ]
    assert runtime.health == [False, True]


@pytest.mark.asyncio
async def test_activation_cancellation_drains_requiesce_and_lock_release(monkeypatch):
    monkeypatch.setenv("DYN_GMS_FAILOVER_SHADOW_MODE", "true")
    runtime = _Runtime()
    warmup_started = asyncio.Event()
    requiesce_started = asyncio.Event()
    allow_requiesce = asyncio.Event()
    release_started = asyncio.Event()
    allow_release = asyncio.Event()
    created = []

    class BlockingRequiesceController(_Controller):
        async def quiesce(self, tags):
            self.quiesce_calls.append(tags)
            if len(self.quiesce_calls) > 1:
                requiesce_started.set()
                await allow_requiesce.wait()
            return True

    class BlockingReleaseLock(_BusyOnTryLock):
        def __init__(self, path):
            super().__init__(path)
            created.append(self)

        async def release(self):
            release_started.set()
            await allow_release.wait()
            self.released += 1

    async def blocking_warmup():
        warmup_started.set()
        await asyncio.Event().wait()

    owner = _Owner()
    owner._quiesce_controller = BlockingRequiesceController()
    activation = asyncio.create_task(
        prepare_gms_failover(
            owner,
            runtime,
            backend_name="test",
            tags=["kv_cache"],
            lock_factory=BlockingReleaseLock,
            promotion_warmup=blocking_warmup,
        )
    )
    await warmup_started.wait()
    activation.cancel()
    await requiesce_started.wait()
    activation.cancel()
    await asyncio.sleep(0)
    assert not activation.done()
    assert not release_started.is_set()

    allow_requiesce.set()
    await release_started.wait()
    activation.cancel()
    await asyncio.sleep(0)
    assert not activation.done()

    allow_release.set()
    with pytest.raises(asyncio.CancelledError):
        await activation

    assert created[0].released == 1
    assert owner._quiesce_controller.quiesce_calls == [
        ["kv_cache"],
        ["kv_cache"],
    ]


@pytest.mark.asyncio
async def test_requiesce_timeout_is_hard_when_cancellation_is_suppressed(monkeypatch):
    monkeypatch.setenv("DYN_GMS_FAILOVER_REQUIESCE_TIMEOUT_SECS", "0.1")
    cancellation_seen = asyncio.Event()
    finish = asyncio.Event()

    class CancellationResistantController:
        async def quiesce(self, tags):
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancellation_seen.set()
                await finish.wait()

    result = await asyncio.wait_for(
        _requiesce_after_activation_error(
            CancellationResistantController(),
            ["kv_cache"],
            backend_name="test",
        ),
        timeout=0.5,
    )

    assert result == (False, None)
    await asyncio.wait_for(cancellation_seen.wait(), timeout=0.1)
    finish.set()
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_failed_requiesce_retains_lock_and_terminates(monkeypatch):
    monkeypatch.setenv("DYN_GMS_FAILOVER_SHADOW_MODE", "true")
    owner = _Owner()
    runtime = _Runtime()
    killed = []
    created = []

    class FailingRequiesceController(_Controller):
        async def quiesce(self, tags):
            self.quiesce_calls.append(tags)
            if len(self.quiesce_calls) > 1:
                raise RuntimeError("cannot quiesce")
            return True

    class RecordingLock(_BusyOnTryLock):
        def __init__(self, path):
            super().__init__(path)
            created.append(self)

    async def failing_warmup():
        raise RuntimeError("warmup failed")

    owner._quiesce_controller = FailingRequiesceController()
    monkeypatch.setattr(
        "dynamo.common.gms_failover.os.kill",
        lambda pid, sig: killed.append((pid, sig)),
    )

    with pytest.raises(RuntimeError, match="warmup failed"):
        await prepare_gms_failover(
            owner,
            runtime,
            backend_name="test",
            tags=["kv_cache"],
            lock_factory=RecordingLock,
            promotion_warmup=failing_warmup,
        )

    assert owner._gms_failover_lock is created[0]
    assert created[0].released == 0
    assert killed == [(os.getpid(), signal.SIGTERM)]
    assert runtime.health[-1] is False


@pytest.mark.asyncio
async def test_gms_failover_warmup_failure_requiesces_and_releases_lock(monkeypatch):
    monkeypatch.setenv("DYN_GMS_FAILOVER_SHADOW_MODE", "true")
    monkeypatch.setenv("DYN_GMS_FAILOVER_KEEP_SHADOW_READY", "false")
    owner = _Owner()
    runtime = _Runtime()
    created = []

    class RecordingLock(_BusyOnTryLock):
        def __init__(self, path):
            super().__init__(path)
            created.append(self)

    async def failing_warmup():
        raise RuntimeError("warmup failed")

    with pytest.raises(RuntimeError, match="warmup failed"):
        await prepare_gms_failover(
            owner,
            runtime,
            backend_name="test",
            tags=["kv_cache"],
            lock_factory=RecordingLock,
            promotion_warmup=failing_warmup,
        )

    assert owner._quiesce_controller.quiesce_calls == [
        ["kv_cache"],
        ["kv_cache"],
    ]
    assert owner._quiesce_controller.resume_calls == [["kv_cache"]]
    assert created[0].released == 1
    assert runtime.health == [False, False]


@pytest.mark.asyncio
async def test_release_attached_gms_failover_lock_releases_and_detaches(monkeypatch):
    class _Handler:
        pass

    handler = _Handler()
    lock = _Lock("/locks/failover.lock")
    setattr(handler, "_gms_failover_lock", lock)
    monkeypatch.setenv("DYN_GMS_FAILOVER_ACTIVE_LOCK_HELD", "1")

    released = await release_attached_gms_failover_lock(handler, backend_name="test")

    assert released is True
    assert lock.released == 1
    assert getattr(handler, "_gms_failover_lock") is None
    assert "DYN_GMS_FAILOVER_ACTIVE_LOCK_HELD" not in os.environ


@pytest.mark.asyncio
async def test_release_attached_gms_failover_lock_without_lock_is_noop():
    class _Handler:
        pass

    handler = _Handler()

    released = await release_attached_gms_failover_lock(handler, backend_name="test")

    assert released is False


def test_release_attached_gms_failover_lock_nowait_releases_and_detaches(monkeypatch):
    class _NowaitLock:
        def __init__(self):
            self.released = 0

        def release_nowait(self):
            self.released += 1
            return True

    handler = _Owner()
    lock = _NowaitLock()
    handler._gms_failover_lock = lock
    monkeypatch.setenv("DYN_GMS_FAILOVER_ACTIVE_LOCK_HELD", "1")

    assert release_attached_gms_failover_lock_nowait(handler, backend_name="test")
    assert lock.released == 1
    assert handler._gms_failover_lock is None
    assert "DYN_GMS_FAILOVER_ACTIVE_LOCK_HELD" not in os.environ


def test_writer_cohort_replaces_timed_quiescence_guess(monkeypatch):
    from dynamo.common.gms_failover import _post_lock_fence_ms

    monkeypatch.delenv("DYN_GMS_FAILOVER_POST_LOCK_FENCE_MS", raising=False)
    monkeypatch.delenv("DYN_SGLANG_GMS_FAILOVER_POST_LOCK_FENCE_MS", raising=False)
    monkeypatch.delenv("DYN_VLLM_GMS_FAILOVER_POST_LOCK_FENCE_MS", raising=False)

    assert _post_lock_fence_ms("sglang") == 0
    assert _post_lock_fence_ms("vllm") == 0


def test_failover_nccl_environment_bounds_default_teardown(monkeypatch):
    monkeypatch.setenv("DYN_GMS_FAILOVER_SHADOW_MODE", "1")
    monkeypatch.delenv("TORCH_NCCL_ASYNC_ERROR_HANDLING", raising=False)
    monkeypatch.delenv("TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC", raising=False)

    assert configure_failover_nccl_environment() == {
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
        "TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC": "1000",
    }


def test_failover_nccl_environment_preserves_operator_values(monkeypatch):
    monkeypatch.setenv("DYN_GMS_FAILOVER_SHADOW_MODE", "true")
    monkeypatch.setenv("TORCH_NCCL_ASYNC_ERROR_HANDLING", "2")
    monkeypatch.setenv("TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC", "2500")

    assert configure_failover_nccl_environment() == {}
    assert os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] == "2"
    assert os.environ["TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC"] == "2500"


def test_failover_nccl_environment_is_inactive_outside_failover(monkeypatch):
    monkeypatch.delenv("DYN_GMS_FAILOVER_SHADOW_MODE", raising=False)
    monkeypatch.delenv("TORCH_NCCL_ASYNC_ERROR_HANDLING", raising=False)
    monkeypatch.delenv("TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC", raising=False)

    assert configure_failover_nccl_environment() == {}
    assert "TORCH_NCCL_ASYNC_ERROR_HANDLING" not in os.environ
    assert "TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC" not in os.environ


def _standby_gate_env(monkeypatch, tmp_path, *, engine_id="0"):
    monkeypatch.setenv("DYN_GMS_FAILOVER_SERVE_AFTER_STANDBY", "1")
    monkeypatch.setenv("FAILOVER_LOCK_PATH", str(tmp_path / "failover.lock"))
    monkeypatch.setenv("ENGINE_ID", engine_id)
    monkeypatch.setenv("DYN_GMS_FAILOVER_PRIMARY_ENGINE_ID", "0")


@pytest.mark.asyncio
async def test_standby_gate_is_opt_in(monkeypatch, tmp_path):
    monkeypatch.delenv("DYN_GMS_FAILOVER_SERVE_AFTER_STANDBY", raising=False)
    monkeypatch.setenv("FAILOVER_LOCK_PATH", str(tmp_path / "failover.lock"))
    assert await gms_failover.wait_for_armed_standby_before_serving("vllm") is False
    assert not (tmp_path / "failover.lock.primary-boot").exists()


@pytest.mark.asyncio
async def test_primary_serves_after_this_boots_standby_arms(monkeypatch, tmp_path):
    _standby_gate_env(monkeypatch, tmp_path)
    armed = tmp_path / "failover.lock.standby-armed"
    armed.mkdir()
    # A marker armed for an earlier primary boot must not release the gate.
    (armed / "engine-9").write_text("stale-boot")
    gate = asyncio.create_task(
        gms_failover.wait_for_armed_standby_before_serving("vllm")
    )
    await asyncio.sleep(0.6)
    assert not gate.done()

    monkeypatch.setenv("ENGINE_ID", "1")
    standby = asyncio.create_task(
        gms_failover.acquire_lock_while_armed("engine-1", asyncio.Event().wait)
    )
    assert await asyncio.wait_for(gate, timeout=5.0) is True
    standby.cancel()
    with pytest.raises(asyncio.CancelledError):
        await standby
    assert (armed / "engine-1").read_text() == (
        tmp_path / "failover.lock.primary-boot"
    ).read_text()


@pytest.mark.asyncio
async def test_primary_serves_without_standby_after_gate_timeout(monkeypatch, tmp_path):
    _standby_gate_env(monkeypatch, tmp_path)
    monkeypatch.setenv("DYN_GMS_FAILOVER_STANDBY_GATE_SECS", "0.3")
    started = time.monotonic()
    assert await gms_failover.wait_for_armed_standby_before_serving("vllm") is False
    assert time.monotonic() - started < 3.0


@pytest.mark.asyncio
async def test_successor_never_waits_for_a_standby(monkeypatch, tmp_path):
    _standby_gate_env(monkeypatch, tmp_path, engine_id="1")
    assert await gms_failover.wait_for_armed_standby_before_serving("vllm") is False
    assert not (tmp_path / "failover.lock.primary-boot").exists()
