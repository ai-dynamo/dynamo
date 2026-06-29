# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared active/shadow gating for GMS-managed KV failover."""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Iterable

logger = logging.getLogger(__name__)

DEFAULT_FAILOVER_LOCK_PATH = "/shared/failover.lock"
DEFAULT_FAILOVER_TAGS = ("kv_cache", "weights")
KEEP_SHADOW_READY_ENV = "DYN_GMS_FAILOVER_KEEP_SHADOW_READY"


def _truthy_env(name: str, *, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r", name, value)
        return default


def _float_env(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r", name, value)
        return default


def _promotion_warmup_enabled(backend_name: str | None = None) -> bool:
    if backend_name:
        backend_env = f"DYN_{backend_name.upper().replace('-', '_')}_GMS_FAILOVER_PROMOTION_WARMUP"
        if backend_env in os.environ:
            return _truthy_env(backend_env)
    return _truthy_env("DYN_GMS_FAILOVER_PROMOTION_WARMUP", default=True)


def _promotion_warmup_attempts() -> int:
    return max(1, _int_env("DYN_GMS_FAILOVER_PROMOTION_WARMUP_ATTEMPTS", 1))


def _promotion_warmup_timeout_s() -> float:
    return max(0.1, _float_env("DYN_GMS_FAILOVER_PROMOTION_WARMUP_TIMEOUT_SECS", 10.0))


def _promotion_warmup_backoff_s() -> float:
    return max(
        0.0, _int_env("DYN_GMS_FAILOVER_PROMOTION_WARMUP_BACKOFF_MS", 100) / 1000.0
    )


def _backend_env_name(backend_name: str, suffix: str) -> str:
    return f"DYN_{backend_name.upper().replace('-', '_')}_{suffix}"


def _post_lock_fence_ms(backend_name: str) -> int:
    backend_env = _backend_env_name(backend_name, "GMS_FAILOVER_POST_LOCK_FENCE_MS")
    if backend_env in os.environ:
        return max(0, _int_env(backend_env, 0))
    return max(0, _int_env("DYN_GMS_FAILOVER_POST_LOCK_FENCE_MS", 0))


class _PromotionWarmupContext:
    def __init__(self) -> None:
        self._id = f"gms-failover-promotion-warmup-{uuid.uuid4()}"
        self.trace_id = uuid.uuid4().hex
        self.span_id = uuid.uuid4().hex[:16]

    def id(self) -> str:
        return self._id

    def is_stopped(self) -> bool:
        return False

    def is_killed(self) -> bool:
        return False

    def trace_headers(self) -> dict[str, str]:
        return {}

    def async_killed_or_stopped(self) -> asyncio.Future[Any]:
        return asyncio.get_running_loop().create_future()


def _warmup_chunk_error(chunk: Any) -> str | None:
    if not isinstance(chunk, dict):
        return None
    status = chunk.get("status")
    if status == "error":
        return str(chunk.get("message") or chunk)
    if chunk.get("error"):
        return str(chunk.get("error"))
    finish_reason = chunk.get("finish_reason")
    if finish_reason == "error":
        return str(chunk.get("message") or chunk)
    if isinstance(finish_reason, dict) and finish_reason.get("error"):
        return str(finish_reason.get("error"))
    return None


async def run_gms_failover_promotion_warmup(
    generate: Callable[[dict[str, Any], Any], Any],
    payload: dict[str, Any],
    *,
    backend_name: str,
) -> None:
    """Run one local canary request before a promoted shadow enters discovery."""

    if not _promotion_warmup_enabled(backend_name):
        return

    attempts = _promotion_warmup_attempts()
    timeout_s = _promotion_warmup_timeout_s()
    backoff_s = _promotion_warmup_backoff_s()
    last_error: Exception | None = None

    async def _run_once(attempt: int) -> None:
        context = _PromotionWarmupContext()
        started = time.monotonic()
        stream = generate(dict(payload), context)
        saw_chunk = False
        try:
            while True:
                chunk = await asyncio.wait_for(anext(stream), timeout=timeout_s)
                saw_chunk = True
                error = _warmup_chunk_error(chunk)
                if error is not None:
                    raise RuntimeError(error)
        except StopAsyncIteration:
            if not saw_chunk:
                raise RuntimeError("promotion warmup stream ended without output")
        finally:
            aclose = getattr(stream, "aclose", None)
            if aclose is not None:
                await aclose()
        logger.info(
            "[GMS failover] %s promotion warmup completed attempt=%d elapsed_ms=%.2f",
            backend_name,
            attempt,
            (time.monotonic() - started) * 1000.0,
        )

    for attempt in range(1, attempts + 1):
        try:
            await _run_once(attempt)
            return
        except Exception as exc:  # noqa: BLE001 - this is a readiness gate.
            last_error = exc
            logger.warning(
                "[GMS failover] %s promotion warmup failed attempt=%d/%d: %s",
                backend_name,
                attempt,
                attempts,
                exc,
            )
            if attempt < attempts and backoff_s > 0:
                await asyncio.sleep(backoff_s)

    raise RuntimeError(
        f"GMS failover promotion warmup failed for {backend_name}: {last_error}"
    )


@dataclass
class GmsFailoverActivation:
    """Result of failover gating.

    Holding ``lock`` keeps the kernel flock fd alive for the active engine.
    Attach it to the long-lived request handler once the handler exists.
    """

    enabled: bool = False
    lock: Any | None = None

    def attach_to(self, target: Any) -> None:
        if self.lock is not None:
            setattr(target, "_gms_failover_lock", self.lock)


async def release_attached_gms_failover_lock(
    target: Any,
    *,
    backend_name: str,
) -> bool:
    """Release a handler's active failover lock for controlled handoff.

    The caller is responsible for first unregistering from discovery and
    quiescing engine memory. Releasing the lock lets the waiting shadow acquire
    ownership and publish its endpoint.
    """

    lock = getattr(target, "_gms_failover_lock", None)
    if lock is None:
        logger.info(
            "[GMS failover] %s controlled handoff requested but no active lock is attached",
            backend_name,
        )
        return False

    release = getattr(lock, "release", None)
    if release is None:
        logger.warning(
            "[GMS failover] %s attached lock does not support release; handoff skipped",
            backend_name,
        )
        return False

    await release()
    setattr(target, "_gms_failover_lock", None)
    logger.info(
        "[GMS failover] %s released active lock for controlled handoff", backend_name
    )
    return True


def _failover_reclaim_foreign_leases_enabled() -> bool:
    return _truthy_env("DYN_GMS_FAILOVER_RECLAIM_FOREIGN_LEASES", default=True)


def _failover_reclaim_max_blocks_per_file() -> int:
    value = os.environ.get("DYN_GMS_FAILOVER_RECLAIM_MAX_BLOCKS_PER_FILE", "0")
    try:
        return max(0, int(value))
    except ValueError:
        logger.warning(
            "Ignoring invalid DYN_GMS_FAILOVER_RECLAIM_MAX_BLOCKS_PER_FILE=%r",
            value,
        )
        return 0


def _normalize_lease_engine_name(backend_name: str) -> str:
    normalized = backend_name.lower().replace("-", "_")
    if normalized in {"trt", "trt_llm", "tensorrt_llm"}:
        return "trtllm"
    return normalized


def _reclaim_foreign_kv_leases_after_fence(backend_name: str, role: str) -> None:
    """Best-effort orphan lease reclaim after this process owns failover.

    The failover lock/epoch is the safety boundary. Before that point the
    previous primary may still write KV. After this process acquired the lock,
    foreign owners in the rank-local lease namespace are fenced leftovers from
    the previous primary and can be reclaimed to provide immediate HBM headroom.
    """

    if not _failover_reclaim_foreign_leases_enabled():
        return
    if not _truthy_env("GMS_KV_LEASES") and not _truthy_env(
        f"GMS_{_normalize_lease_engine_name(backend_name).upper()}_KV_LEASES"
    ):
        return
    try:
        from gpu_memory_service.integrations.common.kv_lease_client import (
            reclaim_foreign_kv_leases_in_shm_dir,
            resolve_lease_device,
        )

        engine = _normalize_lease_engine_name(backend_name)
        device = resolve_lease_device(f"GMS_{engine.upper()}_KV_LEASE_DEVICE")
        started = time.monotonic()
        result = reclaim_foreign_kv_leases_in_shm_dir(
            engine,
            device,
            max_blocks_per_file=_failover_reclaim_max_blocks_per_file(),
        )
        elapsed_ms = (time.monotonic() - started) * 1000.0
        if result.files or result.reclaimed_blocks or result.errors:
            logger.info(
                "[GMS failover] %s %s post-fence KV lease reclaim files=%d "
                "reclaimed_blocks=%d errors=%d elapsed_ms=%.2f",
                backend_name,
                role,
                result.files,
                result.reclaimed_blocks,
                result.errors,
                elapsed_ms,
            )
    except Exception:
        logger.debug(
            "[GMS failover] %s %s post-fence KV lease reclaim skipped",
            backend_name,
            role,
            exc_info=True,
        )


async def run_gms_failover_post_lock_fence(
    *,
    backend_name: str,
    role: str,
) -> None:
    """Fence and reclaim shared-KV lease state after active ownership changes."""

    fence_ms = _post_lock_fence_ms(backend_name)
    if fence_ms > 0:
        logger.info(
            "[GMS failover] %s %s post-lock fence waiting %dms",
            backend_name,
            role,
            fence_ms,
        )
        await asyncio.sleep(fence_ms / 1000.0)
    _reclaim_foreign_kv_leases_after_fence(backend_name, role)


def _controller_from(owner: Any) -> Any:
    controller = getattr(owner, "_quiesce_controller", owner)
    if controller is None:
        raise RuntimeError(
            "GMS failover shadow mode requires an engine quiesce controller"
        )
    return controller


def _lock_factory_or_default(
    lock_factory: Callable[[str], Any] | None
) -> Callable[[str], Any]:
    if lock_factory is not None:
        return lock_factory

    from gpu_memory_service.failover_lock.flock import FlockFailoverLock

    return FlockFailoverLock


def _failover_lock_inputs(
    lock_factory: Callable[[str], Any] | None,
) -> tuple[str, str, bool, Any]:
    factory = _lock_factory_or_default(lock_factory)
    lock_path = os.environ.get("FAILOVER_LOCK_PATH", DEFAULT_FAILOVER_LOCK_PATH)
    engine_id = os.environ.get("ENGINE_ID", "0")
    primary_engine_id = os.environ.get("DYN_GMS_FAILOVER_PRIMARY_ENGINE_ID", "0")
    engine_name = f"engine-{engine_id}"
    is_shadow = engine_id != primary_engine_id
    return lock_path, engine_name, is_shadow, factory(lock_path)


def _private_bootstrap_enabled(backend_name: str) -> bool:
    backend_env = (
        f"DYN_{backend_name.upper().replace('-', '_')}_GMS_PRIVATE_BOOTSTRAP_KV"
    )
    requested = _truthy_env(
        backend_env,
        default=_truthy_env("DYN_GMS_FAILOVER_PRIVATE_BOOTSTRAP_KV"),
    )
    if not requested:
        return False
    _lock_path, _engine_name, is_shadow, _lock = _failover_lock_inputs(None)
    return is_shadow


def _keep_shadow_ready() -> bool:
    return _truthy_env(KEEP_SHADOW_READY_ENV, default=True)


async def _try_acquire_active_lock(lock: Any, engine_name: str) -> bool:
    """Try to become active without blocking.

    Inter-pod failover replacement pods can reuse pod index 0 after a failover,
    while the active holder is now index 1. Static primary-index checks would
    make that replacement block forever without quiescing. The lock is the
    source of truth: if it is free, this worker is active; if it is busy, this
    worker must become a warm shadow and wait.
    """

    try:
        from gpu_memory_service.failover_lock.interface import FailoverLockError
    except ImportError:  # pragma: no cover - default lock import would also fail.
        FailoverLockError = RuntimeError  # type: ignore[assignment]

    try:
        await lock.acquire(engine_id=engine_name, timeout=0.0)
        return True
    except TypeError:
        # Minimal test doubles may not accept keyword arguments. Treat those as
        # uncontended locks so existing unit fakes keep the simple acquire path.
        await lock.acquire(engine_name)
        return True
    except FailoverLockError:
        return False


async def acquire_gms_failover_lock_before_init(
    *,
    backend_name: str,
    lock_factory: Callable[[str], Any] | None = None,
) -> GmsFailoverActivation:
    """Acquire the failover lock before engine initialization.

    Use this when the backend may write shared GMS KV during warmup/init. It
    serializes primary and shadow initialization so only the active engine can
    create CUDA contexts and mutate the shared KV namespace.
    """

    if not _truthy_env("DYN_GMS_FAILOVER_SHADOW_MODE"):
        return GmsFailoverActivation()

    lock_path, engine_name, is_shadow, lock = _failover_lock_inputs(lock_factory)
    role = "shadow" if is_shadow else "primary"
    logger.info(
        "[GMS failover] %s %s acquiring active lock before engine init at %s",
        backend_name,
        role,
        lock_path,
    )
    await lock.acquire(engine_id=engine_name)
    logger.info(
        "[GMS failover] %s %s acquired active lock before engine init",
        backend_name,
        role,
    )
    await run_gms_failover_post_lock_fence(backend_name=backend_name, role=role)
    return GmsFailoverActivation(
        enabled=True,
        lock=lock,
    )


async def prepare_gms_failover(
    owner: Any,
    runtime: Any,
    *,
    backend_name: str,
    tags: Iterable[str] = DEFAULT_FAILOVER_TAGS,
    lock_factory: Callable[[str], Any] | None = None,
    promotion_warmup: Callable[[], Awaitable[None]] | None = None,
) -> GmsFailoverActivation:
    """Gate model registration until this engine owns the shared GMS namespace.

    Bulwark injects ``ENGINE_ID`` and ``FAILOVER_LOCK_PATH`` for all pods. This
    helper only activates when ``DYN_GMS_FAILOVER_SHADOW_MODE`` is true, so
    standalone GMS deployments keep the vanilla Dynamo registration path.
    """

    if not _truthy_env("DYN_GMS_FAILOVER_SHADOW_MODE"):
        return GmsFailoverActivation()

    lock_path, engine_name, _is_shadow, lock = _failover_lock_inputs(lock_factory)

    private_bootstrap = _private_bootstrap_enabled(backend_name)
    if not private_bootstrap:
        logger.info(
            "[GMS failover] %s attempting active lock at %s",
            backend_name,
            lock_path,
        )
        if await _try_acquire_active_lock(lock, engine_name):
            logger.info("[GMS failover] %s active", backend_name)
            await run_gms_failover_post_lock_fence(
                backend_name=backend_name,
                role="active",
            )
            return GmsFailoverActivation(
                enabled=True,
                lock=lock,
            )
        logger.info(
            "[GMS failover] %s active lock is held; preparing warm shadow",
            backend_name,
        )

    controller = _controller_from(owner)
    tag_list = list(tags)

    role = "private-bootstrap member" if private_bootstrap else "shadow"
    logger.info(
        "[GMS failover] %s %s quiescing before discovery registration",
        backend_name,
        role,
    )
    await controller.quiesce(tag_list)

    set_health_status = getattr(runtime, "set_health_status", None)
    keep_shadow_ready = _keep_shadow_ready()
    if set_health_status is not None:
        if keep_shadow_ready:
            # Warm shadows stay Kubernetes-ready but remain out of route
            # discovery until they acquire the active lock.
            set_health_status(True)
        else:
            set_health_status(False)

    logger.info(
        "[GMS failover] %s %s waiting for active lock at %s",
        backend_name,
        role,
        lock_path,
    )
    await lock.acquire(engine_id=engine_name)
    logger.info(
        "[GMS failover] %s %s acquired active lock",
        backend_name,
        role,
    )
    await run_gms_failover_post_lock_fence(backend_name=backend_name, role=role)

    await controller.resume(tag_list)
    mark_resumed = getattr(controller, "mark_resumed", None)
    if mark_resumed is not None:
        mark_resumed()

    if promotion_warmup is not None:
        await promotion_warmup()

    if not keep_shadow_ready and set_health_status is not None:
        set_health_status(True)

    logger.info(
        "[GMS failover] %s %s resumed; registering with discovery",
        backend_name,
        role,
    )
    return GmsFailoverActivation(
        enabled=True,
        lock=lock,
    )
