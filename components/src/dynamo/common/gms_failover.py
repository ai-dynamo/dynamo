# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared active/shadow gating for GMS-managed KV failover."""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from typing import Any, Callable

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
    # Non-zero default: on a SIGKILL failover the kernel releases the flock
    # instantly, but the dead leader's engine subprocesses and already-enqueued
    # CUDA work can keep writing the shared VMM KV pages for a short window. This
    # quiescence gap between acquiring the lock and the new owner writing shared
    # KV mitigates (does not eliminate) that dual-writer overlap; hard
    # elimination needs daemon-side mapping revocation (out of scope here).
    default_ms = 250
    backend_env = _backend_env_name(backend_name, "GMS_FAILOVER_POST_LOCK_FENCE_MS")
    if backend_env in os.environ:
        return max(0, _int_env(backend_env, default_ms))
    return max(0, _int_env("DYN_GMS_FAILOVER_POST_LOCK_FENCE_MS", default_ms))


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


def _directory_socket(backend_name: str) -> str:
    explicit = os.environ.get("GMS_KV_DIRECTORY_SOCKET", "").strip()
    if explicit:
        return explicit
    engine = _normalize_lease_engine_name(backend_name).upper()
    return os.environ.get(f"GMS_{engine}_DAEMON_SOCKET", "").strip()


def _promote_content_directory_after_fence(backend_name: str, role: str) -> "set[int]":
    """Promote this process to directory writer and collect protected HBM slots.

    Constructs the content directory from the environment (so it works without
    the caller threading a directory in), promotes the writer -- fencing the
    crashed one -- and returns the union of HBM-resident slot ids so the
    post-fence reclaim preserves those blocks (recoverable KV) instead of freeing
    them and degrading failover to a full recompute.
    """
    from gms_kv_ring.common.content_directory import (
        ContentDirectory,
        resolve_directory_mode,
    )

    mode = resolve_directory_mode()
    if mode == "off":
        return set()
    socket_path = _directory_socket(backend_name)
    if not socket_path:
        message = "GMS KV directory enabled without GMS_KV_DIRECTORY_SOCKET"
        if mode == "authoritative":
            raise RuntimeError(message)
        logger.warning("[GMS failover] %s %s", backend_name, message)
        return set()
    directory = ContentDirectory(
        socket_path,
        engine=_normalize_lease_engine_name(backend_name),
        block_size=0,
        engine_id=os.environ.get("ENGINE_ID", "0"),
        mode=mode,
    )
    started = time.monotonic()
    protected_blocks: set[int] = set()
    try:
        epoch = directory.promote()
        protected_blocks.update(
            block_id
            for slot_ids in directory.hbm_inventory().values()
            for block_id in slot_ids
        )
        logger.info(
            "[GMS failover] %s %s directory writer promoted epoch=%d "
            "protected_hbm_blocks=%d elapsed_ms=%.2f",
            backend_name,
            role,
            epoch,
            len(protected_blocks),
            (time.monotonic() - started) * 1000.0,
        )
    except Exception:
        if mode == "authoritative":
            raise
        logger.warning(
            "[GMS failover] %s %s directory promotion failed in shadow mode",
            backend_name,
            role,
            exc_info=True,
        )
    finally:
        directory.close()
    return protected_blocks


def _reclaim_foreign_kv_leases_after_fence(
    backend_name: str, role: str, protected_blocks: "set[int] | None" = None
) -> None:
    """Best-effort orphan lease reclaim after this process owns failover.

    The failover lock/epoch is the safety boundary. Before that point the
    previous primary may still write KV. After this process acquired the lock,
    foreign owners in the rank-local lease namespace are fenced leftovers from
    the previous primary and can be reclaimed to provide immediate HBM headroom.

    ``protected_blocks`` (directory-advertised READY HBM slots) are preserved for
    lazy adoption rather than freed.
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
            protected_blocks=protected_blocks,
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
        # A reclaim failure strands the ex-primary's leases and permanently
        # leaks HBM; surface it at WARNING so an operator can see it.
        logger.warning(
            "[GMS failover] %s %s post-fence KV lease reclaim failed",
            backend_name,
            role,
            exc_info=True,
        )


async def run_gms_failover_post_lock_fence(
    *,
    backend_name: str,
    role: str,
) -> None:
    """Fence and reclaim shared-KV lease state after active ownership changes.

    When a content directory is configured, this first promotes the directory
    writer (fencing the crashed one) and collects its HBM-resident slots so the
    reclaim preserves them for adoption instead of freeing them and degrading
    failover to a full recompute.
    """

    fence_ms = _post_lock_fence_ms(backend_name)
    if fence_ms > 0:
        logger.info(
            "[GMS failover] %s %s post-lock fence waiting %dms",
            backend_name,
            role,
            fence_ms,
        )
        await asyncio.sleep(fence_ms / 1000.0)
    protected_blocks: set[int] = set()
    if os.environ.get("GMS_KV_DIRECTORY_MODE", "off").strip().lower() != "off":
        # Blocking socket I/O -- run off the event loop.
        protected_blocks = await asyncio.to_thread(
            _promote_content_directory_after_fence, backend_name, role
        )
    _reclaim_foreign_kv_leases_after_fence(
        backend_name, role, protected_blocks=protected_blocks
    )
