# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared active/shadow gating for GMS-managed KV failover."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import time
import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Iterable

from dynamo.common.utils.env import env_bool as _truthy_env
from dynamo.common.utils.env import env_float as _float_env
from dynamo.common.utils.env import env_int as _int_env
from dynamo.common.utils.env import env_set_unless_false

logger = logging.getLogger(__name__)

DEFAULT_FAILOVER_LOCK_PATH = "/shared/failover.lock"
DEFAULT_FAILOVER_TAGS = ("kv_cache", "weights")
KEEP_SHADOW_READY_ENV = "DYN_GMS_FAILOVER_KEEP_SHADOW_READY"
LEASE_TRANSITION_SERVING_ENV = "DYN_GMS_FAILOVER_LEASE_TRANSITION_SERVING"
_gpu_quiescence_tasks: set[asyncio.Task[None]] = set()


def configure_failover_nccl_environment() -> dict[str, str]:
    """Bound dead-cohort teardown before CUDA/NCCL initialization.

    A surviving TP rank can observe its peer failure immediately yet spend the
    PyTorch default 60 seconds coordinating a flight-recorder dump before the
    watchdog aborts it. Whole-pool takeover must wait for every predecessor
    rank, so that diagnostic default becomes user-visible downtime. Apply a
    bounded failover default early, while preserving every explicit operator
    setting.
    """
    if not _truthy_env("DYN_GMS_FAILOVER_SHADOW_MODE"):
        return {}
    defaults = {
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
        "TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC": "1000",
    }
    applied = {}
    for name, value in defaults.items():
        if name not in os.environ:
            os.environ[name] = value
            applied[name] = value
    return applied


def _promotion_warmup_enabled(backend_name: str | None = None) -> bool:
    if backend_name:
        backend = backend_name.upper().replace("-", "_")
        backend_env = f"DYN_{backend}_GMS_FAILOVER_PROMOTION_WARMUP"
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


def _promotion_warmup_concurrency(backend_name: str) -> int:
    backend_env = _backend_env_name(
        backend_name, "GMS_FAILOVER_PROMOTION_WARMUP_CONCURRENCY"
    )
    if backend_env in os.environ:
        return max(1, _int_env(backend_env, 1))
    return max(1, _int_env("DYN_GMS_FAILOVER_PROMOTION_WARMUP_CONCURRENCY", 1))


def _promotion_warmup_token_counts(backend_name: str) -> tuple[int, ...]:
    """Return opt-in per-request token counts for production-shape warmup."""

    backend_env = _backend_env_name(
        backend_name, "GMS_FAILOVER_PROMOTION_WARMUP_TOKEN_COUNTS"
    )
    raw = os.environ.get(backend_env)
    if raw is None:
        raw = os.environ.get("DYN_GMS_FAILOVER_PROMOTION_WARMUP_TOKEN_COUNTS", "")
    if not raw.strip():
        return ()
    try:
        counts = tuple(dict.fromkeys(int(value.strip()) for value in raw.split(",")))
    except ValueError as exc:
        raise ValueError(
            f"{backend_env} must be a comma-separated integer list"
        ) from exc
    if any(value <= 0 for value in counts):
        raise ValueError(f"{backend_env} token counts must be positive")
    return counts


def _promotion_warmup_payloads(
    payload: dict[str, Any], backend_name: str
) -> tuple[dict[str, Any], ...]:
    """Expand a token-input probe into isolated production-shape requests.

    The caller gives every vLLM stream a distinct cache salt. Without it,
    progressively longer warmups share their prefix and fail to exercise the
    intended prefill shape, leaving Triton MoE kernels for the first request.
    """

    counts = _promotion_warmup_token_counts(backend_name)
    if not counts:
        return (dict(payload),)
    token_ids = payload.get("token_ids")
    if not isinstance(token_ids, list) or not token_ids:
        raise ValueError(
            "promotion warmup token counts require a non-empty token_ids payload"
        )
    variants = []
    for count in counts:
        variant = dict(payload)
        repeats = (count + len(token_ids) - 1) // len(token_ids)
        variant["token_ids"] = (list(token_ids) * repeats)[:count]
        variants.append(variant)
    return tuple(variants)


def _post_lock_fence_ms(backend_name: str) -> int:
    # Cohort guards exclude CPU submitters and permanently close old admission.
    # Their release is NOT proof that previously submitted CUDA work has drained:
    # driver cleanup can outlive the file lock. Nor can a fixed sleep establish
    # that proof. Keep the delay for diagnostics only; crash-time writable reuse
    # still requires a validated GPU-quiescence contract for the deployed driver.
    default_ms = 0
    backend_env = _backend_env_name(backend_name, "GMS_FAILOVER_POST_LOCK_FENCE_MS")
    if backend_env in os.environ:
        return max(0, _int_env(backend_env, default_ms))
    return max(0, _int_env("DYN_GMS_FAILOVER_POST_LOCK_FENCE_MS", default_ms))


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
    """Run local canary requests before an engine enters discovery."""

    if not _promotion_warmup_enabled(backend_name):
        return

    attempts = _promotion_warmup_attempts()
    timeout_s = _promotion_warmup_timeout_s()
    backoff_s = _promotion_warmup_backoff_s()
    last_error: Exception | None = None

    concurrency = _promotion_warmup_concurrency(backend_name)
    warmup_payloads = _promotion_warmup_payloads(payload, backend_name)

    async def _run_stream(run_payload: dict[str, Any]) -> None:
        # Engine handlers accept Dynamo's native Context, not merely a Python
        # object with similarly named methods. SGLang forwards this through a
        # compiled boundary that enforces the concrete type.
        from dynamo._core import Context

        context = Context(f"gms-failover-promotion-warmup-{uuid.uuid4()}")
        request = dict(run_payload)
        if backend_name.lower() == "vllm":
            # Isolate concurrent requests too: an already-completed peer must
            # not turn the remaining probes into prefix-cache hits.
            nvext = dict(request.get("nvext") or {})
            nvext["cache_salt"] = f"gms-promotion-warmup-stream-{uuid.uuid4()}"
            request["nvext"] = nvext
        stream = generate(request, context)
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

    async def _run_once(attempt: int) -> None:
        started = time.monotonic()
        for run_payload in warmup_payloads:
            await asyncio.gather(
                *(_run_stream(run_payload) for _ in range(concurrency))
            )
        logger.info(
            "[GMS failover] %s promotion warmup completed attempt=%d "
            "concurrency=%d token_counts=%s elapsed_ms=%.2f",
            backend_name,
            attempt,
            concurrency,
            [len(item.get("token_ids", ())) for item in warmup_payloads],
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


def release_attached_gms_failover_lock_nowait(
    target: Any,
    *,
    backend_name: str,
) -> bool:
    """Release a concrete flock from a watchdog thread, if supported."""

    lock = getattr(target, "_gms_failover_lock", None)
    release_nowait = getattr(lock, "release_nowait", None)
    if lock is None or not callable(release_nowait):
        return False
    if not release_nowait():
        return False
    setattr(target, "_gms_failover_lock", None)
    logger.info(
        "[GMS failover] %s released active lock from watchdog thread",
        backend_name,
    )
    return True


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
            "[GMS failover] %s controlled handoff requested but no active lock "
            "is attached",
            backend_name,
        )
        return False

    if release_attached_gms_failover_lock_nowait(target, backend_name=backend_name):
        return True

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


def _normalize_lease_engine_name(backend_name: str) -> str:
    normalized = backend_name.lower().replace("-", "_")
    if normalized in {"trt", "trt_llm", "tensorrt_llm"}:
        return "trtllm"
    return normalized


def lease_transition_serving_enabled(
    backend_name: str, *, mapped_standby: bool
) -> bool:
    """Validate the legacy transition-serving configuration.

    This flag no longer selects an earlier admission path: every takeover
    fences CPU writers and classifies predecessor leases before admission.
    Retain the configuration checks for existing deployments that set it.
    Optional GPU-quiescence reclamation remains asynchronous; setting this
    flag never authorizes reuse of quarantined pages.
    """

    if not _truthy_env(LEASE_TRANSITION_SERVING_ENV):
        return False
    engine_id = os.environ.get("ENGINE_ID", "0")
    primary_engine_id = os.environ.get("DYN_GMS_FAILOVER_PRIMARY_ENGINE_ID", "0")
    is_shadow = engine_id != primary_engine_id
    if is_shadow and not mapped_standby:
        raise RuntimeError(
            f"{LEASE_TRANSITION_SERVING_ENV}=1 requires a mapped sleeping standby"
        )
    from gpu_memory_service.integrations.common.kv_lease_client import kv_leases_enabled

    engine = _normalize_lease_engine_name(backend_name)
    if not kv_leases_enabled(engine):
        raise RuntimeError(
            f"{LEASE_TRANSITION_SERVING_ENV}=1 requires {engine} KV leases"
        )
    directory_mode = os.environ.get("GMS_KV_DIRECTORY_MODE", "off").strip().lower()
    if directory_mode == "off":
        raise RuntimeError(
            f"{LEASE_TRANSITION_SERVING_ENV}=1 requires the GMS KV directory"
        )

    if not os.environ.get("GMS_KV_DIRECTORY_MANIFEST", "").strip():
        raise RuntimeError(
            f"{LEASE_TRANSITION_SERVING_ENV}=1 requires GMS_KV_DIRECTORY_MANIFEST"
        )
    if not _directory_socket(backend_name):
        raise RuntimeError(
            f"{LEASE_TRANSITION_SERVING_ENV}=1 requires a GMS KV directory socket"
        )
    directory_is_standby = directory_mode == "shadow" or _truthy_env(
        "GMS_KV_DIRECTORY_STANDBY"
    )
    if is_shadow and not directory_is_standby:
        raise RuntimeError(
            f"{LEASE_TRANSITION_SERVING_ENV}=1 requires the mapped standby "
            "directory to remain read-only before takeover"
        )

    return True


def _directory_socket(backend_name: str) -> str:
    explicit = os.environ.get("GMS_KV_DIRECTORY_SOCKET", "").strip()
    if explicit:
        return explicit
    engine = _normalize_lease_engine_name(backend_name).upper()
    return os.environ.get(f"GMS_{engine}_DAEMON_SOCKET", "").strip()


def _promote_content_directory_after_fence(
    backend_name: str, role: str
) -> "tuple[set[int], set[tuple[int, int]] | None]":
    """Promote the directory writer and collect exact protected HBM leases.

    Constructs the content directory from the environment (so it works without
    the caller threading a directory in), promotes the writer -- fencing the
    crashed one -- and returns HBM slot ids together with their exact lease
    generations. Reclaim validates those generations against each rank-local
    lease ring so stale directory records cannot reserve reused pages.
    """
    from gms_kv_ring.common.content_directory import (
        ContentDirectory,
        resolve_directory_mode,
    )

    mode = resolve_directory_mode()
    if mode == "off":
        return set(), set()
    if not os.environ.get("GMS_KV_DIRECTORY_MANIFEST", "").strip():
        message = (
            "GMS KV failover requires GMS_KV_DIRECTORY_MANIFEST so promotion "
            "uses the same model/layout identity as the inference engine"
        )
        if mode == "authoritative":
            raise RuntimeError(message)
        logger.warning("[GMS failover] %s %s", backend_name, message)
        return set(), set()
    socket_path = _directory_socket(backend_name)
    if not socket_path:
        message = "GMS KV directory enabled without GMS_KV_DIRECTORY_SOCKET"
        if mode == "authoritative":
            raise RuntimeError(message)
        logger.warning("[GMS failover] %s %s", backend_name, message)
        return set(), set()
    directory = ContentDirectory(
        socket_path,
        engine=_normalize_lease_engine_name(backend_name),
        block_size=0,
        engine_id=os.environ.get("ENGINE_ID", "0"),
        mode=mode,
    )
    started = time.monotonic()
    protected_blocks: set[int] = set()
    protected_leases: set[tuple[int, int]] | None = None
    try:
        # ENGINE_ID is stable across process restarts. The external lock now
        # fences the former process, so force a fresh directory epoch even when
        # the writer ID is unchanged. This drops incomplete ACTIVE HBM entries
        # while preserving completion-confirmed READY blocks.
        epoch = directory.promote(force_new_epoch=True)
        blocks_by_engine, leases_by_engine = directory.hbm_lease_inventory()
        protected_blocks.update(
            block_id for slot_ids in blocks_by_engine.values() for block_id in slot_ids
        )
        if leases_by_engine is not None:
            protected_leases = {
                lease for leases in leases_by_engine.values() for lease in leases
            }
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
    return protected_blocks, protected_leases


def _recover_foreign_kv_leases_after_fence(
    backend_name: str,
    role: str,
    *,
    gpu_quiesced: bool,
    recovery_owner_id: str | None = None,
    protected_blocks: "set[int] | None" = None,
    protected_leases: "set[tuple[int, int]] | None" = None,
) -> None:
    """Classify predecessor pages and reclaim only with GPU proof.

    CPU writer fencing makes state immutable but does not prove queued CUDA work
    has drained. IDLE pages provide immediate safe headroom; ambiguous pages are
    quarantined. Only a capability-gated proof permits quarantine reclamation.

    ``protected_leases`` are directory-advertised READY HBM slot generations.
    Only records whose rank-local generation still matches are preserved for
    lazy adoption. ``protected_blocks`` remains for diagnostic compatibility.
    """

    if not _failover_reclaim_foreign_leases_enabled():
        return
    try:
        from gpu_memory_service.integrations.common.kv_lease_client import (
            default_kv_lease_namespace_suffix,
            kv_leases_enabled,
            recover_foreign_kv_leases_in_shm_dir,
            resolve_lease_device,
        )

        engine = _normalize_lease_engine_name(backend_name)
        if not kv_leases_enabled(engine):
            return
        device = resolve_lease_device(f"GMS_{engine.upper()}_KV_LEASE_DEVICE")
        started = time.monotonic()
        result = recover_foreign_kv_leases_in_shm_dir(
            engine,
            device,
            owner_id=recovery_owner_id,
            protected_blocks=protected_blocks,
            protected_leases=protected_leases,
            namespace_suffix=default_kv_lease_namespace_suffix(engine),
            gpu_quiesced=gpu_quiesced,
        )
        elapsed_ms = (time.monotonic() - started) * 1000.0
        if result.errors:
            raise RuntimeError(
                f"GMS KV recovery classification failed in {result.errors} file(s)"
            )
        if (
            result.files
            or result.released_idle_blocks
            or result.quarantined_blocks
            or result.reclaimed_blocks
        ):
            logger.info(
                "[GMS failover] %s %s KV recovery files=%d idle_released=%d "
                "quarantined=%d reclaimed=%d gpu_quiesced=%s elapsed_ms=%.2f",
                backend_name,
                role,
                result.files,
                result.released_idle_blocks,
                result.quarantined_blocks,
                result.reclaimed_blocks,
                gpu_quiesced,
                elapsed_ms,
            )
    except Exception:
        logger.exception(
            "[GMS failover] %s %s KV lease recovery failed closed",
            backend_name,
            role,
        )
        raise


async def _finish_gpu_quiescence_recovery(
    *,
    backend_name: str,
    role: str,
    predecessor_cohort: str | None,
    recovery_owner_id: str,
    lease_device: int,
) -> None:
    """Complete phase two without delaying safe phase-one serving."""
    from gpu_memory_service.integrations.common.gpu_quiescence import (
        prove_predecessor_gpu_quiescence,
    )

    proof = await prove_predecessor_gpu_quiescence(
        backend_name=backend_name,
        predecessor_cohort=predecessor_cohort,
        device=lease_device,
    )
    if not proof.quiesced:
        logger.warning(
            "[GMS failover] %s %s GPU proof rejected; quarantine retained "
            "provider=%s elapsed_ms=%.2f detail=%s",
            backend_name,
            role,
            proof.provider,
            proof.elapsed_ms,
            proof.detail,
        )
        return
    logger.info(
        "[GMS failover] %s %s GPU quiescence proven provider=%s "
        "elapsed_ms=%.2f detail=%s",
        backend_name,
        role,
        proof.provider,
        proof.elapsed_ms,
        proof.detail,
    )
    await asyncio.to_thread(
        _recover_foreign_kv_leases_after_fence,
        backend_name,
        role,
        gpu_quiesced=True,
        recovery_owner_id=recovery_owner_id,
    )


def _phase_two_finished(task: asyncio.Task[None]) -> None:
    _gpu_quiescence_tasks.discard(task)
    if task.cancelled():
        logger.warning("[GMS failover] GPU quiescence recovery was cancelled")
        return
    error = task.exception()
    if error is not None:
        # Phase one is already safe: retain quarantine and keep serving from
        # IDLE/free plus exact SEALED capacity instead of failing the engine.
        logger.error(
            "[GMS failover] GPU quiescence recovery failed; quarantine retained",
            exc_info=(type(error), error, error.__traceback__),
        )


def _schedule_gpu_quiescence_recovery(**kwargs) -> None:
    task = asyncio.create_task(_finish_gpu_quiescence_recovery(**kwargs))
    _gpu_quiescence_tasks.add(task)
    task.add_done_callback(_phase_two_finished)


async def run_gms_failover_post_lock_fence(
    *,
    backend_name: str,
    role: str,
) -> None:
    """Fence and classify shared-KV lease state after active ownership changes.

    When a content directory is configured, this first promotes the directory
    writer (fencing the crashed one) and collects its HBM-resident slots so the
    reclaim preserves them for adoption instead of freeing them and degrading
    failover to a full recompute.
    """

    predecessor = None
    if backend_name == "sglang":
        from gpu_memory_service.integrations.sglang.writer_lifecycle import (
            fence_predecessor_writers,
        )

        predecessor = await fence_predecessor_writers()
    elif backend_name == "vllm":
        from gpu_memory_service.integrations.vllm.writer_lifecycle import (
            fence_predecessor_writers,
        )

        predecessor = await fence_predecessor_writers()

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
    protected_leases: set[tuple[int, int]] | None = None
    if os.environ.get("GMS_KV_DIRECTORY_MODE", "off").strip().lower() != "off":
        # The lock holder must not release ownership while its blocking promotion
        # thread can still publish a generation. Shield it from task cancellation,
        # then drain it before propagating cancellation to the lock owner.
        promotion = asyncio.create_task(
            asyncio.to_thread(
                _promote_content_directory_after_fence, backend_name, role
            )
        )
        cancelled: asyncio.CancelledError | None = None
        while True:
            try:
                protected_blocks, protected_leases = await asyncio.shield(promotion)
                break
            except asyncio.CancelledError as exc:
                if promotion.cancelled():
                    raise
                cancelled = exc
                current = asyncio.current_task()
                if current is not None:
                    current.uncancel()
                if not promotion.done():
                    continue
                try:
                    protected_blocks, protected_leases = promotion.result()
                except Exception:
                    logger.exception(
                        "[GMS failover] %s %s directory promotion failed while "
                        "draining cancellation",
                        backend_name,
                        role,
                    )
                break
            except Exception:
                if cancelled is None:
                    raise
                logger.exception(
                    "[GMS failover] %s %s directory promotion failed while "
                    "draining cancellation",
                    backend_name,
                    role,
                )
                break
        if cancelled is not None:
            raise cancelled
    from gpu_memory_service.integrations.common.kv_lease_client import (
        current_kv_lease_owner_id,
        resolve_lease_device,
    )

    lease_engine = _normalize_lease_engine_name(backend_name)
    lease_device = resolve_lease_device(f"GMS_{lease_engine.upper()}_KV_LEASE_DEVICE")
    recovery_owner_id = current_kv_lease_owner_id(lease_engine, lease_device)
    _recover_foreign_kv_leases_after_fence(
        backend_name,
        role,
        gpu_quiesced=False,
        recovery_owner_id=recovery_owner_id,
        protected_blocks=protected_blocks,
        protected_leases=protected_leases,
    )
    from gpu_memory_service.integrations.common.gpu_quiescence import (
        gpu_quiescence_provider_configured,
    )

    if gpu_quiescence_provider_configured(backend_name):
        # Serving can resume from phase-one IDLE/free capacity and exact SEALED
        # KV immediately. The optional platform proof unlocks ambiguous pages
        # later without putting its latency on the user-visible takeover path.
        _schedule_gpu_quiescence_recovery(
            backend_name=backend_name,
            role=role,
            predecessor_cohort=(None if predecessor is None else str(predecessor)),
            recovery_owner_id=recovery_owner_id,
            lease_device=lease_device,
        )


def _controller_from(owner: Any) -> Any:
    controller = getattr(owner, "_quiesce_controller", owner)
    if controller is None:
        raise RuntimeError(
            "GMS failover shadow mode requires an engine quiesce controller"
        )
    return controller


def _lock_factory_or_default(
    lock_factory: Callable[[str], Any] | None,
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


def _reject_removed_private_bootstrap(backend_name: str) -> None:
    backend = backend_name.upper().replace("-", "_")
    removed = (
        "DYN_GMS_FAILOVER_PRIVATE_BOOTSTRAP_KV",
        f"DYN_{backend}_GMS_PRIVATE_BOOTSTRAP_KV",
        f"GMS_{backend}_PRIVATE_BOOTSTRAP_KV",
    )
    # Permissive on purpose: a banned option must not slip past this
    # guard because it was spelled `enabled` rather than `1`.
    enabled = [name for name in removed if env_set_unless_false(name)]
    if enabled:
        raise RuntimeError(
            "GMS private-bootstrap KV is no longer supported because its "
            "scratch isolation was removed; unset " + ", ".join(enabled)
        )


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
        from gpu_memory_service.failover_lock.interface import FailoverLockContended
    except ImportError:  # pragma: no cover - default lock import would also fail.
        FailoverLockContended = RuntimeError  # type: ignore[assignment]

    try:
        await lock.acquire(engine_id=engine_name, timeout=0.0)
        return True
    except TypeError:
        await lock.acquire(engine_name)
        return True
    except FailoverLockContended:
        return False


async def _release_lock_after_activation_error(lock: Any, *, backend_name: str) -> None:
    release = asyncio.create_task(lock.release())
    cancelled: asyncio.CancelledError | None = None
    while True:
        try:
            await asyncio.shield(release)
            break
        except asyncio.CancelledError as exc:
            cancelled = exc
            if release.cancelled():
                logger.exception(
                    "[GMS failover] %s lock release was cancelled",
                    backend_name,
                )
                break
            current = asyncio.current_task()
            if current is not None:
                current.uncancel()
        except BaseException:
            logger.exception(
                "[GMS failover] %s failed to release lock after activation error",
                backend_name,
            )
            break
    if cancelled is not None:
        raise cancelled


async def _requiesce_after_activation_error(
    controller: Any,
    tags: list[str],
    *,
    backend_name: str,
) -> tuple[bool, asyncio.CancelledError | None]:
    """Re-quiesce within a hard deadline despite repeated cancellation."""
    timeout = max(
        0.1,
        _float_env("DYN_GMS_FAILOVER_REQUIESCE_TIMEOUT_SECS", 30.0),
    )
    task = asyncio.create_task(controller.quiesce(tags))
    deadline = asyncio.get_running_loop().time() + timeout
    cancelled: asyncio.CancelledError | None = None

    def consume_detached_result(completed: asyncio.Task[Any]) -> None:
        try:
            completed.result()
        except asyncio.CancelledError:
            pass
        except BaseException:
            logger.debug(
                "[GMS failover] %s detached re-quiesce task failed",
                backend_name,
                exc_info=True,
            )

    while True:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            task.cancel()
            task.add_done_callback(consume_detached_result)
            logger.critical(
                "[GMS failover] %s re-quiesce exceeded %.1fs hard deadline",
                backend_name,
                timeout,
            )
            return False, cancelled
        try:
            done, _ = await asyncio.wait({task}, timeout=remaining)
        except asyncio.CancelledError as exc:
            cancelled = cancelled or exc
            current = asyncio.current_task()
            if current is not None:
                current.uncancel()
            continue
        if not done:
            continue
        try:
            task.result()
        except asyncio.CancelledError as exc:
            logger.critical(
                "[GMS failover] %s re-quiesce task was cancelled",
                backend_name,
            )
            return False, cancelled or exc
        except BaseException:
            logger.critical(
                "[GMS failover] %s failed to re-quiesce after activation error",
                backend_name,
                exc_info=True,
            )
            return False, cancelled
        return True, cancelled


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
    try:
        await run_gms_failover_post_lock_fence(backend_name=backend_name, role=role)
    except BaseException:
        await _release_lock_after_activation_error(lock, backend_name=backend_name)
        raise
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
    warm_standby_before_quiesce: bool = False,
    activation_barrier: Callable[[], Awaitable[None]] | None = None,
    lease_transition_serving: bool = False,
) -> GmsFailoverActivation:
    """Gate model registration until this engine owns the shared GMS namespace.

    Bulwark injects ``ENGINE_ID`` and ``FAILOVER_LOCK_PATH`` for all pods. This
    helper only activates when ``DYN_GMS_FAILOVER_SHADOW_MODE`` is true, so
    standalone GMS deployments keep the vanilla Dynamo registration path.
    """

    if not _truthy_env("DYN_GMS_FAILOVER_SHADOW_MODE"):
        return GmsFailoverActivation()

    _reject_removed_private_bootstrap(backend_name)
    lock_path, engine_name, _is_shadow, lock = _failover_lock_inputs(lock_factory)

    logger.info(
        "[GMS failover] %s attempting active lock at %s",
        backend_name,
        lock_path,
    )
    if await _try_acquire_active_lock(lock, engine_name):
        logger.info("[GMS failover] %s active", backend_name)
        try:
            await run_gms_failover_post_lock_fence(
                backend_name=backend_name,
                role="active",
            )
            if activation_barrier is not None:
                await activation_barrier()
        except BaseException:
            await _release_lock_after_activation_error(lock, backend_name=backend_name)
            raise
        return GmsFailoverActivation(
            enabled=True,
            lock=lock,
        )
    logger.info(
        "[GMS failover] %s active lock is held; preparing warm shadow",
        backend_name,
    )

    standby_prewarmed = False
    if warm_standby_before_quiesce:
        if promotion_warmup is None:
            raise RuntimeError("warm_standby_before_quiesce requires promotion_warmup")
        logger.info(
            "[GMS failover] %s warming lease-protected standby before quiesce",
            backend_name,
        )
        await promotion_warmup()
        standby_prewarmed = True

    controller = _controller_from(owner)
    tag_list = list(tags)

    role = "shadow"
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
    resume_started = False
    try:
        # Classify predecessor-held pages before admitting any successor read.
        # The shared ring has one reader count, so a background classifier could
        # not distinguish a stale primary pin from a newly admitted shadow pin.
        # This CPU-only phase is the safety boundary; GPU-quiescence reclamation
        # remains asynchronous inside run_gms_failover_post_lock_fence().
        await run_gms_failover_post_lock_fence(backend_name=backend_name, role=role)
        if activation_barrier is not None:
            await activation_barrier()

        resume_started = True
        await controller.resume(tag_list)
        mark_resumed = getattr(controller, "mark_resumed", None)
        if mark_resumed is not None:
            mark_resumed()

        if promotion_warmup is not None and not standby_prewarmed:
            await promotion_warmup()

        if not keep_shadow_ready and set_health_status is not None:
            set_health_status(True)

        if lease_transition_serving:
            logger.info(
                "[GMS failover] %s %s serving from FREE and exact-generation "
                "SEALED leases while GPU-quiescence recovery completes",
                backend_name,
                role,
            )
    except BaseException:
        safe_to_release = not resume_started
        cleanup_cancelled = None
        if resume_started:
            (
                safe_to_release,
                cleanup_cancelled,
            ) = await _requiesce_after_activation_error(
                controller,
                tag_list,
                backend_name=backend_name,
            )
        if set_health_status is not None:
            try:
                set_health_status(False)
            except BaseException:
                logger.exception(
                    "[GMS failover] %s failed to mark activation unhealthy",
                    backend_name,
                )
        if safe_to_release:
            await _release_lock_after_activation_error(lock, backend_name=backend_name)
        else:
            # Keep a strong reference to the lock until SIGTERM completes process
            # teardown. Releasing it while the engine may still write shared KV
            # would allow two physical writers.
            owner._gms_failover_lock = lock
            os.kill(os.getpid(), signal.SIGTERM)
        if cleanup_cancelled is not None:
            raise cleanup_cancelled
        raise

    logger.info(
        "[GMS failover] %s %s resumed; registering with discovery",
        backend_name,
        role,
    )
    return GmsFailoverActivation(
        enabled=True,
        lock=lock,
    )
