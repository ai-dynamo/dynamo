# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared active/shadow gating for GMS-managed KV failover."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Iterable

if TYPE_CHECKING:
    from gpu_memory_service.integrations.common.kv_lease_client import (
        KVLeaseRecoveryResult,
    )

from dynamo.common.utils.env import env_bool as _truthy_env
from dynamo.common.utils.env import env_float as _float_env
from dynamo.common.utils.env import env_int as _int_env
from dynamo.common.utils.env import env_set_unless_false

logger = logging.getLogger(__name__)

DEFAULT_FAILOVER_LOCK_PATH = "/shared/failover.lock"
DEFAULT_FAILOVER_TAGS = ("kv_cache", "weights")
KEEP_SHADOW_READY_ENV = "DYN_GMS_FAILOVER_KEEP_SHADOW_READY"
FROZEN_PREDECESSOR_ENV = "DYN_GMS_FAILOVER_FROZEN_PREDECESSOR"
_gpu_quiescence_tasks: set[asyncio.Task[None]] = set()
RECLAIM_POLICY_ENV = "DYN_GMS_FAILOVER_RECLAIM_POLICY"
PROCESS_DEATH_GRACE_ENV = "DYN_GMS_FAILOVER_PROCESS_DEATH_GRACE_SECS"
_RECLAIM_POLICIES = ("gpu-proof", "process-death-timeout")
_RECLAIM_RETRY_MAX_SECS = 5.0


def reclaim_policy() -> str:
    policy = os.environ.get(RECLAIM_POLICY_ENV, "gpu-proof").strip()
    if policy not in _RECLAIM_POLICIES:
        raise ValueError(
            f"{RECLAIM_POLICY_ENV}={policy!r} must be one of {_RECLAIM_POLICIES}"
        )
    return policy


def process_death_grace_secs() -> float:
    import math

    raw = os.environ.get(PROCESS_DEATH_GRACE_ENV, "2")
    try:
        grace = float(raw)
    except ValueError:
        grace = math.nan
    if not math.isfinite(grace) or grace <= 0:
        raise ValueError(
            f"{PROCESS_DEATH_GRACE_ENV}={raw!r} must be finite and positive"
        )
    return grace


def _record_frozen_reclaim_status(
    backend_name: str,
    role: str,
    status: str,
    detail: str = "",
    *,
    quarantined_blocks: int | None = None,
    **counts: int | None,
) -> None:
    """Publish an opt-in, rank-local health signal for recovery orchestration.

    Logs from a headless TP worker do not necessarily reach its launcher. A
    per-process status file lets tests and pod health checks distinguish
    complete reclamation from a still-quarantined pool without guessing from
    a timeout or another rank's log.
    """
    directory = os.environ.get("DYN_GMS_FAILOVER_RECLAIM_STATUS_DIR")
    if not directory:
        return
    try:
        target_dir = Path(directory)
        target_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        # Primary and shadow containers share this directory and have separate
        # PID namespaces, so the engine identity keeps their records distinct.
        engine_id = os.environ.get("ENGINE_ID", "0")
        target = (
            target_dir / f"{backend_name}-{role}-engine{engine_id}-{os.getpid()}.json"
        )
        pending = target.with_name(f"{target.name}.{uuid.uuid4().hex}.pending")
        payload = {
            "engine_id": engine_id,
            "role": role,
            "status": status,
            "detail": detail,
            "pid": os.getpid(),
            "updated_at_unix_ms": int(time.time() * 1000),
        }
        if quarantined_blocks is not None:
            payload["quarantined_blocks"] = quarantined_blocks
        payload.update(
            {key: value for key, value in counts.items() if value is not None}
        )
        pending.write_text(json.dumps(payload))
        os.replace(pending, target)
    except OSError:
        logger.exception("[GMS failover] could not publish reclaim status")


def _validate_kv_recovery_mode(backend_name: str) -> None:
    """Reject removed recovery modes before they can alter KV ownership."""
    backend = backend_name.upper().replace("-", "_")
    value = (
        os.environ.get(
            f"GMS_{backend}_KV_RECOVERY_MODE",
            os.environ.get("GMS_KV_RECOVERY_MODE", "granular"),
        )
        .strip()
        .lower()
    )
    if value != "granular":
        raise RuntimeError(
            f"unsupported {backend_name} GMS KV recovery mode {value!r}; "
            "only the lease-backed global-writer path remains "
            "(legacy mode value 'granular')"
        )


def frozen_predecessor_enabled(
    backend_name: str, *, mapped_standby: bool | None = None
) -> bool:
    """Allow a fenced shadow to serve while predecessor pages stay frozen.

    The shadow must already have the shared KV pool mapped. Every ordinary
    allocation still goes through the shared lease arbiter; only exact sealed
    generations may be borrowed before GPU quiescence. Under the default
    ``gpu-proof`` policy, neither a PID exit nor this setting makes quarantined
    pages allocatable; ``process-death-timeout`` is the explicit exception.
    """
    if not _truthy_env(FROZEN_PREDECESSOR_ENV):
        return False
    engine = _normalize_lease_engine_name(backend_name)
    # Benchmark and mixed deployments may inherit the global feature flag.
    # It has no meaning for a worker that is not participating in failover.
    if not (
        _truthy_env("DYN_GMS_FAILOVER_SHADOW_MODE")
        or _truthy_env(f"DYN_{engine.upper()}_GMS_SHADOW_MODE")
    ):
        return False
    engine_id = os.environ.get("ENGINE_ID", "0")
    primary_id = os.environ.get("DYN_GMS_FAILOVER_PRIMARY_ENGINE_ID", "0")
    if mapped_standby is False and engine_id != primary_id:
        raise RuntimeError(
            f"{FROZEN_PREDECESSOR_ENV}=1 requires a mapped sleeping standby"
        )
    from gpu_memory_service.integrations.common.kv_lease_client import kv_leases_enabled

    if not kv_leases_enabled(engine):
        raise RuntimeError(f"{FROZEN_PREDECESSOR_ENV}=1 requires {engine} KV leases")
    directory_mode = os.environ.get("GMS_KV_DIRECTORY_MODE", "off").strip().lower()
    if directory_mode != "authoritative" or not _directory_socket(backend_name):
        raise RuntimeError(
            f"{FROZEN_PREDECESSOR_ENV}=1 requires an authoritative GMS KV directory"
        )
    if not os.environ.get("GMS_KV_DIRECTORY_MANIFEST", "").strip():
        raise RuntimeError(
            f"{FROZEN_PREDECESSOR_ENV}=1 requires GMS_KV_DIRECTORY_MANIFEST"
        )
    from gpu_memory_service.integrations.common.gpu_quiescence import (
        gpu_quiescence_provider_configured,
    )

    if not gpu_quiescence_provider_configured(backend_name):
        raise RuntimeError(
            f"{FROZEN_PREDECESSOR_ENV}=1 requires a GPU-quiescence provider "
            "for eventual capacity recovery"
        )
    if engine_id != primary_id and not _truthy_env("GMS_KV_DIRECTORY_STANDBY"):
        raise RuntimeError(
            f"{FROZEN_PREDECESSOR_ENV}=1 requires a read-only standby directory"
        )
    # Fail the boot on a mistyped policy rather than in the background task
    # after takeover, where it would silently strand predecessor pages.
    try:
        if reclaim_policy() == "process-death-timeout":
            process_death_grace_secs()
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    return True


def arm_frozen_shadow_headroom(backend_name: str) -> None:
    """Keep emergency FREE pages unused by either engine until takeover.

    Standby prewarm may use ordinary free pages but cannot consume this reserve.
    Register the reserve before admitting primary traffic; deployments that
    start the shadow late cannot retroactively guarantee free headroom.
    """
    if not frozen_predecessor_enabled(backend_name):
        return
    blocks = max(
        1, int(os.environ.get("DYN_GMS_FAILOVER_FROZEN_HEADROOM_BLOCKS", "16"))
    )
    from gpu_memory_service.integrations.common.kv_lease_client import (
        default_kv_lease_namespace_suffix,
        read_kv_lease_reservation,
        resolve_lease_device,
        set_kv_lease_reservation,
    )

    engine = _normalize_lease_engine_name(backend_name)
    device = resolve_lease_device(f"GMS_{engine.upper()}_KV_LEASE_DEVICE")
    suffix = default_kv_lease_namespace_suffix(engine)
    _namespace, current = read_kv_lease_reservation(
        engine, device, namespace_suffix=suffix
    )
    if current.reserved_blocks >= blocks and current.reserved_for_owner is None:
        return
    set_kv_lease_reservation(
        engine, device, reserved_blocks=blocks, namespace_suffix=suffix
    )


def _release_frozen_shadow_headroom(backend_name: str) -> None:
    from gpu_memory_service.integrations.common.kv_lease_client import (
        default_kv_lease_namespace_suffix,
        resolve_lease_device,
        set_kv_lease_reservation,
    )

    engine = _normalize_lease_engine_name(backend_name)
    device = resolve_lease_device(f"GMS_{engine.upper()}_KV_LEASE_DEVICE")
    set_kv_lease_reservation(
        engine,
        device,
        reserved_blocks=0,
        namespace_suffix=default_kv_lease_namespace_suffix(engine),
    )


def quiesce_local_gpu_cohort_after_rank_loss(
    backend_name: str, *, require_cuda_success: bool = False
) -> bool:
    """Ask GMS/MPS to retire this rank's live CUDA client before host exit.

    A peer-rank crash makes the whole TP cohort unusable, but the local CUDA
    worker may still be alive. Killing that worker first loses the only window
    in which MPS can authoritatively terminate the exact client. Lease-backed
    recovery asks GMS to terminate it while it remains registered.
    The successor still performs its own proof; this helper only preserves the
    capability needed for that proof. Deployments without the GMS-MPS provider
    keep their existing process-fencing path.
    """
    from gpu_memory_service.integrations.common.gpu_quiescence import (
        gms_mps_provider_enabled,
        terminate_current_gpu_cohort_sync,
    )

    if not gms_mps_provider_enabled(backend_name):
        return False
    try:
        proof = terminate_current_gpu_cohort_sync(backend_name=backend_name)
    except Exception:
        logger.exception(
            "[GMS failover] %s could not quiesce its local CUDA cohort after "
            "peer-rank loss; successor must fail closed",
            backend_name,
        )
        return False
    if not proof.quiesced:
        logger.error(
            "[GMS failover] %s local CUDA cohort quiescence was rejected: %s; "
            "successor must fail closed",
            backend_name,
            proof.detail,
        )
        return False
    if require_cuda_success and proof.provider != "gms-mps":
        # Inventory-only retirement is useful for shutting down a surviving
        # rank, but it is not CUDA_SUCCESS and cannot authorize releasing the
        # active-writer lock while its host GPU process is still alive.
        logger.warning(
            "[GMS failover] %s local CUDA cohort has no strict MPS proof: %s",
            backend_name,
            proof.detail,
        )
        return False
    logger.info(
        "[GMS failover] %s local CUDA cohort quiesced after peer-rank loss "
        "provider=%s elapsed_ms=%.1f",
        backend_name,
        proof.provider,
        proof.elapsed_ms,
    )
    return True


def _standby_gate_paths(lock_path: str) -> tuple[Path, Path]:
    """The primary's boot identity and the directory of standby arm markers."""
    return Path(lock_path + ".primary-boot"), Path(lock_path + ".standby-armed")


def _write_atomic(path: Path, text: str) -> None:
    pending = path.with_name(f"{path.name}.{uuid.uuid4().hex}.pending")
    pending.write_text(text)
    os.replace(pending, path)


async def wait_for_armed_standby_before_serving(
    backend_name: str, lock_path: str | None = None
) -> bool:
    """Hold the initial primary out of discovery until a warm standby is armed.

    Opt-in (``DYN_GMS_FAILOVER_SERVE_AFTER_STANDBY``). The standby needs the
    primary's weights and KV geometry, so it cannot finish first; this keeps
    the standby's compilation and graph capture off the serving window. The
    gate is bound to this boot: a marker left by an earlier primary's standby
    never releases it. After ``DYN_GMS_FAILOVER_STANDBY_GATE_SECS`` (default
    1800) the primary serves anyway rather than staying unavailable.
    """
    if not _truthy_env("DYN_GMS_FAILOVER_SERVE_AFTER_STANDBY"):
        return False
    # Only the configured initial primary waits. A successor or replacement
    # that finds the lock free must serve immediately.
    if os.environ.get("ENGINE_ID", "0") != os.environ.get(
        "DYN_GMS_FAILOVER_PRIMARY_ENGINE_ID", "0"
    ):
        return False
    lock_path = lock_path or os.environ.get(
        "FAILOVER_LOCK_PATH", DEFAULT_FAILOVER_LOCK_PATH
    )
    boot_path, armed_dir = _standby_gate_paths(lock_path)
    boot = uuid.uuid4().hex
    await asyncio.to_thread(_write_atomic, boot_path, boot)
    timeout = max(0.0, _float_env("DYN_GMS_FAILOVER_STANDBY_GATE_SECS", 1800.0))
    started = time.monotonic()
    logger.info(
        "[GMS failover] %s primary waiting up to %.0fs for an armed standby "
        "before serving",
        backend_name,
        timeout,
    )

    def armed_by() -> str | None:
        try:
            markers = list(armed_dir.iterdir())
        except FileNotFoundError:
            return None
        for marker in markers:
            if marker.name.endswith(".pending"):
                continue
            try:
                if marker.read_text().strip() == boot:
                    return marker.name
            except OSError:
                continue
        return None

    while True:
        engine = await asyncio.to_thread(armed_by)
        if engine is not None:
            logger.info(
                "[GMS failover] %s standby %s armed after %.1fs; primary serving",
                backend_name,
                engine,
                time.monotonic() - started,
            )
            return True
        if time.monotonic() - started >= timeout:
            logger.warning(
                "[GMS failover] %s no standby armed within %.0fs; primary serving "
                "without a warm standby",
                backend_name,
                timeout,
            )
            return False
        await asyncio.sleep(0.25)


async def keep_standby_armed(engine_name: str, lock_path: str | None = None) -> None:
    """While waiting for the active lock, confirm readiness to the primary.

    Echo the current primary boot identity into this engine's arm marker,
    following a primary that (re)starts while the standby waits. Run as a
    task; cancel it once the lock is acquired.
    """
    if not _truthy_env("DYN_GMS_FAILOVER_SERVE_AFTER_STANDBY"):
        return
    lock_path = lock_path or os.environ.get(
        "FAILOVER_LOCK_PATH", DEFAULT_FAILOVER_LOCK_PATH
    )
    boot_path, armed_dir = _standby_gate_paths(lock_path)
    marker = armed_dir / engine_name
    echoed = None
    while True:
        try:
            boot = await asyncio.to_thread(boot_path.read_text)
        except FileNotFoundError:
            boot = None
        if boot is not None and boot.strip() != echoed:
            await asyncio.to_thread(armed_dir.mkdir, parents=True, exist_ok=True)
            await asyncio.to_thread(_write_atomic, marker, boot.strip())
            echoed = boot.strip()
            logger.info("[GMS failover] standby %s armed", engine_name)
        await asyncio.sleep(0.25)


async def acquire_lock_while_armed(engine_name: str, acquire) -> Any:
    """Await ``acquire()`` while keeping this standby's arm marker current."""
    arm = asyncio.create_task(keep_standby_armed(engine_name))
    try:
        return await acquire()
    finally:
        arm.cancel()
        try:
            await arm
        except asyncio.CancelledError:
            pass
        except Exception:  # noqa: BLE001 - the marker is a startup hint only
            logger.warning("[GMS failover] standby arm marker failed", exc_info=True)


def recycle_idle_mps_servers_before_cuda(backend_name: str) -> int:
    """Give a (re)started engine a fresh MPS server; call before any CUDA init."""
    if not _truthy_env("DYN_GMS_FAILOVER_SHADOW_MODE"):
        return 0
    from gpu_memory_service.integrations.common.gpu_quiescence import (
        recycle_idle_mps_servers,
    )

    return recycle_idle_mps_servers(backend_name)


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


def _promotion_warmup_output_tokens(backend_name: str) -> int:
    backend_env = _backend_env_name(
        backend_name, "GMS_FAILOVER_PROMOTION_WARMUP_OUTPUT_TOKENS"
    )
    raw = os.environ.get(backend_env)
    if raw is None:
        raw = os.environ.get("DYN_GMS_FAILOVER_PROMOTION_WARMUP_OUTPUT_TOKENS", "1")
    try:
        count = int(raw)
    except ValueError as exc:
        raise ValueError(f"{backend_env} must be an integer") from exc
    if not 1 <= count <= 128:
        raise ValueError(f"{backend_env} must be in 1..128")
    return count


def _promotion_warmup_payloads(
    payload: dict[str, Any], backend_name: str
) -> tuple[dict[str, Any], ...]:
    """Expand a token-input probe into isolated production-shape requests.

    The caller gives every vLLM stream a distinct cache salt. Without it,
    progressively longer warmups share their prefix and fail to exercise the
    intended prefill shape, leaving Triton MoE kernels for the first request.
    """

    counts = _promotion_warmup_token_counts(backend_name)
    output_tokens = _promotion_warmup_output_tokens(backend_name)
    if not counts:
        variant = dict(payload)
        if output_tokens != 1:
            variant["max_tokens"] = output_tokens
            variant["ignore_eos"] = True
        return (variant,)
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
        if output_tokens != 1:
            variant["max_tokens"] = output_tokens
            variant["ignore_eos"] = True
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
    request_count = _int_env(
        "DYN_GMS_FAILOVER_PROMOTION_WARMUP_REQUESTS",
        concurrency * len(warmup_payloads),
    )
    if not 1 <= request_count <= 1_000:
        raise ValueError(
            "DYN_GMS_FAILOVER_PROMOTION_WARMUP_REQUESTS must be in 1..1000"
        )

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
        remaining = request_count
        while remaining:
            run_payload = warmup_payloads[
                ((request_count - remaining) // concurrency) % len(warmup_payloads)
            ]
            batch_size = min(concurrency, remaining)
            await asyncio.gather(*(_run_stream(run_payload) for _ in range(batch_size)))
            remaining -= batch_size
        logger.info(
            "[GMS failover] %s promotion warmup completed attempt=%d "
            "requests=%d concurrency=%d token_counts=%s elapsed_ms=%.2f",
            backend_name,
            attempt,
            request_count,
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
    os.environ.pop("DYN_GMS_FAILOVER_ACTIVE_LOCK_HELD", None)
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
    os.environ.pop("DYN_GMS_FAILOVER_ACTIVE_LOCK_HELD", None)
    logger.info(
        "[GMS failover] %s released active lock for controlled handoff", backend_name
    )
    return True


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
    process_death_timeout_elapsed: bool = False,
    inherit_quarantine: bool = False,
) -> KVLeaseRecoveryResult | None:
    """Classify pages; reclaim under an explicitly selected authorization.

    CPU writer fencing makes state immutable but does not prove queued CUDA work
    has drained. IDLE pages provide immediate safe headroom; ambiguous pages are
    quarantined. GPU proof is the default; an explicit process-death grace
    policy permits best-effort reclamation without asserting GPU quiescence.

    ``protected_leases`` are directory-advertised READY HBM slot generations.
    Only records whose rank-local generation still matches are preserved for
    lazy adoption. ``protected_blocks`` remains for diagnostic compatibility.
    """

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
        if not _truthy_env("DYN_GMS_FAILOVER_RECLAIM_FOREIGN_LEASES", default=True):
            raise RuntimeError(
                "GMS KV failover cannot disable foreign lease classification"
            )
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
            process_death_timeout_elapsed=process_death_timeout_elapsed,
            inherit_quarantine=inherit_quarantine,
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
                "quarantined=%d reclaimed=%d gpu_quiesced=%s best_effort=%s elapsed_ms=%.2f",
                backend_name,
                role,
                result.files,
                result.released_idle_blocks,
                result.quarantined_blocks,
                result.reclaimed_blocks,
                gpu_quiesced,
                process_death_timeout_elapsed,
                elapsed_ms,
            )
        return result
    except Exception:
        logger.exception(
            "[GMS failover] %s %s KV lease recovery failed closed",
            backend_name,
            role,
        )
        raise


def _inherit_predecessor_quarantine(backend_name: str) -> bool:
    """Adopt quarantine left by an earlier successor that died before phase two.

    Only the process-death policy can authorize it: a strict GPU proof covers
    the immediate predecessor, not the older cohort whose work was quarantined.
    """
    return (
        frozen_predecessor_enabled(backend_name)
        and reclaim_policy() == "process-death-timeout"
    )


async def _wait_process_death_reclaim_grace(
    predecessor_cohort: str,
    on_waiting: Callable[[str], None] | None = None,
) -> None:
    """Best-effort policy, not proof that an MPS server drained CUDA work.

    Process death is monotonic, so a refused probe (a live guard holder, probe
    contention between TP ranks, or an I/O error) is retried with capped
    backoff until the predecessor is confirmed dead twice, a full grace apart,
    or the task is cancelled. A missing or still-open cohort never counts.
    """
    from gpu_memory_service.integrations.common.process_lifecycle import (
        retired_writer_cohort_has_no_processes,
    )

    grace = process_death_grace_secs()
    path = Path(predecessor_cohort)

    async def confirmed_dead(window: float) -> bool:
        # All TP ranks probe the same retired inode. Their brief exclusive
        # probes contend with each other, so retry briefly before refusing.
        deadline = time.monotonic() + window
        while True:
            try:
                if await asyncio.to_thread(
                    retired_writer_cohort_has_no_processes, path
                ):
                    return True
            except OSError:
                pass
            if time.monotonic() >= deadline:
                return False
            await asyncio.sleep(0.01)

    backoff = 0.1
    while True:
        window = min(grace, 1.0)
        if await confirmed_dead(window):
            await asyncio.sleep(grace)
            if await confirmed_dead(window):
                return
        if on_waiting is not None:
            on_waiting("predecessor writer cohort not confirmed dead; retrying")
        await asyncio.sleep(backoff)
        backoff = min(backoff * 2, _RECLAIM_RETRY_MAX_SECS)


def _accepted_gpu_proof(proof: Any) -> bool:
    # MPS inventory disappearance only shows that the host client can no
    # longer submit work. It is not proof that already-submitted GPU work
    # has retired, so it must never release frozen physical KV pages.
    return bool(
        proof is not None
        and proof.quiesced
        and proof.provider not in {"gms-mps-inventory", "quarantine-only"}
    )


async def _reclaim_frozen_predecessor(
    *,
    backend_name: str,
    role: str,
    predecessor_cohort: str | None,
    recovery_owner_id: str,
    lease_device: int,
    quarantined_blocks: int | None = None,
) -> None:
    """Reclaim this generation under strict proof or opt-in timed policy.

    The status stays ``pending`` until a terminal outcome: ``reclaimed``,
    ``reclaimed-best-effort``, or ``quarantined`` (strict policy without proof,
    or an unknown predecessor). Transient refusals are retried, never final.
    """
    from gpu_memory_service.integrations.common.gpu_quiescence import (
        prove_predecessor_gpu_quiescence,
    )

    policy = reclaim_policy()

    reclaimed_blocks: int | None = None

    def status(state: str, detail: str = "", blocks: int | None = None) -> None:
        # Headless TP workers may suppress INFO logs; keep the phase-one and
        # phase-two counts in the status record so every rank is auditable.
        _record_frozen_reclaim_status(
            backend_name,
            role,
            state,
            detail,
            quarantined_blocks=quarantined_blocks if blocks is None else blocks,
            phase_one_quarantined_blocks=quarantined_blocks,
            reclaimed_blocks=reclaimed_blocks,
        )

    # This is a capacity-reclamation decision, not a serving gate. A stalled
    # MPS/RPC attempt must not leave the outcome undecided for tens of seconds.
    timeout = max(
        0.0,
        float(os.environ.get("DYN_GMS_FAILOVER_BACKGROUND_GPU_PROOF_SECS", "2")),
    )
    status("pending")
    deadline = time.monotonic() + timeout
    proof = None
    refusal = "GPU proof unavailable"
    while True:
        try:
            proof = await asyncio.wait_for(
                prove_predecessor_gpu_quiescence(
                    backend_name=backend_name,
                    predecessor_cohort=predecessor_cohort,
                    device=lease_device,
                ),
                timeout=max(0.0, deadline - time.monotonic()),
            )
        except asyncio.TimeoutError:
            proof = None
            refusal = f"GPU proof exceeded {timeout:.1f}s deadline"
            break
        except Exception:
            logger.warning(
                "[GMS failover] %s %s background GPU proof failed; retrying",
                backend_name,
                role,
                exc_info=True,
            )
            proof = None
        if _accepted_gpu_proof(proof):
            break
        if proof is not None:
            refusal = proof.detail
            # CUDA 201 means the registered context is already gone. It is not
            # quiescence proof, and repeating terminate_client cannot turn it
            # into CUDA_SUCCESS.
            if "cuda_result=201" in proof.detail:
                break
        if time.monotonic() >= deadline:
            break
        await asyncio.sleep(min(0.25, max(0.0, deadline - time.monotonic())))
    gpu_quiesced = _accepted_gpu_proof(proof)
    if not gpu_quiesced:
        if policy != "process-death-timeout" or predecessor_cohort is None:
            status("quarantined", refusal)
            logger.warning(
                "[GMS failover] %s %s predecessor pages remain frozen: policy=%s "
                "predecessor=%s: %s",
                backend_name,
                role,
                policy,
                predecessor_cohort,
                refusal,
            )
            if backend_name == "sglang":
                from gpu_memory_service.integrations.sglang.writer_lifecycle import (
                    mark_gms_reclaim_refused,
                )

                await asyncio.to_thread(mark_gms_reclaim_refused)
            return
        logger.info(
            "[GMS failover] %s %s GPU proof not obtained (%s); waiting for "
            "predecessor process death plus grace",
            backend_name,
            role,
            refusal,
        )
        reported = False

        def waiting(detail: str) -> None:
            nonlocal reported
            status("pending", detail)
            if not reported:
                reported = True
                logger.warning("[GMS failover] %s %s %s", backend_name, role, detail)

        await _wait_process_death_reclaim_grace(predecessor_cohort, waiting)
        logger.warning(
            "[GMS failover] %s %s reclaim authorized by process death plus grace; "
            "GPU quiescence is unproven, including any surviving MPS server",
            backend_name,
            role,
        )
    # A live successor may hold a short lease mutation guard when proof
    # completes. Phase two must wait for it, never steal the activity counter
    # as phase one can after the predecessor's CPU writer fence.
    backoff = 0.1
    reported = False
    while True:
        try:
            result = await asyncio.to_thread(
                _recover_foreign_kv_leases_after_fence,
                backend_name,
                role,
                gpu_quiesced=gpu_quiesced,
                recovery_owner_id=recovery_owner_id,
                **({"process_death_timeout_elapsed": True} if not gpu_quiesced else {}),
            )
            reclaimed_blocks = getattr(result, "reclaimed_blocks", None)
            break
        except Exception:
            if not reported:
                reported = True
                status("pending", "authorized lease reclamation retrying")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 1.0)
    if backend_name == "sglang":
        from gpu_memory_service.integrations.sglang.writer_lifecycle import (
            mark_gms_reclaim_ready,
            mark_gpu_quiescence_ready,
        )

        await asyncio.to_thread(mark_gms_reclaim_ready)
        if gpu_quiesced:
            await asyncio.to_thread(mark_gpu_quiescence_ready)
    logger.info(
        "[GMS failover] %s %s frozen capacity reclaimed "
        "gpu_quiesced=%s provider=%s elapsed_ms=%.2f",
        backend_name,
        role,
        gpu_quiesced,
        proof.provider if gpu_quiesced else "process-death-timeout",
        proof.elapsed_ms if proof is not None else 0.0,
    )
    status(
        "reclaimed" if gpu_quiesced else "reclaimed-best-effort",
        (
            proof.detail
            if gpu_quiesced
            else "retired process cohort plus grace; CUDA unproven"
        ),
        blocks=0,
    )


def _frozen_reclaim_finished(task: asyncio.Task[None]) -> None:
    _gpu_quiescence_tasks.discard(task)
    if task.cancelled():
        return
    error = task.exception()
    if error is not None:
        logger.error(
            "[GMS failover] frozen predecessor reclamation failed; "
            "quarantine retained",
            exc_info=(type(error), error, error.__traceback__),
        )


def classify_frozen_vllm_worker_rank(predecessor_cohort: str | None = None) -> None:
    """Classify a remote TP rank's local ring before the all-rank resume.

    Only rank zero has a Dynamo leader process to run the post-lock fence.
    Other ranks have independent GMS sidecars and /dev/shm lease rings, so
    their workers must promote and classify locally after the shared writer
    cohort has retired. The caller's collective RPC is the admission barrier.
    """
    import threading

    backend_name = "vllm"
    role = "shadow-worker-rank"
    if not frozen_predecessor_enabled(backend_name, mapped_standby=True):
        return
    logger.info(
        "[GMS failover] %s %s starting frozen classification predecessor=%s",
        backend_name,
        role,
        predecessor_cohort,
    )
    protected_blocks, protected_leases = _promote_content_directory_after_fence(
        backend_name, role
    )
    from gpu_memory_service.integrations.common.kv_lease_client import (
        current_kv_lease_owner_id,
        resolve_lease_device,
    )

    lease_device = resolve_lease_device("GMS_VLLM_KV_LEASE_DEVICE")
    recovery_owner_id = current_kv_lease_owner_id(backend_name, lease_device)
    phase_one = _recover_foreign_kv_leases_after_fence(
        backend_name,
        role,
        gpu_quiesced=False,
        recovery_owner_id=recovery_owner_id,
        protected_blocks=protected_blocks,
        protected_leases=protected_leases,
        inherit_quarantine=_inherit_predecessor_quarantine(backend_name),
    )
    _release_frozen_shadow_headroom(backend_name)

    def reclaim() -> None:
        try:
            asyncio.run(
                _reclaim_frozen_predecessor(
                    backend_name=backend_name,
                    role=role,
                    predecessor_cohort=predecessor_cohort,
                    recovery_owner_id=recovery_owner_id,
                    lease_device=lease_device,
                    quarantined_blocks=getattr(phase_one, "quarantined_blocks", None),
                )
            )
        except Exception:
            logger.exception(
                "[GMS failover] vLLM remote rank reclamation failed; "
                "quarantine retained"
            )

    threading.Thread(
        target=reclaim, name="gms-frozen-rank-reclaim", daemon=True
    ).start()


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
    _validate_kv_recovery_mode(backend_name)
    frozen = frozen_predecessor_enabled(backend_name)
    fenced_env = f"DYN_{backend_name.upper().replace('-', '_')}_GMS_POOL_FENCED"
    os.environ.pop(fenced_env, None)
    os.environ.pop("DYN_GMS_FAILOVER_ACTIVE_LOCK_HELD", None)

    predecessor = None
    writer_fence_started = time.monotonic()
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

    if backend_name in {"vllm", "sglang"}:
        logger.info(
            "[GMS failover] %s %s writer-cohort fence completed "
            "predecessor=%s elapsed_ms=%.2f",
            backend_name,
            role,
            predecessor,
            (time.monotonic() - writer_fence_started) * 1000.0,
        )

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

    from gpu_memory_service.integrations.common.gpu_quiescence import (
        gpu_quiescence_provider_configured,
        prove_predecessor_gpu_quiescence,
    )
    from gpu_memory_service.integrations.common.kv_lease_client import (
        current_kv_lease_owner_id,
        resolve_lease_device,
    )

    lease_engine = _normalize_lease_engine_name(backend_name)
    lease_device = resolve_lease_device(f"GMS_{lease_engine.upper()}_KV_LEASE_DEVICE")
    recovery_owner_id = current_kv_lease_owner_id(lease_engine, lease_device)
    # The quiesced recovery call only reclaims pages quarantined by this first
    # pass. It does not inspect or classify the predecessor's live records.
    phase_one = _recover_foreign_kv_leases_after_fence(
        backend_name,
        role,
        gpu_quiesced=False,
        recovery_owner_id=recovery_owner_id,
        protected_blocks=protected_blocks,
        protected_leases=protected_leases,
        inherit_quarantine=_inherit_predecessor_quarantine(backend_name),
    )
    if frozen and (
        predecessor is not None
        or os.environ.get("ENGINE_ID", "0")
        != os.environ.get("DYN_GMS_FAILOVER_PRIMARY_ENGINE_ID", "0")
    ):
        # The active writer lock and CPU fence now make the emergency reserve
        # available to this successor. Future standby registration rearms it.
        _release_frozen_shadow_headroom(backend_name)
    if frozen and predecessor is not None:
        # CPU submitters and directory writes are fenced. Exact committed
        # generations stay readable; ambiguous predecessor pages remain
        # unallocatable until the background proof succeeds. Complete every
        # potentially failing admission step before launching that task.
        if backend_name == "sglang":
            from gpu_memory_service.integrations.sglang.writer_lifecycle import (
                mark_gms_recovery_ready,
            )

            await asyncio.to_thread(mark_gms_recovery_ready)
        os.environ[fenced_env] = "1"
        os.environ["DYN_GMS_FAILOVER_ACTIVE_LOCK_HELD"] = "1"
        # Keep a strong reference so proof survives the promotion request.
        task = asyncio.create_task(
            _reclaim_frozen_predecessor(
                backend_name=backend_name,
                role=role,
                predecessor_cohort=str(predecessor),
                recovery_owner_id=recovery_owner_id,
                lease_device=lease_device,
                quarantined_blocks=getattr(phase_one, "quarantined_blocks", None),
            )
        )
        _gpu_quiescence_tasks.add(task)
        task.add_done_callback(_frozen_reclaim_finished)
        logger.info(
            "[GMS failover] %s %s serving from exact sealed and free KV while "
            "predecessor reclaim runs asynchronously",
            backend_name,
            role,
        )
        return
    gpu_quiesced = False
    if predecessor is not None:
        if not gpu_quiescence_provider_configured(backend_name):
            raise RuntimeError(
                "GMS KV takeover cannot reuse predecessor HBM without a "
                "capability-grade GPU-quiescence provider"
            )
        proof = await prove_predecessor_gpu_quiescence(
            backend_name=backend_name,
            predecessor_cohort=str(predecessor),
            device=lease_device,
        )
        if not proof.quiesced:
            raise RuntimeError(
                "GMS KV takeover could not prove predecessor GPU quiescence "
                f"via {proof.provider}: {proof.detail}"
            )
        gpu_quiesced = True
        logger.info(
            "[GMS failover] %s %s GPU quiescence proven provider=%s " "elapsed_ms=%.2f",
            backend_name,
            role,
            proof.provider,
            proof.elapsed_ms,
        )
    if gpu_quiesced:
        _recover_foreign_kv_leases_after_fence(
            backend_name,
            role,
            gpu_quiesced=True,
            recovery_owner_id=recovery_owner_id,
        )
    if backend_name == "sglang":
        from gpu_memory_service.integrations.sglang.writer_lifecycle import (
            mark_gms_recovery_ready,
        )

        await asyncio.to_thread(mark_gms_recovery_ready)
        if gpu_quiesced:
            from gpu_memory_service.integrations.sglang.writer_lifecycle import (
                mark_gpu_quiescence_ready,
            )

            await asyncio.to_thread(mark_gpu_quiescence_ready)
    os.environ[fenced_env] = "1"
    # SGLang's cache adapter derives its writer/standby role from this marker.
    # Publish it only after the full post-lock fence has succeeded, so warmup
    # and directory writes cannot treat an unproven lock holder as active.
    os.environ["DYN_GMS_FAILOVER_ACTIVE_LOCK_HELD"] = "1"


def promote_rank_local_content_directory(*, backend_name: str, role: str) -> None:
    """Promote a TP rank's directory after its rank-local writer lock is held.

    The request-plane rank performs the cohort-wide writer and GPU fence.  A
    headless TP rank has a distinct GMS daemon and content directory, however,
    so it must publish ownership to that directory before its CUDA worker can
    bind persistent allocations.  Re-running the cohort-wide fence here would
    incorrectly create a second writer generation; the rank-local lock is the
    safety boundary for this directory-only promotion.
    """
    if os.environ.get("GMS_KV_DIRECTORY_MODE", "off").strip().lower() == "off":
        return
    _promote_content_directory_after_fence(backend_name, role)


def rank_local_content_directory_is_fresh(backend_name: str) -> bool:
    """Only a never-promoted, unbound directory can skip predecessor GPU proof."""
    from gms_kv_ring.common.content_directory import ContentDirectory

    socket_path = _directory_socket(backend_name)
    if not socket_path or not os.environ.get("GMS_KV_DIRECTORY_MANIFEST", "").strip():
        return False
    directory = ContentDirectory(
        socket_path,
        engine=_normalize_lease_engine_name(backend_name),
        block_size=0,
        engine_id=os.environ.get("ENGINE_ID", "0"),
    )
    try:
        epoch, writer = directory.status()
        binding = directory.pool_binding(os.environ.get("ENGINE_ID", "0"))
        return epoch == 1 and writer is None and binding is None
    finally:
        directory.close()


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
    os.environ.pop("DYN_GMS_FAILOVER_ACTIVE_LOCK_HELD", None)
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
            await wait_for_armed_standby_before_serving(backend_name, lock_path)
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

    # The request-plane controller has no CUDA client to register with GMS.
    # Establish its proof channel before the failure so the 16-rank takeover
    # does not wait for a new GMS handshake. This is only a latency hint:
    # failure to preconnect never substitutes for the post-lock MPS proof.
    from gpu_memory_service.integrations.common.gpu_quiescence import (
        preconnect_gpu_quiescence_session,
    )

    try:
        await asyncio.to_thread(
            preconnect_gpu_quiescence_session, backend_name=backend_name
        )
    except OSError as exc:
        logger.warning(
            "[GMS failover] %s shadow proof preconnect unavailable: %s",
            backend_name,
            exc,
        )

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
    await acquire_lock_while_armed(
        engine_name, lambda: lock.acquire(engine_id=engine_name)
    )
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
        # Classification and GPU-quiescence proof both finish before admission;
        # a failed proof leaves foreign pages quarantined and prevents resume.
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
