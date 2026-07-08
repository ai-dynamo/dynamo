#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Power Agent DaemonSet — Phase 1 implementation.

Runs as a privileged DaemonSet (hostPID: true) on each GPU node. Every 15s:
  1. Lists pods on this node via the K8s API.
  2. For each physical GPU: nvmlDeviceGetComputeRunningProcesses() → PID list
     (PID discovery always uses NVML, under both actuator modes).
  3. For each PID: reads /proc/{pid}/cgroup → extracts pod UID.
  4. Looks up the pod's dynamo.nvidia.com/gpu-power-limit annotation.
  5. Writes the cap through the selected actuator — NVML
     (nvmlDeviceSetPowerManagementLimit) or DCGM (dcgmConfigSet).

Scope is opt-in: the agent only ever caps a GPU whose pod carries the
dynamo.nvidia.com/gpu-power-limit annotation (set by the planner on
prefill/decode worker pods). A GPU running only unannotated pods — a
non-Dynamo workload, or a Dynamo worker not yet annotated — that the agent
never capped is left at its hardware default and untouched. If the agent had
previously capped that GPU and the opted-in pod is now gone (a non-managed
workload reuses it, or the planner removed the annotation), the cap is
released back to default so it does not strand on the new tenant. See
``_build_uid_to_annotation`` and ``_release_managed_gpu``.

Graceful shutdown: the SIGTERM/SIGINT handler only sets a shutdown flag; the
reconcile loop (``run()``) then restores default TGP on all managed GPUs via
``_shutdown_cleanup`` before exit — heavy NVML/DCGM work never runs inside the
signal handler. A cap is also restored mid-run by ``_release_managed_gpu`` when
a previously-capped GPU is handed to a non-managed tenant.
Cold-start orphan recovery: UUID-gated (persisted to /var/lib/dynamo-power-agent/).
"""

import argparse
import json
import logging
import os
import re
import signal
import threading
from typing import Callable, Optional

import managed_state
from actuator import Actuator, DcgmActuator, NvmlActuator, _GpuIdentityMismatch

# Kubernetes and NVML — imported lazily with clear error messages
try:
    import pynvml
except ImportError:
    pynvml = None  # type: ignore

try:
    from kubernetes import client as k8s_client
    from kubernetes import config as k8s_config
    from kubernetes.config.config_exception import ConfigException
except ImportError:
    k8s_client = None  # type: ignore
    k8s_config = None  # type: ignore
    ConfigException = Exception  # type: ignore

try:
    from prometheus_client import Counter, Gauge, start_http_server

    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("power_agent")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

POWER_ANNOTATION_KEY = "dynamo.nvidia.com/gpu-power-limit"
RECONCILE_INTERVAL_S = 15

# Pod-LIST timeouts. These exist so a SIGTERM that lands while the reconcile
# loop is blocked in the apiserver LIST cannot eat the whole pod termination
# grace period: without a bound, `run()`'s `finally` (which restores caps)
# cannot execute until the LIST returns, so a throttled/stuck apiserver would
# let kubelet SIGKILL the agent with managed caps still live.
#
# Two bounds, defense in depth:
#   * K8S_LIST_SERVER_TIMEOUT_S — the apiserver-side `timeout_seconds` LIST
#     parameter, so the server itself abandons the watch-cache read.
#   * K8S_LIST_CLIENT_TIMEOUT_S — the client-side `_request_timeout`, covering
#     connection/read stalls the server timeout cannot (for example, a dead
#     connection or a server that never answers). It is best-effort rather than
#     a strict wall-clock deadline; kept slightly above the server bound so the
#     server's own timeout is what normally fires.
#
# NOTE: `_request_timeout` is NOT a hard wall-clock deadline: it does not bound
# every source of elapsed time, including DNS and incrementally arriving response
# data. That is exactly why transport retries are disabled on this agent's API
# client (see `_build_k8s_core_v1`) — the 15s reconcile loop is the only retry
# policy, so the LIST delay is substantially limited even though it is not an
# exact wall-clock bound. This is an always-on steady-state timeout, not a
# shutdown-only one; we accept that (superseding the earlier "no client timeout"
# stance) because shutdown-time cap restoration outweighs the steady-state
# retry-amplification risk, and disabling transport retries removes the
# amplification sttts originally warned about. A 20s/25s budget is loose enough
# for normal API Priority and Fairness backpressure while still leaving
# meaningful cleanup headroom inside the default 60s pod grace period.
K8S_LIST_SERVER_TIMEOUT_S = 20
K8S_LIST_CLIENT_TIMEOUT_S = 25
# Sourced from `managed_state` so every launch path (and the actuator's
# separate `import power_agent` module copy) agrees on one location.
_MANAGED_STATE_PATH = managed_state.MANAGED_STATE_PATH

# ---------------------------------------------------------------------------
# cgroup pod-UID extraction
# Handles cgroup v1 (multi-line) and v2 (single-line), systemd / cgroupfs
# drivers, Guaranteed / Burstable / BestEffort QoS, cri-containerd / cri-o.
# ---------------------------------------------------------------------------

_SYSTEMD_RE = re.compile(
    r"kubepods-(?:burstable-|besteffort-)?pod([a-fA-F0-9_]+)\.slice"
)
_CGROUPFS_RE = re.compile(
    r"/kubepods(?:/burstable|/besteffort)?/pod([a-fA-F0-9-]+)(?:/|$)"
)


def _extract_pod_uid_from_cgroup(pid: int) -> Optional[str]:
    """Recover the pod UID from /proc/{pid}/cgroup.

    Iterates lines because cgroup v1 has one line per controller hierarchy
    while cgroup v2 has a single unified line. Uses .search() so wrapper
    segments (cri-containerd, cri-o, dockershim) don't defeat the match.
    Returns None for non-K8s processes — callers skip them silently.
    """
    try:
        with open(f"/proc/{pid}/cgroup") as f:
            lines = f.read().splitlines()
    except OSError:
        return None
    for line in lines:
        m = _SYSTEMD_RE.search(line)
        if m:
            # systemd encodes dashes as underscores in the pod-UID segment
            return m.group(1).replace("_", "-")
        m = _CGROUPFS_RE.search(line)
        if m:
            return m.group(1)
    return None  # non-K8s process — skip


# ---------------------------------------------------------------------------
# Persistent managed-GPU state (UUID-gated orphan recovery)
# ---------------------------------------------------------------------------

# Alias to the single source of truth in `managed_state`. The daemon is
# launched as `python power_agent.py` (so this file is `__main__`) while
# `actuator.py` reaches the same state via `import power_agent` — two distinct
# module objects. Hosting the set in `managed_state` (which both import by
# canonical name) guarantees one copy. NEVER rebind this name; always mutate
# in place (`.add`/`.discard`/`.clear`/`.update`), or the alias splits and the
# dual-copy bug returns. See managed_state.py for the full rationale.
_previously_managed: set[str] = managed_state.previously_managed


def _read_managed_gpus_state() -> tuple[set[str], bool]:
    """Load the persisted managed-UUID set AND whether the read was conclusive.

    Returns ``(uuids, conclusive)``:

      * ``conclusive=True``  — the file was read and parsed cleanly, INCLUDING a
        legitimately empty / absent-file first-boot state. ``uuids`` is
        authoritative and the caller may safely rewrite the file.
      * ``conclusive=False`` — the read or parse FAILED (I/O / permission error,
        corrupt JSON, or a structurally-invalid root / ``managed_uuids`` field).
        The true on-disk state is UNKNOWN, so ``uuids`` is empty and the caller
        MUST NOT rewrite the file (rewriting empty would ERASE state a transient
        failure merely hid) and MUST skip orphan recovery this boot.

    Defensive parsing — corrupt / malformed state files must never crash the
    agent's startup. Per PR #9682 CodeRabbit review this catches a superset of
    the original (FileNotFoundError, JSONDecodeError) cases:

      * OSError (PermissionError, IsADirectoryError, NotADirectoryError, I/O
        errors) — disk problems on the host volume should NOT brick the agent;
        inconclusive so the caller skips recovery + leaves the file untouched.
      * Non-dict JSON root — a top-level list / int / string / null would have
        crashed `.get(...)`; inconclusive (do not clobber a hand-editable file).
      * Non-list `managed_uuids` — a misshapen value would have crashed
        `set(...)`; inconclusive.
      * Non-string entries — bytes / ints / None inside an otherwise valid list
        are dropped at the boundary; this is still a CONCLUSIVE read of a
        mostly-valid file (the kept subset is authoritative).
    """
    try:
        with open(_MANAGED_STATE_PATH) as f:
            raw = json.load(f)
    except FileNotFoundError:
        return set(), True  # first boot — conclusively empty
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(
            "Failed to read managed-GPU state at %s (%s: %s); read is "
            "INCONCLUSIVE — skipping orphan recovery and leaving the file "
            "untouched this startup (retry next boot).",
            _MANAGED_STATE_PATH,
            type(e).__name__,
            e,
        )
        return set(), False

    if not isinstance(raw, dict):
        logger.warning(
            "Managed-GPU state at %s has unexpected root type %s "
            "(expected object); read is INCONCLUSIVE — leaving the file "
            "untouched this startup.",
            _MANAGED_STATE_PATH,
            type(raw).__name__,
        )
        return set(), False

    uuids = raw.get("managed_uuids", [])
    if not isinstance(uuids, list):
        logger.warning(
            "Managed-GPU state at %s has unexpected managed_uuids type "
            "%s (expected list); read is INCONCLUSIVE — leaving the file "
            "untouched this startup.",
            _MANAGED_STATE_PATH,
            type(uuids).__name__,
        )
        return set(), False

    # Count invalid entries directly rather than from len(set) vs
    # len(list): the set comprehension deduplicates,
    # so duplicate-but-valid UUIDs would inflate the false-positive
    # "non-string entries" count. E.g. uuids=["a","a","b"] would
    # wrongly log "1 non-string entry" when there are zero.
    invalid_count = sum(1 for u in uuids if not isinstance(u, str))
    valid = {u for u in uuids if isinstance(u, str)}
    if invalid_count:
        logger.warning(
            "Managed-GPU state at %s contained %d non-string entries; "
            "dropping them. Kept %d valid UUID(s).",
            _MANAGED_STATE_PATH,
            invalid_count,
            len(valid),
        )
    return valid, True


def _load_previously_managed_gpus() -> set[str]:
    """Back-compat wrapper returning ONLY the UUID set (drops the conclusive
    flag). Kept for callers/tests that just need the parsed set; startup orphan
    recovery uses `_read_managed_gpus_state` directly so it can honour read
    conclusiveness (skip recovery + avoid an empty-state rewrite on a failed
    read)."""
    uuids, _ = _read_managed_gpus_state()
    return uuids


def _persist_managed_gpus(uuids: set[str]) -> None:
    os.makedirs(os.path.dirname(_MANAGED_STATE_PATH), exist_ok=True)
    tmp = _MANAGED_STATE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"managed_uuids": sorted(uuids)}, f)
    os.replace(tmp, _MANAGED_STATE_PATH)  # atomic rename


def _nvml_uuid(handle) -> str:
    """Return the GPU UUID as ``str`` regardless of pynvml major version.

    The legacy ``pynvml`` package (NVIDIA bindings) returns ``bytes`` and
    callers ``.decode("ascii")`` themselves.  ``nvidia-ml-py`` (the
    officially supported successor and what newer pip releases install
    under the name ``pynvml``) returns ``str`` directly, and an
    unconditional ``.decode()`` raises ``AttributeError``.  Callers must
    go through this helper.
    """
    uuid = pynvml.nvmlDeviceGetUUID(handle)
    return uuid.decode("ascii") if isinstance(uuid, bytes) else uuid


# UUIDs whose cap ACQUISITION completed (cap live in hardware + in-memory
# ownership recorded) but whose durable ADD to `managed_gpus.json` failed.
# Acquisition-side peer of `_pending_retirement`: without it,
# `_record_managed_gpu_by_uuid` adds the UUID to `_previously_managed` before
# persisting, so a persist failure is masked by the membership guard and never
# retried — a later SIGKILL / node crash then loses the ONLY orphan-recovery
# record and strands the live cap. Retried (persistence ONLY, never a hardware
# re-apply) by `_flush_pending_acquisitions` at the top of every reconcile.
#
# This one must also live in `managed_state`: cap writes run through the
# actuator's canonical `import power_agent` copy, while reconcile flushes the
# entrypoint's `__main__` copy. A module-local queue would split and never flush.
_pending_acquisition: set[str] = managed_state.pending_acquisition


def _record_managed_gpu_by_uuid(uuid: str) -> None:
    """Library-agnostic UUID persistence helper.

    Called by both actuator paths after a successful cap write. The UUID
    is the hardware-level identifier, identical whether obtained from
    NVML (`nvmlDeviceGetUUID`) or DCGM (`DCGM_FI_DEV_UUID`). Separating
    the persistence from the UUID source means DcgmActuator can
    record state without reaching into the NVML helpers.

    Durability: the UUID is added to the in-memory `_previously_managed` mirror
    first (the cap is already live, so the GPU must look managed to this
    process's shutdown / orphan paths). If the durable write then fails we do
    NOT re-raise — the cap write already succeeded and reconcile must continue —
    but we record the UUID in `_pending_acquisition` so
    `_flush_pending_acquisitions` retries the persist next cycle. Otherwise the
    membership guard below would suppress every future persist attempt for this
    UUID, and an ungraceful exit would strand the cap with no recovery record.
    """
    if uuid in _previously_managed:
        return
    _previously_managed.add(uuid)
    try:
        _persist_managed_gpus(_previously_managed)
    except Exception as e:
        _pending_acquisition.add(uuid)
        logger.warning(
            "Recorded managed GPU UUID %s in memory but persisting the durable "
            "set failed; the cap is live and the durable record will be retried "
            "next reconcile (orphan recovery after an ungraceful exit depends on "
            "it): %s",
            uuid,
            e,
        )


def _record_managed_gpu_uuid(handle) -> None:
    """Called from _apply_cap() after every successful NVML write."""
    _record_managed_gpu_by_uuid(_nvml_uuid(handle))


def _flush_pending_acquisitions() -> None:
    """Retry the durable ADD for cap acquisitions whose hardware write +
    in-memory ownership completed but whose `_persist_managed_gpus` write
    failed.

    Retries ONLY the persistence (writes the authoritative in-memory
    `_previously_managed`), never a hardware re-apply, so it can never disturb a
    live cap. Called at the top of every reconcile cycle so it flushes
    independently of Kubernetes API health, mirroring
    `_flush_pending_retirements`."""
    if not _pending_acquisition:
        return
    try:
        _persist_managed_gpus(_previously_managed)
    except Exception as e:
        logger.warning(
            "Deferred acquisition persistence retry failed (%d pending); will "
            "retry next cycle: %s",
            len(_pending_acquisition),
            e,
        )
        return
    logger.info(
        "Flushed %d deferred cap acquisition(s) to durable state.",
        len(_pending_acquisition),
    )
    _pending_acquisition.clear()


# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------


class _NoopMetric:
    def labels(self, **_):
        return self

    def set(self, _):
        pass

    def inc(self, _=1):
        pass


class PowerAgentMetrics:
    def __init__(self, prometheus_port: int = 0) -> None:
        if _PROMETHEUS_AVAILABLE and prometheus_port > 0:
            self.applied_limit_watts = Gauge(
                "dynamo_power_agent_applied_limit_watts",
                "Last applied/observed power limit per GPU index (watts), "
                "updated on every apply and re-synced to the live value on "
                "restore / no-write paths. NOTE: series are labeled by the "
                "actuator's GPU index. On the DCGM actuator a hostengine "
                "re-enumeration can move a GPU to a new index; the new index is "
                "updated but the old index series is NOT deleted and lingers "
                "until the process restarts. NVML indices are process-stable, "
                "so this caveat does not apply to the default actuator.",
                labelnames=("gpu",),
            )
            self.multi_pod_gpu_total = Counter(
                "dynamo_power_agent_multi_pod_gpu_total",
                "Times a physical GPU had multiple pods (agree or conflict).",
                labelnames=("disposition",),
            )
            self.apply_failures_total = Counter(
                "dynamo_power_agent_apply_failures_total",
                "Times a cap could NOT be made live on the GPU: the actuator "
                "write (NVML nvmlDeviceSetPowerManagementLimit or DCGM "
                "dcgmConfigSet) failed, OR the DCGM path refused the write "
                "because it could not verify the target GPU's identity "
                "(re-enumeration / unreadable UUID) before writing. Distinct "
                "from policy fallbacks (tracked by safe_default_applied_total) "
                "where the cap IS applied at safe-default.",
            )
            self.safe_default_applied_total = Counter(
                "dynamo_power_agent_safe_default_applied_total",
                "Times the safe-default cap was used (conflict or cold-start parse failure).",
            )
            self.cap_clamped_total = Counter(
                "dynamo_power_agent_cap_clamped_total",
                "Times a requested cap was clamped to per-SKU constraints.",
                labelnames=("direction",),
            )
            # Pre-fix `_list_pods_on_node` swallowed every API error and returned
            # [], making a transient apiserver outage indistinguishable
            # from a genuinely empty node. Now reconcile_once skips its
            # cycle on list failure and increments this counter so
            # operators can alert (e.g. >0 over 5m → RBAC regression or
            # apiserver outage masking enforcement).
            self.k8s_list_failures_total = Counter(
                "dynamo_power_agent_k8s_list_failures_total",
                "Times the Kubernetes pod-list API call failed during reconcile, "
                "causing the cycle to be skipped (previously-applied caps remain).",
            )
            try:
                start_http_server(prometheus_port)
                logger.info(
                    "Prometheus metrics server started on port %d", prometheus_port
                )
            except Exception as e:
                logger.warning("Failed to start Prometheus server: %s", e)
        else:
            noop = _NoopMetric()
            self.applied_limit_watts = noop
            self.multi_pod_gpu_total = noop
            self.apply_failures_total = noop
            self.safe_default_applied_total = noop
            self.cap_clamped_total = noop
            self.k8s_list_failures_total = noop


# ---------------------------------------------------------------------------
# NVML helpers
# ---------------------------------------------------------------------------


def _clamp_to_constraints(
    handle, requested_w: int, gpu_idx: int, metrics: PowerAgentMetrics
) -> int:
    """Clamp `requested_w` to the SKU-defined NVML power-cap range."""
    try:
        min_mw, max_mw = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)
    except pynvml.NVMLError:
        return requested_w
    min_w, max_w = min_mw // 1000, max_mw // 1000
    if requested_w < min_w:
        logger.warning(
            "Requested cap %d W below SKU min %d W on GPU %d; clamping up.",
            requested_w,
            min_w,
            gpu_idx,
        )
        metrics.cap_clamped_total.labels(direction="min").inc()
        return min_w
    if requested_w > max_w:
        logger.warning(
            "Requested cap %d W above SKU max %d W on GPU %d; clamping down.",
            requested_w,
            max_w,
            gpu_idx,
        )
        metrics.cap_clamped_total.labels(direction="max").inc()
        return max_w
    return requested_w


# Alias to `managed_state` (see `_previously_managed` above for why). Mutate in
# place only; shutdown cleanup and the actuator must see the same set.
_managed_gpu_indices: set[int] = managed_state.managed_gpu_indices

# UUIDs whose runtime release COMPLETED in memory (hardware restored to default,
# actuator + index ownership retired, `_previously_managed` pruned) but whose
# DURABLE prune to `managed_gpus.json` has not yet landed (state-volume write
# failure). Process-local and deliberately NOT shared with the actuator or
# persisted: it exists only to retry the pending `_persist_managed_gpus` write —
# NEVER the hardware restore — on the next reconcile, so a cap another workflow
# installs on the released GPU in the interim is never clobbered by a repeated
# restore.
_pending_retirement: set[str] = set()


def _apply_cap(
    handle, gpu_idx: int, requested_w: int, metrics: PowerAgentMetrics
) -> None:
    """Apply NVML power cap. All writes go through here."""
    effective_w = _clamp_to_constraints(handle, requested_w, gpu_idx, metrics)
    try:
        pynvml.nvmlDeviceSetPowerManagementLimit(handle, effective_w * 1000)
        _managed_gpu_indices.add(gpu_idx)
        _record_managed_gpu_uuid(handle)
        metrics.applied_limit_watts.labels(gpu=str(gpu_idx)).set(effective_w)
    except pynvml.NVMLError as e:
        logger.error(
            "nvmlDeviceSetPowerManagementLimit GPU %d → %d W failed: %s",
            gpu_idx,
            effective_w,
            e,
        )
        metrics.apply_failures_total.inc()


def _retire_actuator_ownership(actuator: Actuator, uuid: str) -> None:
    """Drop `uuid` from the actuator's in-memory ownership set on release, so a
    later shutdown sweep no longer treats a cap we relinquished as ours.

    This MUST run — and complete — before persistent/index ownership is pruned.
    If it were skipped or its failure swallowed, `_capped_uuids` would still
    hold the released UUID while the persisted record was dropped, so a shutdown
    sweep could reset a cap another workflow later installed on that GPU — the
    exact leak this retirement prevents. It is a set operation that does not
    fail in normal use; we deliberately do NOT swallow an exception here — let
    it propagate so `reconcile_once`'s per-GPU guard logs it and retries with
    ownership intact, rather than pruning inconsistently. Only the DCGM actuator
    tracks per-UUID ownership; on others there is nothing to retire (no-op)."""
    retire = getattr(type(actuator), "retire_managed_uuid", None)
    if retire is not None:
        actuator.retire_managed_uuid(uuid)


def _commit_release(actuator: Actuator, gpu_idx: int, release_uuid: str) -> None:
    """Retire ownership of a GPU whose cap was just released to default.

    Ordering and failure handling:

      1. Retire the actuator's in-memory ownership (drops `release_uuid` from
         `_capped_uuids` and every index that projected to it, including the
         shared `_managed_gpu_indices` entries). Pure set ops; a failure is NOT
         swallowed — if it raises, nothing below runs, so no persistent state is
         pruned while `_capped_uuids` still owns the UUID (that would let a
         shutdown sweep clobber a later, unrelated cap).
      2. Drop this call's index (retire already removed any DCGM projections;
         a non-DCGM actuator has none, so this covers the NVML path).
      3. Commit the in-memory retirement (`_previously_managed`) UNCONDITIONALLY.
         The release is already done in hardware, so the GPU must NOT look
         managed again — otherwise the next reconcile would REPEAT the hardware
         restore and, if another workflow capped the GPU after our release,
         erase that new cap. Then persist the reduced set. If persistence fails
         (state-volume outage), record a *pending retirement* and retry ONLY the
         persistence next cycle (`_flush_pending_retirements`) — never the
         hardware restore. Durable ownership and hardware state can't be made
         atomic with the JSON schema, so the residual gap is a crash BEFORE the
         retry lands: on restart disk still owns the UUID and orphan recovery
         reconciles it (idempotent when the GPU is at/above default).
    """
    _retire_actuator_ownership(actuator, release_uuid)
    _managed_gpu_indices.discard(gpu_idx)
    if release_uuid not in _previously_managed:
        return
    _previously_managed.discard(release_uuid)
    try:
        _persist_managed_gpus(_previously_managed)
    except Exception as e:
        _pending_retirement.add(release_uuid)
        logger.warning(
            "Released cap on UUID %s and retired ownership, but persisting the "
            "durable set failed; the hardware restore will NOT be repeated — "
            "only the persistence is retried next cycle: %s",
            release_uuid,
            e,
        )


def _flush_pending_retirements() -> None:
    """Retry the durable prune for releases whose hardware restore + in-memory
    retirement completed but whose `_persist_managed_gpus` write failed.

    Retries ONLY the persistence (writes the authoritative in-memory
    `_previously_managed`), never the hardware restore, so a cap another workflow
    installed on a released GPU after our release is never clobbered. Called at
    the top of every reconcile cycle so it flushes independently of Kubernetes
    API health."""
    if not _pending_retirement:
        return
    try:
        _persist_managed_gpus(_previously_managed)
    except Exception as e:
        logger.warning(
            "Deferred retirement persistence retry failed (%d pending); will "
            "retry next cycle: %s",
            len(_pending_retirement),
            e,
        )
        return
    logger.info(
        "Flushed %d deferred cap retirement(s) to durable state.",
        len(_pending_retirement),
    )
    _pending_retirement.clear()


def _release_managed_gpu(
    actuator: Actuator, gpu_idx: int, expected_uuid: Optional[str] = None
) -> None:
    """Restore default TGP on a GPU we previously capped, and unmanage it.

    Runtime counterpart to ``_shutdown_cleanup`` / ``_restore_orphaned_gpus_on_startup``.
    Invoked from steady-state reconcile when a GPU we previously capped is now
    running only unannotated / non-K8s processes — i.e. the opted-in pod is gone
    and a non-managed workload owns the GPU (or the planner removed the
    annotation to release it). Without this, the agent's last cap would strand
    on the reused GPU until the next agent shutdown (startup orphan recovery
    skips busy GPUs), silently throttling the new tenant. This implements the
    "planner owns cap lifecycle via annotation removal/update" contract at
    runtime.

    Routed through the active ``Actuator`` (not raw ``pynvml``) so the release
    write flows through the same library that applied the cap. On
    ``actuator: dcgm`` this means the restore runs ``dcgmConfigSet(default)``,
    keeping the hostengine's target-config record consistent with the
    driver-level cap — the same reason shutdown cleanup writes through the
    actuator surface. Routing it through raw NVML here would desync the
    DCGM target config on the dcgm path.

    Eligibility is UUID-gated so caps set by other tooling are never touched.
    A GPU is "ours" if it is in ``_managed_gpu_indices`` (capped in THIS process)
    OR its UUID is in the persisted ``_previously_managed`` set (capped in a
    prior process). The latter is essential across restarts: ``_managed_gpu_indices``
    is in-memory and empty after a restart, while ``_previously_managed`` is
    loaded from disk — without it, a GPU capped before the restart and now busy
    with only unannotated work would keep the stale cap (startup orphan recovery
    only restores *idle* GPUs).

    The idle case (no processes at all) is intentionally NOT handled here;
    ``_reconcile_gpu``'s ``not pids`` branch keeps the cap for a briefly-exited
    worker that will return to the same GPU.

    Release acts ONLY on the physical GPU currently occupying ``gpu_idx``, and
    only when that occupant is the GPU we historically capped at this index.
    ``expected_uuid`` is the identity captured BEFORE the PID snapshot that
    routed us here (the "no annotated pod" evidence describes THAT GPU); we
    reverify the occupant still matches it. On the ``dcgm`` path a hostengine
    re-enumeration can move a DIFFERENT physical GPU onto ``gpu_idx`` while the
    GPU we capped moves elsewhere — in that case the observed unannotated
    workload belongs to the new occupant, NOT to the GPU we capped, so we must
    NOT relocate a restore onto the displaced GPU (it may still be running its
    annotated workload at its new index). We skip the stale index→UUID
    projection without pruning; the displaced GPU is evaluated when reconcile
    reaches its current index (or by the UUID-keyed SIGTERM / startup orphan
    recovery), so its cap is never stranded.
    """
    try:
        uuid = actuator.get_uuid(gpu_idx)
    except Exception as e:
        logger.warning(
            "Failed to read UUID for GPU %d during release check: %s", gpu_idx, e
        )
        return

    # Reverify against the pre-snapshot identity that routed us here. The "no
    # annotated pod owns this GPU" evidence was gathered on whatever GPU
    # occupied gpu_idx at snapshot time; if a dcgm re-enumeration swapped the
    # occupant between the snapshot and this check, that evidence no longer
    # describes the current occupant. Bail and let the next reconcile
    # re-evaluate with a fresh, self-consistent snapshot rather than release on
    # stale evidence.
    if expected_uuid is not None and uuid != expected_uuid:
        logger.warning(
            "Skipping cap release for GPU %d: index re-enumerated between the "
            "workload snapshot (%s) and the release check (%s); deferring to "
            "the next reconcile.",
            gpu_idx,
            expected_uuid,
            uuid,
        )
        return

    if gpu_idx not in _managed_gpu_indices and uuid not in _previously_managed:
        return  # not a GPU this agent capped — leave it alone (UUID-gating)

    # Identity we historically capped at this index. On the ``dcgm`` path this
    # can differ from the current occupant after a hostengine re-enumeration
    # (``managed_uuid_for_idx`` returns the UUID recorded at cap time). NVML
    # indices are stable within a process, so the current UUID already matches
    # and the helper is absent there.
    managed_uuid = uuid
    if hasattr(type(actuator), "managed_uuid_for_idx"):
        try:
            managed_uuid = getattr(actuator, "managed_uuid_for_idx")(gpu_idx)
        except Exception as e:
            # DCGM-specific managed-identity lookup failed. We must NOT fall
            # back to the current occupant (uuid): on a re-enumerated index
            # that fallback would make managed_uuid == uuid, silently pass the
            # stale-projection guard below, and release/prune on stale integer
            # membership alone — restoring the wrong GPU and discarding the real
            # managed GPU's ownership. Fail closed: retain ownership and retry
            # next cycle. The current-UUID fallback is sound ONLY for actuators
            # WITHOUT managed_uuid_for_idx (NVML), which never enter this branch
            # .
            logger.warning(
                "Skipping cap release for GPU %d: could not resolve the managed "
                "UUID (retaining ownership, retrying next cycle): %s",
                gpu_idx,
                e,
            )
            return

    # A dcgm re-enumeration can make the index's recorded identity
    # (managed_uuid) differ from the current occupant (uuid). The "unannotated
    # workload" evidence we gathered belongs to the CURRENT occupant, so the
    # release decision is about `uuid`, never about the stale index projection.
    if managed_uuid != uuid:
        # Is the CURRENT occupant itself a GPU we manage? _previously_managed is
        # UUID-keyed and authoritative across re-enumeration. If so, release IT
        # by UUID — this handles the index-SWAP case where two managed GPUs
        # traded indices: each index's recorded map points at the other, so the
        # old index-based skip would block BOTH releases until shutdown. The
        # UUID-addressed restore resolves the occupant's live index and guards
        # the write, so it is re-enumeration safe.
        if uuid in _previously_managed and hasattr(
            type(actuator), "restore_default_by_uuid"
        ):
            result = getattr(actuator, "restore_default_by_uuid")(uuid)
            if result is False:
                # Could not conclusively locate/restore the occupant's cap; keep
                # ownership so a later reconcile or startup orphan recovery
                # retries. Never prune a possibly-live cap.
                logger.warning(
                    "Skipped cap release for GPU %d (occupant %s) via %s "
                    "actuator by UUID (not conclusively located); leaving it "
                    "managed so a later cycle retries.",
                    gpu_idx,
                    uuid,
                    actuator.name,
                )
                return
            logger.info(
                "Released cap on the current occupant of GPU %d (UUID %s) by "
                "UUID: previously managed, now running only unannotated/non-K8s "
                "processes (index re-enumeration; recorded map here was %s).",
                gpu_idx,
                uuid,
                managed_uuid,
            )
            _commit_release(actuator, gpu_idx, uuid)
            return

        # The current occupant is NOT a GPU we manage (a re-enumeration dropped
        # an unrelated GPU onto this index, or the actuator has no UUID-addressed
        # restore). Skip the stale projection WITHOUT restoring or pruning: the
        # GPU we capped is evaluated when reconcile reaches ITS current index
        # (its own snapshot decides keep-vs-release), and meanwhile its cap is
        # held in _previously_managed for the UUID-keyed SIGTERM / startup
        # orphan recovery, so it is never stranded. NVML has no
        # managed_uuid_for_idx (managed_uuid == uuid there), so this whole
        # branch is a no-op on that path.
        logger.info(
            "Skipping cap release for GPU %d: index now hosts %s (not managed by "
            "us) but we capped %s here (dcgm re-enumeration). Deferring the "
            "managed GPU's keep-vs-release decision to its current index.",
            gpu_idx,
            uuid,
            managed_uuid,
        )
        return

    # managed_uuid == uuid: the current occupant IS the GPU we capped here.
    # Delegate the below-default / at-default / restore decision to the
    # actuator's atomic, identity-bound restore. Doing those reads here as
    # SEPARATE default_w()/current_w()/get_uuid() calls left an A->B->A hole:
    # each is a reconnect-capable read, so a re-enumeration mid-sequence could
    # pair GPU-A's default with GPU-B's "already at default" current, skip the
    # restore, and prune GPU-A while its cap stayed live at its new index. The
    # final identity recheck did not close it because A->B->A ends back on A.
    # `restore_default_by_uuid` reads the limits AND their owning UUID from ONE
    # snapshot and guards the write by UUID, so the decision and any write both
    # provably concern `managed_uuid` (same fix already applied to apply,
    # orphan recovery, and the SIGTERM sweep).
    try:
        result = actuator.restore_default_by_uuid(managed_uuid)
    except Exception as e:
        # Leave the GPU in the managed set so a later cycle retries the release.
        logger.warning(
            "Failed to release cap on GPU %d (UUID %s): %s",
            gpu_idx,
            managed_uuid,
            e,
        )
        return
    if result is False:
        # Not conclusively located/restored (a probe raised, or a proven
        # mid-write re-enumeration): the cap may still be LIVE, so keep our
        # ownership state and let a later reconcile or the next startup's orphan
        # recovery retry — never prune a possibly-live cap.
        logger.warning(
            "Skipped cap release for GPU %d (UUID %s) via %s actuator: not "
            "conclusively located; leaving it managed so a later cycle retries.",
            gpu_idx,
            managed_uuid,
            actuator.name,
        )
        return
    # True (restored a live below-default cap) or None (reconfirmed at/above
    # default, or a clean scan proved the GPU gone): either way it is CONCLUSIVE
    # that no cap of ours remains, so retire ownership durably.
    logger.info(
        "Released cap on GPU %d (UUID %s) via %s actuator by UUID: previously "
        "managed, now running only unannotated/non-K8s processes.",
        gpu_idx,
        managed_uuid,
        actuator.name,
    )
    _commit_release(actuator, gpu_idx, managed_uuid)


# ---------------------------------------------------------------------------
# Kubernetes client
# ---------------------------------------------------------------------------


def _build_k8s_core_v1() -> "k8s_client.CoreV1Api":
    """Build a CoreV1Api whose transport does NOT retry.

    The default ``kubernetes`` client retries requests at the urllib3 layer.
    For the pod LIST that is the wrong policy on two counts: (1) a retried LIST
    multiplies apiserver load exactly when it is already throttling under
    Priority & Fairness (the amplification sttts flagged), and (2) transport
    retries stretch the effective wall-clock time of a single
    ``list_pods_on_node`` call, undermining the SIGTERM grace-period budget the
    LIST timeouts are meant to protect.

    The reconcile loop already retries at the application level every
    ``RECONCILE_INTERVAL_S`` and treats a failed LIST as a fail-safe skip, so
    transport-level retries add nothing but risk here. Disable them (retries=0)
    on a dedicated client so this choice is local to the Power Agent and does
    not mutate global client state.
    """
    configuration = k8s_client.Configuration.get_default_copy()
    # urllib3 Retry(0) → no retries; the reconcile loop is the retry policy.
    configuration.retries = 0
    return k8s_client.CoreV1Api(k8s_client.ApiClient(configuration))


# ---------------------------------------------------------------------------
# SIGTERM handler
# ---------------------------------------------------------------------------

_shutdown = threading.Event()


def _handle_sigterm(signum, frame):
    """SIGTERM/SIGINT handler — request shutdown ONLY; do no heavy work here.

    Python delivers signals on the main thread between bytecodes, so this
    handler can interrupt an in-flight reconcile. If it performed the cap
    restores itself, a SIGTERM landing after a `dcgmConfigSet` succeeds but
    before `_record_managed_state` records ownership would sweep, shut down,
    and exit while the interrupted reconcile then records a still-live cap that
    nothing restores — a leak. So we just set
    the event; `run()` runs the one-shot `_shutdown_cleanup` from its `finally`
    once the current reconcile returns, with the in-flight write fully
    recorded.
    """
    logger.info("SIGTERM received — requesting graceful shutdown.")
    _shutdown.set()


def _shutdown_cleanup(actuator: Actuator) -> None:
    """Restore default TGP on every managed GPU, then shut the actuator down.

    Called once from `run()`'s `finally` after the reconcile loop exits — NOT
    from the signal handler — so it never races an in-flight cap write. The
    actuator is always the live one (`run()` passes `self._actuator`), which is
    why there is no None/raw-NVML fallback: cleanup can only run after
    `PowerAgent.__init__` bound and initialised an actuator.

    Dispatches through the actuator so a `dcgm` deployment restores via
    `dcgmConfigSet(default)`, keeping the hostengine's target-config record in
    sync with the driver-level cap (a raw-NVML write would desync them and let
    DCGM re-apply the stale cap after the next GPU reset/reinit).
    """
    for gpu_idx in list(_managed_gpu_indices):
        # Capture the UUID of each GPU we restore so we can prune it from
        # `_previously_managed`; otherwise the next startup's orphan recovery
        # would "restore" a GPU we no longer own, clobbering another workflow's
        # cap (different DGD, manual `nvidia-smi -pl`, vendor default).
        restored_uuid: Optional[str] = None
        try:
            restore_result = actuator.restore_default(gpu_idx)
            if restore_result is False:
                logger.warning(
                    "Skipped default TGP restore for GPU %d via %s actuator "
                    "(cap not conclusively released: GPU no longer locatable, "
                    "or its identity could not be confirmed at write time — "
                    "re-enumeration/unverifiable); leaving it managed so the "
                    "UUID sweep and next-startup orphan recovery retry.",
                    gpu_idx,
                    actuator.name,
                )
                continue
            logger.info(
                "Restored GPU %d to default TGP via %s actuator",
                gpu_idx,
                actuator.name,
            )
            try:
                if hasattr(type(actuator), "managed_uuid_for_idx"):
                    restored_uuid = getattr(actuator, "managed_uuid_for_idx")(gpu_idx)
                else:
                    restored_uuid = actuator.get_uuid(gpu_idx)
            except Exception as e:
                # Benign: the GPU is already at default, so a stale entry just
                # makes the next startup see current_w >= default_w and skip.
                logger.warning(
                    "Could not resolve UUID for restored GPU %d "
                    "(state file may retain stale entry): %s",
                    gpu_idx,
                    e,
                )
        except Exception as e:
            logger.exception("Failed to restore TGP on GPU %d: %s", gpu_idx, e)
            # Do NOT prune on failure — the cap may still be live and the next
            # startup's orphan recovery is our only chance to reset it.
            continue
        if restored_uuid is not None:
            _previously_managed.discard(restored_uuid)
            # Retire the per-process ownership too, so the UUID sweep below does
            # not redundantly re-restore this already-defaulted GPU — and, more
            # importantly, cannot ERASE a fresh cap an external writer installs
            # on it between this indexed restore and the sweep (the stale
            # `_capped_uuids` entry would otherwise make the sweep reset it).
            # Best-effort: a failure here only forfeits that optimisation (the
            # sweep falls back to reprocessing), so it must not abort shutdown.
            try:
                _retire_actuator_ownership(actuator, restored_uuid)
            except Exception as e:
                logger.warning(
                    "Could not retire ownership for restored UUID %s at "
                    "shutdown (sweep will reprocess it): %s",
                    restored_uuid,
                    e,
                )

    # UUID-complete safety net: the index-keyed loop above misses a still-capped
    # GPU when DCGM re-enumerated and a later reconcile re-capped its old index
    # onto a different physical GPU (the displaced GPU drops out of
    # `_managed_gpu_indices`). Resolve each UUID this process capped to its
    # CURRENT index instead. Scope to `actuator.managed_uuids()` — NOT the
    # cross-incarnation `_previously_managed` set, which can hold UUIDs startup
    # recovery kept but we never capped; sweeping those would reset another
    # workflow's cap. Only DcgmActuator can relocate by UUID and track capped
    # UUIDs, so gate on BOTH methods (NvmlActuator has `restore_default_by_uuid`
    # but no `managed_uuids`, and must not enter the sweep).
    if hasattr(type(actuator), "managed_uuids") and hasattr(
        type(actuator), "restore_default_by_uuid"
    ):
        for uuid in getattr(actuator, "managed_uuids")():
            try:
                sweep_result = getattr(actuator, "restore_default_by_uuid")(uuid)
            except Exception as e:
                # Keep the UUID: a write/relocation failure means the cap may
                # still be live; the next startup's orphan recovery retries.
                logger.exception(
                    "SIGTERM UUID sweep failed to restore managed UUID %s: %s",
                    uuid,
                    e,
                )
                continue
            # Retire ownership on any CONCLUSIVE result (True = restored a live
            # cap; None = reconfirmed at/above default or proved gone), retain
            # only on the inconclusive False. Leaving a proven-gone UUID would
            # let a future startup clobber a later unrelated cap on the same
            # GPU.
            if sweep_result is not False:
                _previously_managed.discard(uuid)

    # Persist the pruned state so the next startup only touches GPUs we still
    # own. A write failure is non-fatal — log and proceed to shutdown.
    try:
        _persist_managed_gpus(_previously_managed)
    except Exception as e:
        logger.warning(
            "Failed to persist pruned managed_gpus state at shutdown: %s "
            "(next startup may briefly re-restore already-default GPUs).",
            e,
        )
    try:
        actuator.shutdown()
    except Exception:
        # Log (with traceback) but never re-raise: shutdown must complete so
        # the container exits cleanly (PR #9682 CodeRabbit review).
        logger.exception(
            "Actuator shutdown raised; proceeding with agent exit anyway.",
        )


# ---------------------------------------------------------------------------
# Orphan cap restoration on startup (UUID-gated)
# ---------------------------------------------------------------------------


def _restore_orphaned_gpus_on_startup(actuator: Actuator) -> None:
    """Restore default TGP only on GPUs this agent previously capped AND that are now idle.

    Runs through the actuator surface rather than inline NVML: on the
    DCGM path, orphan recovery must write through `nvidia-dcgm`
    too, not bypass it via raw NVML — otherwise the hostengine's
    target-configuration record (and its reset/reinit auto-reapply
    behaviour) drifts from the driver-level reality. Going through
    `actuator.restore_default` keeps a single write path per actuator.

    Two guards are preserved verbatim from the earlier NVML-only
    implementation:

      1. UUID-gating — only touch GPUs whose UUID is in the persisted
         `managed_gpus.json`. Prevents stepping on caps applied by
         other workflows (different DGD, manual `nvidia-smi -pl`,
         vendor firmware defaults).
      2. `current_w < default_w` — only write when the cap is
         actually below default. Skips a redundant privileged write
         (and the audit-log entry it produces) when the previous
         shutdown left the GPU at default, or when something else
         already restored it.

    The Protocol now carries `current_w` and `default_w` methods
    expressly so the guard survives the migration; see actuator.py
    `Actuator` Protocol.
    """
    # Reload IN PLACE — never rebind `_previously_managed`, or the alias to
    # `managed_state.previously_managed` (shared with the actuator's module
    # copy) would split and re-introduce the dual-copy bug.
    reloaded, conclusive = _read_managed_gpus_state()
    _previously_managed.clear()
    _previously_managed.update(reloaded)
    if not conclusive:
        # The durable state could not be read conclusively (I/O error or corrupt
        # JSON). We do NOT know which GPUs a prior incarnation owned, so acting
        # now would be unsafe on both sides: recovering from an empty view could
        # strand real orphaned caps, and persisting the empty set would ERASE
        # the (possibly intact) on-disk record a transient failure merely hid.
        # Skip recovery AND the rewrite; the next boot retries the read.
        logger.warning(
            "Managed-GPU state read was inconclusive; skipping startup orphan "
            "recovery and leaving the on-disk state untouched this boot."
        )
        return

    persisted_snapshot = set(_previously_managed)
    # UUIDs this pass removed from `_previously_managed` (restored or pruned).
    # Queued for the reconcile-loop retirement flush if the final persist fails,
    # so a pruned record can't linger on disk indefinitely.
    retired_uuids: set[str] = set()

    # ONE conclusive identity snapshot instead of a `range(device_count())` +
    # per-index `get_uuid` loop. That naive loop was falsely conclusive on the
    # DCGM path: the range is fixed pre-reconnect, but each `get_uuid` can
    # reconnect and GROW discovery, so a persisted UUID that MOVED to a
    # newly-enumerated index was never visited (no probe raised) and got pruned
    # as absent. `scan_uuid_index_map` builds the map inside a single
    # `_with_reconnect` that re-materializes the topology length, so a mid-scan
    # growth is fully rescanned (see actuator.py `_resolve_idx_for_uuid` /
    # `scan_uuid_index_map`). `scan_complete` is True only when the whole
    # topology was enumerated cleanly, so an absent UUID is provably gone.
    uuid_to_idx, scan_complete = actuator.scan_uuid_index_map()

    # Restore pass — hardware writes, only on GPUs we positively identified as
    # ours in the snapshot. Runs regardless of `scan_complete`: restoring a live
    # orphan cap on a visible idle GPU is always safe (idle-gated + identity-
    # bound), and `restore_default_by_uuid` re-resolves identity itself.
    for uuid in sorted(persisted_snapshot & set(uuid_to_idx)):
        gpu_idx = uuid_to_idx[uuid]
        try:
            # Bind the idle check to `uuid` (same reason as the reconcile path):
            # resolving PIDs by bare `gpu_idx` on the DCGM path could read a
            # re-enumerated GPU's workload and let us restore/retire the wrong
            # GPU. A mismatch raises `_GpuIdentityMismatch`, caught below, which
            # keeps the UUID for a later startup — fail closed.
            if actuator.list_running_pids(gpu_idx, expected_uuid=uuid):
                continue  # workload running — let normal reconcile handle it
            # Delegate the ENTIRE below-default / at-default / gone decision to
            # `restore_default_by_uuid`, which resolves `uuid` to its current
            # index and reads the power limits + identity from ONE snapshot.
            # Doing the current_w/default_w/identity checks here at the caller
            # would reintroduce an A->B->A re-enumeration hole: current_w from
            # GPU-A, default_w from GPU-B, final identity back on A could
            # falsely conclude "at default" and retire ownership while A's cap
            # is still live. It also re-resolves the index (never trusts
            # `gpu_idx`), so a DCGM re-enumeration between the snapshot above and
            # the write lands on the GPU that actually carries `uuid`.
            restore_result = actuator.restore_default_by_uuid(uuid)
            if restore_result is False:
                # Inconclusive: a relocation-scan probe raised / the resolved
                # index's identity could not be confirmed at write time / a mid
                # re-enumeration. The GPU may still carry our cap, so keep the
                # UUID and retry on the next startup rather than prematurely
                # pruning it.
                continue
            if restore_result is True:
                logger.info(
                    "Restored orphaned cap for idle managed GPU (UUID %s).",
                    uuid,
                )
            # True -> we restored a live below-default cap.
            # None -> restore_default_by_uuid confirmed (in one snapshot) the
            #         GPU is already at/above default. Either outcome is
            #         CONCLUSIVE that no cap of ours remains, so retire
            #         ownership: retaining a UUID whose cap is proven gone would
            #         let a LATER unrelated cap on the same physical GPU be
            #         clobbered by a future startup's orphan recovery.
            _previously_managed.discard(uuid)
            retired_uuids.add(uuid)
        except Exception as e:
            logger.warning("orphan-restore failed for UUID %s: %s", uuid, e)

    # Absent-UUID prune — STATE ONLY, never a hardware write. A persisted UUID
    # missing from the snapshot is provably absent ONLY when the scan was
    # CONCLUSIVE; then the GPU is gone, there is nothing to restore, and we drop
    # the stale record. We deliberately do NOT call `restore_default_by_uuid`
    # here: a UUID missing from an INCONCLUSIVE snapshot can be a transiently
    # unreadable but PRESENT (possibly busy) GPU, and that call would strip its
    # live cap with no idle check. On an inconclusive scan we retain everything
    # and retry next boot.
    if scan_complete:
        for uuid in persisted_snapshot - set(uuid_to_idx):
            _previously_managed.discard(uuid)
            retired_uuids.add(uuid)
            logger.info(
                "Retiring stale managed-GPU record for absent UUID %s (no "
                "visible GPU reports it; GPU replaced / removed / moved).",
                uuid,
            )

    # Persisting must never brick startup. `_read_managed_gpus_state` already
    # tolerates a read failure without rewriting, so the matching write here
    # must be equally defensive: a read-only / permission-denied state volume
    # would otherwise raise, escape `PowerAgent.__init__`, and CrashLoop the pod
    # — the exact outcome the read path was written to avoid. On failure, queue
    # this pass's retirements so the reconcile-loop retirement flush retries the
    # durable write (otherwise a pruned UUID would linger on disk indefinitely).
    try:
        _persist_managed_gpus(_previously_managed)
    except Exception as e:
        if retired_uuids:
            _pending_retirement.update(retired_uuids)
        logger.warning(
            "Failed to persist managed-GPU state during startup orphan recovery "
            "(%s: %s); queued %d retirement(s) for the reconcile-loop flush.",
            type(e).__name__,
            e,
            len(retired_uuids),
        )


# ---------------------------------------------------------------------------
# Multi-pod-per-GPU policy
# ---------------------------------------------------------------------------


def _resolve_cap_for_gpu(
    gpu_idx: int,
    pod_annotations: list[tuple[str, Optional[str]]],
    safe_default_watts: int,
    metrics: PowerAgentMetrics,
) -> int:
    """Determine the NVML cap to apply for a GPU given the pod annotations on it.

    Policy:
      - 1 pod with parseable int annotation       → use that value.
      - 1 pod with missing/invalid annotation     → safe_default_watts, ERROR.
      - 2+ pods, ALL parseable AND all agree      → agreed value, WARNING.
      - 2+ pods, any missing/invalid/disagreement → safe_default_watts, ERROR.

    A multi-pod GPU
    where pod A has cap 480 and pod B has no annotation must NOT inherit
    pod A's cap. The pre-fix code filtered None before computing the
    agree-set, so the "all agree" branch fired whenever the surviving
    non-None values agreed, even if other pods on the same GPU had no
    parseable cap. That let one pod's annotation silently govern
    another pod's GPU usage — the exact cross-workload policy failure
    the multi-pod guard is meant to contain.

    Returns the cap in watts.
    """
    # Parse each pod's raw annotation. Track missing (None) and invalid
    # (non-int) separately so the log message tells operators which
    # pathology triggered the safe-default fallback.
    parsed: list[int] = []
    has_missing = False
    has_invalid = False
    for _, raw in pod_annotations:
        if raw is None:
            has_missing = True
            continue
        try:
            parsed.append(int(raw))
        except (ValueError, TypeError):
            has_invalid = True

    if len(pod_annotations) > 1:
        # Multi-pod-per-GPU: this is always an operator misconfig (we
        # don't support pod-pool topologies on the same physical GPU).
        # Either all pods agree on a parseable cap and we propagate it
        # with a WARNING, or we fail safe.
        if has_missing or has_invalid or len(set(parsed)) > 1:
            logger.error(
                "GPU %d: %d pods with missing/invalid/conflicting caps "
                "(parsed=%s, has_missing=%s, has_invalid=%s); applying "
                "safe default (%d W).",
                gpu_idx,
                len(pod_annotations),
                sorted(set(parsed)),
                has_missing,
                has_invalid,
                safe_default_watts,
            )
            metrics.multi_pod_gpu_total.labels(disposition="conflict").inc()
            # Do NOT tick apply_failures_total — the caller WILL apply
            # the cap at safe-default, so the cap WILL be live. That
            # metric's contract is "cap NOT live"; policy-fallback is
            # tracked by safe_default_applied_total.
            metrics.safe_default_applied_total.inc()
            return safe_default_watts
        logger.warning(
            "GPU %d: %d pods all agree on cap %d W (multi-pod-per-GPU is unsupported topology).",
            gpu_idx,
            len(pod_annotations),
            parsed[0],
        )
        metrics.multi_pod_gpu_total.labels(disposition="agree").inc()
        return parsed[0]

    # Single pod from here. Either parsed has exactly one entry (happy
    # path) or it's empty (pod's annotation is missing or non-int).
    if not parsed:
        if has_missing:
            logger.error(
                "GPU %d: pod has no power-limit annotation; applying safe default (%d W).",
                gpu_idx,
                safe_default_watts,
            )
        else:
            logger.error(
                "GPU %d: pod annotation is not an integer; applying safe default (%d W).",
                gpu_idx,
                safe_default_watts,
            )
        metrics.safe_default_applied_total.inc()
        return safe_default_watts
    return parsed[0]


# ---------------------------------------------------------------------------
# Main reconcile loop
# ---------------------------------------------------------------------------


class PowerAgent:
    def __init__(
        self,
        safe_default_watts: int,
        node_name: Optional[str] = None,
        k8s_namespace: Optional[str] = None,
        prometheus_port: int = 0,
        actuator: Optional[Actuator] = None,
        actuator_factory: Optional[Callable[["PowerAgentMetrics"], Actuator]] = None,
    ) -> None:
        self.safe_default_watts = safe_default_watts
        self.node_name = node_name or os.environ.get("NODE_NAME", "")
        self.k8s_namespace = k8s_namespace
        self.metrics = PowerAgentMetrics(prometheus_port)

        if pynvml is None:
            raise RuntimeError("pynvml is required — install pynvml or nvidia-ml-py")
        if k8s_client is None:
            raise RuntimeError("kubernetes Python SDK is required — install kubernetes")

        # NVML init is owned by the actuator now:
        # `NvmlActuator.init()` runs `nvmlInit()`, `DcgmActuator.init()` runs
        # its own guarded `nvmlInit()`, and each pairs it with a single
        # `nvmlShutdown()` in `shutdown()`. Calling `nvmlInit()` here as well
        # would re-introduce the process-wide init/shutdown imbalance in DCGM
        # mode (two inits, one shutdown). `_actuator.init()` below does it.

        # Bind the actuator. Resolution order: explicit instance >
        # factory(metrics) > default NvmlActuator(metrics). The factory
        # form is used by `main()`/`_make_actuator` because the
        # PowerAgentMetrics object isn't constructible until __init__
        # runs (the Prometheus server starts in its constructor).
        # Tests typically pass an explicit MagicMock actuator instance.
        if actuator is not None:
            self._actuator: Actuator = actuator
        elif actuator_factory is not None:
            self._actuator = actuator_factory(self.metrics)
        else:
            self._actuator = NvmlActuator(self.metrics)
        self._actuator.init()

        self.device_count = self._actuator.device_count()
        logger.info(
            "Actuator initialized: %s. %d GPU(s) found on this node.",
            self._actuator.name,
            self.device_count,
        )

        _restore_orphaned_gpus_on_startup(self._actuator)

        # K8s client
        try:
            k8s_config.load_incluster_config()
        except ConfigException:
            k8s_config.load_kube_config()
        self._core_v1 = _build_k8s_core_v1()

    def _list_pods_on_node(self) -> Optional[list]:
        """List all pods scheduled on this node.

        Returns the pod list on success (an empty list is a *valid* success
        result, meaning this node genuinely hosts no pods), or ``None`` to
        signal that the listing FAILED (API error).

        The ``None`` sentinel is deliberate and load-bearing: callers MUST
        distinguish "the API call failed" from "this node has zero pods".
        Returning ``[]`` for both would let a transient apiserver error look
        identical to an empty node, silently re-deriving every GPU's cap from
        a zero-pod view. ``reconcile_once`` keys its fail-safe (skip the cycle,
        freeze each GPU at its last-known-good cap) off this ``None`` — so do
        NOT collapse the failure path back to ``[]``.
        """
        try:
            field_selector = (
                f"spec.nodeName={self.node_name}" if self.node_name else None
            )
            # TODO(#9682 follow-up): this polls a full pod LIST per agent every
            # RECONCILE_INTERVAL_S. Even with the node field-selector that is one
            # apiserver request per node per cycle, so aggregate request rate
            # grows linearly with cluster size (~N/interval LISTs/s fleet-wide:
            # ~66/s at 1000 nodes, ~330/s at 5000). It will not surface in tests
            # or small clusters, only at production scale. The real fix is a
            # watch/informer-backed local pod cache (one initial LIST + a
            # streamed watch per node, as kubelet does) so steady-state cost is
            # N idle watch connections instead of N LISTs every cycle. Tracked
            # for a follow-up PR; see PR #9682 @sttts review.
            #
            # Interim mitigation: resource_version="0" lets the apiserver serve
            # the LIST from its watch cache instead of reading through to etcd,
            # which relieves etcd pressure (it does NOT change the request-rate
            # shape). The tradeoff is "Any" list consistency: the result may be
            # slightly stale and is not a quorum-consistent "most recent" read
            # (https://kubernetes.io/docs/reference/using-api/api-concepts/#semantics-for-list-and-watch).
            # That is acceptable for this MVP because reconcile is periodic, live
            # GPU ownership is still checked from host PIDs each cycle, and a
            # stale pod view delays convergence rather than changing the
            # failure-path contract.
            # Bound the LIST on BOTH sides so a stuck/throttled apiserver
            # substantially limits the delay before graceful shutdown cleanup:
            #   * timeout_seconds — apiserver-side LIST deadline;
            #   * _request_timeout — client-side (connect, read) deadline for
            #     stalls the server timeout can't cover.
            # A timeout raises, which the `except` below maps to the `None`
            # fail-safe (skip the cycle, freeze last-known-good caps). Transport
            # retries are disabled on this client (see `_build_k8s_core_v1`), so
            # a timeout is not silently multiplied into extra apiserver load.
            if self.k8s_namespace:
                result = self._core_v1.list_namespaced_pod(
                    namespace=self.k8s_namespace,
                    field_selector=field_selector,
                    resource_version="0",
                    timeout_seconds=K8S_LIST_SERVER_TIMEOUT_S,
                    _request_timeout=K8S_LIST_CLIENT_TIMEOUT_S,
                )
            else:
                result = self._core_v1.list_pod_for_all_namespaces(
                    field_selector=field_selector,
                    resource_version="0",
                    timeout_seconds=K8S_LIST_SERVER_TIMEOUT_S,
                    _request_timeout=K8S_LIST_CLIENT_TIMEOUT_S,
                )
            return result.items
        except Exception as e:
            # Explicit failure result — see the contract in the docstring.
            # Returning None (not []) is what keeps the reconcile fail-safe.
            logger.warning("Failed to list pods on node: %s", e)
            return None

    def _build_uid_to_annotation(self, pods: list) -> dict[str, Optional[str]]:
        """Map pod UID → power-limit annotation value, for opted-in pods only.

        Scope-by-annotation-key: a pod is in scope **only** if it actually
        carries ``POWER_ANNOTATION_KEY``. Pods without the key are omitted
        from the map entirely.

        This omission is load-bearing on shared/multi-tenant nodes.
        ``_reconcile_gpu`` decides whether a GPU is managed by testing
        ``uid in uid_to_annotation``; if an unannotated pod were added here
        with a ``None`` value, a GPU running only that pod would still build a
        non-empty ``pod_annotations`` and fall through to the "no parseable
        annotation → safe default" branch in ``_resolve_cap_for_gpu`` — i.e.
        the agent would silently power-cap a co-located non-Dynamo workload (or
        a Dynamo worker the planner has not yet annotated). Gating on key
        presence is what keeps the agent from touching GPUs it was never asked
        to manage. The planner is the sole writer of this key and stamps it
        only on prefill/decode worker pods. Do NOT reintroduce unannotated pods
        with a ``None`` value.

        A pod that carries the key but with a malformed/empty value IS kept
        (value as-is) so the safe-default fail-safe still applies to a
        genuinely-managed pod whose annotation is broken.
        """
        result: dict[str, Optional[str]] = {}
        for pod in pods:
            annotations = pod.metadata.annotations or {}
            if POWER_ANNOTATION_KEY in annotations:
                result[pod.metadata.uid] = annotations[POWER_ANNOTATION_KEY]
        return result

    def reconcile_once(self) -> None:
        """Run one reconcile cycle: list pods, map PIDs→UIDs, apply caps.

        On Kubernetes API failure during the pod list we skip the cycle
        rather than treating the apiserver outage as "no pods on this
        node" (which would silently drop enforcement for the duration of
        the outage). Previously-applied caps remain live; a NEW pod
        arriving during the outage runs at whatever cap was last set on
        its GPU. Operators should alert on
        `k8s_list_failures_total > 0 over 5m`.
        """
        # Shutdown fast-path: if SIGTERM already landed, do NOT start a new
        # cycle. A cycle issues a pod LIST (network I/O that can block under
        # apiserver throttling) and GPU writes; starting one here would delay
        # `run()`'s `finally` cleanup and risk kubelet SIGKILL before caps are
        # restored. `run()` still executes `_shutdown_cleanup` from its
        # `finally`, so returning early loses no restoration work.
        if _shutdown.is_set():
            logger.info("Shutdown requested — skipping reconcile cycle.")
            return

        # Flush any release OR acquisition whose durable write failed on a
        # previous cycle BEFORE listing pods, so it retries even during a
        # Kubernetes API outage (the retry touches only the state volume, not
        # the apiserver).
        _flush_pending_retirements()
        _flush_pending_acquisitions()

        pods = self._list_pods_on_node()
        if pods is None:
            # Fail-safe: the pod listing failed (API error), so we have no
            # trustworthy view of which pods own which GPUs this cycle. We
            # deliberately SKIP the reconcile rather than proceed with an
            # empty view — skipping freezes each GPU at its last-known-good
            # cap until the next successful cycle, which is strictly safer
            # than un-capping or re-deriving caps from a zero-pod snapshot.
            # The cap state lives on the GPU (NVML) and the agent's managed
            # set, so a skipped cycle loses nothing.
            self.metrics.k8s_list_failures_total.inc()
            logger.error(
                "Kubernetes pod-list failed; skipping reconcile cycle to "
                "preserve last-known-good caps. Previously-applied caps "
                "remain in effect; alert on k8s_list_failures_total > 0 "
                "over 5m."
            )
            return
        uid_to_annotation = self._build_uid_to_annotation(pods)

        # Re-snapshot the device count every cycle rather than trusting the
        # value cached at startup. A DCGM hostengine reconnect rebuilds the
        # discovered-GPU set, so the count can change at runtime: if it GREW,
        # a startup-frozen count would never reconcile the new GPUs; if it
        # SHRANK, iterating the stale (larger) range raises per-index errors.
        # Best-effort — on a transient read failure keep the last-known count
        # for this cycle rather than skipping enforcement entirely.
        try:
            self.device_count = self._actuator.device_count()
        except Exception as e:
            logger.warning(
                "Could not refresh GPU count this cycle; using last-known %d: %s",
                self.device_count,
                e,
            )

        for gpu_idx in range(self.device_count):
            # Stop enforcing the moment shutdown is requested: cleanup runs from
            # run()'s finally only after this cycle returns, so continuing
            # through the remaining GPUs' (possibly slow) DCGM/NVML calls could
            # eat the pod's termination grace period before any cap is restored.
            # Breaking hands control back to the finally after the current GPU.
            if _shutdown.is_set():
                logger.info(
                    "Shutdown requested mid-reconcile; stopping the GPU loop "
                    "before GPU %d (and any later) so cleanup can restore caps "
                    "within the grace period.",
                    gpu_idx,
                )
                break
            try:
                self._reconcile_gpu(gpu_idx, uid_to_annotation)
            except Exception as e:
                logger.error("Reconcile failed for GPU %d: %s", gpu_idx, e)

    def _reconcile_gpu(
        self,
        gpu_idx: int,
        uid_to_annotation: dict[str, Optional[str]],
    ) -> None:
        """Apply the policy-resolved cap for one GPU via the active actuator.

        Routes through `self._actuator.list_running_pids` and
        `self._actuator.apply_cap` instead of inline `pynvml`. On
        `actuator: dcgm` this means the cap write actually flows through
        `nvidia-dcgm` via `dcgmConfigSet`, which is the entire point of
        selecting that actuator. An earlier revision hard-coded
        `pynvml.nvmlDeviceGetHandleByIndex` + module-level `_apply_cap`
        in the reconcile loop, so `agent.actuator=dcgm` only changed
        cold-start orphan recovery — the steady-state cap-write path
        silently used NVML regardless.

        The PID read still happens through the actuator because
        `DcgmActuator.list_running_pids` performs the UUID-keyed
        cross-library identity lookup (DCGM gpuId -> UUID -> NVML index)
        before calling `pynvml.nvmlDeviceGetComputeRunningProcesses`.
        Bypassing the actuator here would skip that lookup and read PIDs
        from the wrong physical GPU on any node where DCGM and NVML
        disagree on enumeration order — see actuator.py
        `_ensure_identity_map`.

        Caps are persistent by design: when a managed GPU is merely idle this
        cycle (no processes at all) we deliberately DO NOT restore default TGP
        here. A managed worker may exit briefly (OOM, reschedule) and return to
        the same GPU; restoring during that gap would violate the planner's
        power budget, and the planner owns cap lifecycle via annotation
        removal/update.

        A cap IS restored to default in three places:
          - ``_release_managed_gpu`` during ordinary reconcile, when a GPU we
            previously capped now runs only unannotated / non-K8s processes
            (the opted-in pod is gone and a non-managed tenant owns the GPU,
            or the planner removed the annotation);
          - ``_shutdown_cleanup`` at agent shutdown (invoked from ``run()``
            after SIGTERM); and
          - ``_restore_orphaned_gpus_on_startup`` (previously-managed +
            now-idle GPUs at agent start).
        Per PR #9682 @sttts review.
        """
        # Anchor the whole policy decision to one physical GPU. Capture the
        # identity BEFORE the PID snapshot that feeds the cap, then pass it to
        # `apply_cap`, which re-verifies it in-transaction immediately before
        # the Set. Without this anchor, a DCGM reconnect between PID
        # attribution here and the identity capture inside `apply_cap` could
        # re-enumerate the index and apply GPU-A's workload-derived cap to
        # GPU-B (both reads self-consistently see GPU-B). If the identity is
        # unreadable now we cannot safely attribute PIDs or a cap to this
        # index at all, so skip the GPU this cycle and retry next reconcile —
        # fail closed. The actuator re-verification turns a
        # mid-sequence re-enumeration into a skipped write rather than a
        # misapplied cap.
        try:
            expected_uuid = self._actuator.get_uuid(gpu_idx)
        except Exception as e:
            logger.warning(
                "Skipping reconcile for GPU %d this cycle: could not read its "
                "identity to anchor the cap decision (retrying next cycle): %s",
                gpu_idx,
                e,
            )
            return

        # Bind the PID snapshot to the anchored identity: on the DCGM path a
        # re-enumeration could otherwise attribute a different GPU's workload to
        # this cap, which the identity-guarded `apply_cap` would then write onto
        # the anchored GPU (its guard only checks the write destination). Bound
        # here so the whole attribution -> cap -> write sequence refers to ONE
        # physical GPU. A mismatch means the anchored GPU is no longer
        # resolvable, so skip this cycle and retry next reconcile — fail closed
        # .
        try:
            pids = self._actuator.list_running_pids(
                gpu_idx, expected_uuid=expected_uuid
            )
        except _GpuIdentityMismatch as e:
            logger.warning(
                "Skipping reconcile for GPU %d this cycle: the PID snapshot "
                "could not be bound to the anchored identity %s "
                "(re-enumeration or hot-unplug); retrying next cycle: %s",
                gpu_idx,
                expected_uuid,
                e,
            )
            return
        if not pids:
            return  # no K8s workload on this GPU

        # Deduplicate by pod UID before building `pod_annotations`. A
        # single pod commonly runs multiple GPU processes (one per rank
        # in a TP/PP/EP topology, helper workers, profilers, etc.); the
        # pre-fix code emitted one entry per PID and would treat a
        # one-pod / two-PID GPU as if two pods were colocated. That
        # both fired the spurious "multi-pod-per-GPU" WARNING and, when
        # the pod's annotation was missing/invalid, took the
        # conflict-resolution branch in `_resolve_cap_for_gpu` (since
        # `len(pod_annotations) > 1` was true), incorrectly applying
        # safe_default + bumping multi_pod_gpu_total. Per PR #9682
        # CodeRabbit review (power_agent.py:636).
        seen_uids: set[str] = set()
        pod_annotations: list[tuple[str, Optional[str]]] = []
        for pid in pids:
            uid = _extract_pod_uid_from_cgroup(pid)
            if uid is None:
                continue  # non-K8s process — skip
            if uid in seen_uids:
                continue  # already counted this pod via an earlier PID
            if uid in uid_to_annotation:  # opted-in: carries POWER_ANNOTATION_KEY
                seen_uids.add(uid)
                pod_annotations.append((uid, uid_to_annotation[uid]))

        if not pod_annotations:
            # No opted-in pod owns this GPU (every process is either non-K8s or
            # belongs to a pod without POWER_ANNOTATION_KEY). Two sub-cases,
            # both handled by _release_managed_gpu's UUID-gated eligibility:
            #   * never managed by us → left at hardware default (the scope
            #     boundary — see _build_uid_to_annotation).
            #   * previously managed by us (this process OR a prior one, via the
            #     persisted UUID set) → the opted-in pod is gone and a
            #     non-managed workload now runs here, so release our cap rather
            #     than strand it on the new tenant until shutdown.
            # The idle case (no processes) is handled by the `not pids` branch
            # above, which keeps the cap for a briefly-exited worker.
            # Pass the pre-snapshot identity so the release only acts on the
            # GPU whose (absent) annotated workload we actually observed — a
            # re-enumeration that swapped this index's occupant is detected and
            # the stale projection skipped.
            _release_managed_gpu(self._actuator, gpu_idx, expected_uuid=expected_uuid)
            return

        cap_w = _resolve_cap_for_gpu(
            gpu_idx, pod_annotations, self.safe_default_watts, self.metrics
        )
        # Pass the pre-snapshot identity so the actuator re-verifies it is
        # still the GPU on this index before writing; a re-enumeration
        # anywhere across attribution → resolve → write is detected and the
        # write skipped.
        self._actuator.apply_cap(gpu_idx, cap_w, expected_uuid=expected_uuid)

    def run(self) -> None:
        """Main reconcile loop. Blocks until SIGTERM, then cleans up once.

        The signal handler only sets `_shutdown`; the shutdown restore + sweep
        run here from the `finally`, after the in-flight reconcile has fully
        returned, so cleanup never races an in-flight cap write.
        """
        signal.signal(signal.SIGTERM, _handle_sigterm)
        signal.signal(signal.SIGINT, _handle_sigterm)

        logger.info(
            "Power Agent started. Node=%s, safe_default=%dW, interval=%ds",
            self.node_name or "(all)",
            self.safe_default_watts,
            RECONCILE_INTERVAL_S,
        )

        try:
            while not _shutdown.is_set():
                try:
                    self.reconcile_once()
                except Exception as e:
                    logger.exception("Unexpected error in reconcile loop: %s", e)
                _shutdown.wait(timeout=RECONCILE_INTERVAL_S)
        finally:
            _shutdown_cleanup(self._actuator)

        logger.info("Power Agent shut down.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _make_actuator(args, metrics) -> Actuator:
    """Construct the actuator declared by `--actuator`.

    Strict binary choice — `nvml` or `dcgm`. There is no auto-detection
    and no runtime probe. The operator
    declares the actuator at chart-install time based on whether their
    cluster runs `nvidia-dcgm`; this function honors that declaration
    without modification. argparse's `choices=` guarantees `args.actuator`
    is one of the two values below, but we re-check defensively so a
    future refactor that loosens the choices doesn't silently no-op.
    """
    if args.actuator == "nvml":
        return NvmlActuator(metrics=metrics)
    if args.actuator == "dcgm":
        return DcgmActuator(
            host=args.dcgm_host,
            port=args.dcgm_port,
            metrics=metrics,
        )
    raise ValueError(f"Unknown actuator {args.actuator!r}; expected 'nvml' or 'dcgm'.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Dynamo Power Agent DaemonSet")
    parser.add_argument(
        "--safe-default-watts",
        type=int,
        required=True,
        help="Per-GPU fail-closed cap (watts) applied when annotation parsing fails.",
    )
    parser.add_argument(
        "--node-name",
        type=str,
        default=os.environ.get("NODE_NAME", ""),
        help="K8s node name (defaults to NODE_NAME env var).",
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default=None,
        help="Restrict pod watch to this K8s namespace. Default: all namespaces.",
    )
    parser.add_argument(
        "--prometheus-port",
        type=int,
        default=int(os.environ.get("PROMETHEUS_PORT", "0")),
        help="Port for Prometheus metrics (0 = disabled).",
    )
    parser.add_argument(
        "--actuator",
        choices=["nvml", "dcgm"],
        default="nvml",
        help=(
            "Power-cap actuator. 'nvml' (default) calls "
            "nvmlDeviceSetPowerManagementLimit directly — used on clusters "
            "where the GPU Operator runs with dcgm.enabled=false (the "
            "upstream default). 'dcgm' connects to the operator-managed "
            "nvidia-dcgm hostengine via TCP and uses dcgmConfigSet — used "
            "on clusters where the operator set dcgm.enabled=true. The "
            "two are mutually exclusive: a given chart deployment uses "
            "exactly one. The chart's agent.actuator value is the single "
            "source of truth; no auto-detection."
        ),
    )
    parser.add_argument(
        "--dcgm-host",
        type=str,
        default=DcgmActuator.DEFAULT_HOST,
        help=(
            "DCGM hostengine host. Default matches the upstream GPU "
            "Operator's nvidia-dcgm Service. Only consulted when "
            "--actuator=dcgm."
        ),
    )
    parser.add_argument(
        "--dcgm-port",
        type=int,
        default=DcgmActuator.DEFAULT_PORT,
        help=(
            "DCGM hostengine port. Default matches the upstream nvidia-dcgm "
            "hostPort. Only consulted when --actuator=dcgm."
        ),
    )

    args = parser.parse_args()

    agent = PowerAgent(
        safe_default_watts=args.safe_default_watts,
        node_name=args.node_name,
        k8s_namespace=args.namespace,
        prometheus_port=args.prometheus_port,
        actuator_factory=lambda metrics: _make_actuator(args, metrics),
    )
    agent.run()


if __name__ == "__main__":
    # Launched as `python /app/power_agent.py`, this file is module `__main__`
    # while the actuator reaches the agent via `import power_agent` — two
    # distinct module objects. That is SAFE here because cross-module mutable
    # state lives in `managed_state` (imported by canonical name from both),
    # so the two module copies' `_managed_gpu_indices` / `_previously_managed`
    # / `_pending_acquisition` aliases converge on one set. See managed_state.py.
    main()
