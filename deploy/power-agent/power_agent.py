#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Power Agent DaemonSet — Phase 1 implementation.

Runs as a privileged DaemonSet (hostPID: true) on each GPU node. Every 15s:
  1. Lists pods on this node via the K8s API.
  2. For each physical GPU: nvmlDeviceGetComputeRunningProcesses() → PID list.
  3. For each PID: reads /proc/{pid}/cgroup → extracts pod UID.
  4. Looks up the pod's dynamo.nvidia.com/gpu-power-limit annotation.
  5. Calls nvmlDeviceSetPowerManagementLimit(handle, watts × 1000).

Scope is opt-in: the agent only ever caps a GPU whose pod carries the
dynamo.nvidia.com/gpu-power-limit annotation (set by the planner on
prefill/decode worker pods). A GPU running only unannotated pods — a
non-Dynamo workload, or a Dynamo worker not yet annotated — that the agent
never capped is left at its hardware default and untouched. If the agent had
previously capped that GPU and the opted-in pod is now gone (a non-managed
workload reuses it, or the planner removed the annotation), the cap is
released back to default so it does not strand on the new tenant. See
``_build_uid_to_annotation`` and ``_release_managed_gpu``.

SIGTERM handler: restores default TDP on all managed GPUs before shutdown.
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
from actuator import Actuator, DcgmActuator, NvmlActuator

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


def _load_previously_managed_gpus() -> set[str]:
    """Load the persisted set of UUIDs this agent previously capped.

    Defensive parsing — corrupt / malformed state files must never crash
    the agent's startup. Per PR #9682 CodeRabbit review, this catches a
    superset of the original (FileNotFoundError, JSONDecodeError) cases:

      * OSError (PermissionError, IsADirectoryError, NotADirectoryError,
        I/O errors) — disk problems on the host volume should NOT brick
        the agent. Returning empty means we lose the orphan-recovery
        opportunity for this restart, which is strictly better than
        CrashLoopBackOff with no caps actuated.
      * Non-dict JSON root — a file containing a top-level list / int /
        string / null would have crashed `.get(...)` with AttributeError.
      * Non-list `managed_uuids` — a misshapen value would have crashed
        `set(...)` (int) or silently iterated characters (string).
      * Non-string entries — bytes / ints / None inside the list could
        flow through to downstream `in _previously_managed` checks
        comparing against `str` UUIDs and silently never match. Coerce
        the type guard at the boundary instead.

    Every malformed-state branch logs at WARNING and returns ``set()``
    so the operator can spot the corruption in pod logs without losing
    cap-write availability.
    """
    try:
        with open(_MANAGED_STATE_PATH) as f:
            raw = json.load(f)
    except FileNotFoundError:
        return set()
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(
            "Failed to read managed-GPU state at %s (%s: %s); "
            "treating as empty. Orphan recovery will skip this startup.",
            _MANAGED_STATE_PATH,
            type(e).__name__,
            e,
        )
        return set()

    if not isinstance(raw, dict):
        logger.warning(
            "Managed-GPU state at %s has unexpected root type %s "
            "(expected object); treating as empty.",
            _MANAGED_STATE_PATH,
            type(raw).__name__,
        )
        return set()

    uuids = raw.get("managed_uuids", [])
    if not isinstance(uuids, list):
        logger.warning(
            "Managed-GPU state at %s has unexpected managed_uuids type "
            "%s (expected list); treating as empty.",
            _MANAGED_STATE_PATH,
            type(uuids).__name__,
        )
        return set()

    # Count invalid entries directly rather than from len(set) vs
    # len(list) (PR9790 review): the set comprehension deduplicates,
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
    return valid


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


def _record_managed_gpu_by_uuid(uuid: str) -> None:
    """Library-agnostic UUID persistence helper.

    Called by both actuator paths after a successful cap write. The UUID
    is the hardware-level identifier, identical whether obtained from
    NVML (`nvmlDeviceGetUUID`) or DCGM (`DCGM_FI_DEV_UUID`). Separating
    the persistence from the UUID source means DcgmActuator (PR B) can
    record state without reaching into the NVML helpers.
    """
    if uuid not in _previously_managed:
        _previously_managed.add(uuid)
        _persist_managed_gpus(_previously_managed)


def _record_managed_gpu_uuid(handle) -> None:
    """Called from _apply_cap() after every successful NVML write."""
    _record_managed_gpu_by_uuid(_nvml_uuid(handle))


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
                "Power cap currently applied per physical GPU (watts).",
                labelnames=("gpu",),
            )
            self.multi_pod_gpu_total = Counter(
                "dynamo_power_agent_multi_pod_gpu_total",
                "Times a physical GPU had multiple pods (agree or conflict).",
                labelnames=("disposition",),
            )
            self.apply_failures_total = Counter(
                "dynamo_power_agent_apply_failures_total",
                "Times an actuator write (NVML nvmlDeviceSetPowerManagementLimit "
                "or DCGM dcgmConfigSet) raised — the cap was NOT applied to the "
                "GPU. Distinct from policy fallbacks (tracked by "
                "safe_default_applied_total) where the cap IS applied at safe-default.",
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
            # DCGM-only. Distinct from apply_failures_total because the
            # underlying dcgmConfigSet succeeded (cap IS live on the GPU)
            # — only the optional dcgmConfigEnforce that registers it as
            # DCGM's target configuration for auto-reapply-after-reset
            # failed. Stays at 0 on actuator=nvml and on
            # actuator=dcgm + agent.dcgm.enforce=false.
            self.dcgm_enforce_failures_total = Counter(
                "dynamo_power_agent_dcgm_enforce_failures_total",
                "Times dcgmConfigEnforce failed AFTER a successful dcgmConfigSet "
                "(cap is live and tracked; auto-reapply-after-GPU-reset is not).",
            )
            # Per PR9790 Codex adversarial review (finding #3). Pre-fix
            # `_list_pods_on_node` swallowed every API error and returned
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
            self.dcgm_enforce_failures_total = noop
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
# place only; the SIGTERM handler and the actuator must see the same set.
_managed_gpu_indices: set[int] = managed_state.managed_gpu_indices


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


def _release_managed_gpu(actuator: Actuator, gpu_idx: int) -> None:
    """Restore default TGP on a GPU we previously capped, and unmanage it.

    Runtime counterpart to ``_handle_sigterm`` / ``_restore_orphaned_gpus_on_startup``.
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
    driver-level cap — the same reason ``_handle_sigterm`` was lifted onto the
    actuator surface in v1.6. Routing it through raw NVML here would desync the
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
    """
    try:
        uuid = actuator.get_uuid(gpu_idx)
    except Exception as e:
        logger.warning(
            "Failed to read UUID for GPU %d during release check: %s", gpu_idx, e
        )
        return
    if gpu_idx not in _managed_gpu_indices and uuid not in _previously_managed:
        return  # not a GPU this agent capped — leave it alone (UUID-gating)

    # UUID to prune from the persisted set once the release succeeds. On the
    # ``dcgm`` path a hostengine re-enumeration can move the originally-capped
    # GPU to a different index, and ``restore_default`` relocates the write by
    # the recorded UUID (``actuator._resolve_managed_idx``). Pruning by ``uuid``
    # above — the CURRENT occupant of ``gpu_idx`` — would then drop the wrong
    # entry and strand the GPU we actually restored in ``managed_gpus.json``.
    # Mirror ``_handle_sigterm``: prune the originally-managed UUID via
    # ``managed_uuid_for_idx``. NVML indices are stable within a process, so the
    # current UUID already matches and the helper is absent there.
    managed_uuid = uuid
    if hasattr(type(actuator), "managed_uuid_for_idx"):
        try:
            managed_uuid = getattr(actuator, "managed_uuid_for_idx")(gpu_idx)
        except Exception as e:
            logger.warning(
                "Could not resolve managed UUID for GPU %d during release; "
                "falling back to current UUID for state pruning: %s",
                gpu_idx,
                e,
            )

    try:
        default_w = actuator.default_w(gpu_idx)
        current_w = actuator.current_w(gpu_idx)
        # Attempt the restore when EITHER the current index still shows a live
        # cap (``current_w < default_w``) OR a dcgm re-enumeration relocated the
        # GPU we capped to a different index (``managed_uuid != uuid``). The
        # ``current_w``/``default_w`` probe reads the CURRENT occupant of
        # ``gpu_idx``; in the relocation case that occupant (``uuid``) can sit
        # "already at default" while the GPU we actually manage (``managed_uuid``)
        # is still capped at its new index. Gating solely on
        # ``current_w < default_w`` would then skip the restore yet still prune
        # ``managed_uuid`` below, stranding that live cap permanently.
        # ``restore_default`` relocates by the recorded UUID
        # (``_resolve_managed_idx``) and is idempotent, so a redundant call when
        # the managed GPU is already at default is a harmless no-op write.
        if managed_uuid != uuid or current_w < default_w:
            # ``restore_default`` returns False when it could not conclusively
            # locate the managed GPU (e.g. a dcgm re-enumeration it cannot
            # resolve → ``_resolve_managed_idx`` returns None). The cap is then
            # still LIVE, so keep our ownership state and let a later reconcile
            # or the next startup's orphan recovery retry — never prune here, or
            # we lose the only record that the GPU still needs restoring. This
            # mirrors ``_handle_sigterm``'s ``is False`` guard.
            if actuator.restore_default(gpu_idx) is False:
                logger.warning(
                    "Skipped cap release for GPU %d via %s actuator (managed "
                    "GPU not conclusively located); leaving it managed so a "
                    "later cycle retries.",
                    gpu_idx,
                    actuator.name,
                )
                return
            logger.info(
                "Released cap on GPU %d (managed UUID %s, index observed %d W / "
                "default %d W): previously managed, now running only "
                "unannotated/non-K8s processes.",
                gpu_idx,
                managed_uuid,
                current_w,
                default_w,
            )
    except Exception as e:
        # Leave the GPU in the managed set so a later cycle retries the release.
        logger.warning("Failed to release cap on GPU %d: %s", gpu_idx, e)
        return
    _managed_gpu_indices.discard(gpu_idx)
    if managed_uuid in _previously_managed:
        _previously_managed.discard(managed_uuid)
        _persist_managed_gpus(_previously_managed)


# ---------------------------------------------------------------------------
# SIGTERM handler
# ---------------------------------------------------------------------------

_shutdown = threading.Event()

# Module-level reference to the active actuator. Populated by
# `PowerAgent.__init__` (line ~478, immediately after `self._actuator.init()`),
# read by the module-level `_handle_sigterm` because Python's `signal.signal`
# hands the handler a `(signum, frame)` tuple with no other context. v1.6
# wiring per review comment #6 (SIGTERM previously bypassed the actuator and
# went straight to `pynvml`, leaving DCGM's target-config record stale on the
# `actuator: dcgm` path).
_active_actuator: Optional[Actuator] = None


def _handle_sigterm(signum, frame):
    """Restore default TGP on managed GPUs via the active actuator, then shut down.

    Dispatches through `_active_actuator` so that:
      - On `actuator: nvml`, `NvmlActuator.restore_default` runs the same
        `nvmlDeviceSetPowerManagementLimit(default)` call the pre-v1.6 inline
        handler did. Externally observable behaviour is unchanged on the NVML
        path (`test_shutdown.py` covers this).
      - On `actuator: dcgm`, `DcgmActuator.restore_default` runs
        `dcgmConfigSet(mPowerLimit.val=default)` so the hostengine's
        "target configuration" record stays consistent with the driver-level
        cap. Pre-v1.6 the raw-NVML write would have desynced them: the driver
        cap returns to default, but DCGM still holds the old cap as target
        config, and DCGM would re-apply the *old* cap after the next GPU
        reset/reinit. With `enforce: true` that mismatch was particularly
        nasty because the auto-reapply specifically uses the (now-stale)
        target config.

    Defensive fallback: if `_active_actuator` is None (SIGTERM fires before
    `PowerAgent.__init__` finished registering), we go through raw NVML so the
    GPU isn't left at a custom cap — better to ungracefully restore via the
    wrong library than to abandon a cap'd GPU.
    """
    logger.info(
        "SIGTERM received — restoring default TGP on managed GPUs and shutting down."
    )
    actuator = _active_actuator
    for gpu_idx in list(_managed_gpu_indices):
        # Record the UUID of GPUs we successfully restore so we can prune
        # them from `_previously_managed` after the loop. Without this,
        # `managed_gpus.json` retains the stale UUID across restarts and
        # startup orphan recovery would later "restore" a GPU this agent
        # no longer owns — clobbering a cap applied by another workflow
        # (different DGD, manual `nvidia-smi -pl`, vendor defaults).
        restored_uuid: Optional[str] = None
        try:
            if actuator is not None:
                restore_result = actuator.restore_default(gpu_idx)
                if restore_result is False:
                    logger.warning(
                        "Skipped default TGP restore for GPU %d via %s actuator "
                        "(managed GPU no longer visible)",
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
                        restored_uuid = getattr(actuator, "managed_uuid_for_idx")(
                            gpu_idx
                        )
                    else:
                        restored_uuid = actuator.get_uuid(gpu_idx)
                except Exception as e:
                    # UUID lookup failure post-restore is benign: the GPU
                    # is already at default, so a stale entry in
                    # managed_gpus.json just means the next startup's
                    # orphan recovery sees `current_w >= default_w` and
                    # skips the redundant write. Log so it's visible.
                    logger.warning(
                        "Could not resolve UUID for restored GPU %d "
                        "(state file may retain stale entry): %s",
                        gpu_idx,
                        e,
                    )
            else:
                # Fallback: actuator not yet registered. Should be rare —
                # only happens if SIGTERM fires during PowerAgent.__init__
                # before the `_active_actuator = self._actuator` line runs.
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)
                default_mw = pynvml.nvmlDeviceGetPowerManagementDefaultLimit(handle)
                pynvml.nvmlDeviceSetPowerManagementLimit(handle, default_mw)
                logger.warning(
                    "Restored GPU %d via raw NVML (actuator not registered "
                    "yet at SIGTERM time)",
                    gpu_idx,
                )
                try:
                    restored_uuid = _nvml_uuid(handle)
                except Exception as e:
                    logger.warning(
                        "Could not resolve UUID for restored GPU %d via "
                        "raw NVML (state file may retain stale entry): %s",
                        gpu_idx,
                        e,
                    )
        except Exception as e:
            logger.exception("Failed to restore TGP on GPU %d: %s", gpu_idx, e)
            # Do NOT prune `_previously_managed` on failure — the cap may
            # still be live, and the next startup's orphan recovery is
            # our only chance to reset it.
            continue
        if restored_uuid is not None:
            _previously_managed.discard(restored_uuid)

    # UUID-complete safety net (PR9790 review follow-up). The index-keyed
    # loop above can MISS a still-capped GPU when DCGM re-enumerated and a
    # later reconcile re-capped that GPU's old index onto a *different*
    # physical GPU: `_managed_uuid_by_idx[old_idx]` gets overwritten and the
    # displaced GPU drops out of the index-keyed `_managed_gpu_indices`, so
    # the loop never visits it. Resolve each UUID we capped to its CURRENT
    # index instead. Without this, the displaced cap leaks whenever the agent
    # is removed without a restart (no future orphan recovery runs).
    #
    # Scope the sweep to `actuator.managed_uuids()` — the UUIDs THIS process
    # actually capped — NOT the persisted `_previously_managed` set. The
    # latter is cross-incarnation and can hold UUIDs that startup orphan
    # recovery KEPT but this process never capped (e.g. a GPU with a running
    # workload, skipped at `_restore_orphaned_gpus_on_startup`'s
    # `list_running_pids` guard). Sweeping those would reset a cap owned by
    # another workflow on shutdown; `current_w < default_w` prevents a
    # redundant write but is not an ownership guard.
    #
    # Only the DCGM actuator can relocate by UUID; NVML indices are stable
    # within a process so its index loop is already complete. Gate on the
    # FULL sweep surface — both `managed_uuids()` (the per-process ownership
    # set the sweep iterates) and `restore_default_by_uuid()` (the relocating
    # restore). `restore_default_by_uuid` alone is NOT a sufficient gate:
    # NvmlActuator gained it (for identity-stable orphan recovery) but
    # deliberately does NOT track capped UUIDs, so gating on it alone would
    # enter the sweep on NVML and then `AttributeError` on the missing
    # `managed_uuids()`, skipping persist/shutdown/_shutdown.set() (PR9790
    # review follow-up).
    if (
        actuator is not None
        and hasattr(type(actuator), "managed_uuids")
        and hasattr(type(actuator), "restore_default_by_uuid")
    ):
        for uuid in getattr(actuator, "managed_uuids")():
            try:
                sweep_result = getattr(actuator, "restore_default_by_uuid")(uuid)
            except Exception as e:
                # Keep the UUID: a write/relocation failure means the cap may
                # still be live, and the next startup's orphan recovery is our
                # only remaining chance to reset it.
                logger.exception(
                    "SIGTERM UUID sweep failed to restore managed UUID %s: %s",
                    uuid,
                    e,
                )
                continue
            # Prune ONLY when we actively restored a live below-default cap
            # (True), mirroring the index loop's invariant: "discard a UUID
            # only after restoring its GPU to default." A stale persisted
            # UUID that resolves to an already-at-default GPU, a GPU that
            # left the node, or one we could not locate conclusively
            # (None / False) is left in place — exactly as orphan recovery
            # leaves at-default UUIDs (`_restore_orphaned_gpus_on_startup`
            # only discards when `current_w < default_w`). This keeps the
            # sweep from aggressively dropping persisted UUIDs it did not
            # just act on, which `_previously_managed` legitimately holds
            # across incarnations.
            if sweep_result is True:
                _previously_managed.discard(uuid)

    # Persist the pruned state so the next startup's orphan recovery
    # only touches GPUs we still own. Failure to write is non-fatal:
    # log and proceed to shutdown.
    try:
        _persist_managed_gpus(_previously_managed)
    except Exception as e:
        logger.warning(
            "Failed to persist pruned managed_gpus state at SIGTERM: %s "
            "(next startup may briefly re-restore already-default GPUs).",
            e,
        )
    try:
        if actuator is not None:
            actuator.shutdown()
        else:
            pynvml.nvmlShutdown()
    except Exception:
        # We MUST proceed to `_shutdown.set()` so the run loop unblocks
        # and the container exits cleanly — re-raising here would leave
        # the agent hung on SIGTERM. But silently dropping the failure
        # made shutdown-time NVML/DCGM faults impossible to diagnose
        # from pod logs (PR #9682 CodeRabbit review on power_agent.py:355).
        # `logger.exception` writes the full traceback at ERROR level so
        # operators can correlate with hostengine / driver events.
        logger.exception(
            "Actuator/NVML shutdown raised; proceeding with agent exit anyway.",
        )
    _shutdown.set()


# ---------------------------------------------------------------------------
# Orphan cap restoration on startup (UUID-gated)
# ---------------------------------------------------------------------------


def _restore_orphaned_gpus_on_startup(actuator: Actuator) -> None:
    """Restore default TDP only on GPUs this agent previously capped AND that are now idle.

    Migrated from inline NVML to the actuator surface in v1.5 (Fix #5):
    on the DCGM path, orphan recovery must write through `nvidia-dcgm`
    too, not bypass it via raw NVML — otherwise the hostengine's
    target-configuration record (and its reset/reinit auto-reapply
    behaviour) drifts from the driver-level reality. Going through
    `actuator.restore_default` keeps a single write path per actuator.

    Two guards are preserved verbatim from the pre-v1.5 NVML-only
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
    reloaded = _load_previously_managed_gpus()
    _previously_managed.clear()
    _previously_managed.update(reloaded)
    for gpu_idx in range(actuator.device_count()):
        try:
            uuid = actuator.get_uuid(gpu_idx)
            if uuid not in _previously_managed:
                continue
            if actuator.list_running_pids(gpu_idx):
                continue  # workload running — let normal reconcile handle it
            current_w = actuator.current_w(gpu_idx)
            default_w = actuator.default_w(gpu_idx)
            if current_w < default_w:
                # UUID-stable restore. The cheap guards above (managed,
                # idle, below-default) are read at `gpu_idx`, but the WRITE
                # must resolve identity from the UUID we just confirmed: a
                # DCGM hostengine reconnect/re-enumeration inside the restore
                # write can move `gpu_idx` onto a different physical GPU, and
                # `_managed_uuid_by_idx` is empty at cold start so the index-
                # keyed `restore_default` cannot self-verify here. Writing by
                # index would then restore (and the discard below would prune)
                # the wrong GPU, clobbering an unrelated cap and leaking ours.
                # `restore_default_by_uuid` re-resolves the index at write time
                # so both land on the GPU that actually carries `uuid`.
                restore_result = actuator.restore_default_by_uuid(uuid)
                if not restore_result:
                    # None  -> nothing of ours to restore (already at/above
                    #          default, or the GPU is gone on a clean scan).
                    # False -> location inconclusive (a probe raised, e.g. a
                    #          transient outage); the GPU may still carry our
                    #          cap. Either way keep the UUID and retry on the
                    #          next startup rather than prematurely pruning it.
                    continue
                logger.info(
                    "Restored orphaned cap for idle managed GPU "
                    "(index %d at probe time, UUID %s, %d W → %d W).",
                    gpu_idx,
                    uuid,
                    current_w,
                    default_w,
                )
                _previously_managed.discard(uuid)
        except Exception as e:
            logger.warning("orphan-restore failed for GPU %d: %s", gpu_idx, e)
    _persist_managed_gpus(_previously_managed)


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

    Per PR9790 Codex adversarial review (finding #2): a multi-pod GPU
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

        # NVML init still happens here for the NVML path because
        # PR #9682's reconcile loop calls pynvml directly. The DCGM
        # path runs `pynvml.nvmlInit()` again inside `DcgmActuator.init()`
        # — `nvmlInit` is idempotent so the double call is harmless.
        pynvml.nvmlInit()

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

        # Register the actuator for the module-level SIGTERM handler
        # (v1.6, per review comment #6). signal.signal-registered callbacks
        # receive only (signum, frame) — they need a module-level handle to
        # reach this actuator, and we set it as soon as init() succeeds so
        # the window between actuator-ready and SIGTERM-handler-ready is as
        # short as possible. Tests may overwrite this to inject a mock.
        global _active_actuator
        _active_actuator = self._actuator

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
        self._core_v1 = k8s_client.CoreV1Api()

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
            if self.k8s_namespace:
                result = self._core_v1.list_namespaced_pod(
                    namespace=self.k8s_namespace,
                    field_selector=field_selector,
                    resource_version="0",
                )
            else:
                result = self._core_v1.list_pod_for_all_namespaces(
                    field_selector=field_selector,
                    resource_version="0",
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
        `k8s_list_failures_total > 0 over 5m`. Per PR9790 Codex
        adversarial review (finding #3).
        """
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

        for gpu_idx in range(self.device_count):
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

        v1.6 wiring per review comment #4: routes through
        `self._actuator.list_running_pids` and `self._actuator.apply_cap`
        instead of inline `pynvml`. On `actuator: dcgm` this means the
        cap write actually flows through `nvidia-dcgm` via `dcgmConfigSet`,
        which is the entire point of selecting that actuator. Pre-v1.6
        the reconcile loop hard-coded `pynvml.nvmlDeviceGetHandleByIndex`
        + module-level `_apply_cap`, so `agent.actuator=dcgm` only changed
        cold-start orphan recovery — the steady-state cap-write path
        silently used NVML regardless.

        The PID read still happens through the actuator because
        `DcgmActuator.list_running_pids` performs the v1.5 UUID-keyed
        cross-library identity lookup (DCGM gpuId -> UUID -> NVML index)
        before calling `pynvml.nvmlDeviceGetComputeRunningProcesses`.
        Bypassing the actuator here would skip that lookup and read PIDs
        from the wrong physical GPU on any node where DCGM and NVML
        disagree on enumeration order — see actuator.py
        `_ensure_identity_map`.

        Caps are persistent by design: when a GPU has no K8s workload this
        cycle we deliberately DO NOT restore default TDP here. A managed
        worker may exit briefly (OOM, reschedule) and return to the same
        GPU; restoring during that gap would violate the planner's power
        budget, and the planner owns cap lifecycle via annotation
        removal/update. The cap is only restored to default by
        ``_handle_sigterm`` (agent shutdown) and
        ``_restore_orphaned_gpus_on_startup`` (previously-managed +
        now-idle GPUs at agent start). Per PR #9682 @sttts review.
        """
        pids = self._actuator.list_running_pids(gpu_idx)
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
            _release_managed_gpu(self._actuator, gpu_idx)
            return

        cap_w = _resolve_cap_for_gpu(
            gpu_idx, pod_annotations, self.safe_default_watts, self.metrics
        )
        self._actuator.apply_cap(gpu_idx, cap_w)

    def run(self) -> None:
        """Main reconcile loop. Blocks until SIGTERM."""
        signal.signal(signal.SIGTERM, _handle_sigterm)
        signal.signal(signal.SIGINT, _handle_sigterm)

        logger.info(
            "Power Agent started. Node=%s, safe_default=%dW, interval=%ds",
            self.node_name or "(all)",
            self.safe_default_watts,
            RECONCILE_INTERVAL_S,
        )

        while not _shutdown.is_set():
            try:
                self.reconcile_once()
            except Exception as e:
                logger.exception("Unexpected error in reconcile loop: %s", e)
            _shutdown.wait(timeout=RECONCILE_INTERVAL_S)

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
            enforce=args.dcgm_enforce,
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

    def _parse_bool_strict(x: str) -> bool:
        """Strict bool parser — `treu` (typo) errors out, doesn't silently → False.

        Pre-v1.6 used `str(x).lower() in ("true","1","yes")` which mapped any
        unknown string to False, so `--dcgm-enforce treu` silently produced
        enforce=False and the operator got no feedback on the typo. argparse
        propagates the ArgumentTypeError as a parser exit-with-error, which is
        what we want at chart-install / `kubectl describe pod` time.
        """
        s = str(x).strip().lower()
        truthy = {"true", "1", "yes", "on"}
        falsy = {"false", "0", "no", "off"}
        if s in truthy:
            return True
        if s in falsy:
            return False
        raise argparse.ArgumentTypeError(
            f"--dcgm-enforce expects one of " f"{sorted(truthy | falsy)!r}; got {x!r}"
        )

    parser.add_argument(
        "--dcgm-enforce",
        type=_parse_bool_strict,
        default=False,
        help=(
            "Call dcgmConfigEnforce after each dcgmConfigSet. Default false "
            "(set-and-forget, matches NVML's semantics). Set true to "
            "register the cap as DCGM's target configuration so the "
            "hostengine re-applies it automatically after a GPU reset or "
            "reinit (DcgmConfigManager.h:113-117). This is the only "
            "automatic re-enforcement DCGM provides; it is NOT a "
            "tick-driven loop and does NOT make the cap survive Power "
            "Agent restart (the agent's SIGTERM handler restores default "
            "on every managed GPU regardless of --dcgm-enforce). Cost: "
            "one extra DCGM RPC per agent reconcile per GPU. Recommended "
            "for sites that see frequent GPU resets (XID-driven recovery, "
            "partition rebuilds, manual nvidia-smi --gpu-reset)."
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
    # distinct module objects. That is SAFE here because all shared mutable
    # state lives in `managed_state` (imported by canonical name from both),
    # so the two module copies' `_managed_gpu_indices` / `_previously_managed`
    # aliases converge on one set. See managed_state.py.
    main()
