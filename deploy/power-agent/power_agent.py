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
from typing import Optional

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
_MANAGED_STATE_PATH = "/var/lib/dynamo-power-agent/managed_gpus.json"

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

_previously_managed: set[str] = set()


def _load_previously_managed_gpus() -> set[str]:
    """Load the persisted set of UUIDs this agent previously capped.

    Defensive parsing — corrupt / malformed state files must never crash
    the agent's startup. Per PR #9682 CodeRabbit review, this catches a
    superset of the original (FileNotFoundError, JSONDecodeError) cases:

      * OSError (PermissionError, IsADirectoryError, I/O errors) — disk
        problems on the host volume should NOT brick the agent.
      * Non-dict JSON root — a file with a top-level list / int / string
        / null would have crashed ``.get(...)`` with ``AttributeError``.
      * Non-list ``managed_uuids`` — a misshapen value would have crashed
        ``set(non_iterable)`` with ``TypeError``.

    Returning empty means we lose the orphan-recovery opportunity for
    this restart, which is strictly better than CrashLoopBackOff with
    no caps actuated.
    """
    try:
        with open(_MANAGED_STATE_PATH) as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Failed to read managed GPU state: %s", e)
        return set()
    if not isinstance(payload, dict):
        logger.warning(
            "Managed GPU state has non-dict root: %s", type(payload).__name__
        )
        return set()
    uuids = payload.get("managed_uuids", [])
    if not isinstance(uuids, list):
        logger.warning("managed_uuids is not a list: %s", type(uuids).__name__)
        return set()
    return set(uuids)


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


def _record_managed_gpu_uuid(handle) -> None:
    """Called from _apply_cap() after every successful NVML write."""
    uuid = _nvml_uuid(handle)
    if uuid not in _previously_managed:
        _previously_managed.add(uuid)
        _persist_managed_gpus(_previously_managed)


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
                "NVML cap currently applied per physical GPU (watts).",
                labelnames=("gpu",),
            )
            self.multi_pod_gpu_total = Counter(
                "dynamo_power_agent_multi_pod_gpu_total",
                "Times a physical GPU had multiple pods (agree or conflict).",
                labelnames=("disposition",),
            )
            self.apply_failures_total = Counter(
                "dynamo_power_agent_apply_failures_total",
                "Times the agent failed to set an NVML power cap.",
            )
            self.safe_default_applied_total = Counter(
                "dynamo_power_agent_safe_default_applied_total",
                "Times the safe-default cap was used (conflict or cold-start parse failure).",
            )
            self.cap_clamped_total = Counter(
                "dynamo_power_agent_cap_clamped_total",
                "Times a requested cap was clamped to SKU NVML constraints.",
                labelnames=("direction",),
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


_managed_gpu_indices: set[int] = set()


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


def _release_managed_gpu(handle, gpu_idx: int) -> None:
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

    Eligibility is UUID-gated so caps set by other tooling are never touched.
    A GPU is "ours" if it is in ``_managed_gpu_indices`` (capped in THIS process)
    OR its UUID is in the persisted ``_previously_managed`` set (capped in a
    prior process). The latter is essential across restarts: ``_managed_gpu_indices``
    is in-memory and empty after a restart, while ``_previously_managed`` is
    loaded from disk — without it, a GPU capped before the restart and now busy
    with only unannotated work would keep the stale cap (startup orphan recovery
    only restores *idle* GPUs).

    The idle case (no processes at all) is intentionally NOT handled here;
    ``_reconcile_gpu``'s ``not procs`` branch keeps the cap for a briefly-exited
    worker that will return to the same GPU.
    """
    try:
        uuid = _nvml_uuid(handle)
    except Exception as e:
        logger.warning(
            "Failed to read UUID for GPU %d during release check: %s", gpu_idx, e
        )
        return
    if gpu_idx not in _managed_gpu_indices and uuid not in _previously_managed:
        return  # not a GPU this agent capped — leave it alone (UUID-gating)
    try:
        default_mw = pynvml.nvmlDeviceGetPowerManagementDefaultLimit(handle)
        current_mw = pynvml.nvmlDeviceGetPowerManagementLimit(handle)
        if current_mw < default_mw:
            pynvml.nvmlDeviceSetPowerManagementLimit(handle, default_mw)
            logger.info(
                "Released cap on GPU %d (%d W → %d W): previously managed, now "
                "running only unannotated/non-K8s processes.",
                gpu_idx,
                current_mw // 1000,
                default_mw // 1000,
            )
    except Exception as e:
        # Leave the GPU in the managed set so a later cycle retries the release.
        logger.warning("Failed to release cap on GPU %d: %s", gpu_idx, e)
        return
    _managed_gpu_indices.discard(gpu_idx)
    if uuid in _previously_managed:
        _previously_managed.discard(uuid)
        _persist_managed_gpus(_previously_managed)


# ---------------------------------------------------------------------------
# SIGTERM handler
# ---------------------------------------------------------------------------

_shutdown = threading.Event()


def _handle_sigterm(signum, frame):
    logger.info(
        "SIGTERM received — restoring default TGP on managed GPUs and shutting down."
    )
    for gpu_idx in list(_managed_gpu_indices):
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)
            default_mw = pynvml.nvmlDeviceGetPowerManagementDefaultLimit(handle)
            pynvml.nvmlDeviceSetPowerManagementLimit(handle, default_mw)
            logger.info(
                "Restored GPU %d to default TGP (%d W)", gpu_idx, default_mw // 1000
            )
        except Exception as e:
            logger.exception("Failed to restore TGP on GPU %d: %s", gpu_idx, e)
    try:
        pynvml.nvmlShutdown()
    except Exception:
        # We MUST proceed to ``_shutdown.set()`` so the run loop unblocks
        # and the container exits cleanly — re-raising here would leave
        # the agent hung on SIGTERM. But silently dropping the failure
        # made shutdown-time NVML faults impossible to diagnose from pod
        # logs (PR #9682 CodeRabbit review). ``logger.exception`` writes
        # the full traceback at ERROR level so operators can correlate
        # with driver / hostengine events.
        logger.exception("nvmlShutdown raised; proceeding with agent exit anyway.")
    _shutdown.set()


# ---------------------------------------------------------------------------
# Orphan cap restoration on startup (UUID-gated)
# ---------------------------------------------------------------------------


def _restore_orphaned_gpus_on_startup(device_count: int) -> None:
    """Restore default TDP only on GPUs this agent previously capped AND that are now idle.

    UUID-gating prevents touching caps applied by other workflows (different DGD,
    manual nvidia-smi -pl, vendor firmware defaults).
    """
    global _previously_managed
    _previously_managed = _load_previously_managed_gpus()
    for gpu_idx in range(device_count):
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)
            uuid = _nvml_uuid(handle)
            if uuid not in _previously_managed:
                continue
            procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            if procs:
                continue  # workload running — let normal reconcile handle it
            current_mw = pynvml.nvmlDeviceGetPowerManagementLimit(handle)
            default_mw = pynvml.nvmlDeviceGetPowerManagementDefaultLimit(handle)
            if current_mw < default_mw:
                pynvml.nvmlDeviceSetPowerManagementLimit(handle, default_mw)
                logger.info(
                    "Restored orphaned cap on idle GPU %d (%d W → %d W).",
                    gpu_idx,
                    current_mw // 1000,
                    default_mw // 1000,
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

    Policy (``None`` annotations are filtered out before conflict resolution):
      - 1 pod with annotation  → use that value.
      - 2+ pods, all agree      → use agreed value, WARNING (multi-pod is misconfig).
      - 2+ pods, conflict       → use safe_default_watts, ERROR.
      - No parseable annotation → use safe_default_watts, ERROR.

    Mixed ``None`` + valid (e.g. ``[None, "480"]``) is intentionally lenient:
    the ``None`` pod is dropped and the single remaining value (480) is
    applied (this still fires the "all agree" multi-pod WARNING because
    ``len(pod_annotations) > 1``). We apply the only constraint available
    rather than fail-closed to safe_default — we do not penalise an annotated
    pod for a co-located unannotated one. Multi-pod-per-GPU remains an
    unsupported topology regardless. Per PR #9682 @sttts review.

    Returns the cap in watts.
    """
    values = [v for _, v in pod_annotations if v is not None]
    if not values:
        logger.error(
            "GPU %d: no parseable annotation on any pod; applying safe default (%d W).",
            gpu_idx,
            safe_default_watts,
        )
        metrics.apply_failures_total.inc()
        metrics.safe_default_applied_total.inc()
        return safe_default_watts

    unique = set(values)
    if len(pod_annotations) > 1:
        if len(unique) == 1:
            logger.warning(
                "GPU %d: %d pods all agree on cap %s W (multi-pod-per-GPU is unsupported topology).",
                gpu_idx,
                len(pod_annotations),
                values[0],
            )
            metrics.multi_pod_gpu_total.labels(disposition="agree").inc()
        else:
            logger.error(
                "GPU %d: %d pods with conflicting caps %s; applying safe default (%d W).",
                gpu_idx,
                len(pod_annotations),
                sorted(unique),
                safe_default_watts,
            )
            metrics.multi_pod_gpu_total.labels(disposition="conflict").inc()
            metrics.safe_default_applied_total.inc()
            return safe_default_watts

    try:
        return int(values[0])
    except (ValueError, TypeError):
        logger.error(
            "GPU %d: annotation value %r is not an integer; applying safe default (%d W).",
            gpu_idx,
            values[0],
            safe_default_watts,
        )
        metrics.apply_failures_total.inc()
        metrics.safe_default_applied_total.inc()
        return safe_default_watts


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
    ) -> None:
        self.safe_default_watts = safe_default_watts
        self.node_name = node_name or os.environ.get("NODE_NAME", "")
        self.k8s_namespace = k8s_namespace
        self.metrics = PowerAgentMetrics(prometheus_port)

        if pynvml is None:
            raise RuntimeError("pynvml is required — install pynvml or nvidia-ml-py")
        if k8s_client is None:
            raise RuntimeError("kubernetes Python SDK is required — install kubernetes")

        pynvml.nvmlInit()
        self.device_count = pynvml.nvmlDeviceGetCount()
        logger.info(
            "NVML initialized. %d GPU(s) found on this node.", self.device_count
        )

        _restore_orphaned_gpus_on_startup(self.device_count)

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
        """Run one reconcile cycle: list pods, map PIDs→UIDs, apply caps."""
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
            logger.warning(
                "Pod listing unavailable this cycle; skipping reconcile to "
                "preserve last-known-good caps."
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
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)
        procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        if not procs:
            # No process on this GPU this cycle. We deliberately DO NOT restore
            # the cap to default TDP here — caps are persistent by design while
            # the GPU is idle. A managed worker may exit briefly (OOM,
            # reschedule) and return to the same GPU; restoring during that gap
            # would violate the planner's power budget. An idle previously-
            # managed GPU is restored by ``_handle_sigterm`` (agent shutdown)
            # and ``_restore_orphaned_gpus_on_startup`` (at agent start). A
            # previously-managed GPU that is *busy* with only unannotated work
            # is released below via ``_release_managed_gpu`` (the opted-in pod
            # is gone, so a stale cap must not strand on the new tenant). Per
            # PR #9682 @sttts review.
            return

        # Deduplicate by pod UID before building ``pod_annotations``. A
        # single pod commonly runs multiple GPU processes (one per rank
        # in a TP/PP/EP topology, helper workers, profilers, etc.); the
        # pre-fix code emitted one entry per PID and would treat a
        # one-pod / two-PID GPU as if two pods were colocated. That
        # both fired the spurious "multi-pod-per-GPU" WARNING and, when
        # the pod's annotation was missing/invalid, took the
        # conflict-resolution branch in ``_resolve_cap_for_gpu`` (since
        # ``len(pod_annotations) > 1`` was true), incorrectly applying
        # safe_default + bumping multi_pod_gpu_total. Per PR #9682
        # CodeRabbit review.
        seen_uids: set[str] = set()
        pod_annotations: list[tuple[str, Optional[str]]] = []
        for proc in procs:
            uid = _extract_pod_uid_from_cgroup(proc.pid)
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
            # The idle case (no processes) is handled by the `not procs` branch
            # above, which keeps the cap for a briefly-exited worker.
            _release_managed_gpu(handle, gpu_idx)
            return

        cap_w = _resolve_cap_for_gpu(
            gpu_idx, pod_annotations, self.safe_default_watts, self.metrics
        )
        _apply_cap(handle, gpu_idx, cap_w, self.metrics)

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
    args = parser.parse_args()

    agent = PowerAgent(
        safe_default_watts=args.safe_default_watts,
        node_name=args.node_name,
        k8s_namespace=args.namespace,
        prometheus_port=args.prometheus_port,
    )
    agent.run()


if __name__ == "__main__":
    main()
