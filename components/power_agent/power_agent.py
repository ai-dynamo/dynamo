#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Power Agent DaemonSet — Phase 1 implementation.

Runs as a privileged DaemonSet (hostPID: true) on each GPU node. Every 15s:
  1. Lists all pods on this node via the K8s API.
  2. For each physical GPU: nvmlDeviceGetComputeRunningProcesses() → PID list.
  3. For each PID: reads /proc/{pid}/cgroup → extracts pod UID.
  4. Looks up the pod's dynamo.nvidia.com/gpu-power-limit annotation.
  5. Calls nvmlDeviceSetPowerManagementLimit(handle, watts × 1000).

SIGTERM handler: restores default TDP on all managed GPUs before shutdown.
Cold-start orphan recovery: UUID-gated (persisted to /var/lib/dynamo-power-agent/).
"""

import json
import logging
import os
import re
import signal
import sys
import threading
import time
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

_SYSTEMD_RE = re.compile(r"kubepods-(?:burstable-|besteffort-)?pod([a-fA-F0-9_]+)\.slice")
_CGROUPFS_RE = re.compile(r"/kubepods(?:/burstable|/besteffort)?/pod([a-fA-F0-9-]+)(?:/|$)")


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
    try:
        with open(_MANAGED_STATE_PATH) as f:
            return set(json.load(f).get("managed_uuids", []))
    except (FileNotFoundError, json.JSONDecodeError):
        return set()


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
    def labels(self, **_): return self
    def set(self, _): pass
    def inc(self, _=1): pass


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
                logger.info("Prometheus metrics server started on port %d", prometheus_port)
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

def _clamp_to_constraints(handle, requested_w: int, gpu_idx: int, metrics: PowerAgentMetrics) -> int:
    """Clamp `requested_w` to the SKU-defined NVML power-cap range."""
    try:
        min_mw, max_mw = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)
    except pynvml.NVMLError:
        return requested_w
    min_w, max_w = min_mw // 1000, max_mw // 1000
    if requested_w < min_w:
        logger.warning(
            "Requested cap %d W below SKU min %d W on GPU %d; clamping up.",
            requested_w, min_w, gpu_idx,
        )
        metrics.cap_clamped_total.labels(direction="min").inc()
        return min_w
    if requested_w > max_w:
        logger.warning(
            "Requested cap %d W above SKU max %d W on GPU %d; clamping down.",
            requested_w, max_w, gpu_idx,
        )
        metrics.cap_clamped_total.labels(direction="max").inc()
        return max_w
    return requested_w


_managed_gpu_indices: set[int] = set()


def _apply_cap(handle, gpu_idx: int, requested_w: int, metrics: PowerAgentMetrics) -> None:
    """Apply NVML power cap. All writes go through here."""
    effective_w = _clamp_to_constraints(handle, requested_w, gpu_idx, metrics)
    try:
        pynvml.nvmlDeviceSetPowerManagementLimit(handle, effective_w * 1000)
        _managed_gpu_indices.add(gpu_idx)
        _record_managed_gpu_uuid(handle)
        metrics.applied_limit_watts.labels(gpu=str(gpu_idx)).set(effective_w)
    except pynvml.NVMLError as e:
        logger.error("nvmlDeviceSetPowerManagementLimit GPU %d → %d W failed: %s", gpu_idx, effective_w, e)
        metrics.apply_failures_total.inc()


# ---------------------------------------------------------------------------
# SIGTERM handler
# ---------------------------------------------------------------------------

_shutdown = threading.Event()


def _handle_sigterm(signum, frame):
    logger.info("SIGTERM received — restoring default TGP on managed GPUs and shutting down.")
    for gpu_idx in list(_managed_gpu_indices):
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)
            default_mw = pynvml.nvmlDeviceGetPowerManagementDefaultLimit(handle)
            pynvml.nvmlDeviceSetPowerManagementLimit(handle, default_mw)
            logger.info("Restored GPU %d to default TGP (%d W)", gpu_idx, default_mw // 1000)
        except Exception as e:
            logger.exception("Failed to restore TGP on GPU %d: %s", gpu_idx, e)
    try:
        pynvml.nvmlShutdown()
    except Exception:
        pass
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
                    gpu_idx, current_mw // 1000, default_mw // 1000,
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
      - 1 pod with annotation  → use that value.
      - 2+ pods, all agree      → use agreed value, WARNING (multi-pod is misconfig).
      - 2+ pods, conflict       → use safe_default_watts, ERROR.
      - No parseable annotation → use safe_default_watts, ERROR.
    Returns the cap in watts.
    """
    values = [v for _, v in pod_annotations if v is not None]
    if not values:
        logger.error(
            "GPU %d: no parseable annotation on any pod; applying safe default (%d W).",
            gpu_idx, safe_default_watts,
        )
        metrics.apply_failures_total.inc()
        metrics.safe_default_applied_total.inc()
        return safe_default_watts

    unique = set(values)
    if len(pod_annotations) > 1:
        if len(unique) == 1:
            logger.warning(
                "GPU %d: %d pods all agree on cap %s W (multi-pod-per-GPU is unsupported topology).",
                gpu_idx, len(pod_annotations), values[0],
            )
            metrics.multi_pod_gpu_total.labels(disposition="agree").inc()
        else:
            logger.error(
                "GPU %d: %d pods with conflicting caps %s; applying safe default (%d W).",
                gpu_idx, len(pod_annotations), sorted(unique), safe_default_watts,
            )
            metrics.multi_pod_gpu_total.labels(disposition="conflict").inc()
            metrics.safe_default_applied_total.inc()
            return safe_default_watts

    try:
        return int(values[0])
    except (ValueError, TypeError):
        logger.error(
            "GPU %d: annotation value %r is not an integer; applying safe default (%d W).",
            gpu_idx, values[0], safe_default_watts,
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
        logger.info("NVML initialized. %d GPU(s) found on this node.", self.device_count)

        _restore_orphaned_gpus_on_startup(self.device_count)

        # K8s client
        try:
            k8s_config.load_incluster_config()
        except ConfigException:
            k8s_config.load_kube_config()
        self._core_v1 = k8s_client.CoreV1Api()

    def _list_pods_on_node(self) -> list:
        """List all pods scheduled on this node."""
        try:
            field_selector = f"spec.nodeName={self.node_name}" if self.node_name else None
            kwargs: dict = {}
            if self.k8s_namespace:
                result = self._core_v1.list_namespaced_pod(
                    namespace=self.k8s_namespace,
                    field_selector=field_selector,
                )
            else:
                result = self._core_v1.list_pod_for_all_namespaces(
                    field_selector=field_selector,
                )
            return result.items
        except Exception as e:
            logger.warning("Failed to list pods: %s", e)
            return []

    def _build_uid_to_annotation(self, pods: list) -> dict[str, Optional[str]]:
        """Map pod UID → power-limit annotation value (or None if absent/malformed)."""
        result: dict[str, Optional[str]] = {}
        for pod in pods:
            uid = pod.metadata.uid
            annotations = pod.metadata.annotations or {}
            result[uid] = annotations.get(POWER_ANNOTATION_KEY)
        return result

    def reconcile_once(self) -> None:
        """Run one reconcile cycle: list pods, map PIDs→UIDs, apply caps."""
        pods = self._list_pods_on_node()
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
            return  # no K8s workload on this GPU

        pod_annotations: list[tuple[str, Optional[str]]] = []
        for proc in procs:
            uid = _extract_pod_uid_from_cgroup(proc.pid)
            if uid is None:
                continue  # non-K8s process — skip
            if uid in uid_to_annotation:
                pod_annotations.append((uid, uid_to_annotation[uid]))

        if not pod_annotations:
            return  # all processes are non-K8s

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
            self.node_name or "(all)", self.safe_default_watts, RECONCILE_INTERVAL_S,
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
    import argparse

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
