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

"""
Power Agent DaemonSet - Node-local GPU power limit enforcement.

This agent runs on each GPU node and applies power limits based on
Kubernetes pod annotations. It maps running processes to pods via
cgroup inspection (standard Kubernetes pattern).

Source: MR3_REFACTORED_ARCHITECTURE.md
"""

import logging
import os
import re
import signal
import time
from typing import Dict

import pynvml
from kubernetes import client, config

# Configure Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
ANNOTATION_KEY = "dynamo.nvidia.com/gpu-power-limit"
RECONCILE_INTERVAL = 15  # seconds
NODE_NAME = os.getenv("NODE_NAME")


class NodePowerAgent:
    """
    Node-local agent that enforces GPU power limits based on pod annotations.

    Workflow:
    1. Query K8s API for pods on this node with power limit annotations
    2. For each GPU: get running processes (via NVML)
    3. Map each process PID to its pod UID (via /proc/{pid}/cgroup)
    4. If pod has annotation: apply power limit via NVML
    5. If GPU was previously throttled but pod is gone: restore to default TGP
    """

    def __init__(self):
        self.node_name = NODE_NAME
        if not self.node_name:
            raise ValueError("NODE_NAME environment variable is required")

        # Initialize K8s Client
        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config()

        self.v1 = client.CoreV1Api()

        # Initialize NVML
        try:
            pynvml.nvmlInit()
            self.device_count = pynvml.nvmlDeviceGetCount()
            logger.info(
                f"Initialized NVML. Found {self.device_count} GPUs on node {self.node_name}."
            )
        except pynvml.NVMLError:
            logger.exception("Failed to initialize NVML")
            raise

        # Cache default power limits for each GPU and track throttled GPUs
        self.default_power_limits: Dict[int, int] = {}  # gpu_idx -> default_limit_watts
        self.throttled_gpus: set = set()  # gpu_idx that have been throttled
        self._cache_default_power_limits()

        # Restore any GPUs left at reduced power from previous sessions
        self._restore_orphaned_gpus_on_startup()

    def _cache_default_power_limits(self):
        """
        Cache the default (maximum) power limit for each GPU at startup.
        This is used to restore GPUs to full TGP when pods are removed.
        """
        for gpu_idx in range(self.device_count):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)
                # Get the default power management limit (TGP)
                default_limit_mw = pynvml.nvmlDeviceGetPowerManagementDefaultLimit(
                    handle
                )
                default_limit_w = default_limit_mw // 1000
                self.default_power_limits[gpu_idx] = default_limit_w
                logger.info(
                    f"GPU {gpu_idx}: Default power limit (TGP) = {default_limit_w}W"
                )
            except pynvml.NVMLError:
                logger.exception(f"Failed to get default power limit for GPU {gpu_idx}")
                # Fallback: use a high value that won't accidentally throttle
                self.default_power_limits[
                    gpu_idx
                ] = 700  # Conservative default for H200

    def _restore_orphaned_gpus_on_startup(self):
        """
        On startup, check for GPUs that were left at reduced power limits from
        a previous session (orphaned throttling). Restore any idle GPU that is
        below its default TGP.

        This handles the case where:
        - A previous power-agent session throttled a GPU
        - The pod was deleted but the power-agent crashed/restarted before restoring
        - The GPU is now idle but still at reduced power
        """
        logger.info("Checking for orphaned throttled GPUs on startup...")
        restored_count = 0

        for gpu_idx in range(self.device_count):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)
                uuid = pynvml.nvmlDeviceGetUUID(handle)

                # Check if GPU has any running processes
                procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)

                if not procs:
                    # No processes - check if power limit is below default
                    current_limit = (
                        pynvml.nvmlDeviceGetPowerManagementLimit(handle) // 1000
                    )
                    default_limit = self.default_power_limits.get(gpu_idx, 700)

                    if current_limit < default_limit:
                        logger.info(
                            f"GPU {gpu_idx} ({uuid}): Found orphaned throttling - "
                            f"restoring from {current_limit}W to {default_limit}W"
                        )
                        pynvml.nvmlDeviceSetPowerManagementLimit(
                            handle, default_limit * 1000
                        )
                        restored_count += 1

            except pynvml.NVMLError:
                logger.exception(f"Failed to check/restore GPU {gpu_idx} on startup")

        if restored_count > 0:
            logger.info(
                f"Restored {restored_count} orphaned throttled GPU(s) to default TGP"
            )
        else:
            logger.info("No orphaned throttled GPUs found")

    def _restore_gpu_to_default(self, gpu_idx: int, handle, uuid: str):
        """
        Restore a GPU to its default power limit (TGP).

        Args:
            gpu_idx: GPU index
            handle: NVML device handle
            uuid: GPU UUID for logging
        """
        default_limit = self.default_power_limits.get(gpu_idx, 700)
        current_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) // 1000

        if current_limit < default_limit:
            logger.info(
                f"GPU {gpu_idx} ({uuid}): Restoring power limit to default {default_limit}W "
                f"(was {current_limit}W) - pod removed or no longer annotated"
            )
            pynvml.nvmlDeviceSetPowerManagementLimit(handle, default_limit * 1000)
            self.throttled_gpus.discard(gpu_idx)
        elif gpu_idx in self.throttled_gpus:
            # GPU was throttled but is now at default - clean up tracking
            logger.debug(
                f"GPU {gpu_idx} ({uuid}): Already at default {current_limit}W, clearing throttle tracking"
            )
            self.throttled_gpus.discard(gpu_idx)

    def get_local_pods(self) -> Dict[str, int]:
        """
        Get pods scheduled to this node that have power limit annotations.

        Returns:
            {pod_uid: power_limit_watts}
        """
        try:
            # Field selector ensures we only get pods on THIS node
            pods = self.v1.list_pod_for_all_namespaces(
                field_selector=f"spec.nodeName={self.node_name}"
            )

            targets = {}
            for pod in pods.items:
                if (
                    pod.metadata.annotations
                    and ANNOTATION_KEY in pod.metadata.annotations
                ):
                    try:
                        limit = int(pod.metadata.annotations[ANNOTATION_KEY])
                        targets[pod.metadata.uid] = limit
                        logger.debug(
                            f"Pod {pod.metadata.namespace}/{pod.metadata.name} "
                            f"({pod.metadata.uid}): power limit = {limit}W"
                        )
                    except ValueError:
                        logger.warning(
                            f"Invalid power limit format for pod "
                            f"{pod.metadata.namespace}/{pod.metadata.name}"
                        )

            return targets

        except Exception:
            logger.exception("Failed to list pods")
            return {}

    def map_pids_to_pod_uids(self, pids: list) -> Dict[int, str]:
        """
        Map process IDs to Kubernetes Pod UIDs by reading /proc/{pid}/cgroup.

        This is the standard pattern used by monitoring tools (cadvisor, etc.).
        Kubernetes creates cgroup paths containing the Pod UID.

        Args:
            pids: List of process IDs

        Returns:
            {pid: pod_uid}
        """
        # Determine proc path: check /host/proc first (Minikube with mount), then /proc (real K8s)
        proc_base = "/host/proc" if os.path.exists("/host/proc") else "/proc"
        logger.info(f"Using proc_base: {proc_base} for PID mapping of {len(pids)} PIDs")

        pid_map = {}
        for pid in pids:
            try:
                with open(f"{proc_base}/{pid}/cgroup", "r") as f:
                    content = f.read()
                    # Look for kubepods pattern with pod UID
                    # Regex handles both cgroupfs and systemd drivers
                    # Matches both formats:
                    #   /kubepods/burstable/pod12345678-1234-1234-1234-123456789abc/...
                    #   /kubepods.slice/kubepods-burstable.slice/kubepods-burstable-pod12345678_1234_1234_1234_123456789abc.slice/...
                    match = re.search(
                        r"pod([a-f0-9]{8}[-_][a-f0-9]{4}[-_][a-f0-9]{4}[-_][a-f0-9]{4}[-_][a-f0-9]{12})",
                        content,
                    )
                    if match:
                        # Normalize UID (replace underscores with hyphens)
                        uid = match.group(1).replace("_", "-")
                        pid_map[pid] = uid
                        logger.info(f"PID {pid} → Pod UID {uid}")
                    else:
                        logger.warning(
                            f"PID {pid}: No pod UID found in cgroup. Content: {content[:200]}"
                        )

            except (FileNotFoundError, ProcessLookupError) as e:
                # Process exited between query and cgroup read, or PID not visible
                logger.warning(f"PID {pid} not found in {proc_base}: {e}")
                continue
            except Exception as e:
                logger.warning(f"Error reading cgroup for PID {pid}: {e}")
                continue

        return pid_map

    def enforce_limits(self):
        """
        Main reconciliation logic.

        For each GPU on this node:
        1. Get running processes
        2. Map processes to pods
        3. If pod has power limit annotation: apply via NVML
        4. If GPU was previously throttled but no longer needs throttling: restore to default TGP
        """
        desired_state = self.get_local_pods()

        if desired_state:
            logger.info(
                f"Enforcing limits for {len(desired_state)} pods: {desired_state}"
            )
        else:
            logger.debug("No pods with power limit annotations on this node")

        for gpu_idx in range(self.device_count):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)
                uuid = pynvml.nvmlDeviceGetUUID(handle)

                # Get all processes running on this GPU
                procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                pids = [p.pid for p in procs]

                if not pids:
                    # No processes running - check if GPU was previously throttled
                    if gpu_idx in self.throttled_gpus:
                        logger.info(
                            f"GPU {gpu_idx} ({uuid}): No processes running, "
                            "restoring previously throttled GPU to default"
                        )
                        self._restore_gpu_to_default(gpu_idx, handle, uuid)
                    else:
                        logger.debug(f"GPU {gpu_idx} ({uuid}): No processes running")
                    continue

                logger.info(
                    f"GPU {gpu_idx} ({uuid}): {len(pids)} processes running, PIDs={pids}"
                )

                # Map PIDs to Pod UIDs
                pid_pod_map = self.map_pids_to_pod_uids(pids)
                logger.info(
                    f"GPU {gpu_idx} ({uuid}): PID-to-Pod mapping: {pid_pod_map}"
                )

                # Check if any process belongs to a pod with power limit
                target_limit = None
                target_pod_uid = None
                for pid, pod_uid in pid_pod_map.items():
                    if pod_uid in desired_state:
                        target_limit = desired_state[pod_uid]
                        target_pod_uid = pod_uid
                        break  # Assume 1 pod per GPU (exclusive mode)

                # Apply limit if needed
                if target_limit:
                    current_limit = (
                        pynvml.nvmlDeviceGetPowerManagementLimit(handle) // 1000
                    )

                    if current_limit != target_limit:
                        logger.info(
                            f"GPU {gpu_idx} ({uuid}): Setting power limit to {target_limit}W "
                            f"(was {current_limit}W) for pod {target_pod_uid}"
                        )
                        # NVML expects milliwatts
                        pynvml.nvmlDeviceSetPowerManagementLimit(
                            handle, target_limit * 1000
                        )
                        # Track this GPU as throttled
                        self.throttled_gpus.add(gpu_idx)
                    else:
                        logger.debug(
                            f"GPU {gpu_idx} ({uuid}): Power limit already at {target_limit}W"
                        )
                        # Ensure tracking is correct
                        self.throttled_gpus.add(gpu_idx)
                else:
                    # No power limit annotation for processes on this GPU
                    # Check if this GPU was previously throttled and needs restoration
                    if gpu_idx in self.throttled_gpus:
                        logger.info(
                            f"GPU {gpu_idx} ({uuid}): Processes running but no power limit "
                            "annotation - restoring previously throttled GPU to default"
                        )
                        self._restore_gpu_to_default(gpu_idx, handle, uuid)
                    else:
                        logger.debug(
                            f"GPU {gpu_idx} ({uuid}): No power limit annotation for running processes"
                        )

            except pynvml.NVMLError:
                logger.exception(f"NVML error on GPU {gpu_idx}")
            except Exception:
                logger.exception(f"Unexpected error on GPU {gpu_idx}")

    def restore_all_gpus_to_default(self):
        """
        Restore all GPUs to their default power limits.
        Called on shutdown or when cleaning up.
        """
        logger.info("Restoring all GPUs to default power limits...")
        for gpu_idx in range(self.device_count):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)
                uuid = pynvml.nvmlDeviceGetUUID(handle)
                self._restore_gpu_to_default(gpu_idx, handle, uuid)
            except pynvml.NVMLError:
                logger.exception(f"Failed to restore GPU {gpu_idx} to default")
        self.throttled_gpus.clear()
        logger.info("All GPUs restored to default power limits")

    def run(self):
        """Main control loop."""
        logger.info(f"Starting Power Agent on node {self.node_name}")
        logger.info(f"Reconcile interval: {RECONCILE_INTERVAL}s")
        logger.info(f"Annotation key: {ANNOTATION_KEY}")
        logger.info(f"Default power limits: {self.default_power_limits}")

        # Track if we should keep running
        self._running = True

        def handle_shutdown(signum, frame):
            """Handle SIGTERM/SIGINT for graceful shutdown."""
            sig_name = signal.Signals(signum).name
            logger.info(f"Received {sig_name}, initiating graceful shutdown...")
            self._running = False

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, handle_shutdown)
        signal.signal(signal.SIGINT, handle_shutdown)

        try:
            while self._running:
                try:
                    self.enforce_limits()
                except Exception:
                    logger.exception("Error in reconciliation loop")

                # Use shorter sleep intervals to respond to shutdown faster
                for _ in range(RECONCILE_INTERVAL):
                    if not self._running:
                        break
                    time.sleep(1)
        finally:
            # Restore all GPUs to default on shutdown
            self.restore_all_gpus_to_default()
            logger.info("Power Agent shutdown complete")


if __name__ == "__main__":
    agent = NodePowerAgent()
    agent.run()
