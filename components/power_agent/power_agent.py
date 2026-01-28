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
        """
        desired_state = self.get_local_pods()
        if not desired_state:
            logger.debug("No pods with power limit annotations on this node")
            return

        logger.info(f"Enforcing limits for {len(desired_state)} pods: {desired_state}")

        for gpu_idx in range(self.device_count):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)
                uuid = pynvml.nvmlDeviceGetUUID(handle)

                # Get all processes running on this GPU
                procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                pids = [p.pid for p in procs]

                if not pids:
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
                    else:
                        logger.debug(
                            f"GPU {gpu_idx} ({uuid}): Power limit already at {target_limit}W"
                        )
                else:
                    logger.debug(
                        f"GPU {gpu_idx} ({uuid}): No power limit annotation for running processes"
                    )

            except pynvml.NVMLError:
                logger.exception(f"NVML error on GPU {gpu_idx}")
            except Exception:
                logger.exception(f"Unexpected error on GPU {gpu_idx}")

    def run(self):
        """Main control loop."""
        logger.info(f"Starting Power Agent on node {self.node_name}")
        logger.info(f"Reconcile interval: {RECONCILE_INTERVAL}s")
        logger.info(f"Annotation key: {ANNOTATION_KEY}")

        while True:
            try:
                self.enforce_limits()
            except Exception:
                logger.exception("Error in reconciliation loop")

            time.sleep(RECONCILE_INTERVAL)


if __name__ == "__main__":
    agent = NodePowerAgent()
    agent.run()
