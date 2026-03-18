# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
"""
Kubernetes operations helpers for fault tolerance testing.

Provides utilities for managing nodes, pods, and deployments
during fault injection scenarios.
"""

import logging
import time
from typing import Dict, List, Optional

from kubernetes import client
from kubernetes.client.rest import ApiException

logger = logging.getLogger(__name__)


class NodeOperations:
    """Helper for node-level Kubernetes operations."""

    def __init__(self, k8s_core: client.CoreV1Api):
        """
        Initialize node operations helper.

        Args:
            k8s_core: Kubernetes CoreV1Api client
        """
        self.k8s_core = k8s_core
        # Track original schedulable state for each node before cordoning
        self._original_node_states: Dict[str, bool] = {}

    def cordon_node(self, node_name: str, reason: str = "fault-injection-test") -> bool:
        """
        Cordon a node (make it unschedulable).

        Stores the node's original schedulable state before cordoning,
        which can be restored later with uncordon_node.

        Args:
            node_name: Name of the node to cordon
            reason: Reason for cordoning (used in label)

        Returns:
            True if successful, False otherwise
        """
        try:
            # Read and store the original state before cordoning
            node = self.k8s_core.read_node(node_name)
            original_unschedulable = node.spec.unschedulable or False
            self._original_node_states[node_name] = original_unschedulable

            self.k8s_core.patch_node(
                node_name,
                {
                    "spec": {"unschedulable": True},
                    "metadata": {
                        "labels": {
                            "test.fault-injection/cordoned": "true",
                            "test.fault-injection/reason": reason,
                        }
                    },
                },
            )

            # Verify cordon took effect
            node = self.k8s_core.read_node(node_name)
            return node.spec.unschedulable is True

        except Exception as e:
            logger.error(f"Failed to cordon node {node_name}: {e}")
            return False

    def uncordon_node(
        self, node_name: str, restore_previous_state: bool = False
    ) -> bool:
        """
        Uncordon a node (make it schedulable again).

        Args:
            node_name: Name of the node to uncordon
            restore_previous_state: If True, restore the node to its original
                schedulable state before cordoning. If False (default), always
                make the node schedulable (unschedulable=False).

        Returns:
            True if successful, False otherwise
        """
        try:
            # Determine the target unschedulable state
            if restore_previous_state and node_name in self._original_node_states:
                # Restore to the state before we cordoned it
                target_unschedulable = self._original_node_states[node_name]
                # Clean up stored state
                del self._original_node_states[node_name]
            else:
                # Default behavior: make node schedulable
                target_unschedulable = False

            self.k8s_core.patch_node(
                node_name,
                {
                    "spec": {"unschedulable": target_unschedulable},
                    "metadata": {
                        "labels": {
                            "test.fault-injection/cordoned": None,
                            "test.fault-injection/reason": None,
                        }
                    },
                },
            )
            return True

        except Exception as e:
            logger.error(f"Failed to uncordon node {node_name}: {e}")
            return False

    def is_node_cordoned(self, node_name: str) -> bool:
        """Check if a node is cordoned (unschedulable)."""
        try:
            node = self.k8s_core.read_node(node_name)
            return node.spec.unschedulable or False
        except Exception:
            return False

    def restart_gpu_driver(self, node_name: str, wait_timeout: int = 300) -> bool:
        """
        Restart the NVIDIA GPU driver pod on a specific node.

        Simulates NVSentinel fault-remediation-module behavior:
        - Reference: fault-remediation-module/pkg/reconciler/reconciler.go:184-232
        - Action: COMPONENT_RESET for XID 79

        Args:
            node_name: Name of the node to restart GPU driver on
            wait_timeout: Max seconds to wait for driver to be ready (default: 300)

        Returns:
            True if driver restart succeeded, False otherwise
        """
        try:
            # Find the nvidia-driver-daemonset pod on this node
            pods = self.k8s_core.list_namespaced_pod(
                namespace="gpu-operator", label_selector="app=nvidia-driver-daemonset"
            )

            target_pod = None
            for pod in pods.items:
                if pod.spec.node_name == node_name:
                    target_pod = pod.metadata.name
                    break

            if not target_pod:
                logger.error(f"No GPU driver pod found on node {node_name}")
                return False

            logger.info(f"Found driver pod: {target_pod}")

            # Get the current pod's creation timestamp before deletion
            old_pod = self.k8s_core.read_namespaced_pod(
                name=target_pod, namespace="gpu-operator"
            )
            old_creation_time = old_pod.metadata.creation_timestamp

            # Delete the pod to force restart
            logger.info("Deleting pod to trigger restart...")
            self.k8s_core.delete_namespaced_pod(
                name=target_pod, namespace="gpu-operator", grace_period_seconds=0
            )

            # Wait for new pod to be ready
            logger.info(
                f"Waiting for new driver pod to be ready (max {wait_timeout}s)..."
            )
            start_time = time.time()

            while time.time() - start_time < wait_timeout:
                try:
                    # List pods again since DaemonSet creates a new pod with a different name
                    pods = self.k8s_core.list_namespaced_pod(
                        namespace="gpu-operator",
                        label_selector="app=nvidia-driver-daemonset",
                    )

                    # Find the new pod on this node
                    pod = None
                    for p in pods.items:
                        if (
                            p.spec.node_name == node_name
                            and p.metadata.creation_timestamp > old_creation_time
                        ):
                            pod = p
                            break

                    if not pod:
                        # New pod not yet created
                        time.sleep(5)
                        continue

                    # Check if pod is ready
                    if pod.status.phase == "Running":
                        # Check all containers are ready
                        all_ready = True
                        if pod.status.container_statuses:
                            for container in pod.status.container_statuses:
                                if not container.ready:
                                    all_ready = False
                                    break

                        if all_ready:
                            elapsed = int(time.time() - start_time)
                            logger.info(
                                f"New driver pod ready: {pod.metadata.name} (took {elapsed}s)"
                            )

                            # Wait a bit more for GPU initialization
                            logger.info(
                                "Waiting additional 30s for GPU initialization..."
                            )
                            time.sleep(30)

                            logger.info("GPU driver restarted successfully")
                            return True
                except Exception:
                    pass

                time.sleep(5)

            logger.error(f"GPU driver pod did not become ready within {wait_timeout}s")
            return False

        except Exception as e:
            logger.error(f"Failed to restart GPU driver: {e}")
            return False


class PodOperations:
    """Helper for pod-level Kubernetes operations."""

    def __init__(self, k8s_core: client.CoreV1Api):
        """
        Initialize pod operations helper.

        Args:
            k8s_core: Kubernetes CoreV1Api client
        """
        self.k8s_core = k8s_core

    def drain_pods(
        self, namespace: str, label_selector: str, node_name: Optional[str] = None
    ) -> int:
        """
        Drain (delete) pods matching selector, optionally filtered by node.

        Simulates NVSentinel node-drainer-module behavior:
        - Reference: node-drainer-module/pkg/informers/informers.go:471-535

        Args:
            namespace: Kubernetes namespace
            label_selector: Label selector for pods to drain
            node_name: If provided, only drain pods on this node

        Returns:
            Number of pods successfully drained
        """
        try:
            # Build field selector
            field_selector = f"spec.nodeName={node_name}" if node_name else None

            # Get pods to drain
            pods = self.k8s_core.list_namespaced_pod(
                namespace=namespace,
                label_selector=label_selector,
                field_selector=field_selector,
            )

            drained_count = 0
            for pod in pods.items:
                try:
                    self.k8s_core.delete_namespaced_pod(
                        name=pod.metadata.name,
                        namespace=namespace,
                        grace_period_seconds=0,
                    )
                    logger.info(f"Evicted: {pod.metadata.name}")
                    drained_count += 1
                except ApiException as e:
                    if e.status != 404:  # Ignore if already deleted
                        logger.warning(f"Failed to evict {pod.metadata.name}: {e}")

            return drained_count

        except Exception as e:
            logger.error(f"Failed to drain pods: {e}")
            return 0

    def get_pod_distribution(
        self, namespace: str, label_selector: str
    ) -> Dict[str, int]:
        """
        Get distribution of pods across nodes.

        Args:
            namespace: Kubernetes namespace
            label_selector: Label selector for pods

        Returns:
            Dict mapping node names to pod counts
        """
        try:
            pods = self.k8s_core.list_namespaced_pod(
                namespace=namespace, label_selector=label_selector
            )

            distribution: Dict[str, int] = {}
            for pod in pods.items:
                if pod.status.phase == "Running":
                    node = pod.spec.node_name
                    distribution[node] = distribution.get(node, 0) + 1

            return distribution

        except Exception as e:
            logger.error(f"Failed to get pod distribution: {e}")
            return {}

    def wait_for_pods_ready(
        self,
        namespace: str,
        label_selector: str,
        expected_count: int,
        timeout: int = 900,
        exclude_node: Optional[str] = None,
    ) -> bool:
        """
        Wait for pods to become ready.

        Args:
            namespace: Kubernetes namespace
            label_selector: Label selector for pods
            expected_count: Expected number of ready pods
            timeout: Max seconds to wait (default: 900 = 15 minutes)
            exclude_node: If provided, only count pods NOT on this node

        Returns:
            True if expected pods became ready, False if timeout
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                pods = self.k8s_core.list_namespaced_pod(
                    namespace=namespace, label_selector=label_selector
                )

                ready_count = 0
                for pod in pods.items:
                    # Skip if on excluded node
                    if exclude_node and pod.spec.node_name == exclude_node:
                        continue

                    # Check if ready (all containers must be ready)
                    if pod.status.phase == "Running" and pod.status.container_statuses:
                        # Check all containers are ready
                        all_ready = all(
                            container.ready
                            for container in pod.status.container_statuses
                        )
                        if all_ready:
                            ready_count += 1

                elapsed = int(time.time() - start_time)
                logger.debug(f"{elapsed}s: {ready_count}/{expected_count} ready")

                if ready_count >= expected_count:
                    logger.info(f"All {expected_count} pods ready after {elapsed}s!")
                    return True

            except Exception as e:
                logger.warning(f"Error checking pods: {e}")

            time.sleep(30)

        return False

    def get_pod_status_details(
        self, namespace: str, label_selector: str, node_name: Optional[str] = None
    ) -> List[Dict]:
        """
        Get detailed status for each pod.

        Args:
            namespace: Kubernetes namespace
            label_selector: Label selector for pods
            node_name: If provided, only get pods on this node

        Returns:
            List of dicts with keys: name (pod name), node (node name), and state (pod state)
        """
        try:
            field_selector = f"spec.nodeName={node_name}" if node_name else None

            pods = self.k8s_core.list_namespaced_pod(
                namespace=namespace,
                label_selector=label_selector,
                field_selector=field_selector,
            )

            details = []
            for pod in pods.items:
                pod_name = pod.metadata.name
                node = pod.spec.node_name

                if pod.status.container_statuses:
                    cs = pod.status.container_statuses[0]
                    if cs.state.waiting:
                        state = cs.state.waiting.reason
                    elif cs.state.terminated:
                        state = f"Terminated ({cs.state.terminated.reason})"
                    elif cs.state.running:
                        state = "Running"
                    else:
                        state = "Unknown"
                else:
                    state = f"{pod.status.phase} (no container status)"

                details.append({"name": pod_name, "node": node, "state": state})

            return details

        except Exception as e:
            logger.error(f"Failed to get pod status details: {e}")
            return []
