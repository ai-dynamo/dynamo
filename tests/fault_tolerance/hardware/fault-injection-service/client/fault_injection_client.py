# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
"""
Fault Injection Client Library for Pytest Integration.

This library provides a clean Python interface to the Fault Injection API,
making it easy to inject faults, monitor recovery, and collect metrics in tests.
"""

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import httpx
import yaml
from kubernetes import client, config
from kubernetes.client.rest import ApiException

logger = logging.getLogger(__name__)


# ============================================================================
# Enums
# ============================================================================


class GPUFault(str, Enum):
    """GPU fault types"""

    XID_ERROR = "xid_error"
    THROTTLE = "throttle"
    MEMORY_PRESSURE = "memory_pressure"
    OVERHEAT = "overheat"
    COMPUTE_OVERLOAD = "compute_overload"


class NetworkPartition(str, Enum):
    """Network partition types"""

    FRONTEND_WORKER = "frontend_worker"
    WORKER_NATS = "worker_nats"
    WORKER_WORKER = "worker_worker"
    CUSTOM = "custom"


class NetworkMode(str, Enum):
    """Network fault modes"""

    NETWORKPOLICY = "networkpolicy"  # Use Kubernetes NetworkPolicy (complete blocking)
    CHAOS_MESH = (
        "chaos_mesh"  # Use ChaosMesh for advanced faults (packet loss, delay, etc.)
    )


class FaultSeverity(str, Enum):
    """Fault severity levels"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class FaultInfo:
    """Information about an injected fault"""

    fault_id: str
    status: str
    fault_type: str
    target: str
    injected_at: str
    message: Optional[str] = None


@dataclass
class Metrics:
    """Collected metrics"""

    timestamp: str
    namespace: str
    gpu_metrics: Optional[dict[str, Any]] = None
    network_metrics: Optional[dict[str, Any]] = None
    inference_metrics: Optional[dict[str, Any]] = None
    node_health: Optional[dict[str, Any]] = None


# ============================================================================
# Fault Injection Client
# ============================================================================


class FaultInjectionClient:
    """Client for Fault Injection API"""

    def __init__(self, api_url: str = "http://localhost:8080", timeout: int = 30):
        """
        Initialize Fault Injection Client.

        Args:
            api_url: Fault Injection API URL
            timeout: Request timeout in seconds
        """
        self.api_url = api_url.rstrip("/")
        self.timeout = timeout
        self.http_client = httpx.Client(timeout=timeout)

        # Initialize Kubernetes client
        try:
            config.load_kube_config()
        except Exception:
            try:
                config.load_incluster_config()
            except Exception as e:
                logger.warning(f"Failed to load Kubernetes config: {e}")

        self.core_v1 = client.CoreV1Api()
        self.apps_v1 = client.AppsV1Api()

        logger.info(f"Fault Injection Client initialized: {api_url}")

    def health_check(self) -> bool:
        """Check if API service is healthy"""
        try:
            response = self.http_client.get(f"{self.api_url}/health")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    # ========================================================================
    # DCGM Infrastructure Management
    # ========================================================================

    def deploy_dcgm(
        self,
        namespace: str = "gpu-operator",
        yaml_path: Optional[str] = None,
        wait_ready: bool = True,
        timeout: int = 120,
    ) -> dict[str, Any]:
        """
        Deploy DCGM DaemonSet for GPU monitoring.

        This automatically deploys the DCGM DaemonSet which is required for
        GPU health monitoring and fault injection.

        Args:
            namespace: Kubernetes namespace (default: gpu-operator)
            yaml_path: Path to dcgm-daemonset.yaml (auto-detected if None)
            wait_ready: Wait for DaemonSet to be ready
            timeout: Timeout in seconds

        Returns:
            Deployment status
        """
        # Auto-detect DCGM YAML path if not provided
        if yaml_path is None:
            # Try common locations
            search_paths = [
                Path(__file__).parent.parent.parent.parent / "dcgm-daemonset.yaml",
                Path(__file__).parent.parent / "deploy" / "dcgm-daemonset.yaml",
                Path.cwd() / "dcgm-daemonset.yaml",
                Path.cwd()
                / "dynamo"
                / "tests"
                / "fault_tolerance"
                / "hardware"
                / "dcgm-daemonset.yaml",
            ]

            for path in search_paths:
                if path.exists():
                    yaml_path = str(path)
                    break

            if yaml_path is None:
                raise FileNotFoundError(
                    "dcgm-daemonset.yaml not found. Please provide yaml_path parameter."
                )

        logger.info(f"Deploying DCGM DaemonSet from {yaml_path}...")

        # Load and apply YAML
        with open(yaml_path, "r") as f:
            yaml_docs = list(yaml.safe_load_all(f))

        deployed_resources = []

        for doc in yaml_docs:
            if not doc:
                continue

            kind = doc.get("kind")
            metadata = doc.get("metadata", {})
            name = metadata.get("name")

            # Ensure namespace matches
            if "namespace" in metadata:
                metadata["namespace"] = namespace

            try:
                if kind == "DaemonSet":
                    self.apps_v1.create_namespaced_daemon_set(
                        namespace=namespace, body=doc
                    )
                    deployed_resources.append(f"DaemonSet/{name}")
                    logger.info(f"Created DaemonSet: {name}")

                elif kind == "Service":
                    self.core_v1.create_namespaced_service(
                        namespace=namespace, body=doc
                    )
                    deployed_resources.append(f"Service/{name}")
                    logger.info(f"Created Service: {name}")

            except ApiException as e:
                if e.status == 409:  # Already exists
                    logger.info(f"{kind}/{name} already exists, skipping...")
                    deployed_resources.append(f"{kind}/{name} (existing)")
                else:
                    raise

        result = {
            "status": "deployed",
            "namespace": namespace,
            "resources": deployed_resources,
        }

        # Wait for DaemonSet to be ready
        if wait_ready:
            logger.info("Waiting for DCGM DaemonSet to be ready...")
            if self._wait_for_dcgm_ready(namespace, timeout):
                result["ready"] = True
                logger.info("DCGM DaemonSet is ready")
            else:
                result["ready"] = False
                logger.warning("DCGM DaemonSet did not become ready within timeout")

        return result

    def _wait_for_dcgm_ready(self, namespace: str, timeout: int) -> bool:
        """Wait for DCGM DaemonSet to be ready"""
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                ds = self.apps_v1.read_namespaced_daemon_set(
                    name="nvidia-dcgm", namespace=namespace
                )

                desired = ds.status.desired_number_scheduled or 0
                ready = ds.status.number_ready or 0

                logger.debug(f"DCGM DaemonSet: {ready}/{desired} pods ready")

                if desired > 0 and ready == desired:
                    return True

            except ApiException as e:
                logger.debug(f"Error checking DaemonSet status: {e}")

            time.sleep(5)

        return False

    def undeploy_dcgm(self, namespace: str = "gpu-operator") -> dict[str, Any]:
        """
        Undeploy DCGM DaemonSet.

        Args:
            namespace: Kubernetes namespace

        Returns:
            Undeploy status
        """
        logger.info(f"Undeploying DCGM DaemonSet from namespace {namespace}...")

        deleted_resources = []

        # Delete DaemonSet
        try:
            self.apps_v1.delete_namespaced_daemon_set(
                name="nvidia-dcgm", namespace=namespace
            )
            deleted_resources.append("DaemonSet/nvidia-dcgm")
            logger.info("Deleted DaemonSet: nvidia-dcgm")
        except ApiException as e:
            if e.status != 404:
                logger.warning(f"Failed to delete DaemonSet: {e}")

        # Delete Service
        try:
            self.core_v1.delete_namespaced_service(
                name="nvidia-dcgm", namespace=namespace
            )
            deleted_resources.append("Service/nvidia-dcgm")
            logger.info("Deleted Service: nvidia-dcgm")
        except ApiException as e:
            if e.status != 404:
                logger.warning(f"Failed to delete Service: {e}")

        return {
            "status": "undeployed",
            "namespace": namespace,
            "resources": deleted_resources,
        }

    def check_dcgm_status(self, namespace: str = "gpu-operator") -> dict[str, Any]:
        """
        Check DCGM DaemonSet status.

        Args:
            namespace: Kubernetes namespace

        Returns:
            DCGM status information
        """
        # Try different DCGM DaemonSet names (varies by installation method)
        dcgm_names = ["nvidia-dcgm-exporter", "nvidia-dcgm", "dcgm-exporter"]

        for dcgm_name in dcgm_names:
            try:
                ds = self.apps_v1.read_namespaced_daemon_set(
                    name=dcgm_name, namespace=namespace
                )

                return {
                    "deployed": True,
                    "namespace": namespace,
                    "daemonset_name": dcgm_name,
                    "desired_pods": ds.status.desired_number_scheduled,
                    "ready_pods": ds.status.number_ready,
                    "available_pods": ds.status.number_available,
                    "unavailable_pods": ds.status.number_unavailable,
                    "ready": ds.status.number_ready
                    == ds.status.desired_number_scheduled,
                }
            except ApiException as e:
                if e.status == 404:
                    continue  # Try next name
                raise

        # None of the DCGM names found
        return {
            "deployed": False,
            "namespace": namespace,
            "message": f"DCGM DaemonSet not found (tried: {', '.join(dcgm_names)})",
        }

    @contextmanager
    def dcgm_infrastructure(
        self,
        namespace: str = "gpu-operator",
        yaml_path: Optional[str] = None,
        auto_cleanup: bool = False,
    ):
        """
        Context manager for DCGM infrastructure with optional automatic cleanup.

        Usage:
            with client.dcgm_infrastructure(namespace="gpu-operator"):
                # Run tests that need DCGM
                pass
            # DCGM optionally cleaned up if auto_cleanup=True

        Args:
            namespace: Kubernetes namespace
            yaml_path: Path to dcgm-daemonset.yaml
            auto_cleanup: Whether to undeploy DCGM after tests (default: False)
        """
        # Check if already deployed
        status = self.check_dcgm_status(namespace)
        already_deployed = status.get("deployed", False)

        if not already_deployed:
            logger.info("DCGM not deployed, deploying now...")
            self.deploy_dcgm(namespace, yaml_path, wait_ready=True)
        else:
            logger.info("DCGM already deployed, reusing existing deployment")

        try:
            yield namespace
        finally:
            # Only cleanup if we deployed it AND auto_cleanup is True
            if auto_cleanup and not already_deployed:
                logger.info("Cleaning up DCGM DaemonSet...")
                try:
                    self.undeploy_dcgm(namespace)
                except Exception as e:
                    logger.error(f"Failed to cleanup DCGM: {e}")
            else:
                logger.info("Keeping DCGM DaemonSet for future tests")

    # ========================================================================
    # GPU Fault Injection
    # ========================================================================

    def inject_gpu_fault(
        self,
        node: str,
        fault_type: GPUFault,
        duration: Optional[int] = None,
        severity: FaultSeverity = FaultSeverity.MEDIUM,
        **parameters,
    ) -> FaultInfo:
        """
        Inject GPU fault.

        Args:
            node: Target node name
            fault_type: Type of GPU fault
            duration: Duration in seconds (None = permanent)
            severity: Fault severity
            **parameters: Additional fault parameters

        Returns:
            FaultInfo with fault details
        """
        payload = {
            "node_name": node,
            "fault_type": fault_type.value,
            "duration": duration,
            "severity": severity.value,
            "parameters": parameters,
        }

        response = self.http_client.post(
            f"{self.api_url}/api/v1/faults/gpu/inject", json=payload
        )
        response.raise_for_status()
        data = response.json()

        return FaultInfo(
            fault_id=data["fault_id"],
            status=data["status"],
            fault_type=data["fault_type"],
            target=data["target"],
            injected_at=data["injected_at"],
            message=data.get("message"),
        )

    @contextmanager
    def gpu_fault(
        self,
        node: str,
        fault_type: GPUFault,
        duration: Optional[int] = None,
        severity: FaultSeverity = FaultSeverity.MEDIUM,
        **parameters,
    ):
        """
        Context manager for GPU fault injection with automatic recovery.

        Usage:
            with client.gpu_fault(node="gpu-node-1", fault_type=GPUFault.XID_ERROR):
                # Test recovery behavior
                pass
            # Fault automatically recovered
        """
        fault = self.inject_gpu_fault(
            node, fault_type, duration, severity, **parameters
        )
        logger.info(f"Injected GPU fault: {fault.fault_id}")

        try:
            yield fault
        finally:
            logger.info(f"Recovering GPU fault: {fault.fault_id}")
            self.recover_fault(fault.fault_id)

    # ========================================================================
    # Network Partition Injection
    # ========================================================================

    def inject_network_partition(
        self,
        partition_type: NetworkPartition,
        source: str,
        target: str,
        mode: NetworkMode = NetworkMode.NETWORKPOLICY,
        duration: Optional[int] = None,
        **parameters,
    ) -> FaultInfo:
        """
        Inject network partition or network fault.

        Args:
            partition_type: Type of network partition
            source: Source namespace or pod selector
            target: Target namespace or pod selector
            mode: Network fault mode (NETWORKPOLICY or CHAOS_MESH)
            duration: Duration in seconds
            **parameters: Mode-specific parameters

        NetworkPolicy Mode Parameters:
            - namespace: Kubernetes namespace
            - target_pod_prefix: Pod name prefix to target
            - block_nats: Block NATS traffic (default: True)
            - block_specific_pods: List of pod label selectors to block
            - block_all_egress: Block all egress traffic

        ChaosMesh Mode Parameters:
            - namespace: Kubernetes namespace
            - target_pod_prefix: Pod name prefix to target
            - packet_loss_percent: Percentage of packets to drop (0-100)
            - delay_ms: Delay to add in milliseconds
            - delay_jitter_ms: Jitter for delay in milliseconds
            - bandwidth_limit: Bandwidth limit (e.g., "1mbps")
            - corrupt_percent: Percentage of packets to corrupt (0-100)
            - duplicate_percent: Percentage of packets to duplicate (0-100)
            - target_nats: Target NATS traffic (default: True)
            - target_specific_pods: List of pod label selectors to target

        Returns:
            FaultInfo with partition details
        """
        payload = {
            "partition_type": partition_type.value,
            "source": source,
            "target": target,
            "mode": mode.value,
            "parameters": parameters,
            "duration": duration,
        }

        response = self.http_client.post(
            f"{self.api_url}/api/v1/faults/network/inject", json=payload
        )
        response.raise_for_status()
        data = response.json()

        return FaultInfo(
            fault_id=data["fault_id"],
            status=data["status"],
            fault_type=data["fault_type"],
            target=data["target"],
            injected_at=data["injected_at"],
            message=data.get("message"),
        )

    @contextmanager
    def network_partition(
        self,
        partition_type: NetworkPartition,
        source: str = "dynamo-oviya",
        target: str = "dynamo-oviya",
        mode: NetworkMode = NetworkMode.NETWORKPOLICY,
        duration: Optional[int] = None,
        **parameters,
    ):
        """
        Context manager for network partition with automatic recovery.

        Usage:
            with client.network_partition(NetworkPartition.FRONTEND_WORKER):
                # Test recovery behavior
                pass
            # Partition automatically removed
        """
        fault = self.inject_network_partition(
            partition_type, source, target, mode, duration, **parameters
        )
        logger.info(f"Injected network partition: {fault.fault_id}")

        try:
            yield fault
        finally:
            logger.info(f"Recovering network partition: {fault.fault_id}")
            self.recover_fault(fault.fault_id)

    # ========================================================================
    # Fault Recovery
    # ========================================================================

    def recover_fault(self, fault_id: str) -> dict[str, Any]:
        """
        Recover from fault.

        Args:
            fault_id: Fault ID to recover

        Returns:
            Recovery status
        """
        response = self.http_client.post(
            f"{self.api_url}/api/v1/faults/{fault_id}/recover"
        )
        response.raise_for_status()
        return response.json()

    def list_faults(self, active_only: bool = False) -> list[dict[str, Any]]:
        """
        List all faults.

        Args:
            active_only: Only list active faults

        Returns:
            List of faults
        """
        response = self.http_client.get(
            f"{self.api_url}/api/v1/faults", params={"active_only": active_only}
        )
        response.raise_for_status()
        return response.json()["faults"]

    def cleanup_network_policies(
        self, namespace: str = "dynamo-oviya"
    ) -> dict[str, Any]:
        """
        Clean up orphaned NetworkPolicies created by fault injection.

        This method manually triggers cleanup of NetworkPolicy resources that may have
        been left behind from previous test runs or API restarts.

        Args:
            namespace: Kubernetes namespace to clean up

        Returns:
            Cleanup result with count and list of deleted policies
        """
        response = self.http_client.post(
            f"{self.api_url}/api/v1/faults/network/cleanup",
            params={"namespace": namespace},
        )
        response.raise_for_status()
        result = response.json()

        logger.info(
            f"Cleaned up {result['policies_deleted']} NetworkPolicy(ies) in namespace {namespace}"
        )

        return result

    # ========================================================================
    # Metrics Collection
    # ========================================================================

    def collect_metrics(self, namespace: str, duration: int = 60) -> Metrics:
        """
        Collect metrics from monitoring agents.

        Args:
            namespace: Target namespace to monitor
            duration: Collection duration in seconds

        Returns:
            Metrics data
        """
        response = self.http_client.get(
            f"{self.api_url}/api/v1/metrics/collect",
            params={"namespace": namespace, "duration": duration},
        )
        response.raise_for_status()
        data = response.json()

        return Metrics(
            timestamp=data["timestamp"],
            namespace=data["namespace"],
            gpu_metrics=data.get("gpu_metrics"),
            network_metrics=data.get("network_metrics"),
            inference_metrics=data.get("inference_metrics"),
            node_health=data.get("node_health"),
        )

    # ========================================================================
    # Kubernetes Helpers
    # ========================================================================

    def wait_for_node_cordoned(self, node_name: str, timeout: int = 180) -> bool:
        """
        Wait for a node to be cordoned.

        Args:
            node_name: Node name to check
            timeout: Timeout in seconds

        Returns:
            True if node was cordoned within timeout
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                node = self.core_v1.read_node(node_name)
                if node.spec.unschedulable:
                    return True
            except ApiException as e:
                logger.warning(f"Failed to check node status: {e}")
            time.sleep(5)
        return False

    def wait_for_pod_rescheduled(
        self,
        namespace: str,
        original_pod_name: str,
        original_node: str,
        timeout: int = 120,
    ) -> Optional[str]:
        """
        Wait for a pod to be rescheduled to a different node.

        Args:
            namespace: Namespace of pod
            original_pod_name: Original pod name
            original_node: Original node name
            timeout: Timeout in seconds

        Returns:
            New pod name if rescheduled, None otherwise
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                pods = self.core_v1.list_namespaced_pod(namespace=namespace)
                for pod in pods.items:
                    # Look for worker pods on different nodes
                    if (
                        "worker" in pod.metadata.name
                        and pod.spec.node_name != original_node
                        and pod.status.phase == "Running"
                    ):
                        return pod.metadata.name
            except ApiException as e:
                logger.warning(f"Failed to list pods: {e}")
            time.sleep(5)
        return None

    def verify_pod_rescheduled(
        self,
        namespace: str,
        original_pod_name: str = "",
        original_node: str = "",
        timeout: int = 120,
    ) -> bool:
        """
        Verify that a pod was rescheduled to a different node.

        Args:
            namespace: Namespace of pod
            original_pod_name: Original pod name (optional)
            original_node: Original node name (optional)
            timeout: Timeout in seconds

        Returns:
            True if pod was rescheduled
        """
        if original_pod_name and original_node:
            new_pod = self.wait_for_pod_rescheduled(
                namespace, original_pod_name, original_node, timeout
            )
            return new_pod is not None
        else:
            # Just check if there are running worker pods
            try:
                pods = self.core_v1.list_namespaced_pod(namespace=namespace)
                for pod in pods.items:
                    if "worker" in pod.metadata.name and pod.status.phase == "Running":
                        return True
            except ApiException:
                pass
            return False

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.http_client.close()


# ============================================================================
# Convenience Functions
# ============================================================================


def create_client(api_url: str = "http://localhost:8080") -> FaultInjectionClient:
    """
    Create a Fault Injection Client.

    Args:
        api_url: Fault Injection API URL

    Returns:
        FaultInjectionClient instance
    """
    return FaultInjectionClient(api_url)
