# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
"""
Fault Injection API Service - Central orchestrator for hardware fault testing.

This service provides REST APIs to inject GPU faults, network partitions, and
monitor system behavior during fault scenarios. It coordinates with DaemonSets
running on cluster nodes for privileged operations.
"""

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

import httpx
from fastapi import FastAPI, HTTPException, Query
from kubernetes import client, config
from kubernetes.client.rest import ApiException
from kubernetes.stream import stream
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
FAULT_INJECTION_NAMESPACE = "fault-injection-system"
DEFAULT_DYNAMO_NAMESPACE = "dynamo"


# ============================================================================
# Enums and Models
# ============================================================================


class GPUFaultType(str, Enum):
    """Types of GPU faults that can be injected"""

    XID_ERROR = "xid_error"
    THROTTLE = "throttle"
    MEMORY_PRESSURE = "memory_pressure"
    OVERHEAT = "overheat"
    COMPUTE_OVERLOAD = "compute_overload"


class NetworkPartitionType(str, Enum):
    """Types of network partitions"""

    FRONTEND_WORKER = "frontend_worker"
    WORKER_NATS = (
        "worker_nats"  # Partition between worker pods and NATS messaging service
    )
    WORKER_WORKER = (
        "worker_worker"  # Partition between worker pods (inter-worker communication)
    )
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


class FaultStatus(str, Enum):
    """Status of injected fault"""

    PENDING = "pending"
    INJECTED = "injected"  # Fault has been successfully injected into the system
    ACTIVE = "active"  # Fault is currently affecting the system (actively monitored)
    RECOVERING = "recovering"
    RECOVERED = "recovered"
    FAILED = "failed"


class GPUFaultRequest(BaseModel):
    """Request to inject GPU fault"""

    node_name: str = Field(..., description="Target GPU node name")
    fault_type: GPUFaultType = Field(..., description="Type of GPU fault")
    duration: Optional[int] = Field(
        None, description="Duration in seconds (None = permanent)"
    )
    severity: FaultSeverity = Field(FaultSeverity.MEDIUM, description="Fault severity")
    parameters: Optional[dict[str, Any]] = Field(
        default_factory=dict, description="Additional parameters"
    )


class NetworkFaultRequest(BaseModel):
    """Request to inject network partition/fault"""

    partition_type: NetworkPartitionType = Field(
        ..., description="Type of network partition"
    )
    source: str = Field(..., description="Source namespace or pod selector")
    target: str = Field(..., description="Target namespace or pod selector")
    mode: NetworkMode = Field(
        NetworkMode.NETWORKPOLICY, description="Network fault mode"
    )
    parameters: Optional[dict[str, Any]] = Field(
        default_factory=dict,
        description="Mode-specific parameters (delay_ms, packet_loss_pct, bandwidth_mbps)",
    )
    duration: Optional[int] = Field(None, description="Duration in seconds")


class FaultResponse(BaseModel):
    """Response after injecting fault"""

    fault_id: str
    status: FaultStatus
    fault_type: str
    target: str
    injected_at: str
    message: Optional[str] = None


class MetricsResponse(BaseModel):
    """Response with collected metrics"""

    timestamp: str
    namespace: str
    gpu_metrics: Optional[
        dict[str, Any]
    ] = None  # GPU utilization, memory, temperature, power
    network_metrics: Optional[dict[str, Any]] = None  # Latency, packet loss, throughput
    inference_metrics: Optional[
        dict[str, Any]
    ] = None  # Inference latency, throughput, accuracy
    node_health: Optional[dict[str, Any]] = None  # Node status, resource availability


# ============================================================================
# Fault Tracker
# ============================================================================


class FaultTracker:
    """Track active faults and their lifecycle"""

    def __init__(self):
        self.faults: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def create_fault(
        self, fault_type: str, target: str, details: dict[str, Any]
    ) -> str:
        """Create and track a new fault"""
        async with self._lock:
            fault_id = f"{fault_type}-{uuid.uuid4().hex[:8]}"
            self.faults[fault_id] = {
                "id": fault_id,
                "type": fault_type,
                "target": target,
                "status": FaultStatus.PENDING,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "injected_at": None,
                "recovered_at": None,
                "details": details,
            }
            return fault_id

    async def update_status(self, fault_id: str, status: FaultStatus):
        """Update fault status"""
        async with self._lock:
            if fault_id not in self.faults:
                raise ValueError(f"Fault {fault_id} not found")
            self.faults[fault_id]["status"] = status
            if status == FaultStatus.INJECTED:
                self.faults[fault_id]["injected_at"] = datetime.now(
                    timezone.utc
                ).isoformat()
            elif status == FaultStatus.RECOVERED:
                self.faults[fault_id]["recovered_at"] = datetime.now(
                    timezone.utc
                ).isoformat()

    async def get_fault(self, fault_id: str) -> Optional[dict[str, Any]]:
        """Get fault details"""
        async with self._lock:
            return self.faults.get(fault_id)

    async def list_active_faults(self) -> list[dict[str, Any]]:
        """List all active faults"""
        async with self._lock:
            return [
                f
                for f in self.faults.values()
                if f["status"]
                in [FaultStatus.PENDING, FaultStatus.INJECTED, FaultStatus.ACTIVE]
            ]


# ============================================================================
# Kubernetes Helper
# ============================================================================


class KubernetesHelper:
    """Helper for Kubernetes operations"""

    def __init__(self):
        try:
            config.load_incluster_config()
            logger.info("Loaded in-cluster Kubernetes config")
        except Exception:
            config.load_kube_config()
            logger.info("Loaded local Kubernetes config")

        self.core_v1 = client.CoreV1Api()
        self.apps_v1 = client.AppsV1Api()
        self.networking_v1 = client.NetworkingV1Api()

    async def cleanup_orphaned_network_policies(
        self, namespace: str = DEFAULT_DYNAMO_NAMESPACE
    ) -> tuple[int, list[str]]:
        """
        Clean up orphaned NetworkPolicies created by fault injection.

        Returns:
            Tuple of (count_deleted, list_of_deleted_policy_names)
        """
        deleted_policies = []

        try:
            # List NetworkPolicies with our label selector
            policies = await asyncio.to_thread(
                self.networking_v1.list_namespaced_network_policy,
                namespace=namespace,
                label_selector="managed-by=fault-injection-api",
            )

            for policy in policies.items:
                policy_name = policy.metadata.name
                logger.info(
                    f"Found orphaned NetworkPolicy: {policy_name} in namespace {namespace}"
                )

                try:
                    # Delete the policy
                    await asyncio.to_thread(
                        self.networking_v1.delete_namespaced_network_policy,
                        name=policy_name,
                        namespace=namespace,
                    )
                    deleted_policies.append(policy_name)
                    logger.info(f"Deleted orphaned NetworkPolicy: {policy_name}")

                except ApiException as e:
                    if e.status != 404:  # Ignore if already deleted
                        logger.exception(
                            f"Failed to delete NetworkPolicy {policy_name}: {e}"
                        )

            return len(deleted_policies), deleted_policies

        except ApiException as e:
            logger.error(f"Failed to list NetworkPolicies in {namespace}: {e}")
            return 0, []

    async def get_daemonset_pod(
        self, namespace: str, label_selector: str
    ) -> Optional[str]:
        """Get a pod from DaemonSet by label"""
        try:
            pods = await asyncio.to_thread(
                self.core_v1.list_namespaced_pod,
                namespace=namespace,
                label_selector=label_selector,
            )
            if pods.items:
                if len(pods.items) > 1:
                    logger.warning(
                        f"Multiple pods ({len(pods.items)}) found for label '{label_selector}', using first: {pods.items[0].metadata.name}"
                    )
                return pods.items[0].metadata.name
            return None
        except ApiException as e:
            logger.error(f"Failed to get DaemonSet pod: {e}")
            return None

    async def exec_in_pod(
        self, namespace: str, pod_name: str, command: list[str]
    ) -> tuple[bool, str]:
        """Execute command in pod and return (success, output)"""
        try:
            resp = stream(
                self.core_v1.connect_get_namespaced_pod_exec,
                pod_name,
                namespace,
                command=command,
                stderr=True,
                stdin=False,
                stdout=True,
                tty=False,
            )
            return True, resp
        except ApiException as e:
            logger.error(f"Failed to exec in pod {pod_name}: {e}")
            return False, str(e)

    async def get_node_with_pod(
        self, namespace: str, pod_selector: str
    ) -> Optional[str]:
        """Get node name where a pod is running"""
        try:
            pods = await asyncio.to_thread(
                self.core_v1.list_namespaced_pod,
                namespace=namespace,
                label_selector=pod_selector,
            )
            if pods.items:
                if len(pods.items) > 1:
                    logger.warning(
                        f"Multiple pods ({len(pods.items)}) found for selector '{pod_selector}', using first: {pods.items[0].metadata.name}"
                    )
                return pods.items[0].spec.node_name
            return None
        except ApiException as e:
            logger.error(f"Failed to get node for pod: {e}")
            return None

    async def get_pod_by_prefix(self, namespace: str, pod_prefix: str) -> Optional[str]:
        """Get full pod name by prefix. Returns first match if multiple pods found."""
        try:
            pods = await asyncio.to_thread(self.core_v1.list_namespaced_pod, namespace)
            matching_pods = [
                p for p in pods.items if p.metadata.name.startswith(pod_prefix)
            ]
            if matching_pods:
                if len(matching_pods) > 1:
                    logger.warning(
                        f"Multiple pods ({len(matching_pods)}) found with prefix '{pod_prefix}', using first: {matching_pods[0].metadata.name}"
                    )
                return matching_pods[0].metadata.name
        except ApiException as e:
            logger.error(f"Error listing pods: {e}")
        return None

    async def get_pod_labels(self, namespace: str, pod_name: str) -> dict:
        """Get labels from a pod"""
        try:
            pod = await asyncio.to_thread(
                self.core_v1.read_namespaced_pod, pod_name, namespace
            )
            return pod.metadata.labels or {}
        except ApiException as e:
            logger.error(f"Error reading pod {pod_name}: {e}")
            return {}


# ============================================================================
# Agent Clients
# ============================================================================


class GPUFaultInjectorClient:
    """Client for GPU Fault Injector DaemonSet"""

    def __init__(
        self, k8s: KubernetesHelper, namespace: str = FAULT_INJECTION_NAMESPACE
    ):
        self.k8s = k8s
        self.namespace = namespace
        self.agent_port = 8083

    async def inject_fault(
        self,
        node_name: str,
        fault_type: GPUFaultType,
        duration: Optional[int],
        severity: FaultSeverity,
        parameters: dict[str, Any],
    ) -> tuple[bool, str]:
        """Inject GPU fault via agent on target node"""

        # Find agent pod on target node
        pods = await asyncio.to_thread(
            self.k8s.core_v1.list_namespaced_pod,
            namespace=self.namespace,
            label_selector="app=gpu-fault-injector",
            field_selector=f"spec.nodeName={node_name}",
        )

        if not pods.items:
            return False, f"No GPU fault injector found on node {node_name}"

        pod_ip = pods.items[0].status.pod_ip
        agent_url = f"http://{pod_ip}:{self.agent_port}"

        payload = {
            "fault_type": fault_type.value,
            "duration": duration,
            "severity": severity.value,
            "parameters": parameters,
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(f"{agent_url}/inject", json=payload)
                response.raise_for_status()
                return True, response.json().get("message", "Fault injected")
        except Exception as e:
            logger.error(f"Failed to inject GPU fault: {e}")
            return False, str(e)

    async def inject_xid_error(
        self,
        node_name: str,
        xid_type: int,
        gpu_id: int,
        duration: Optional[int],
        fault_id: str,
    ) -> tuple[bool, str]:
        """Inject specific XID error via agent on target node"""
        logger.info(f"[CLIENT-DEBUG] Step A: Looking for agent pod on node {node_name}")

        # Find agent pod on target node (use kernel agent label)
        pods = await asyncio.to_thread(
            self.k8s.core_v1.list_namespaced_pod,
            namespace=self.namespace,
            label_selector="app=gpu-fault-injector-kernel",
            field_selector=f"spec.nodeName={node_name}",
        )

        if not pods.items:
            logger.error(f"[CLIENT-DEBUG] No agent pod found on {node_name}")
            return False, f"No GPU fault injector found on node {node_name}"

        # We expect exactly one pod per node for DaemonSet
        pod_ip = pods.items[0].status.pod_ip
        pod_name = pods.items[0].metadata.name
        agent_url = f"http://{pod_ip}:{self.agent_port}"
        logger.info(f"[CLIENT-DEBUG] Step B: Found agent pod {pod_name} at {pod_ip}")

        payload = {
            "fault_id": fault_id,
            "xid_type": xid_type,
            "gpu_id": gpu_id,
            "duration": duration,
        }

        try:
            logger.info(
                f"[CLIENT-DEBUG] Step C: Sending POST to {agent_url}/inject-xid with payload: {payload}"
            )
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(f"{agent_url}/inject-xid", json=payload)
                logger.info(
                    f"[CLIENT-DEBUG] Step D: Received response status {response.status_code}"
                )
                response.raise_for_status()
                result = response.json().get(
                    "message", f"XID {xid_type} injected on GPU {gpu_id}"
                )
                logger.info(f"[CLIENT-DEBUG] Step E: Success - {result}")
                return True, result
        except Exception as e:
            logger.error(
                f"[CLIENT-DEBUG] Step F: Exception occurred - {type(e).__name__}: {e}"
            )
            return False, str(e)

    async def recover_fault(self, node_name: str, fault_id: str) -> tuple[bool, str]:
        """Recover from GPU fault"""

        pods = await asyncio.to_thread(
            self.k8s.core_v1.list_namespaced_pod,
            namespace=self.namespace,
            label_selector="app=gpu-fault-injector",
            field_selector=f"spec.nodeName={node_name}",
        )

        if not pods.items:
            return False, f"No GPU fault injector found on node {node_name}"

        pod_ip = pods.items[0].status.pod_ip
        agent_url = f"http://{pod_ip}:{self.agent_port}"

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{agent_url}/recover", json={"fault_id": fault_id}
                )
                response.raise_for_status()
                return True, response.json().get("message", "Fault recovered")
        except Exception as e:
            logger.error(f"Failed to recover GPU fault: {e}")
            return False, str(e)


class NetworkFaultInjectorClient:
    """Client for managing network faults via NetworkPolicy and ChaosMesh (no agents required)"""

    def __init__(
        self, k8s: KubernetesHelper, namespace: str = FAULT_INJECTION_NAMESPACE
    ):
        self.k8s = k8s
        self.namespace = namespace
        # Track NetworkPolicies for cleanup: {fault_id: {"policy_name": str, "namespace": str}}
        self.active_policies: dict[str, dict] = {}
        # Track ChaosMesh resources for cleanup: {fault_id: {"chaos_name": str, "namespace": str, "action": str}}
        self.active_chaos: dict[str, dict] = {}

        # Initialize Custom Object API for ChaosMesh
        try:
            self.custom_api = client.CustomObjectsApi()
        except Exception as e:
            logger.warning(f"Failed to initialize Custom Objects API: {e}")

    async def _get_target_pod_details(
        self, namespace: str, target_pod_prefix: str
    ) -> tuple[bool, str | dict[str, str], str]:
        """
        Looks up target Pod name and labels.
        Returns (success: bool, labels: dict[str, str] | error_msg: str, pod_name: str)
        """
        if not target_pod_prefix:
            return False, "target_pod_prefix parameter is required", ""

        target_pod_name = await self.k8s.get_pod_by_prefix(namespace, target_pod_prefix)
        if not target_pod_name:
            return (
                False,
                f"Could not find pod with prefix '{target_pod_prefix}' in namespace '{namespace}'",
                "",
            )

        target_labels = await self.k8s.get_pod_labels(namespace, target_pod_name)
        if not target_labels:
            return False, f"Could not get labels for pod '{target_pod_name}'", ""

        logger.info(f"Found target pod: {target_pod_name} with labels: {target_labels}")
        return True, target_labels, target_pod_name

    def _build_egress_match_expressions(
        self, block_nats: bool, block_specific_pods: list[dict[str, Any]]
    ) -> list:
        """Build match expressions for egress rules"""
        match_expressions = []

        # Block NATS if requested
        if block_nats:
            match_expressions.append(
                client.V1LabelSelectorRequirement(
                    key="app.kubernetes.io/name",
                    operator="NotIn",
                    values=["nats", "dynamo-platform-nats"],
                )
            )

        # Block specific pods if requested
        for pod_label_selector in block_specific_pods:
            for key, values in pod_label_selector.items():
                if not isinstance(values, list):
                    values = [values]
                match_expressions.append(
                    client.V1LabelSelectorRequirement(
                        key=key, operator="NotIn", values=values
                    )
                )

        return match_expressions

    def _build_egress_peers(
        self,
        match_expressions: list,
        allow_namespaces: list[str],
    ) -> list:
        """Build egress peers for NetworkPolicy"""
        egress_peers = []

        if match_expressions:
            # Allow pods that don't match the blocked criteria
            egress_peers.append(
                client.V1NetworkPolicyPeer(
                    pod_selector=client.V1LabelSelector(
                        match_expressions=match_expressions
                    )
                )
            )

        if allow_namespaces:
            for ns in allow_namespaces:
                egress_peers.append(
                    client.V1NetworkPolicyPeer(
                        namespace_selector=client.V1LabelSelector(
                            match_labels={"kubernetes.io/metadata.name": ns}
                        )
                    )
                )

        return egress_peers

    async def inject_partition(
        self,
        partition_type: NetworkPartitionType,
        source: str,
        target: str,
        mode: NetworkMode,
        parameters: dict[str, Any],
        duration: Optional[int],
    ) -> tuple[bool, str]:
        """Inject network partition using NetworkPolicy or ChaosMesh"""

        if mode == NetworkMode.NETWORKPOLICY:
            # Create NetworkPolicy directly (no agent needed)
            return await self._create_networkpolicy(
                partition_type, source, target, parameters
            )
        elif mode == NetworkMode.CHAOS_MESH:
            # Create ChaosMesh NetworkChaos resource
            return await self._create_chaos_mesh_network_fault(
                partition_type, source, target, parameters, duration
            )
        else:
            return (
                False,
                f"Unsupported mode: {mode.value}. Supported modes: 'networkpolicy', 'chaos_mesh'",
            )

    async def _create_networkpolicy(
        self,
        partition_type: NetworkPartitionType,
        source: str,
        target: str,
        parameters: dict[str, Any],
    ) -> tuple[bool, str]:
        """Create NetworkPolicy directly (no agent needed)"""

        namespace = parameters.get("namespace", source)
        target_pod_prefix = parameters.get("target_pod_prefix", "")

        # Validate target_pod_prefix early
        if not target_pod_prefix:
            return False, "target_pod_prefix parameter is required"

        # Network policy configuration
        block_nats = parameters.get("block_nats", True)
        block_all_egress = parameters.get("block_all_egress", False)
        block_ingress = parameters.get("block_ingress", False)
        block_specific_pods = parameters.get(
            "block_specific_pods", []
        )  # List of label selectors
        block_ports = parameters.get("block_ports", [])  # List of port numbers
        allow_namespaces = parameters.get(
            "allow_namespaces", []
        )  # List of namespace names

        # Replace underscores with hyphens for RFC 1123 compliance
        safe_partition_type = partition_type.value.replace("_", "-")
        policy_name = parameters.get(
            "policy_name", f"fault-injection-{safe_partition_type}-{id(self)}"
        )
        fault_id = parameters.get("fault_id", policy_name)

        # Get the actual pod and its labels
        success, labels_or_error, target_pod_name = await self._get_target_pod_details(
            namespace, target_pod_prefix
        )
        if not success:
            return False, labels_or_error  # type: ignore
        target_labels = labels_or_error  # type: ignore

        try:
            # Build NetworkPolicy
            policy_types: list[str] = []
            egress_rules: list[client.V1NetworkPolicyEgressRule] = []
            ingress_rules: list[client.V1NetworkPolicyIngressRule] = []

            # === EGRESS RULES ===
            if not block_all_egress:
                # Always allow DNS unless blocking all egress
                dns_rule = client.V1NetworkPolicyEgressRule(
                    to=[
                        client.V1NetworkPolicyPeer(
                            namespace_selector=client.V1LabelSelector(
                                match_labels={
                                    "kubernetes.io/metadata.name": "kube-system"
                                }
                            )
                        )
                    ],
                    ports=[client.V1NetworkPolicyPort(protocol="UDP", port=53)],
                )
                egress_rules.append(dns_rule)

                # Build egress rule based on configuration
                match_expressions = self._build_egress_match_expressions(
                    block_nats, block_specific_pods
                )

                # Build the main egress rule
                if match_expressions or allow_namespaces:
                    egress_peers = self._build_egress_peers(
                        match_expressions, allow_namespaces
                    )

                    # Build egress rule with optional port restrictions
                    egress_ports = None
                    if block_ports:
                        # Allow all ports except blocked ones (requires inverse logic)
                        # Note: NetworkPolicy doesn't support "NotIn" for ports directly
                        # So we log a warning and don't apply port blocking for now
                        logger.warning(
                            f"Port blocking requested but not fully supported: {block_ports}"
                        )

                    allow_rule = client.V1NetworkPolicyEgressRule(
                        to=egress_peers, ports=egress_ports
                    )
                    egress_rules.append(allow_rule)
                else:
                    # No restrictions - allow all egress
                    egress_rules.append(client.V1NetworkPolicyEgressRule())

            if egress_rules or block_all_egress:
                policy_types.append("Egress")

            # === INGRESS RULES ===
            if block_ingress:
                # Block all ingress (empty ingress list = deny all)
                policy_types.append("Ingress")
                # ingress_rules remains empty = block all

            # Create policy using actual pod labels
            # Note: For policy types in policy_types, we must provide the corresponding rules
            # - If we want to block all, pass empty list []
            # - If we don't have that policy type, pass None
            policy = client.V1NetworkPolicy(
                api_version="networking.k8s.io/v1",
                kind="NetworkPolicy",
                metadata=client.V1ObjectMeta(
                    name=policy_name,
                    namespace=namespace,
                    labels={
                        "managed-by": "fault-injection-api",
                        "fault-type": "network-partition",
                    },
                ),
                spec=client.V1NetworkPolicySpec(
                    pod_selector=client.V1LabelSelector(match_labels=target_labels),
                    policy_types=policy_types,
                    egress=egress_rules if "Egress" in policy_types else None,
                    ingress=ingress_rules if "Ingress" in policy_types else None,
                ),
            )

            # Create the NetworkPolicy
            await asyncio.to_thread(
                self.k8s.networking_v1.create_namespaced_network_policy,
                namespace=namespace,
                body=policy,
            )

            # Store for cleanup
            self.active_policies[fault_id] = {
                "policy_name": policy_name,
                "namespace": namespace,
            }

            logger.info(
                f"Created NetworkPolicy: {policy_name} in namespace {namespace}"
            )
            return True, f"NetworkPolicy {policy_name} created in {namespace}"

        except Exception as e:
            logger.error(f"Failed to create NetworkPolicy: {e}")
            return False, str(e)

    async def _create_chaos_mesh_network_fault(
        self,
        partition_type: NetworkPartitionType,
        source: str,
        target: str,
        parameters: dict[str, Any],
        duration: Optional[int],
    ) -> tuple[bool, str]:
        """Create ChaosMesh NetworkChaos resource for packet loss, delay, etc."""

        namespace = parameters.get("namespace", source)
        target_pod_prefix = parameters.get("target_pod_prefix", "")
        fault_id = parameters.get("fault_id", f"chaos-{uuid.uuid4().hex[:8]}")

        # ChaosMesh parameters
        packet_loss_percent = parameters.get("packet_loss_percent", 0)
        delay_ms = parameters.get("delay_ms", 0)
        delay_jitter_ms = parameters.get("delay_jitter_ms", 0)
        bandwidth_limit = parameters.get("bandwidth_limit", "")
        corrupt_percent = parameters.get("corrupt_percent", 0)
        duplicate_percent = parameters.get("duplicate_percent", 0)

        # Target configuration
        target_nats = parameters.get("target_nats", True)  # Block NATS by default
        target_specific_pods = parameters.get(
            "target_specific_pods", []
        )  # Target specific pod labels
        target_ports = parameters.get("target_ports", [])  # Target specific ports

        # Get the actual pod and its labels using helper
        success, labels_or_error, target_pod_name = await self._get_target_pod_details(
            namespace, target_pod_prefix
        )
        if not success:
            return False, labels_or_error  # type: ignore
        target_labels = labels_or_error  # type: ignore

        try:
            # Build the NetworkChaos spec
            safe_partition_type = partition_type.value.replace("_", "-")
            chaos_name = f"network-chaos-{safe_partition_type}-{uuid.uuid4().hex[:8]}"

            # Build selector for the source pod (the pod that will have faults injected)
            selector = {
                "namespaces": [namespace],
                "labelSelectors": target_labels,
            }

            # Build target selector (where the packets are going TO)
            target_spec: dict[str, Any] | None = None
            if target_nats or target_specific_pods:
                target_selector: dict[str, Any] = {
                    "mode": "all"
                }  # Default to all matching pods

                # Build label selector for target
                target_match_labels: dict[str, str] = {}
                target_match_expressions: list[dict[str, Any]] = []

                if target_nats:
                    # Target NATS services
                    # We'll use externalTargets for NATS service IPs or target pods with NATS labels
                    target_match_expressions.append(
                        {
                            "key": "app.kubernetes.io/name",
                            "operator": "In",
                            "values": ["nats", "dynamo-platform-nats"],
                        }
                    )

                # Add specific pod targets
                for pod_selector in target_specific_pods:
                    for key, value in pod_selector.items():
                        if not isinstance(value, list):
                            value = [value]
                        target_match_expressions.append(
                            {
                                "key": key,
                                "operator": "In",
                                "values": value,
                            }
                        )

                if target_match_labels or target_match_expressions:
                    label_selectors: dict[str, Any] = {}
                    if target_match_labels:
                        label_selectors["matchLabels"] = target_match_labels
                    if target_match_expressions:
                        label_selectors["matchExpressions"] = target_match_expressions
                    target_selector["labelSelectors"] = label_selectors
                    target_selector["namespaces"] = [namespace]

                target_spec = target_selector

            # Build action spec (packet loss, delay, etc.)
            action = "loss"  # Default action
            action_spec = {}

            if packet_loss_percent > 0:
                action = "loss"
                action_spec = {
                    "loss": {
                        "loss": str(packet_loss_percent),
                        "correlation": "0",  # Independent loss for each packet
                    }
                }
            elif delay_ms > 0:
                action = "delay"
                delay_spec = {
                    "latency": f"{delay_ms}ms",
                }
                if delay_jitter_ms > 0:
                    delay_spec["jitter"] = f"{delay_jitter_ms}ms"
                    delay_spec["correlation"] = "0"
                action_spec = {"delay": delay_spec}
            elif bandwidth_limit:
                action = "bandwidth"
                action_spec = {
                    "bandwidth": {
                        "rate": bandwidth_limit,
                        "limit": "20480",
                        "buffer": "10000",
                    }
                }
            elif corrupt_percent > 0:
                action = "corrupt"
                action_spec = {
                    "corrupt": {
                        "corrupt": str(corrupt_percent),
                        "correlation": "0",
                    }
                }
            elif duplicate_percent > 0:
                action = "duplicate"
                action_spec = {
                    "duplicate": {
                        "duplicate": str(duplicate_percent),
                        "correlation": "0",
                    }
                }

            # Build the full NetworkChaos resource
            chaos_resource: dict[str, Any] = {
                "apiVersion": "chaos-mesh.org/v1alpha1",
                "kind": "NetworkChaos",
                "metadata": {
                    "name": chaos_name,
                    "namespace": namespace,
                    "labels": {
                        "managed-by": "fault-injection-api",
                        "fault-type": "network-chaos",
                    },
                },
                "spec": {
                    "action": action,
                    "mode": "all",  # Apply to all matching pods
                    "selector": selector,
                    "direction": "to",  # Default: affect traffic going TO target
                    **action_spec,
                },
            }

            # Add target if specified
            if target_spec:
                chaos_resource["spec"]["target"] = target_spec

            # Add duration if specified
            if duration:
                chaos_resource["spec"]["duration"] = f"{duration}s"

            # Add port targeting if specified
            if target_ports:
                chaos_resource["spec"]["externalTargets"] = []
                # Note: For port-specific targeting, you may need to adjust based on your requirements

            # Create the NetworkChaos resource
            await asyncio.to_thread(
                self.custom_api.create_namespaced_custom_object,
                group="chaos-mesh.org",
                version="v1alpha1",
                namespace=namespace,
                plural="networkchaos",
                body=chaos_resource,
            )

            # Store for cleanup
            self.active_chaos[fault_id] = {
                "chaos_name": chaos_name,
                "namespace": namespace,
                "action": action,
            }

            logger.info(f"Created NetworkChaos: {chaos_name} in namespace {namespace}")

            message = f"NetworkChaos {chaos_name} created in {namespace}"
            if packet_loss_percent > 0:
                message += f" (packet loss: {packet_loss_percent}%)"
            elif delay_ms > 0:
                message += f" (delay: {delay_ms}ms"
                if delay_jitter_ms > 0:
                    message += f" ± {delay_jitter_ms}ms"
                message += ")"

            return True, message

        except Exception as e:
            logger.error(f"Failed to create NetworkChaos: {e}")
            import traceback

            traceback.print_exc()
            return False, f"Failed to create NetworkChaos: {str(e)}"

    async def recover_partition(self, fault_id: str) -> tuple[bool, str]:
        """Recover from network partition by deleting NetworkPolicy or ChaosMesh resource"""

        # Check if it's a NetworkPolicy
        if fault_id in self.active_policies:
            return await self._delete_networkpolicy(fault_id)
        # Check if it's a ChaosMesh resource
        elif fault_id in self.active_chaos:
            return await self._delete_chaos_mesh(fault_id)
        else:
            return False, f"Fault resource not found for fault_id: {fault_id}"

    async def _delete_networkpolicy(self, fault_id: str) -> tuple[bool, str]:
        """Delete NetworkPolicy directly (no agent needed)"""

        if fault_id not in self.active_policies:
            return False, f"NetworkPolicy not found for fault_id: {fault_id}"

        policy_info = self.active_policies[fault_id]
        policy_name = policy_info["policy_name"]
        namespace = policy_info["namespace"]

        try:
            await asyncio.to_thread(
                self.k8s.networking_v1.delete_namespaced_network_policy,
                name=policy_name,
                namespace=namespace,
            )

            # Remove from tracking
            del self.active_policies[fault_id]

            logger.info(
                f"Deleted NetworkPolicy: {policy_name} from namespace {namespace}"
            )
            return True, f"NetworkPolicy {policy_name} deleted from {namespace}"

        except Exception as e:
            logger.error(f"Failed to delete NetworkPolicy: {e}")
            return False, str(e)

    async def _delete_chaos_mesh(self, fault_id: str) -> tuple[bool, str]:
        """Delete ChaosMesh NetworkChaos resource"""

        if fault_id not in self.active_chaos:
            return False, f"NetworkChaos resource not found for fault_id: {fault_id}"

        chaos_info = self.active_chaos[fault_id]
        chaos_name = chaos_info["chaos_name"]
        namespace = chaos_info["namespace"]

        try:
            await asyncio.to_thread(
                self.custom_api.delete_namespaced_custom_object,
                group="chaos-mesh.org",
                version="v1alpha1",
                namespace=namespace,
                plural="networkchaos",
                name=chaos_name,
            )

            # Remove from tracking
            del self.active_chaos[fault_id]

            logger.info(
                f"Deleted NetworkChaos: {chaos_name} from namespace {namespace}"
            )
            return True, f"NetworkChaos {chaos_name} deleted from {namespace}"

        except Exception as e:
            logger.error(f"Failed to delete NetworkChaos: {e}")
            return False, str(e)


class MonitoringAgentClient:
    """Client for Monitoring Agent DaemonSet"""

    def __init__(
        self, k8s: KubernetesHelper, namespace: str = FAULT_INJECTION_NAMESPACE
    ):
        self.k8s = k8s
        self.namespace = namespace
        self.agent_port = 8083

    async def collect_metrics(
        self, target_namespace: str, duration: int = 60
    ) -> dict[str, Any]:
        """Collect metrics from all monitoring agents"""

        pods = await asyncio.to_thread(
            self.k8s.core_v1.list_namespaced_pod,
            namespace=self.namespace,
            label_selector="app=monitoring-agent",
        )

        if not pods.items:
            return {"error": "No monitoring agents found"}

        # Aggregate metrics from all agents
        all_metrics: dict[str, list] = {
            "gpu_metrics": [],
            "network_metrics": [],
            "inference_metrics": [],
        }

        for pod in pods.items:
            pod_ip = pod.status.pod_ip
            agent_url = f"http://{pod_ip}:{self.agent_port}"

            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.get(
                        f"{agent_url}/metrics",
                        params={"namespace": target_namespace, "duration": duration},
                    )
                    response.raise_for_status()
                    metrics = response.json()

                    if "gpu_metrics" in metrics:
                        all_metrics["gpu_metrics"].append(metrics["gpu_metrics"])
                    if "network_metrics" in metrics:
                        all_metrics["network_metrics"].append(
                            metrics["network_metrics"]
                        )
                    if "inference_metrics" in metrics:
                        all_metrics["inference_metrics"].append(
                            metrics["inference_metrics"]
                        )

            except Exception as e:
                logger.warning(
                    f"Failed to collect metrics from {pod.metadata.name}: {e}"
                )

        # Aggregate and average metrics
        return self._aggregate_metrics(all_metrics)

    def _aggregate_metrics(self, all_metrics: dict[str, list]) -> dict[str, Any]:
        """Aggregate metrics from multiple agents"""
        aggregated: dict[str, Any] = {}

        # Average GPU metrics
        if all_metrics["gpu_metrics"]:
            gpu_list = all_metrics["gpu_metrics"]
            aggregated["gpu_metrics"] = {
                "utilization": sum(m.get("utilization", 0) for m in gpu_list)
                / len(gpu_list),
                "memory_used_mb": sum(m.get("memory_used_mb", 0) for m in gpu_list)
                / len(gpu_list),
                "temperature_c": sum(m.get("temperature_c", 0) for m in gpu_list)
                / len(gpu_list),
                "power_watts": sum(m.get("power_watts", 0) for m in gpu_list)
                / len(gpu_list),
            }

        # Aggregate network metrics
        if all_metrics["network_metrics"]:
            net_list = all_metrics["network_metrics"]
            aggregated["network_metrics"] = {
                "latency_ms": sum(m.get("latency_ms", 0) for m in net_list)
                / len(net_list),
                "packet_loss_pct": sum(m.get("packet_loss_pct", 0) for m in net_list)
                / len(net_list),
                "throughput_mbps": sum(m.get("throughput_mbps", 0) for m in net_list)
                / len(net_list),
            }

        # Take first inference metrics (should be consistent across nodes)
        if all_metrics["inference_metrics"]:
            aggregated["inference_metrics"] = all_metrics["inference_metrics"][0]

        return aggregated


# ============================================================================
# Monitoring Service Verification
# ============================================================================


async def verify_monitoring_services(k8s: KubernetesHelper):
    """Verify that monitoring services are deployed and running"""
    logger.info("Verifying monitoring services...")

    # Define services with their namespace, name, and type
    required_services = [
        ("fault-injection-system", "dcgm-exporter", "DaemonSet"),
        ("monitoring", "prometheus-kube-prometheus-stack-prometheus", "StatefulSet"),
        ("fault-injection-system", "monitoring-agent", "DaemonSet"),
    ]

    apps_v1 = client.AppsV1Api()
    missing_services = []

    for namespace, service_name, service_type in required_services:
        try:
            if service_type == "DaemonSet":
                ds = apps_v1.read_namespaced_daemon_set(service_name, namespace)
                desired = ds.status.desired_number_scheduled or 0
                ready = ds.status.number_ready or 0
                if ready < desired:
                    logger.warning(
                        f"[WARN] {service_name} ({namespace}): {ready}/{desired} pods ready"
                    )
                else:
                    logger.info(
                        f"[CHECK] {service_name} ({namespace}): {ready}/{desired} pods ready"
                    )
            elif service_type == "Deployment":
                deploy = apps_v1.read_namespaced_deployment(service_name, namespace)
                desired = deploy.spec.replicas or 0
                ready = deploy.status.ready_replicas or 0
                if ready < desired:
                    logger.warning(
                        f"[WARN] {service_name} ({namespace}): {ready}/{desired} replicas ready"
                    )
                else:
                    logger.info(
                        f"[CHECK] {service_name} ({namespace}): {ready}/{desired} replicas ready"
                    )
            elif service_type == "StatefulSet":
                sts = apps_v1.read_namespaced_stateful_set(service_name, namespace)
                desired = sts.spec.replicas or 0
                ready = sts.status.ready_replicas or 0
                if ready < desired:
                    logger.warning(
                        f"[WARN] {service_name} ({namespace}): {ready}/{desired} replicas ready"
                    )
                else:
                    logger.info(
                        f"[CHECK] {service_name} ({namespace}): {ready}/{desired} replicas ready"
                    )
        except ApiException as e:
            if e.status == 404:
                logger.warning(
                    f"[WARN] {service_name} ({service_type}) not found in namespace {namespace}"
                )
                missing_services.append(f"{service_name} ({namespace})")
            else:
                logger.error(f"Error checking {service_name}: {e}")

    if missing_services:
        logger.warning(
            f"Missing monitoring services: {', '.join(missing_services)}. "
            f"Some monitoring features may not work. "
            f"Deploy with: kubectl apply -k deploy/"
        )
    else:
        logger.info("[CHECK] All monitoring services verified!")


# ============================================================================
# FastAPI Application
# ============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI"""
    # Startup
    logger.info("Starting Fault Injection API Service")
    app.state.k8s = KubernetesHelper()
    app.state.fault_tracker = FaultTracker()
    app.state.gpu_client = GPUFaultInjectorClient(app.state.k8s)
    app.state.network_client = NetworkFaultInjectorClient(app.state.k8s)
    app.state.monitoring_client = MonitoringAgentClient(app.state.k8s)

    # Verify monitoring services are running
    await verify_monitoring_services(app.state.k8s)

    # Clean up orphaned NetworkPolicies from previous runs
    logger.info("Checking for orphaned NetworkPolicies...")
    count, deleted = await app.state.k8s.cleanup_orphaned_network_policies()
    if count > 0:
        logger.warning(f"Cleaned up {count} orphaned NetworkPolicy(ies): {deleted}")
    else:
        logger.info("No orphaned NetworkPolicies found")

    yield

    # Shutdown
    logger.info("Shutting down Fault Injection API Service")
    # Recover all active faults
    active_faults = await app.state.fault_tracker.list_active_faults()
    for fault in active_faults:
        fault_type = fault.get("type", "")
        fault_id = fault["id"]
        logger.warning(
            f"Recovering active fault on shutdown: {fault_id} (type: {fault_type})"
        )
        try:
            if fault_type == "network":
                # Recover network partition
                await app.state.network_client.recover_partition(fault_id)
            elif fault_type.startswith("gpu"):
                # Recover GPU fault - extract node name from target
                target = fault.get("target", "")
                # Target format is "node_name" or "node_name/gpu0"
                node_name = target.split("/")[0] if "/" in target else target
                if node_name:
                    await app.state.gpu_client.recover_fault(
                        node_name=node_name, fault_id=fault_id
                    )
                else:
                    logger.error(
                        f"Cannot recover GPU fault {fault_id}: invalid target '{target}'"
                    )
        except Exception:
            logger.exception(f"Failed to recover fault {fault_id} during shutdown")


app = FastAPI(
    title="Fault Injection API",
    description="Hardware fault injection service for NVIDIA Dynamo testing",
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================================
# API Endpoints
# ============================================================================


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "fault-injection-api",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/api/v1/faults/gpu/inject", response_model=FaultResponse)
async def inject_gpu_fault(request: GPUFaultRequest):
    """Inject GPU fault on target node"""

    # Create fault tracking
    fault_id = await app.state.fault_tracker.create_fault(
        fault_type="gpu",
        target=request.node_name,
        details={
            "fault_type": request.fault_type.value,
            "duration": request.duration,
            "severity": request.severity.value,
            "parameters": request.parameters,
        },
    )

    # Inject fault via agent
    success, message = await app.state.gpu_client.inject_fault(
        node_name=request.node_name,
        fault_type=request.fault_type,
        duration=request.duration,
        severity=request.severity,
        parameters=request.parameters,
    )

    if not success:
        await app.state.fault_tracker.update_status(fault_id, FaultStatus.FAILED)
        raise HTTPException(
            status_code=500, detail=f"Failed to inject GPU fault: {message}"
        )

    await app.state.fault_tracker.update_status(fault_id, FaultStatus.INJECTED)

    return FaultResponse(
        fault_id=fault_id,
        status=FaultStatus.INJECTED,
        fault_type=f"gpu_{request.fault_type.value}",
        target=request.node_name,
        injected_at=datetime.now(timezone.utc).isoformat(),
        message=message,
    )


@app.post("/api/v1/faults/network/inject", response_model=FaultResponse)
async def inject_network_fault(request: NetworkFaultRequest):
    """Inject network partition/fault"""

    # Create fault tracking
    fault_id = await app.state.fault_tracker.create_fault(
        fault_type="network",
        target=f"{request.source} -> {request.target}",
        details={
            "partition_type": request.partition_type.value,
            "mode": request.mode.value,
            "duration": request.duration,
            "parameters": request.parameters,
        },
    )

    # Inject partition via agent (pass fault_id in parameters for tracking)
    parameters_with_id = {**(request.parameters or {}), "fault_id": fault_id}
    success, message = await app.state.network_client.inject_partition(
        partition_type=request.partition_type,
        source=request.source,
        target=request.target,
        mode=request.mode,
        parameters=parameters_with_id,
        duration=request.duration,
    )

    if not success:
        await app.state.fault_tracker.update_status(fault_id, FaultStatus.FAILED)
        raise HTTPException(
            status_code=500, detail=f"Failed to inject network fault: {message}"
        )

    await app.state.fault_tracker.update_status(fault_id, FaultStatus.INJECTED)

    return FaultResponse(
        fault_id=fault_id,
        status=FaultStatus.INJECTED,
        fault_type=f"network_{request.partition_type.value}",
        target=f"{request.source} -> {request.target}",
        injected_at=datetime.now(timezone.utc).isoformat(),
        message=message,
    )


@app.post("/api/v1/faults/{fault_id}/recover")
async def recover_fault(fault_id: str):
    """Recover from injected fault"""

    fault = await app.state.fault_tracker.get_fault(fault_id)
    if not fault:
        raise HTTPException(status_code=404, detail=f"Fault {fault_id} not found")

    await app.state.fault_tracker.update_status(fault_id, FaultStatus.RECOVERING)

    success = False
    message = ""

    if fault["type"] == "gpu":
        success, message = await app.state.gpu_client.recover_fault(
            node_name=fault["target"], fault_id=fault_id
        )
    elif fault["type"] == "network":
        success, message = await app.state.network_client.recover_partition(fault_id)

    if not success:
        await app.state.fault_tracker.update_status(fault_id, FaultStatus.FAILED)
        raise HTTPException(
            status_code=500, detail=f"Failed to recover fault: {message}"
        )

    await app.state.fault_tracker.update_status(fault_id, FaultStatus.RECOVERED)

    return {
        "fault_id": fault_id,
        "status": "recovered",
        "recovered_at": datetime.now(timezone.utc).isoformat(),
        "message": message,
    }


@app.get("/api/v1/faults")
async def list_faults(active_only: bool = Query(False)):
    """List all faults"""
    if active_only:
        faults = await app.state.fault_tracker.list_active_faults()
    else:
        faults = list(app.state.fault_tracker.faults.values())

    return {"faults": faults, "count": len(faults)}


@app.get("/api/v1/faults/{fault_id}")
async def get_fault(fault_id: str):
    """Get fault details"""
    fault = await app.state.fault_tracker.get_fault(fault_id)
    if not fault:
        raise HTTPException(status_code=404, detail=f"Fault {fault_id} not found")
    return fault


@app.get("/api/v1/metrics/collect", response_model=MetricsResponse)
async def collect_metrics(
    namespace: str = Query(..., description="Target namespace to monitor"),
    duration: int = Query(60, description="Collection duration in seconds"),
):
    """Collect metrics from monitoring agents"""

    metrics = await app.state.monitoring_client.collect_metrics(
        target_namespace=namespace, duration=duration
    )

    return MetricsResponse(
        timestamp=datetime.now(timezone.utc).isoformat(),
        namespace=namespace,
        gpu_metrics=metrics.get("gpu_metrics"),
        network_metrics=metrics.get("network_metrics"),
        inference_metrics=metrics.get("inference_metrics"),
        node_health=metrics.get("node_health"),
    )


@app.get("/api/v1/faults/gpu/xid-types")
async def list_xid_types():
    """List all available XID error types"""
    return {
        "xid_types": [
            {
                "code": 43,
                "name": "Kernel Assert",
                "severity": "high",
                "description": "GPU kernel assertion failure",
            },
            {
                "code": 48,
                "name": "Double Bit ECC Error",
                "severity": "critical",
                "description": "Uncorrectable memory error",
            },
            {
                "code": 74,
                "name": "NVLink Error",
                "severity": "high",
                "description": "NVLink communication failure",
            },
            {
                "code": 79,
                "name": "GPU Fell Off Bus",
                "severity": "critical",
                "description": "Complete GPU hardware failure",
            },
            {
                "code": 94,
                "name": "Contained ECC Error",
                "severity": "medium",
                "description": "Correctable memory error",
            },
            {
                "code": 95,
                "name": "Uncontained Error",
                "severity": "critical",
                "description": "Severe uncontained memory corruption",
            },
            {
                "code": 119,
                "name": "GSP Error",
                "severity": "high",
                "description": "GPU firmware/GSP failure",
            },
            {
                "code": 120,
                "name": "GSP Resource Manager Error",
                "severity": "high",
                "description": "GPU resource exhaustion",
            },
        ]
    }


class XIDFaultRequest(BaseModel):
    """Request to inject specific XID error"""

    node_name: str = Field(..., description="Target GPU node name")
    xid_type: int = Field(..., description="XID error code")
    gpu_id: int = Field(0, description="GPU device ID")
    duration: Optional[int] = Field(
        None, description="Duration in seconds (None = permanent)"
    )


@app.post("/api/v1/faults/gpu/inject/xid-43")
async def inject_xid_43(request: XIDFaultRequest):
    """Inject XID 43: Kernel Assert"""
    return await _inject_xid_error(request, 43, "Kernel Assert")


@app.post("/api/v1/faults/gpu/inject/xid-48")
async def inject_xid_48(request: XIDFaultRequest):
    """Inject XID 48: Double Bit ECC Error"""
    return await _inject_xid_error(request, 48, "Double Bit ECC Error")


@app.post("/api/v1/faults/gpu/inject/xid-74")
async def inject_xid_74(request: XIDFaultRequest):
    """Inject XID 74: NVLink Error"""
    return await _inject_xid_error(request, 74, "NVLink Error")


@app.post("/api/v1/faults/gpu/inject/xid-79")
async def inject_xid_79(request: XIDFaultRequest):
    """Inject XID 79: GPU Fell Off Bus"""
    return await _inject_xid_error(request, 79, "GPU Fell Off Bus")


@app.post("/api/v1/faults/gpu/inject/xid-94")
async def inject_xid_94(request: XIDFaultRequest):
    """Inject XID 94: Contained ECC Error"""
    return await _inject_xid_error(request, 94, "Contained ECC Error")


@app.post("/api/v1/faults/gpu/inject/xid-95")
async def inject_xid_95(request: XIDFaultRequest):
    """Inject XID 95: Uncontained Error"""
    return await _inject_xid_error(request, 95, "Uncontained Error")


@app.post("/api/v1/faults/gpu/inject/xid-119")
async def inject_xid_119(request: XIDFaultRequest):
    """Inject XID 119: GSP Error"""
    return await _inject_xid_error(request, 119, "GSP Error")


@app.post("/api/v1/faults/gpu/inject/xid-120")
async def inject_xid_120(request: XIDFaultRequest):
    """Inject XID 120: GSP Resource Manager Error"""
    return await _inject_xid_error(request, 120, "GSP RM Error")


async def _inject_xid_error(request: XIDFaultRequest, xid_code: int, xid_name: str):
    """Helper function to inject XID errors"""
    logger.info(
        f"[API-DEBUG] Step 1: Creating fault tracker entry for XID {xid_code} on {request.node_name}"
    )
    fault_id = await app.state.fault_tracker.create_fault(
        fault_type=f"gpu_xid_{xid_code}",
        target=f"{request.node_name}/gpu{request.gpu_id}",
        details={
            "xid_code": xid_code,
            "xid_name": xid_name,
            "gpu_id": request.gpu_id,
            "duration": request.duration,
        },
    )
    logger.info(f"[API-DEBUG] Step 2: Fault created with ID: {fault_id}")

    # Inject via GPU agent
    logger.info(
        f"[API-DEBUG] Step 3: Calling GPU client inject_xid_error for node {request.node_name}"
    )
    success, message = await app.state.gpu_client.inject_xid_error(
        node_name=request.node_name,
        xid_type=xid_code,
        gpu_id=request.gpu_id,
        duration=request.duration,
        fault_id=fault_id,
    )
    logger.info(
        f"[API-DEBUG] Step 4: GPU client returned - success={success}, message={message}"
    )

    if not success:
        await app.state.fault_tracker.update_status(fault_id, FaultStatus.FAILED)
        raise HTTPException(
            status_code=500, detail=f"Failed to inject XID {xid_code}: {message}"
        )

    logger.info("[API-DEBUG] Step 5: Updating fault status to INJECTED")
    await app.state.fault_tracker.update_status(fault_id, FaultStatus.INJECTED)

    return FaultResponse(
        fault_id=fault_id,
        status=FaultStatus.INJECTED,
        fault_type=f"gpu_xid_{xid_code}",
        target=f"{request.node_name}/gpu{request.gpu_id}",
        injected_at=datetime.now(timezone.utc).isoformat(),
        message=message,
    )


@app.post("/api/v1/faults/network/cleanup")
async def cleanup_network_policies(namespace: str = Query(DEFAULT_DYNAMO_NAMESPACE)):
    """
    Clean up orphaned NetworkPolicies created by fault injection.

    This endpoint manually triggers cleanup of NetworkPolicy resources that may have
    been left behind from previous test runs or API restarts.
    """
    logger.info(f"Manual cleanup requested for namespace: {namespace}")

    try:
        count, deleted = await app.state.k8s.cleanup_orphaned_network_policies(
            namespace
        )

        return {
            "status": "success",
            "namespace": namespace,
            "policies_deleted": count,
            "deleted_policy_names": deleted,
            "message": f"Cleaned up {count} orphaned NetworkPolicy(ies)",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"Failed to cleanup NetworkPolicies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        log_level="info",
    )
