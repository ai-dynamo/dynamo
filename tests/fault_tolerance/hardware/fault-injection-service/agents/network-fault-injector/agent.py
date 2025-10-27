# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
"""
Network Fault Injector Agent - Runs as DaemonSet on all nodes.

This agent provides network fault injection capabilities:
- Network partitions via Kubernetes NetworkPolicy
- Packet loss via tc (traffic control)
- Latency injection via tc netem
- Bandwidth throttling via tc tbf
- Connection drops via iptables
"""

import logging
import os
import subprocess
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from kubernetes import client, config
from kubernetes.client.rest import ApiException
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# Models and Enums
# ============================================================================


class NetworkPartitionType(str, Enum):
    FRONTEND_WORKER = "frontend_worker"
    WORKER_NATS = "worker_nats"
    WORKER_WORKER = "worker_worker"
    CUSTOM = "custom"


class NetworkMode(str, Enum):
    DROP = "drop"
    DELAY = "delay"
    THROTTLE = "throttle"
    CORRUPT = "corrupt"
    NETWORKPOLICY = "networkpolicy"  # Use Kubernetes NetworkPolicy


class PartitionInjectRequest(BaseModel):
    partition_type: NetworkPartitionType
    source: str
    target: str
    mode: NetworkMode
    parameters: dict[str, Any] = {}
    duration: Optional[int] = None


class PartitionRecoverRequest(BaseModel):
    fault_id: str


# ============================================================================
# Network Fault Injector
# ============================================================================


class NetworkFaultInjector:
    """Network fault injection using tc, iptables, and NetworkPolicy"""

    def __init__(self):
        self.active_faults: dict[str, dict[str, Any]] = {}
        self.node_name = os.getenv("NODE_NAME", "unknown")
        self.tc_available = self._check_tc()
        self.iptables_available = self._check_iptables()

        # Initialize Kubernetes client
        try:
            config.load_incluster_config()
            self.k8s_core = client.CoreV1Api()
            self.k8s_network = client.NetworkingV1Api()
            self.k8s_available = True
            logger.info("Kubernetes client initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Kubernetes client: {e}")
            self.k8s_core = None
            self.k8s_network = None
            self.k8s_available = False

        logger.info(f"Network Fault Injector initialized on node: {self.node_name}")
        logger.info(f"tc available: {self.tc_available}")
        logger.info(f"iptables available: {self.iptables_available}")
        logger.info(f"Kubernetes available: {self.k8s_available}")

    def _check_tc(self) -> bool:
        """Check if tc (traffic control) is available"""
        try:
            result = subprocess.run(["tc", "-V"], capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except Exception as e:
            logger.warning(f"tc not available: {e}")
            return False

    def _check_iptables(self) -> bool:
        """Check if iptables is available"""
        try:
            result = subprocess.run(
                ["iptables", "--version"], capture_output=True, text=True, timeout=5
            )
            return result.returncode == 0
        except Exception as e:
            logger.warning(f"iptables not available: {e}")
            return False

    def _run_command(self, command: list[str], timeout: int = 30) -> tuple[bool, str]:
        """Run shell command with timeout"""
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
            success = result.returncode == 0
            output = result.stdout if success else result.stderr
            return success, output.strip()
        except subprocess.TimeoutExpired:
            return False, "Command timed out"
        except Exception as e:
            return False, str(e)

    async def inject_packet_drop(
        self, interface: str, loss_pct: float, target_ip: Optional[str] = None
    ) -> tuple[bool, str]:
        """
        Inject packet loss using tc netem.

        Args:
            interface: Network interface (e.g., eth0)
            loss_pct: Packet loss percentage (0-100)
            target_ip: Optional target IP to apply loss to specific destination
        """
        logger.info(f"Injecting {loss_pct}% packet loss on {interface}")

        # Add qdisc with packet loss
        commands = [
            ["tc", "qdisc", "add", "dev", interface, "root", "netem", "loss", f"{loss_pct}%"]
        ]

        # If target IP specified, use iptables to mark packets
        if target_ip:
            commands.insert(
                0, ["iptables", "-A", "OUTPUT", "-d", target_ip, "-j", "MARK", "--set-mark", "1"]
            )
            # Modify tc command to only affect marked packets
            commands[1] = [
                "tc",
                "qdisc",
                "add",
                "dev",
                interface,
                "root",
                "handle",
                "1:",
                "netem",
                "loss",
                f"{loss_pct}%",
            ]

        for cmd in commands:
            success, output = self._run_command(cmd)
            if not success:
                logger.error(f"Failed to inject packet loss: {output}")
                return False, f"Failed to inject packet loss: {output}"

        return True, f"Packet loss {loss_pct}% injected on {interface}"

    async def inject_latency(
        self, interface: str, delay_ms: int, jitter_ms: int = 0, target_ip: Optional[str] = None
    ) -> tuple[bool, str]:
        """
        Inject network latency using tc netem.

        Args:
            interface: Network interface
            delay_ms: Delay in milliseconds
            jitter_ms: Jitter in milliseconds (variance)
            target_ip: Optional target IP
        """
        logger.info(f"Injecting {delay_ms}ms latency (±{jitter_ms}ms jitter) on {interface}")

        # Build netem command
        netem_cmd = [
            "tc",
            "qdisc",
            "add",
            "dev",
            interface,
            "root",
            "netem",
            "delay",
            f"{delay_ms}ms",
        ]
        if jitter_ms > 0:
            netem_cmd.extend([f"{jitter_ms}ms"])

        success, output = self._run_command(netem_cmd)
        if not success:
            return False, f"Failed to inject latency: {output}"

        return True, f"Latency {delay_ms}ms injected on {interface}"

    async def inject_bandwidth_throttle(
        self, interface: str, bandwidth_mbps: int
    ) -> tuple[bool, str]:
        """
        Throttle bandwidth using tc tbf (token bucket filter).

        Args:
            interface: Network interface
            bandwidth_mbps: Bandwidth limit in Mbps
        """
        logger.info(f"Throttling bandwidth to {bandwidth_mbps}Mbps on {interface}")

        # Convert Mbps to kbps
        rate_kbps = bandwidth_mbps * 1000

        # Add tbf qdisc
        success, output = self._run_command(
            [
                "tc",
                "qdisc",
                "add",
                "dev",
                interface,
                "root",
                "tbf",
                "rate",
                f"{rate_kbps}kbit",
                "burst",
                "32kbit",
                "latency",
                "400ms",
            ]
        )

        if not success:
            return False, f"Failed to throttle bandwidth: {output}"

        return True, f"Bandwidth throttled to {bandwidth_mbps}Mbps on {interface}"

    async def inject_connection_drop(
        self, source_ip: str, dest_ip: str, port: Optional[int] = None
    ) -> tuple[bool, str]:
        """
        Drop connections using iptables.

        Args:
            source_ip: Source IP or CIDR
            dest_ip: Destination IP or CIDR
            port: Optional destination port
        """
        logger.info(f"Dropping connections: {source_ip} -> {dest_ip}:{port or 'any'}")

        # Build iptables rule
        cmd = ["iptables", "-A", "OUTPUT", "-s", source_ip, "-d", dest_ip, "-j", "DROP"]

        if port:
            cmd.extend(["-p", "tcp", "--dport", str(port)])

        success, output = self._run_command(cmd)
        if not success:
            return False, f"Failed to drop connections: {output}"

        return True, f"Connections dropped: {source_ip} -> {dest_ip}"

    async def inject_networkpolicy_partition(
        self,
        partition_type: NetworkPartitionType,
        source: str,
        target: str,
        parameters: dict[str, Any],
    ) -> tuple[bool, str, Optional[str]]:
        """
        Inject network partition using Kubernetes NetworkPolicy.

        Parameters:
            block_nats: bool - If True, blocks target's access to NATS (default: True for frontend-worker)
            source_pod_prefix: str - Source pod name prefix
            target_pod_prefix: str - Target pod name prefix
            policy_name: str - Custom policy name (optional)
            namespace: str - Kubernetes namespace (default: source)

        Returns:
            (success, message, policy_name)
        """
        if not self.k8s_available:
            return False, "Kubernetes client not available", None

        namespace = parameters.get("namespace", source)
        source_pod_prefix = parameters.get("source_pod_prefix", "")
        target_pod_prefix = parameters.get("target_pod_prefix", "")
        block_nats = parameters.get("block_nats", True)
        policy_name = parameters.get("policy_name")

        # Determine what to block based on partition type
        if partition_type == NetworkPartitionType.FRONTEND_WORKER:
            block_nats = parameters.get("block_nats", True)
            if not target_pod_prefix:
                target_pod_prefix = "worker"
        elif partition_type == NetworkPartitionType.WORKER_NATS:
            block_nats = True
            if not target_pod_prefix:
                target_pod_prefix = "worker"

        # Find target pod
        try:
            pods = self.k8s_core.list_namespaced_pod(namespace)
            target_pod = None
            for pod in pods.items:
                if target_pod_prefix and pod.metadata.name.startswith(target_pod_prefix):
                    target_pod = pod
                    break

            if not target_pod:
                return (
                    False,
                    f"Could not find pod with prefix '{target_pod_prefix}' in namespace '{namespace}'",
                    None,
                )

            target_labels = target_pod.metadata.labels or {}
            if not target_labels:
                return False, f"Target pod {target_pod.metadata.name} has no labels", None

            # Generate policy name if not provided
            if not policy_name:
                target_short = target_pod_prefix or target_pod.metadata.name.split("-")[-1]
                policy_name = f"fault-injector-{target_short}-partition"

            # Create NetworkPolicy based on configuration
            if block_nats:
                # Block target pod's egress to NATS
                policy = client.V1NetworkPolicy(
                    api_version="networking.k8s.io/v1",
                    kind="NetworkPolicy",
                    metadata=client.V1ObjectMeta(
                        name=policy_name,
                        namespace=namespace,
                        labels={
                            "managed-by": "fault-injector-api",
                            "fault-type": "network-partition",
                        },
                    ),
                    spec=client.V1NetworkPolicySpec(
                        pod_selector=client.V1LabelSelector(
                            match_labels={
                                "app.kubernetes.io/name": target_labels.get(
                                    "app.kubernetes.io/name"
                                )
                            }
                        ),
                        policy_types=["Egress"],
                        egress=[
                            # Allow DNS
                            client.V1NetworkPolicyEgressRule(
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
                            ),
                            # Allow all traffic EXCEPT to NATS
                            client.V1NetworkPolicyEgressRule(
                                to=[
                                    client.V1NetworkPolicyPeer(
                                        pod_selector=client.V1LabelSelector(
                                            match_expressions=[
                                                client.V1LabelSelectorRequirement(
                                                    key="app.kubernetes.io/name",
                                                    operator="NotIn",
                                                    values=["nats", "dynamo-platform-nats"],
                                                )
                                            ]
                                        )
                                    )
                                ]
                            ),
                        ],
                    ),
                )
                effect = f"Blocked {target_pod.metadata.name} egress to NATS"
            else:
                # Block direct pod-to-pod traffic (requires source pod labels)
                if not source_pod_prefix:
                    return False, "source_pod_prefix required for direct pod-to-pod blocking", None

                # Find source pod
                source_pod = None
                for pod in pods.items:
                    if pod.metadata.name.startswith(source_pod_prefix):
                        source_pod = pod
                        break

                if not source_pod:
                    return False, f"Could not find pod with prefix '{source_pod_prefix}'", None

                source_labels = source_pod.metadata.labels or {}
                source_selector_label = None
                source_selector_value = None
                for label_key in [
                    "app.kubernetes.io/name",
                    "app.kubernetes.io/component",
                    "grove.io/podclique",
                ]:
                    if label_key in source_labels:
                        source_selector_label = label_key
                        source_selector_value = source_labels[label_key]
                        break

                if not source_selector_label:
                    return (
                        False,
                        f"Could not find suitable label on source pod {source_pod.metadata.name}",
                        None,
                    )

                policy = client.V1NetworkPolicy(
                    api_version="networking.k8s.io/v1",
                    kind="NetworkPolicy",
                    metadata=client.V1ObjectMeta(
                        name=policy_name,
                        namespace=namespace,
                        labels={
                            "managed-by": "fault-injector-api",
                            "fault-type": "network-partition",
                        },
                    ),
                    spec=client.V1NetworkPolicySpec(
                        pod_selector=client.V1LabelSelector(
                            match_labels={
                                "app.kubernetes.io/name": target_labels.get(
                                    "app.kubernetes.io/name"
                                )
                            }
                        ),
                        policy_types=["Ingress"],
                        ingress=[
                            # Allow traffic from all pods EXCEPT the source pod
                            client.V1NetworkPolicyIngressRule(
                                _from=[
                                    client.V1NetworkPolicyPeer(
                                        pod_selector=client.V1LabelSelector(
                                            match_expressions=[
                                                client.V1LabelSelectorRequirement(
                                                    key=source_selector_label,
                                                    operator="NotIn",
                                                    values=[source_selector_value],
                                                )
                                            ]
                                        )
                                    )
                                ]
                            )
                        ],
                    ),
                )
                effect = f"Blocked {source_pod.metadata.name} -> {target_pod.metadata.name}"

            # Create the NetworkPolicy
            self.k8s_network.create_namespaced_network_policy(namespace=namespace, body=policy)

            logger.info(f"Created NetworkPolicy: {policy_name} in {namespace}")
            return True, effect, policy_name

        except ApiException as e:
            if e.status == 409:
                return True, f"NetworkPolicy {policy_name} already exists", policy_name
            return False, f"Failed to create NetworkPolicy: {e}", None
        except Exception as e:
            return False, f"Error creating NetworkPolicy: {e}", None

    async def inject_network_partition(
        self,
        partition_type: NetworkPartitionType,
        source: str,
        target: str,
        mode: NetworkMode,
        parameters: dict[str, Any],
    ) -> tuple[bool, str]:
        """
        Inject network partition based on type and mode.
        """
        # Use NetworkPolicy mode if specified
        if mode == NetworkMode.NETWORKPOLICY:
            success, message, policy_name = await self.inject_networkpolicy_partition(
                partition_type, source, target, parameters
            )
            if success and policy_name:
                # Store policy name for cleanup
                fault_id = parameters.get("fault_id", f"fault-{id(self)}")
                self.active_faults[fault_id] = {
                    "type": "networkpolicy",
                    "policy_name": policy_name,
                    "namespace": parameters.get("namespace", source),
                }
            return success, message

        # For tc-based modes, get network interface
        interface = self._get_primary_interface()
        if not interface:
            return False, "Failed to determine primary network interface"

        if mode == NetworkMode.DROP:
            # Full packet drop - simulates complete partition
            loss_pct = parameters.get("packet_loss_pct", 100)
            return await self.inject_packet_drop(interface, loss_pct)

        elif mode == NetworkMode.DELAY:
            # Add latency
            delay_ms = parameters.get("delay_ms", 500)
            jitter_ms = parameters.get("jitter_ms", 50)
            return await self.inject_latency(interface, delay_ms, jitter_ms)

        elif mode == NetworkMode.THROTTLE:
            # Bandwidth throttling
            bandwidth_mbps = parameters.get("bandwidth_mbps", 10)
            return await self.inject_bandwidth_throttle(interface, bandwidth_mbps)

        return False, f"Unsupported network mode: {mode}"

    def _get_primary_interface(self) -> Optional[str]:
        """Get primary network interface"""
        try:
            # Get default route interface
            result = subprocess.run(
                ["ip", "route", "show", "default"], capture_output=True, text=True, timeout=5
            )

            if result.returncode == 0:
                # Parse output: "default via X.X.X.X dev eth0"
                parts = result.stdout.split()
                if "dev" in parts:
                    idx = parts.index("dev")
                    if idx + 1 < len(parts):
                        return parts[idx + 1]

            # Fallback to eth0
            return "eth0"

        except Exception as e:
            logger.warning(f"Failed to get primary interface: {e}")
            return "eth0"

    async def recover_fault(self, fault_id: str) -> tuple[bool, str]:
        """Recover from network fault"""
        if fault_id not in self.active_faults:
            return False, f"Fault {fault_id} not found"

        fault = self.active_faults[fault_id]
        fault_type = fault.get("type", "tc")

        if fault_type == "networkpolicy":
            # Delete NetworkPolicy
            policy_name = fault.get("policy_name")
            namespace = fault.get("namespace", "default")

            if not policy_name:
                return False, "Policy name not found in fault record"

            try:
                self.k8s_network.delete_namespaced_network_policy(
                    name=policy_name, namespace=namespace
                )
                del self.active_faults[fault_id]
                logger.info(f"Deleted NetworkPolicy: {policy_name} from {namespace}")
                return True, f"NetworkPolicy {policy_name} deleted"
            except ApiException as e:
                if e.status == 404:
                    # Policy already deleted, clean up tracking
                    del self.active_faults[fault_id]
                    return True, f"NetworkPolicy {policy_name} already deleted"
                return False, f"Failed to delete NetworkPolicy: {e}"
            except Exception as e:
                return False, f"Error deleting NetworkPolicy: {e}"
        else:
            # TC-based fault recovery
            interface = fault.get("interface", "eth0")

            # Remove tc qdisc
            success, output = self._run_command(["tc", "qdisc", "del", "dev", interface, "root"])

            # Flush iptables rules (be careful in production!)
            # In production, we should track specific rules and remove only those
            self._run_command(["iptables", "-F", "OUTPUT"])

            del self.active_faults[fault_id]

            return True, f"Network fault recovered on {interface}"

    async def inject_fault(self, request: PartitionInjectRequest) -> tuple[bool, str]:
        """Inject network partition/fault"""

        # Use fault_id from parameters if provided (from API), otherwise generate one
        fault_id = request.parameters.get("fault_id")
        if not fault_id:
            fault_id = f"net_fault_{len(self.active_faults)}_{id(request)}"

        # Ensure fault_id is in parameters for NetworkPolicy tracking
        params_with_id = {**request.parameters, "fault_id": fault_id}

        success, message = await self.inject_network_partition(
            partition_type=request.partition_type,
            source=request.source,
            target=request.target,
            mode=request.mode,
            parameters=params_with_id,
        )

        # For tc-based faults, we need to track them here
        # (NetworkPolicy faults are tracked inside inject_networkpolicy_partition)
        if success and request.mode != NetworkMode.NETWORKPOLICY:
            self.active_faults[fault_id] = {
                "type": request.partition_type.value,
                "mode": request.mode.value,
                "interface": self._get_primary_interface(),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }

        return success, message


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(title="Network Fault Injector Agent", version="1.0.0")
injector = NetworkFaultInjector()


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "node": injector.node_name,
        "tc_available": injector.tc_available,
        "iptables_available": injector.iptables_available,
        "active_faults": len(injector.active_faults),
    }


@app.post("/inject")
async def inject_partition(request: PartitionInjectRequest):
    """Inject network partition"""
    logger.info(f"Received partition injection request: {request.partition_type}")

    success, message = await injector.inject_fault(request)

    if not success:
        raise HTTPException(status_code=500, detail=message)

    return {
        "status": "injected",
        "node": injector.node_name,
        "message": message,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/recover")
async def recover_partition(request: PartitionRecoverRequest):
    """Recover from network partition"""
    logger.info(f"Received partition recovery request: {request.fault_id}")

    try:
        success, message = await injector.recover_fault(request.fault_id)

        if not success:
            logger.error(f"Recovery failed for {request.fault_id}: {message}")
            raise HTTPException(status_code=500, detail=message)

        logger.info(f"Successfully recovered fault {request.fault_id}: {message}")
        return {
            "status": "recovered",
            "node": injector.node_name,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error recovering fault {request.fault_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@app.get("/faults")
async def list_active_faults():
    """List active faults on this node"""
    return {
        "node": injector.node_name,
        "active_faults": list(injector.active_faults.keys()),
        "count": len(injector.active_faults),
    }


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8082,
        log_level="info",
    )
