"""
NVSentinel workflow simulation helpers.

Provides high-level functions that simulate the complete NVSentinel
fault tolerance workflow for testing purposes.
"""

import time
from typing import Optional

from kubernetes import client

try:
    # Try relative import first (when used as package)
    from .inference_testing import InferenceLoadTester
    from .k8s_operations import NodeOperations, PodOperations
except ImportError:
    # Fall back to absolute import (when helpers dir is in sys.path)
    from inference_testing import InferenceLoadTester
    from k8s_operations import NodeOperations, PodOperations


class NVSentinelWorkflowSimulator:
    """
    Simulates NVSentinel fault tolerance workflow.

    In production, these steps are automatic when NVSentinel is configured:
    1. Fault Detection (syslog-health-monitor)
    2. Node Cordoning (fault-quarantine-module)
    3. Pod Draining (node-drainer-module)
    4. Component Reset (fault-remediation-module)
    5. Node Uncordoning (fault-quarantine-module)
    """

    def __init__(
        self, k8s_core: client.CoreV1Api, namespace: str, deployment_name: str
    ):
        """
        Initialize NVSentinel workflow simulator.

        Args:
            k8s_core: Kubernetes CoreV1Api client
            namespace: Kubernetes namespace for workload
            deployment_name: Name of deployment to monitor
        """
        self.namespace = namespace
        self.deployment_name = deployment_name
        self.node_ops = NodeOperations(k8s_core)
        self.pod_ops = PodOperations(k8s_core)

        # Track state for cleanup
        self.cordoned_node = None
        self.node_was_initially_cordoned = False

    def cordon_faulty_node(self, node_name: str) -> bool:
        """
        Cordon the faulty node (Step 2: Node Quarantine).

        Simulates: NVSentinel/fault-quarantine-module
        Reference: fault-quarantine-module/pkg/reconciler/node_quarantine.go:274

        Args:
            node_name: Name of node to cordon

        Returns:
            True if successful
        """
        print("\n" + "=" * 80)
        print("STEP: Cordon Node (Prevent New Pods on Faulty Node)")
        print("=" * 80)
        print("[MANUAL] This step done by test")
        print(
            "        In production: NVSentinel fault-quarantine-module does this automatically"
        )
        print(f"[→] Cordoning node: {node_name}")

        # Track initial state for cleanup
        self.cordoned_node = node_name
        self.node_was_initially_cordoned = self.node_ops.is_node_cordoned(node_name)

        success = self.node_ops.cordon_node(node_name, reason="xid-fault-test")

        if success:
            print("[✓] Node cordoned successfully")
        else:
            print("[✗] Failed to cordon node")

        return success

    def drain_faulty_node(self, node_name: str) -> int:
        """
        Drain pods from the faulty node (Step 3: Pod Eviction).

        Simulates: NVSentinel/node-drainer-module
        Reference: node-drainer-module/pkg/informers/informers.go:471-535

        Args:
            node_name: Name of node to drain

        Returns:
            Number of pods drained
        """
        print(
            "\n[MANUAL] Draining node (test simulates NVSentinel node-drainer-module)"
        )
        print(
            "        In production: NVSentinel node-drainer-module does this automatically"
        )
        print(f"[→] Evicting all pods from {node_name}")

        # Show all worker pods and their distribution
        all_pods = self.pod_ops.get_pod_distribution(
            self.namespace,
            f"nvidia.com/dynamo-component-type=worker,nvidia.com/dynamo-graph-deployment-name={self.deployment_name}",
        )
        print(f"    All worker pods: {sum(all_pods.values())} total")
        for node, count in all_pods.items():
            marker = "← TARGET NODE" if node == node_name else ""
            print(f"      - {node}: {count} pod(s) {marker}")

        # Drain pods from target node
        label_selector = f"nvidia.com/dynamo-component-type=worker,nvidia.com/dynamo-graph-deployment-name={self.deployment_name}"
        drained_count = self.pod_ops.drain_pods(
            self.namespace, label_selector, node_name
        )

        print(f"[✓] Node drained - {drained_count} pods evicted")
        print("    Kubernetes will reschedule evicted pods to healthy nodes")

        return drained_count

    def restart_gpu_driver(self, node_name: str, wait_timeout: int = 300) -> bool:
        """
        Restart GPU driver on the node (Step 4: Component Reset).

        Simulates: NVSentinel/fault-remediation-module
        Reference: fault-remediation-module/pkg/reconciler/reconciler.go:204
        Action: COMPONENT_RESET for XID 79

        Args:
            node_name: Name of node to restart GPU driver on
            wait_timeout: Max seconds to wait for driver ready

        Returns:
            True if driver restart succeeded
        """
        print("\n" + "=" * 80)
        print("STEP: GPU Driver Restart (Reset GPU State)")
        print("=" * 80)
        print("[MANUAL] Simulating NVSentinel fault-remediation-module")
        print(
            "        In production: fault-remediation-module reads event from MongoDB"
        )
        print(
            "        Reference: fault-remediation-module/pkg/reconciler/reconciler.go:204"
        )
        print("        Action: COMPONENT_RESET (restarts GPU driver)")

        success = self.node_ops.restart_gpu_driver(node_name, wait_timeout)

        if success:
            print("[✓] GPU driver restarted - GPU state reset")
            print("    Node is now healthy and ready for pod rescheduling")
        else:
            print("[⚠] GPU driver restart failed or timed out")
            print("    Leaving node cordoned - GPU may still be unhealthy")

        return success

    def uncordon_node(self, node_name: str) -> bool:
        """
        Uncordon the node (Step 5: Node Uncordon).

        Simulates: NVSentinel/fault-quarantine-module uncordon logic
        Reference: fault-quarantine-module/pkg/reconciler/reconciler.go:840-844
        Trigger: All health checks recovered

        Args:
            node_name: Name of node to uncordon

        Returns:
            True if successful
        """
        print(f"\n[MANUAL] Uncordoning node: {node_name}")
        print(
            "        In production: fault-quarantine-module does this when health checks pass"
        )
        print(
            "        Reference: fault-quarantine-module/pkg/reconciler/reconciler.go:840-844"
        )

        success = self.node_ops.uncordon_node(
            node_name, self.node_was_initially_cordoned
        )

        if success:
            print("[✓] Node uncordoned - ready for rescheduling")
        else:
            print("[✗] Failed to uncordon node")

        return success

    def wait_for_pod_rescheduling(
        self,
        expected_count: int = 3,
        exclude_node: Optional[str] = None,
        timeout: int = 900,
    ) -> bool:
        """
        Wait for pods to reschedule to healthy nodes (Step 6: Rescheduling).

        This is automatic in Kubernetes.

        Args:
            expected_count: Expected number of ready pods
            exclude_node: Node to exclude from count (faulty node)
            timeout: Max seconds to wait

        Returns:
            True if pods rescheduled successfully
        """
        print("\n" + "=" * 80)
        print("STEP: Pod Rescheduling")
        print("=" * 80)
        print("[AUTOMATED] Kubernetes reschedules pods to healthy nodes")
        print("            Dynamo operator manages pod lifecycle")
        print(
            f"[→] Waiting for {expected_count} pods to reschedule (max {timeout//60} minutes)"
        )

        label_selector = f"nvidia.com/dynamo-component-type=worker,nvidia.com/dynamo-graph-deployment-name={self.deployment_name}"

        success = self.pod_ops.wait_for_pods_ready(
            self.namespace, label_selector, expected_count, timeout, exclude_node
        )

        if not success:
            print(f"[⚠] Not all pods ready after {timeout//60} minutes")

        # Show final distribution
        distribution = self.pod_ops.get_pod_distribution(self.namespace, label_selector)

        print("\n[→] Final pod distribution:")
        if exclude_node:
            on_faulty = distribution.get(exclude_node, 0)
            on_healthy = sum(
                count for node, count in distribution.items() if node != exclude_node
            )
            print(f"    Pods on faulty node: {on_faulty}")
            print(f"    Pods on healthy nodes: {on_healthy}")

        if distribution:
            print("\n[→] Distribution across nodes:")
            for node, count in sorted(distribution.items()):
                marker = "(faulty)" if node == exclude_node else ""
                print(f"    • {node}: {count} pod(s) {marker}")

        return success

    def wait_for_inference_recovery(
        self, load_tester: InferenceLoadTester, timeout: int = 900
    ) -> float:
        """
        Wait for inference to recover (Step 7: Recovery Validation).

        Args:
            load_tester: InferenceLoadTester instance
            timeout: Max seconds to wait

        Returns:
            Recovery rate (0-100)
        """
        print("\n" + "=" * 80)
        print("STEP: Inference Recovery Validation")
        print("=" * 80)
        print("[AUTOMATED] Dynamo routing layer reconnects to new workers")
        print("            vLLM workers register with frontend")
        print(f"[→] Waiting for inference to recover (max {timeout//60} minutes)")

        # Test inference periodically
        inference_recovered = False
        for i in range(timeout // 30):
            time.sleep(30)
            elapsed = (i + 1) * 30

            result = load_tester.send_inference_request()
            status = "✓" if result["success"] else "✗"
            print(f"    ... {elapsed}s: inference test {status}")

            if result["success"]:
                print(f"    ✓ Inference recovered after {elapsed}s!")
                inference_recovered = True
                break

        if not inference_recovered:
            print(f"[⚠] Inference did not recover within {timeout//60} minutes")

        # Test recovery with multiple requests
        print("\n[→] Testing inference recovery (10 requests)...")
        recovery_successes = 0
        for i in range(10):
            result = load_tester.send_inference_request()
            status = "✓" if result["success"] else "✗"
            print(f"    Request {i+1}/10: {status}")
            if result["success"]:
                recovery_successes += 1
            time.sleep(1)

        recovery_rate = (recovery_successes / 10) * 100
        print(f"\n[→] Recovery rate: {recovery_rate:.0f}%")

        if recovery_rate >= 70:
            print(f"[✓] Recovery successful: {recovery_rate:.0f}%")
        else:
            print(f"[⚠] Partial recovery: {recovery_rate:.0f}%")

        return recovery_rate

    def cleanup(self):
        """Cleanup any resources created during simulation."""
        if self.cordoned_node:
            print(f"\n[→] Cleanup: Uncordoning {self.cordoned_node}")
            self.node_ops.uncordon_node(
                self.cordoned_node, self.node_was_initially_cordoned
            )
