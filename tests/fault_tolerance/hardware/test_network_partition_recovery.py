# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
"""
Network partition fault tolerance test with recovery validation.

Tests Dynamo's ability to handle network partitions between components
and recover gracefully when connectivity is restored.
"""

import logging
import sys
import time
from typing import List, Optional

import pytest
import requests
from kubernetes import client, config
from kubernetes.client.rest import ApiException

logger = logging.getLogger(__name__)

# ANSI color codes
YELLOW = "\033[93m"
RESET = "\033[0m"


class NetworkFaultInjector:
    """Inject network faults using Kubernetes NetworkPolicies"""

    def __init__(self, namespace: str = "dynamo-oviya"):
        """Initialize the fault injector"""
        self.namespace = namespace
        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config()

        self.v1 = client.CoreV1Api()
        self.network_v1 = client.NetworkingV1Api()

    def get_pod_by_prefix(self, pod_prefix: str) -> Optional[str]:
        """Get full pod name by prefix"""
        try:
            pods = self.v1.list_namespaced_pod(self.namespace)
            for pod in pods.items:
                if pod.metadata.name.startswith(pod_prefix):
                    return pod.metadata.name
        except ApiException as e:
            print(f"Error listing pods: {e}", file=sys.stderr)
        return None

    def wait_for_pod_ready(self, pod_name: str, timeout: int = 120) -> bool:
        """Wait for a pod to be ready"""
        import time

        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                pod = self.v1.read_namespaced_pod(pod_name, self.namespace)
                if pod.status.conditions:
                    for condition in pod.status.conditions:
                        if condition.type == "Ready" and condition.status == "True":
                            return True
            except ApiException as e:
                print(f"Error checking pod status: {e}", file=sys.stderr)
                return False
            time.sleep(2)
        return False

    def get_pod_labels(self, pod_name: str) -> dict:
        """Get labels from a pod"""
        try:
            pod = self.v1.read_namespaced_pod(pod_name, self.namespace)
            return pod.metadata.labels or {}
        except ApiException as e:
            print(f"Error reading pod {pod_name}: {e}", file=sys.stderr)
            return {}

    def inject_partition(
        self, source_pod: str, target_pod: str, policy_name: Optional[str] = None
    ) -> str:
        """Inject a network partition between two pods"""
        source_full = self.get_pod_by_prefix(source_pod) or source_pod
        target_full = self.get_pod_by_prefix(target_pod) or target_pod

        target_labels = self.get_pod_labels(target_full)
        if not target_labels:
            raise ValueError(f"Could not find labels for target pod: {target_full}")

        if not policy_name:
            source_short = (
                source_pod.split("-")[-1] if "-" in source_pod else source_pod
            )
            target_short = (
                target_pod.split("-")[-1] if "-" in target_pod else target_pod
            )
            policy_name = f"network-fault-{source_short}-to-{target_short}"

        policy = client.V1NetworkPolicy(
            api_version="networking.k8s.io/v1",
            kind="NetworkPolicy",
            metadata=client.V1ObjectMeta(
                name=policy_name,
                namespace=self.namespace,
                labels={
                    "managed-by": "fault-injector",
                    "fault-type": "network-partition",
                },
            ),
            spec=client.V1NetworkPolicySpec(
                pod_selector=client.V1LabelSelector(match_labels=target_labels),
                policy_types=["Ingress"],
                ingress=[
                    client.V1NetworkPolicyIngressRule(
                        _from=[
                            client.V1NetworkPolicyPeer(
                                pod_selector=client.V1LabelSelector(
                                    match_expressions=[
                                        client.V1LabelSelectorRequirement(
                                            key="app.kubernetes.io/instance",
                                            operator="NotIn",
                                            values=[source_full.split("-")[0]],
                                        )
                                    ]
                                )
                            )
                        ]
                    )
                ],
            ),
        )

        try:
            self.network_v1.create_namespaced_network_policy(
                namespace=self.namespace, body=policy
            )
            print(f"Network partition injected: {policy_name}")
            return policy_name
        except ApiException as e:
            if e.status == 409:
                logger.warning(f"NetworkPolicy {policy_name} already exists")
                return policy_name
            else:
                raise

    def list_policies(self) -> List[dict]:
        """List all fault injection NetworkPolicies"""
        try:
            policies = self.network_v1.list_namespaced_network_policy(
                namespace=self.namespace, label_selector="managed-by=fault-injector"
            )
            return [
                {
                    "name": policy.metadata.name,
                    "namespace": policy.metadata.namespace,
                }
                for policy in policies.items
            ]
        except ApiException as e:
            logger.error(f"Error listing NetworkPolicies: {e}")
            return []

    def clear_policy(self, policy_name: str) -> bool:
        """Clear a specific NetworkPolicy"""
        try:
            self.network_v1.delete_namespaced_network_policy(
                name=policy_name, namespace=self.namespace
            )
            print(f"Cleared network fault: {policy_name}")
            return True
        except ApiException as e:
            if e.status == 404:
                logger.warning(f"NetworkPolicy {policy_name} not found")
            else:
                logger.error(f"Error deleting NetworkPolicy: {e}")
            return False

    def clear_all_policies(self) -> int:
        """Clear all fault injection NetworkPolicies"""
        policies = self.list_policies()
        count = 0
        for policy in policies:
            if self.clear_policy(policy["name"]):
                count += 1
        return count


# Test configuration
FRONTEND_PORT = 8000
REQUEST_TIMEOUT = 30
PARTITION_SETTLE_TIME = 5
RECOVERY_SETTLE_TIME = 5


def get_frontend_url():
    """Get the frontend URL (assumes port-forward to localhost:8000)"""
    return f"http://localhost:{FRONTEND_PORT}"


def get_available_model():
    """Get the first available model from /v1/models endpoint"""
    url = f"{get_frontend_url()}/v1/models"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if "data" in data and len(data["data"]) > 0:
                return data["data"][0]["id"]
    except Exception as e:
        logger.warning(f"Failed to fetch model list: {e}")
    # Default fallback
    return "Qwen/Qwen3-0.6B"


def send_completion_request(
    prompt: str,
    max_tokens: int,
    timeout: int = REQUEST_TIMEOUT,
    model: Optional[str] = None,
):
    """
    Send a completion request to the frontend.

    Args:
        prompt: The prompt text
        max_tokens: Maximum tokens to generate
        timeout: Request timeout in seconds
        model: Model name (auto-detected if None)

    Returns:
        Response object
    """
    if model is None:
        model = get_available_model()

    url = f"{get_frontend_url()}/v1/completions"
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }

    print(
        f"Sending completion request: model='{model}', prompt='{prompt}', max_tokens={max_tokens}"
    )
    response = requests.post(url, json=payload, timeout=timeout)
    return response


def validate_completion_response(response: requests.Response) -> None:
    """Validate that the response is a proper completion response"""
    assert (
        response.status_code == 200
    ), f"Request failed with status {response.status_code}: {response.text}"

    try:
        data = response.json()
    except ValueError:
        pytest.fail(f"Response is not valid JSON: {response.text}")

    assert "choices" in data, f"Response missing 'choices' field: {data}"
    assert len(data["choices"]) > 0, f"Response has empty 'choices': {data}"
    assert "text" in data["choices"][0], f"Response choice missing 'text' field: {data}"
    assert data["choices"][0]["text"], f"Response text is empty: {data}"

    print(f"Received valid completion: {data['choices'][0]['text'][:50]}...")


def check_frontend_reachable() -> bool:
    """Check if frontend is reachable via health check"""
    try:
        response = requests.get(f"{get_frontend_url()}/health", timeout=5)
        return response.status_code == 200
    except Exception as e:
        logger.warning(f"Frontend health check failed: {e}")
        return False


@pytest.mark.fault_tolerance
def test_network_partition_frontend_to_worker_with_recovery():
    """
    Test network partition between frontend and worker with recovery validation.

    This test:
    1. Verifies baseline connectivity (request succeeds)
    2. Injects network partition (frontend cannot reach worker)
    3. Verifies request fails or times out during partition
    4. Clears the partition
    5. Verifies connectivity is restored (request succeeds again)
    6. Validates that the system recovered gracefully

    Expected behavior:
    - Initial request succeeds
    - Requests fail during partition (single worker) or migrate (multiple workers)
    - Requests succeed after partition is cleared
    - No lingering effects from the partition
    """

    print("\n" + "=" * 80)
    print("TEST: Network Partition Recovery (Frontend to Worker)")
    print("=" * 80)

    # Initialize injector
    injector = NetworkFaultInjector(namespace="dynamo-oviya")

    # BEFORE
    print("\n[BEFORE] Establish Baseline")
    print("-" * 80)
    if not check_frontend_reachable():
        pytest.skip("Frontend not reachable")
    print("[OK] Frontend reachable")

    frontend_pod = injector.get_pod_by_prefix("vllm-agg-0-frontend")
    worker_pod = injector.get_pod_by_prefix("vllm-agg-0-vllmdecodeworker")

    if not frontend_pod or not worker_pod:
        pytest.skip("Frontend or worker pod not found")

    print(f"[OK] Frontend: {frontend_pod}")
    print(f"[OK] Worker: {worker_pod}")

    # Wait for worker pod to be ready (may have been recently rescheduled by GPU test)
    print("Waiting for worker pod to be ready (timeout: 5 minutes)...")
    if not injector.wait_for_pod_ready(worker_pod, timeout=300):
        pytest.skip(f"Worker pod {worker_pod} not ready within timeout")
    print("[OK] Worker pod is ready")

    try:
        response = send_completion_request("Hello", 10, timeout=30)
        validate_completion_response(response)
        print("[OK] Baseline request succeeded")
    except Exception as e:
        pytest.fail(f"Baseline request failed: {e}")

    # DURING
    print("\n[DURING] Inject Fault")
    print("-" * 80)
    policy_name = injector.inject_partition(
        source_pod=frontend_pod, target_pod=worker_pod
    )
    print("[OK] Network partition injected")
    time.sleep(PARTITION_SETTLE_TIME)

    policies = injector.list_policies()
    active_policy_names = [p["name"] for p in policies]
    assert policy_name in active_policy_names
    print("[OK] Partition active")

    print("\nTesting request during partition...")
    try:
        response = send_completion_request("Test during fault", 10, timeout=15)
        validate_completion_response(response)
        print("[OK] Request succeeded (migrated to another worker)")
    except (requests.Timeout, requests.RequestException):
        print("[EXPECTED] Request failed (worker unreachable)")

    # AFTER
    print("\n[AFTER] Validate Recovery")
    print("-" * 80)
    success = injector.clear_policy(policy_name)
    assert success
    print("[OK] Partition cleared")
    time.sleep(RECOVERY_SETTLE_TIME)

    policies = injector.list_policies()
    active_policy_names = [p["name"] for p in policies]
    assert policy_name not in active_policy_names
    print("[OK] Partition removed")

    try:
        response = send_completion_request("Hello after recovery", 10, timeout=30)
        validate_completion_response(response)
        print("[OK] Recovery request succeeded")
    except Exception as e:
        pytest.fail(f"Recovery failed: {e}")

    try:
        response = send_completion_request("Final validation", 10, timeout=30)
        validate_completion_response(response)
        print("[OK] No lingering effects - system stable")
    except Exception as e:
        pytest.fail(f"System still degraded: {e}")

    print("=" * 80)


@pytest.fixture(scope="module", autouse=True)
def setup_and_teardown():
    """Setup before tests and cleanup after tests"""
    yield

    # Cleanup
    print("\n[CLEANUP] Removing network policies...")
    injector = NetworkFaultInjector(namespace="dynamo-oviya")
    count = injector.clear_all_policies()
    if count > 0:
        print(f"[OK] Cleaned up {count} network policies")
