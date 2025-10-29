"""
Template for Writing XID Error Tests

This template shows how to write tests for different XID error types.
Copy and modify for your specific test scenarios.

Supported XIDs: 79, 48, 94, 95, 43, 74
"""

import sys
import time
from pathlib import Path

from kubernetes import client, config

# Add cuda-fault-injection to path
cuda_injection_dir = Path(__file__).parent
sys.path.insert(0, str(cuda_injection_dir))

from inject_into_pods import (
    create_cuda_fault_configmap,
    delete_cuda_fault_configmap,
    patch_deployment_env,
)

# Initialize Kubernetes client
config.load_kube_config()
k8s_core = client.CoreV1Api()


def test_xid79_gpu_fell_off_bus():
    """
    Test XID 79: GPU fell off bus

    Expected behavior:
    - Pods crash with CUDA_ERROR_NO_DEVICE
    - All CUDA calls fail immediately
    - Most severe node-level failure
    """
    deployment_name = "vllm-worker"
    namespace = "default"
    target_node = "node-with-gpu"

    try:
        # Enable XID 79 fault injection
        create_cuda_fault_configmap(namespace)
        patch_deployment_env(
            deployment_name,
            namespace,
            enable=True,
            use_configmap=True,
            target_node=target_node,
            xid_type=79,  # GPU fell off bus
        )

        # Wait for pods to crash
        time.sleep(30)

        # Verify pods are crashing
        pods = k8s_core.list_namespaced_pod(
            namespace=namespace, field_selector=f"spec.nodeName={target_node}"
        )

        crashed_pods = [
            p
            for p in pods.items
            if p.status.container_statuses
            and any(
                cs.state.waiting and cs.state.waiting.reason == "CrashLoopBackOff"
                for cs in p.status.container_statuses
            )
        ]

        print(f"✓ {len(crashed_pods)} pods crashed (expected behavior)")

        # Your test assertions here
        # ...

    finally:
        # Cleanup
        patch_deployment_env(
            deployment_name, namespace, enable=False, use_configmap=True
        )
        delete_cuda_fault_configmap(namespace)


def test_xid48_ecc_double_bit_error():
    """
    Test XID 48: Double-bit ECC error

    Expected behavior:
    - Pods crash with CUDA_ERROR_ECC_UNCORRECTABLE
    - Memory operations fail
    - Indicates hardware memory corruption
    """
    deployment_name = "vllm-worker"
    namespace = "default"

    try:
        create_cuda_fault_configmap(namespace)
        patch_deployment_env(
            deployment_name,
            namespace,
            enable=True,
            use_configmap=True,
            xid_type=48,  # Double-bit ECC error
        )

        # Your test logic here
        # ...

    finally:
        patch_deployment_env(
            deployment_name, namespace, enable=False, use_configmap=True
        )
        delete_cuda_fault_configmap(namespace)


def test_xid94_contained_ecc_error():
    """
    Test XID 94: Contained ECC error

    Expected behavior:
    - Pods crash with CUDA_ERROR_ECC_UNCORRECTABLE
    - Error is contained (less severe than XID 48)
    - May be recoverable with driver restart
    """
    deployment_name = "vllm-worker"
    namespace = "default"

    try:
        create_cuda_fault_configmap(namespace)
        patch_deployment_env(
            deployment_name,
            namespace,
            enable=True,
            use_configmap=True,
            xid_type=94,  # Contained ECC error
        )

        # Your test logic here
        # ...

    finally:
        patch_deployment_env(
            deployment_name, namespace, enable=False, use_configmap=True
        )
        delete_cuda_fault_configmap(namespace)


def test_xid95_uncontained_error():
    """
    Test XID 95: Uncontained error

    Expected behavior:
    - Pods crash with CUDA_ERROR_UNKNOWN
    - Fatal GPU error
    - System instability
    """
    deployment_name = "vllm-worker"
    namespace = "default"

    try:
        create_cuda_fault_configmap(namespace)
        patch_deployment_env(
            deployment_name,
            namespace,
            enable=True,
            use_configmap=True,
            xid_type=95,  # Uncontained error
        )

        # Your test logic here
        # ...

    finally:
        patch_deployment_env(
            deployment_name, namespace, enable=False, use_configmap=True
        )
        delete_cuda_fault_configmap(namespace)


def test_xid43_gpu_stopped_responding():
    """
    Test XID 43: GPU stopped responding

    Expected behavior:
    - Pods crash with CUDA_ERROR_LAUNCH_TIMEOUT
    - Kernel launches time out
    - GPU appears hung
    """
    deployment_name = "vllm-worker"
    namespace = "default"

    try:
        create_cuda_fault_configmap(namespace)
        patch_deployment_env(
            deployment_name,
            namespace,
            enable=True,
            use_configmap=True,
            xid_type=43,  # GPU stopped responding
        )

        # Your test logic here
        # ...

    finally:
        patch_deployment_env(
            deployment_name, namespace, enable=False, use_configmap=True
        )
        delete_cuda_fault_configmap(namespace)


def test_xid74_nvlink_error():
    """
    Test XID 74: NVLink error

    Expected behavior:
    - Pods crash with CUDA_ERROR_PEER_ACCESS_UNSUPPORTED
    - Multi-GPU communication fails
    - Affects distributed workloads
    """
    deployment_name = "vllm-worker"
    namespace = "default"

    try:
        create_cuda_fault_configmap(namespace)
        patch_deployment_env(
            deployment_name,
            namespace,
            enable=True,
            use_configmap=True,
            xid_type=74,  # NVLink error
        )

        # Your test logic here
        # ...

    finally:
        patch_deployment_env(
            deployment_name, namespace, enable=False, use_configmap=True
        )
        delete_cuda_fault_configmap(namespace)


# Example: Pytest-style test with fixture
def test_with_pytest_fixture():
    """
    Example using pytest fixture for automatic cleanup

    Add this to conftest.py:

    @pytest.fixture
    def cuda_fault_injection(request):
        deployment = request.param.get('deployment')
        namespace = request.param.get('namespace')
        xid_type = request.param.get('xid_type', 79)

        create_cuda_fault_configmap(namespace)
        patch_deployment_env(deployment, namespace, enable=True,
                           use_configmap=True, xid_type=xid_type)

        yield

        patch_deployment_env(deployment, namespace, enable=False, use_configmap=True)
        delete_cuda_fault_configmap(namespace)

    Then use it:

    @pytest.mark.parametrize('cuda_fault_injection', [
        {'deployment': 'vllm-worker', 'namespace': 'default', 'xid_type': 79}
    ], indirect=True)
    def test_with_fixture(cuda_fault_injection):
        # Your test logic here
        pass
    """
    pass


if __name__ == "__main__":
    print("XID Error Test Templates")
    print("=" * 80)
    print("\nThis file contains templates for testing different XID errors.")
    print("Copy the functions and modify for your test scenarios.")
    print("\nSupported XIDs:")
    print("  79 - GPU fell off bus")
    print("  48 - Double-bit ECC error")
    print("  94 - Contained ECC error")
    print("  95 - Uncontained error")
    print("  43 - GPU stopped responding")
    print("  74 - NVLink error")
    print("\nSee README.md for more details.")
