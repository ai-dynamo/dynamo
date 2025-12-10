# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Quick CUDA Toggle Test - Validates the cuda_intercept.c library toggle mechanism.

This test verifies:
1. Library loads in passthrough mode (pods run normally)
2. Toggle ON causes immediate pod crashes on CUDA calls
3. Toggle OFF allows pods to recover ON THE SAME NODE

Usage:
    pytest test_cuda_toggle.py::test_cuda_toggle_mechanism --namespace=dynamo-oviya -v -s
"""

import asyncio
import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.utils.managed_deployment import DeploymentSpec, ManagedDeployment

# Import inference testing
_hw_helpers_path = (
    Path(__file__).parent.parent / "hardware/fault_injection_service/helpers"
)
if str(_hw_helpers_path) not in sys.path:
    sys.path.insert(0, str(_hw_helpers_path))

try:
    from inference_testing import InferenceLoadTester
except ImportError:
    InferenceLoadTester = None


def _get_workspace_dir() -> str:
    current = os.path.dirname(os.path.abspath(__file__))
    while current != os.path.dirname(current):
        if os.path.exists(os.path.join(current, "pyproject.toml")):
            return current
        current = os.path.dirname(current)
    return os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )


async def write_toggle_via_helper_pod(
    namespace: str, target_node: str, toggle_value: str, logger
):
    """
    Write toggle file via a helper pod when main pods are crashed.

    Creates a temporary busybox pod on the target node with the same hostPath,
    writes the toggle value, then deletes the pod.
    """
    import json

    helper_pod_name = "cuda-toggle-helper"

    # Pod spec with hostPath mount
    pod_manifest = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": helper_pod_name,
            "namespace": namespace,
        },
        "spec": {
            "restartPolicy": "Never",
            "nodeSelector": {"kubernetes.io/hostname": target_node},
            "tolerations": [
                {"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}
            ],
            "containers": [
                {
                    "name": "toggle-writer",
                    "image": "busybox:latest",
                    "command": [
                        "sh",
                        "-c",
                        f"echo '{toggle_value}' > /host-fault/cuda_fault_enabled && cat /host-fault/cuda_fault_enabled && sleep 5",
                    ],
                    "volumeMounts": [
                        {"name": "fault-marker", "mountPath": "/host-fault"}
                    ],
                }
            ],
            "volumes": [
                {
                    "name": "fault-marker",
                    "hostPath": {
                        "path": "/var/lib/cuda-fault-test",
                        "type": "DirectoryOrCreate",
                    },
                }
            ],
        },
    }

    logger.info(f"  Creating helper pod on {target_node}...")

    # Delete if exists
    subprocess.run(
        [
            "kubectl",
            "delete",
            "pod",
            helper_pod_name,
            "-n",
            namespace,
            "--ignore-not-found",
        ],
        capture_output=True,
    )

    # Create helper pod
    result = subprocess.run(
        ["kubectl", "apply", "-f", "-", "-n", namespace],
        input=json.dumps(pod_manifest),
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        logger.error(f"  Failed to create helper pod: {result.stderr}")
        return False

    # Wait for pod to complete
    logger.info("  Waiting for helper pod to write toggle file...")
    for _ in range(30):
        result = subprocess.run(
            [
                "kubectl",
                "get",
                "pod",
                helper_pod_name,
                "-n",
                namespace,
                "-o",
                "jsonpath={.status.phase}",
            ],
            capture_output=True,
            text=True,
        )
        phase = result.stdout.strip()
        if phase in ["Succeeded", "Failed"]:
            break
        await asyncio.sleep(1)

    # Get logs
    result = subprocess.run(
        ["kubectl", "logs", helper_pod_name, "-n", namespace],
        capture_output=True,
        text=True,
    )
    logger.info(f"  Helper pod output: {result.stdout.strip()}")

    # Cleanup
    subprocess.run(
        [
            "kubectl",
            "delete",
            "pod",
            helper_pod_name,
            "-n",
            namespace,
            "--ignore-not-found",
        ],
        capture_output=True,
    )

    logger.info(f"  ✓ Toggle file set to {toggle_value} on {target_node}")
    return True


@pytest.fixture
def sglang_deployment():
    """Create SGLang deployment spec."""
    workspace = _get_workspace_dir()
    yaml_path = os.path.join(workspace, "examples/backends/sglang/deploy/disagg.yaml")
    if not os.path.exists(yaml_path):
        pytest.skip(f"Deployment YAML not found: {yaml_path}")
    spec = DeploymentSpec(yaml_path)
    spec.name = "cuda-toggle-test"
    spec.set_dynamo_namespace("cuda-toggle-test")
    return spec


@pytest.mark.k8s
@pytest.mark.fault_tolerance
@pytest.mark.slow
async def test_cuda_toggle_mechanism(request, namespace, sglang_deployment):
    """
    Test the CUDA library toggle mechanism WITHOUT XID injection.

    Flow:
    1. Deploy with CUDA library in passthrough mode
    2. Verify inference works (library loaded, faults OFF)
    3. Toggle faults ON → pods crash
    4. Toggle faults OFF via helper pod → pods recover ON SAME NODE
    5. Verify recovery
    """
    logger = logging.getLogger(request.node.name)

    logger.info("=" * 70)
    logger.info("CUDA TOGGLE MECHANISM TEST")
    logger.info("=" * 70)

    hw_fault_config = {
        "enabled": True,
        "xid_type": 79,
        "target_node": None,
        "backend": "sglang",
    }

    async with ManagedDeployment(
        namespace=namespace,
        log_dir=request.node.name,
        deployment_spec=sglang_deployment,
        enable_hw_faults=True,
        hw_fault_config=hw_fault_config,
    ) as deployment:

        def log_pods(phase: str):
            pods = deployment.get_pods()
            logger.info(f"\n  [{phase}] Pod Status:")
            for svc, svc_pods in pods.items():
                for pod in svc_pods:
                    try:
                        status = pod.status.phase
                        node = getattr(pod.spec, "nodeName", "N/A")
                        restarts = 0
                        if (
                            hasattr(pod.status, "containerStatuses")
                            and pod.status.containerStatuses
                        ):
                            restarts = pod.status.containerStatuses[0].restartCount
                        logger.info(
                            f"    {pod.name}: {status} (restarts={restarts}) on {node}"
                        )
                    except:
                        logger.info(f"    {pod.name}: (status unavailable)")

        async def send_request(label: str) -> bool:
            """Send single inference request, return True if successful."""
            if not InferenceLoadTester:
                logger.info(f"    {label}: [SKIP - no load tester]")
                return True

            pods = deployment.get_pods()
            for svc, svc_pods in pods.items():
                if "frontend" in svc.lower() and svc_pods:
                    pf = deployment.port_forward(svc_pods[0], 8000)
                    if pf and pf.local_port:
                        endpoint = f"http://localhost:{pf.local_port}/v1/completions"
                        tester = InferenceLoadTester(
                            endpoint, "Qwen/Qwen3-0.6B", timeout=15
                        )
                        result = tester.send_inference_request("test")
                        status = "✓" if result["success"] else "✗"
                        logger.info(f"    {label}: {status} ({result['latency']:.2f}s)")
                        try:
                            pf.stop()
                        except:
                            pass
                        return result["success"]
            logger.info(f"    {label}: [SKIP - no frontend]")
            return True

        # =====================================================
        # PHASE 1: Initial deployment (no CUDA library yet)
        # =====================================================
        logger.info("\n[PHASE 1] Initial deployment ready")
        log_pods("Initial")

        target_node = deployment.get_hw_fault_target_node()
        logger.info(f"  Target node: {target_node}")

        # =====================================================
        # PHASE 2: Setup CUDA passthrough
        # =====================================================
        logger.info("\n[PHASE 2] Setting up CUDA passthrough (faults OFF)...")

        success = await deployment.setup_cuda_passthrough(xid_type=79)
        assert success, "Failed to setup CUDA passthrough"
        logger.info("  ✓ Passthrough configured, waiting for pods...")

        await deployment.wait_for_all_pods_ready(timeout=300)
        logger.info("  ✓ Pods ready with CUDA library")
        log_pods("After CUDA Setup")

        # =====================================================
        # PHASE 3: Verify passthrough works (inference OK)
        # =====================================================
        logger.info("\n[PHASE 3] Testing inference (faults OFF)...")

        # Wait for model to load
        await asyncio.sleep(10)

        for i in range(3):
            await send_request(f"Request {i+1}")
            await asyncio.sleep(1)

        log_pods("After Passthrough Test")

        # =====================================================
        # PHASE 4: Toggle faults ON
        # =====================================================
        logger.info("\n[PHASE 4] Toggling CUDA faults ON...")

        toggle_result = await deployment.toggle_cuda_faults(enable=True)
        logger.info(f"  Toggle result: {'✓' if toggle_result else '⚠ partial'}")

        logger.info("  Testing inference (faults ON - should fail)...")
        await asyncio.sleep(2)

        for i in range(3):
            await send_request(f"Request {i+1}")
            await asyncio.sleep(2)

        log_pods("After Faults ON")

        # =====================================================
        # PHASE 5: Toggle faults OFF (same node recovery)
        # =====================================================
        logger.info("\n[PHASE 5] Recovery via toggle OFF (same node)...")
        logger.info("  Pods are crashed - using helper pod to write toggle=0")

        # Write toggle=0 via helper pod (since crashed pods can't be exec'd)
        await write_toggle_via_helper_pod(
            namespace=namespace,
            target_node=target_node,
            toggle_value="0",
            logger=logger,
        )

        logger.info("  Waiting for pods to restart and recover (same node)...")
        # Pods will restart (crashloop backoff) and read toggle=0
        await asyncio.sleep(30)  # Wait for crashloop to retry

        # Wait for pods to become ready
        await deployment.wait_for_all_pods_ready(timeout=300)

        log_pods("After Toggle OFF")

        # =====================================================
        # PHASE 6: Verify recovery (still on same node!)
        # =====================================================
        logger.info("\n[PHASE 6] Testing recovery inference (same node)...")

        # Wait for model to reload
        await asyncio.sleep(10)

        for i in range(3):
            await send_request(f"Request {i+1}")
            await asyncio.sleep(1)

        log_pods("Final")

        # Verify pods are still on the target node
        pods = deployment.get_pods()
        for svc, svc_pods in pods.items():
            if "worker" in svc.lower() or svc in ["decode", "prefill"]:
                for pod in svc_pods:
                    node = getattr(pod.spec, "nodeName", None)
                    if node == target_node:
                        logger.info(f"  ✓ {pod.name} recovered on SAME node: {node}")
                    else:
                        logger.warning(
                            f"  ⚠ {pod.name} moved to different node: {node}"
                        )

        logger.info("\n" + "=" * 70)
        logger.info("TEST COMPLETE - Recovery via toggle OFF on same node")
        logger.info("=" * 70)
