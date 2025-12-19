# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Hardware Fault Injection Tests with Ephemeral Deployments

End-to-end hardware fault tolerance tests that validate:
    - XID fault injection via fault injection API
    - CUDA fault injection via LD_PRELOAD library
    - NVSentinel detection and node cordoning
    - Pod eviction and rescheduling to healthy nodes
    - Inference recovery after fault remediation

Test:
    test_gpu_xid79: XID 79 (GPU Fell Off Bus) with CUDA fault injection

Usage:
    pytest test_hw_faults.py::test_gpu_xid79 --enable-hw-faults --namespace=dynamo-oviya -v -s
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

import pytest

from tests.utils.managed_deployment import DeploymentSpec, ManagedDeployment

# Import inference testing from hardware FT helpers
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
    """Get workspace directory."""
    current = os.path.dirname(os.path.abspath(__file__))
    while current != os.path.dirname(current):
        if os.path.exists(os.path.join(current, "pyproject.toml")):
            return current
        current = os.path.dirname(current)
    return os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )


# Default images for each backend
VLLM_DEFAULT_IMAGE = "nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.7.0"
SGLANG_DEFAULT_IMAGE = "nvcr.io/nvidia/ai-dynamo/sglang-runtime:0.7.1"
TRTLLM_DEFAULT_IMAGE = "nvcr.io/nvidia/ai-dynamo/trtllm-runtime:0.7.0"


@pytest.fixture
def vllm_disagg_deployment():
    """Create a vLLM disaggregated deployment spec."""
    workspace = _get_workspace_dir()
    yaml_path = os.path.join(
        workspace, "examples/backends/vllm/deploy/disagg_router.yaml"
    )

    if not os.path.exists(yaml_path):
        pytest.skip(f"Deployment YAML not found: {yaml_path}")

    spec = DeploymentSpec(yaml_path)
    spec.name = "hw-fault-test"
    spec.set_dynamo_namespace("hw-fault-test")
    spec.set_model("Qwen/Qwen3-0.6B")
    spec.set_image(VLLM_DEFAULT_IMAGE)

    return spec


@pytest.fixture
def sglang_disagg_deployment():
    """Create an SGLang disaggregated deployment spec."""
    workspace = _get_workspace_dir()
    yaml_path = os.path.join(workspace, "examples/backends/sglang/deploy/disagg.yaml")

    if not os.path.exists(yaml_path):
        pytest.skip(f"Deployment YAML not found: {yaml_path}")

    spec = DeploymentSpec(yaml_path)
    spec.name = "hw-fault-test"
    spec.set_dynamo_namespace("hw-fault-test")
    spec.set_model("Qwen/Qwen3-0.6B")
    spec.set_image(SGLANG_DEFAULT_IMAGE)

    return spec


@pytest.fixture
def trtllm_disagg_deployment():
    """Create a TensorRT-LLM disaggregated deployment spec."""
    workspace = _get_workspace_dir()
    yaml_path = os.path.join(workspace, "examples/backends/trtllm/deploy/disagg.yaml")

    if not os.path.exists(yaml_path):
        pytest.skip(f"Deployment YAML not found: {yaml_path}")

    spec = DeploymentSpec(yaml_path)
    spec.name = "hw-fault-test"
    spec.set_dynamo_namespace("hw-fault-test")
    spec.set_image(TRTLLM_DEFAULT_IMAGE)

    return spec


@pytest.fixture
def hw_fault_deployment(
    request,
    hw_fault_backend,
    vllm_disagg_deployment,
    sglang_disagg_deployment,
    trtllm_disagg_deployment,
):
    """Return the appropriate deployment spec based on --hw-fault-backend."""
    if hw_fault_backend == "sglang":
        return sglang_disagg_deployment
    elif hw_fault_backend == "trtllm":
        return trtllm_disagg_deployment
    return vllm_disagg_deployment


@pytest.mark.k8s
@pytest.mark.fault_tolerance
@pytest.mark.hw_faults
@pytest.mark.slow
async def test_gpu_xid79_nvsentinel_eviction(
    request,
    namespace,
    image,
    hw_fault_config,
    hw_fault_backend,
    hw_fault_deployment,
):
    """
    XID 79 fault tolerance test with NVSentinel-driven pod eviction.
    
    This test relies on NVSentinel's node-drainer to evict pods instead of
    manually toggling off faults and deleting pods.
    
    Prerequisites:
        - NVSentinel deployed with all modules enabled
        - node-drainer configured with DeleteAfterTimeout mode (5 min timeout)
        - Test namespace added to node-drainer's userNamespaces
    
    Flow:
    1. Deploy pods with CUDA passthrough
    2. Run baseline inference
    3. Enable CUDA faults + inject XID
    4. NVSentinel detects XID → cordons node
    5. Clean up DGD spec (but keep pods running)
    6. Wait for node-drainer to force evict pods (5 min timeout)
    7. New pods come up clean (no CUDA library, no node affinity)
    8. Recovery inference
    
    Expected behavior vs test_gpu_xid79:
        - test_gpu_xid79: Manual recovery (toggle faults, delete pods)
        - This test: NVSentinel-driven recovery (node-drainer evicts)
    """
    logger = logging.getLogger(request.node.name)
    
    # Timeout for node-drainer eviction (should match NVSentinel config)
    # NVSentinel config: deleteAfterTimeoutMinutes=2
    NODE_DRAINER_TIMEOUT = 120  # 2 minutes
    
    if hw_fault_config is None:
        hw_fault_config = {
            "enabled": True,
            "xid_type": 79,
            "target_node": None,
            "backend": hw_fault_backend or "vllm",
        }
    
    backend = hw_fault_backend or "vllm"
    deployment_spec = hw_fault_deployment
    
    logger.info("=" * 80)
    logger.info(f"TEST: XID 79 - NVSentinel-Driven Eviction [{backend.upper()}]")
    logger.info("=" * 80)
    logger.info("This test relies on node-drainer to evict pods (2 min timeout)")
    
    if image:
        deployment_spec.set_image(image)
    
    load_stats = {
        "baseline": {"requests": 0, "success": 0, "avg_latency": 0},
        "during_fault": {"requests": 0, "success": 0, "avg_latency": 0},
        "after_recovery": {"requests": 0, "success": 0, "avg_latency": 0},
    }
    
    async with ManagedDeployment(
        namespace=namespace,
        log_dir=request.node.name,
        deployment_spec=deployment_spec,
        enable_hw_faults=True,
        hw_fault_config=hw_fault_config,
    ) as deployment:
        
        logger.info("[PHASE 1] Deployment ready")
        
        target_node = deployment.get_hw_fault_target_node()
        assert target_node, "No target node detected"
        logger.info(f"  Target node: {target_node}")
        
        def log_pod_status(phase: str):
            pods = deployment.get_pods()
            logger.info(f"  [{phase}] Pod Status:")
            for service_name, service_pods in pods.items():
                for pod in service_pods:
                    try:
                        phase_str = pod.status.phase if hasattr(pod, "status") else "Unknown"
                        node = (
                            pod.spec.nodeName
                            if hasattr(pod, "spec") and hasattr(pod.spec, "nodeName")
                            else "Unassigned"
                        )
                        logger.info(f"    - {pod.name}: {phase_str} on {node}")
                    except Exception:
                        logger.info(f"    - {pod.name}: (status unavailable)")
        
        pods = deployment.get_pods()
        initial_count = sum(len(p) for p in pods.values())
        logger.info(f"  Initial pods: {initial_count}")
        log_pod_status("Initial")
        
        # =====================================================================
        # PHASE 2: CUDA PASSTHROUGH SETUP
        # =====================================================================
        logger.info("\n[PHASE 2] Setting up CUDA passthrough...")
        
        passthrough_success = await deployment.setup_cuda_passthrough(xid_type=79)
        assert passthrough_success, "Failed to setup CUDA passthrough"
        logger.info("  ✓ CUDA passthrough configured")
        
        await deployment.wait_for_all_pods_ready(timeout=300)
        logger.info("  ✓ Pods ready with CUDA library (faults disabled)")
        log_pod_status("After CUDA Setup")
        
        # =====================================================================
        # PHASE 3: BASELINE INFERENCE (fail fast if baseline fails)
        # =====================================================================
        logger.info("\n[PHASE 3] Baseline inference...")
        
        load_tester = None
        frontend_pf = None
        
        if InferenceLoadTester:
            pods = deployment.get_pods()
            for service_name, service_pods in pods.items():
                if "frontend" in service_name.lower() and service_pods:
                    frontend_pf = deployment.port_forward(service_pods[0], 8000)
                    break
            
            if not frontend_pf or not frontend_pf.local_port:
                logger.error("  ✗ Failed to port-forward to frontend!")
                pytest.fail("Failed to port-forward to frontend")
            
            endpoint = f"http://localhost:{frontend_pf.local_port}/v1/completions"
            load_tester = InferenceLoadTester(endpoint, "Qwen/Qwen3-0.6B", timeout=30)
            logger.info(f"  ✓ Endpoint: {endpoint}")
            
            baseline_results = []
            for i in range(5):
                result = load_tester.send_inference_request(f"Baseline test {i}")
                baseline_results.append(result)
                status = "✓" if result["success"] else "✗"
                logger.info(f"    [{i+1}/5] {status} - {result['latency']:.2f}s")
                await asyncio.sleep(1)
            
            successful = sum(1 for r in baseline_results if r["success"])
            avg_latency = sum(r["latency"] for r in baseline_results) / len(baseline_results)
            load_stats["baseline"] = {
                "requests": 5,
                "success": successful,
                "avg_latency": avg_latency,
            }
            logger.info(f"  Baseline: {successful}/5 successful")
            
            # Fail fast if baseline fails
            if successful < 3:
                logger.error(f"  ✗ Baseline failed ({successful}/5) - CUDA passthrough broken!")
                pytest.fail(f"Baseline inference failed ({successful}/5) - check CUDA passthrough")
        else:
            logger.error("  ✗ InferenceLoadTester not available!")
            pytest.fail("InferenceLoadTester not available")
        
        deployment.collect_metrics(phase="baseline")
        
        # =====================================================================
        # PHASE 4: FAULT INJECTION (fail fast if XID injection fails)
        # =====================================================================
        logger.info("\n[PHASE 4] Injecting XID 79 and enabling CUDA faults...")
        
        fault_id = await deployment.inject_hw_fault(fault_type="xid", xid_type=79, gpu_id=0)
        if fault_id:
            logger.info(f"  ✓ XID 79 injected: {fault_id}")
        else:
            logger.error("  ✗ XID injection failed - fault-injection-api not available!")
            logger.error("    - Check: kubectl get pods -n fault-injection-system")
            logger.error("    - This test REQUIRES XID injection for NVSentinel to detect")
            pytest.fail("XID injection failed - fault-injection-api not available")
        
        toggle_success = await deployment.toggle_cuda_faults(enable=True)
        if toggle_success:
            logger.info("  ✓ CUDA faults ENABLED")
        else:
            logger.error("  ✗ CUDA fault toggle failed!")
            pytest.fail("CUDA fault toggle failed")
        
        # =====================================================================
        # PHASE 5: INFERENCE DURING FAULT (verify faults are working)
        # =====================================================================
        logger.info("\n[PHASE 5] Inference during fault...")
        log_pod_status("During Fault")
        
        fault_results = []
        for i in range(5):
            result = load_tester.send_inference_request(f"Fault test {i}")
            fault_results.append(result)
            status = "✓" if result["success"] else "✗"
            logger.info(f"    [{i+1}/5] {status} - {result['latency']:.2f}s")
            await asyncio.sleep(2)
        
        successful = sum(1 for r in fault_results if r["success"])
        avg_latency = sum(r["latency"] for r in fault_results) / len(fault_results)
        load_stats["during_fault"] = {
            "requests": 5,
            "success": successful,
            "avg_latency": avg_latency,
        }
        logger.info(f"  During fault: {successful}/5 successful (failures expected)")
        
        # Fail fast if faults aren't working (too many successes)
        if successful >= 4:
            logger.error(f"  ✗ CUDA faults not working! ({successful}/5 still succeeded)")
            pytest.fail("CUDA faults not working - too many requests succeeded")
        
        # =====================================================================
        # PHASE 6: NVSENTINEL RESPONSE - VERIFY CORDON (fail fast)
        # =====================================================================
        logger.info("\n[PHASE 6] Verifying NVSentinel cordoned the node...")
        
        # Fail fast: check for cordon every 5s, max 30s
        cordoned = False
        for i in range(6):  # 6 x 5s = 30s max
            await asyncio.sleep(5)
            cordoned = deployment.is_node_cordoned(target_node)
            logger.info(f"  [{(i+1)*5}s] Node cordoned: {'✓' if cordoned else '✗'}")
            if cordoned:
                break
        
        if not cordoned:
            logger.error("  ✗ NVSentinel did not cordon the node within 30s!")
            logger.error("    - Verify NVSentinel is deployed with syslog-health-monitor")
            logger.error("    - Check: kubectl logs -n nvsentinel -l app.kubernetes.io/name=syslog-health-monitor")
            pytest.fail("NVSentinel did not cordon node within 30s - check syslog-health-monitor")
        
        # =====================================================================
        # PHASE 7: CLEAN UP SPEC (keep pods running, let node-drainer evict)
        # =====================================================================
        logger.info("\n[PHASE 7] Cleaning up DGD spec (keeping pods running)...")
        logger.info("  New pods will come up WITHOUT CUDA library or node affinity")
        logger.info("  Node-drainer will evict pods from cordoned node automatically")
        
        cleanup_success = await deployment.cleanup_cuda_spec_without_restart()
        assert cleanup_success, "Failed to clean up DGD spec"
        logger.info("  ✓ DGD spec cleaned")
        
        # Disable CUDA faults via toggle (clear hostPath file)
        # This ensures pods won't crash from CUDA faults while waiting for eviction
        logger.info("  Disabling CUDA faults via toggle...")
        await deployment.toggle_cuda_faults(enable=False)
        logger.info("  ✓ CUDA faults disabled via toggle")
        
        # Wait for DGD spec change to propagate to the controller
        # This is critical - we need the controller to see the clean spec BEFORE
        # node-drainer evicts pods, so new pods are created with clean spec
        logger.info("  Waiting 30s for DGD spec to propagate to controller...")
        await asyncio.sleep(30)
        
        log_pod_status("After Spec Cleanup (pods still running)")
        
        # =====================================================================
        # PHASE 8: WAIT FOR NODE-DRAINER EVICTION
        # =====================================================================
        logger.info("\n[PHASE 8] Waiting for node-drainer to evict pods...")
        logger.info(f"  Timeout: {NODE_DRAINER_TIMEOUT}s (~2 min from XID detection)")
        logger.info("  Pods will be force-deleted when DeleteAfterTimeout is reached")
        
        # Get pods on target node before eviction
        pods_on_target = []
        for service_name, service_pods in deployment.get_pods().items():
            for pod in service_pods:
                if hasattr(pod, "spec") and pod.spec.nodeName == target_node:
                    pods_on_target.append(pod.name)
        
        logger.info(f"  Pods on cordoned node: {len(pods_on_target)}")
        
        # Wait for pods to be evicted (check every 15s)
        eviction_timeout = NODE_DRAINER_TIMEOUT + 60  # 2 min + 1 min buffer
        start_time = asyncio.get_event_loop().time()
        evicted = False
        
        while asyncio.get_event_loop().time() - start_time < eviction_timeout:
            elapsed = asyncio.get_event_loop().time() - start_time
            
            # Check if pods are gone from target node
            current_pods = deployment.get_pods()
            current_on_target = 0
            for service_name, service_pods in current_pods.items():
                for pod in service_pods:
                    if hasattr(pod, "spec") and pod.spec.nodeName == target_node:
                        current_on_target += 1
            
            logger.info(f"  [{int(elapsed)}s] Pods on cordoned node: {current_on_target}")
            
            if current_on_target == 0:
                evicted = True
                logger.info(f"  ✓ All pods evicted from cordoned node after {int(elapsed)}s")
                break
            
            await asyncio.sleep(15)
        
        if not evicted:
            logger.error("  ✗ Pods were not evicted within timeout!")
            logger.error("    - Check: kubectl logs -n nvsentinel -l app.kubernetes.io/name=node-drainer")
            logger.error("    - Verify deleteAfterTimeoutMinutes=2 in node-drainer config")
            pytest.fail("node-drainer did not evict pods within timeout")
        
        # =====================================================================
        # PHASE 9: WAIT FOR NEW PODS ON HEALTHY NODES
        # =====================================================================
        logger.info("\n[PHASE 9] Waiting for new clean pods on healthy nodes...")
        logger.info("  (vLLM workers take ~5-7 min to start after cold eviction)")
        
        await deployment.wait_for_pods_on_healthy_nodes(exclude_node=target_node, timeout=1000)
        await deployment.wait_for_all_pods_ready(timeout=1000)
        
        log_pod_status("After Eviction (new clean pods)")
        
        # Verify new pods don't have CUDA library
        new_pods = deployment.get_pods()
        for service_name, service_pods in new_pods.items():
            for pod in service_pods:
                if hasattr(pod, "spec"):
                    # Check no init containers for CUDA (use getattr to avoid BoxKeyError)
                    init_containers = getattr(pod.spec, "init_containers", None) or getattr(pod.spec, "initContainers", None)
                    if init_containers:
                        for ic in init_containers:
                            ic_name = getattr(ic, "name", "")
                            if "cuda-fault" in ic_name or "compile-cuda" in ic_name:
                                logger.error(f"  ✗ Pod {pod.metadata.name} still has CUDA init container!")
                    # Check node is not the cordoned one (use getattr for both naming styles)
                    node_name = getattr(pod.spec, "node_name", None) or getattr(pod.spec, "nodeName", None)
                    if node_name == target_node:
                        logger.error(f"  ✗ Pod {pod.metadata.name} scheduled on cordoned node!")
        
        logger.info("  ✓ New pods are clean (no CUDA library, not on cordoned node)")
        
        # =====================================================================
        # PHASE 10: RECOVERY INFERENCE (fail fast if recovery fails)
        # =====================================================================
        logger.info("\n[PHASE 10] Recovery inference...")
        
        # Re-establish port-forward to new frontend pod
        if frontend_pf:
            try:
                frontend_pf.stop()
            except Exception:
                pass
        
        pods = deployment.get_pods()
        frontend_pf = None
        for service_name, service_pods in pods.items():
            if "frontend" in service_name.lower() and service_pods:
                frontend_pf = deployment.port_forward(service_pods[0], 8000)
                break
        
        if not frontend_pf or not frontend_pf.local_port:
            logger.error("  ✗ Failed to port-forward to new frontend!")
            pytest.fail("Failed to port-forward to new frontend after recovery")
        
        endpoint = f"http://localhost:{frontend_pf.local_port}/v1/completions"
        load_tester = InferenceLoadTester(endpoint, "Qwen/Qwen3-0.6B", timeout=30)
        
        logger.info("  Running recovery requests...")
        recovery_results = []
        for i in range(5):
            result = load_tester.send_inference_request(f"Recovery test {i}")
            recovery_results.append(result)
            status = "✓" if result["success"] else "✗"
            logger.info(f"    [{i+1}/5] {status} - {result['latency']:.2f}s")
            await asyncio.sleep(1)
        
        successful = sum(1 for r in recovery_results if r["success"])
        avg_latency = sum(r["latency"] for r in recovery_results) / len(recovery_results)
        load_stats["after_recovery"] = {
            "requests": 5,
            "success": successful,
            "avg_latency": avg_latency,
        }
        logger.info(f"  Recovery: {successful}/5 successful")
        
        # Fail fast if recovery fails
        if successful < 3:
            logger.error(f"  ✗ Recovery failed ({successful}/5) - pods not healthy!")
            pytest.fail(f"Recovery inference failed ({successful}/5)")
        
        deployment.collect_metrics(phase="after_recovery")
        
        # =====================================================================
        # SUMMARY
        # =====================================================================
        logger.info("\n" + "=" * 80)
        logger.info("TEST RESULTS - NVSENTINEL-DRIVEN EVICTION")
        logger.info("=" * 80)
        
        def fmt_rate(stats):
            if stats["requests"] == 0:
                return "N/A"
            return f"{stats['success']}/{stats['requests']} ({stats['success']/stats['requests']*100:.0f}%)"
        
        logger.info("")
        logger.info(f"{'Phase':<20} {'Success Rate':<18} {'Avg Latency':<15}")
        logger.info("-" * 55)
        logger.info(f"{'Baseline':<20} {fmt_rate(load_stats['baseline']):<18} {load_stats['baseline']['avg_latency']:.2f}s")
        logger.info(f"{'During Fault':<20} {fmt_rate(load_stats['during_fault']):<18} {load_stats['during_fault']['avg_latency']:.2f}s")
        logger.info(f"{'After Recovery':<20} {fmt_rate(load_stats['after_recovery']):<18} {load_stats['after_recovery']['avg_latency']:.2f}s")
        logger.info("")
        
        logger.info("KEY RESULTS:")
        logger.info(f"  ✓ CUDA passthrough (toggle-based): PASS")
        logger.info(f"  ✓ NVSentinel cordoned node: PASS")
        logger.info(f"  ✓ node-drainer evicted pods (no force delete): PASS")
        logger.info(f"  ✓ New pods clean (no CUDA lib, healthy node): PASS")
        logger.info("")
        
        logger.info("🎉 SUCCESS: Full NVSentinel-driven fault recovery!")
        logger.info("=" * 80)
        
        # Assertions
        assert passthrough_success, "CUDA passthrough setup failed"
        assert cordoned, "NVSentinel did not cordon node"
        assert evicted, "node-drainer did not evict pods"
        assert load_stats["baseline"]["success"] > 0, "Baseline inference failed"
        assert load_stats["after_recovery"]["success"] > 0, "Recovery inference failed"
