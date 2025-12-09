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
_hw_helpers_path = Path(__file__).parent.parent / "hardware/fault_injection_service/helpers"
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
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


@pytest.fixture
def vllm_disagg_deployment():
    """Create a vLLM disaggregated deployment spec."""
    workspace = _get_workspace_dir()
    yaml_path = os.path.join(workspace, "examples/backends/vllm/deploy/disagg_router.yaml")
    
    if not os.path.exists(yaml_path):
        pytest.skip(f"Deployment YAML not found: {yaml_path}")
    
    spec = DeploymentSpec(yaml_path)
    spec.name = "hw-fault-test"
    spec.set_dynamo_namespace("hw-fault-test")
    spec.set_model("Qwen/Qwen3-0.6B")
    
    return spec


@pytest.mark.k8s
@pytest.mark.fault_tolerance
@pytest.mark.hw_faults
@pytest.mark.slow
async def test_gpu_xid79(
    request,
    namespace,
    image,
    hw_fault_config,
    vllm_disagg_deployment,
):
    """
    XID 79 (GPU Fell Off Bus) fault tolerance test with CUDA fault injection.
    
    Flow:
    1. Deploy pods
    2. Setup CUDA passthrough (faults disabled, library loaded)
    3. Run baseline inference (with library loaded)
    4. Inject XID 79 + enable CUDA faults
    5. Inference during fault (should fail)
    6. Verify NVSentinel cordoned node
    7. Recovery: disable faults, remove affinity, wait for reschedule
    8. Recovery inference (should succeed)
    """
    logger = logging.getLogger(request.node.name)
    
    if hw_fault_config is None:
        pytest.skip("Hardware faults not enabled (use --enable-hw-faults)")
    
    logger.info("=" * 80)
    logger.info("TEST: XID 79 (GPU Fell Off Bus) - CUDA Fault Injection")
    logger.info("=" * 80)
    
    if image:
        vllm_disagg_deployment.set_image(image)
    
    load_stats = {
        "baseline": {"requests": 0, "success": 0, "avg_latency": 0},
        "during_fault": {"requests": 0, "success": 0, "avg_latency": 0},
        "after_recovery": {"requests": 0, "success": 0, "avg_latency": 0},
    }
    
    async with ManagedDeployment(
        namespace=namespace,
        log_dir=request.node.name,
        deployment_spec=vllm_disagg_deployment,
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
                        phase_str = pod.status.phase if hasattr(pod, 'status') else 'Unknown'
                        node = pod.spec.nodeName if hasattr(pod, 'spec') and hasattr(pod.spec, 'nodeName') else 'Unassigned'
                        logger.info(f"    - {pod.name}: {phase_str} on {node}")
                    except Exception:
                        logger.info(f"    - {pod.name}: (status unavailable)")
        
        pods = deployment.get_pods()
        initial_count = sum(len(p) for p in pods.values())
        logger.info(f"  Initial pods: {initial_count}")
        log_pod_status("Initial")
        
        # =====================================================================
        # PHASE 2: CUDA PASSTHROUGH SETUP (Library loaded, faults disabled)
        # =====================================================================
        logger.info("\n[PHASE 2] Setting up CUDA passthrough...")
        logger.info("  (Pods will restart with library loaded, faults DISABLED)")
        
        passthrough_success = await deployment.setup_cuda_passthrough(xid_type=79)
        assert passthrough_success, "Failed to setup CUDA passthrough"
        logger.info("  ✓ CUDA passthrough configured")
        
        logger.info("  Waiting for pods to restart with CUDA library...")
        await deployment.wait_for_all_pods_ready(timeout=300)
        logger.info("  ✓ Pods ready with CUDA library (faults disabled)")
        
        log_pod_status("After CUDA Setup")
        
        # =====================================================================
        # PHASE 3: BASELINE INFERENCE (with CUDA library loaded, faults OFF)
        # =====================================================================
        logger.info("\n[PHASE 3] Baseline inference (CUDA library loaded, faults OFF)...")
        
        load_tester = None
        frontend_pf = None
        
        if InferenceLoadTester:
            pods = deployment.get_pods()
            for service_name, service_pods in pods.items():
                if "frontend" in service_name.lower() and service_pods:
                    frontend_pf = deployment.port_forward(service_pods[0], 8000)
                    break
            
            if frontend_pf and frontend_pf.local_port:
                endpoint = f"http://localhost:{frontend_pf.local_port}/v1/completions"
                load_tester = InferenceLoadTester(endpoint, "Qwen/Qwen3-0.6B", timeout=30)
                logger.info(f"  ✓ Endpoint: {endpoint}")
                
                logger.info("  Running baseline requests...")
                baseline_results = []
                for i in range(5):
                    result = load_tester.send_inference_request(f"Baseline test {i}")
                    baseline_results.append(result)
                    status = "✓" if result["success"] else "✗"
                    logger.info(f"    [{i+1}/5] {status} - {result['latency']:.2f}s")
                    await asyncio.sleep(1)
                
                successful = sum(1 for r in baseline_results if r["success"])
                avg_latency = sum(r["latency"] for r in baseline_results) / len(baseline_results)
                load_stats["baseline"] = {"requests": 5, "success": successful, "avg_latency": avg_latency}
                logger.info(f"  Baseline: {successful}/5 successful, {avg_latency:.2f}s avg")
        
        deployment.collect_metrics(phase="baseline")
        
        # =====================================================================
        # PHASE 4: FAULT INJECTION (XID + enable CUDA faults)
        # =====================================================================
        logger.info("\n[PHASE 4] Injecting XID 79 and enabling CUDA faults...")
        
        fault_id = await deployment.inject_hw_fault(fault_type='xid', xid_type=79, gpu_id=0)
        logger.info(f"  ✓ XID 79 injected: {fault_id or 'API not available'}")
        
        toggle_success = await deployment.toggle_cuda_faults(enable=True)
        assert toggle_success, "Failed to toggle CUDA faults"
        logger.info("  ✓ CUDA faults ENABLED - pods will crash on CUDA calls")
        
        # =====================================================================
        # PHASE 5: INFERENCE DURING FAULT (should fail)
        # =====================================================================
        logger.info("\n[PHASE 5] Inference during fault...")
        log_pod_status("During Fault")
        
        if load_tester:
            fault_results = []
            for i in range(5):
                result = load_tester.send_inference_request(f"Fault test {i}")
                fault_results.append(result)
                status = "✓" if result["success"] else "✗"
                logger.info(f"    [{i+1}/5] {status} - {result['latency']:.2f}s")
                await asyncio.sleep(2)
            
            successful = sum(1 for r in fault_results if r["success"])
            avg_latency = sum(r["latency"] for r in fault_results) / len(fault_results)
            load_stats["during_fault"] = {"requests": 5, "success": successful, "avg_latency": avg_latency}
            logger.info(f"  During fault: {successful}/5 successful (failures expected)")
        
        # =====================================================================
        # PHASE 6: NVSENTINEL RESPONSE
        # =====================================================================
        logger.info("\n[PHASE 6] Checking NVSentinel response...")
        await asyncio.sleep(10)
        
        cordoned = deployment.is_node_cordoned(target_node)
        logger.info(f"  Node cordoned: {'✓' if cordoned else '✗'}")
        
        # =====================================================================
        # PHASE 7: RECOVERY
        # =====================================================================
        logger.info("\n[PHASE 7] Recovery...")
        
        await deployment.toggle_cuda_faults(enable=False)
        logger.info("  ✓ CUDA faults DISABLED")
        
        logger.info("  Removing node affinity to allow rescheduling...")
        await deployment.remove_node_affinity()
        logger.info("  ✓ Node affinity removed")
        
        logger.info("  Waiting for pods to reschedule to healthy nodes...")
        await deployment.wait_for_pods_on_healthy_nodes(exclude_node=target_node, timeout=360)
        
        logger.info("  Waiting for pods to become Ready...")
        await deployment.wait_for_all_pods_ready(timeout=360)
        
        log_pod_status("After Recovery")
        
        # =====================================================================
        # PHASE 8: RECOVERY INFERENCE
        # =====================================================================
        logger.info("\n[PHASE 8] Recovery inference...")
        
        if load_tester:
            if frontend_pf:
                try:
                    frontend_pf.stop()
                except Exception:
                    pass
            
            pods = deployment.get_pods()
            for service_name, service_pods in pods.items():
                if "frontend" in service_name.lower() and service_pods:
                    frontend_pf = deployment.port_forward(service_pods[0], 8000)
                    if frontend_pf and frontend_pf.local_port:
                        endpoint = f"http://localhost:{frontend_pf.local_port}/v1/completions"
                        load_tester = InferenceLoadTester(endpoint, "Qwen/Qwen3-0.6B", timeout=30)
                    break
            
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
            load_stats["after_recovery"] = {"requests": 5, "success": successful, "avg_latency": avg_latency}
            logger.info(f"  Recovery: {successful}/5 successful, {avg_latency:.2f}s avg")
        
        deployment.collect_metrics(phase="after_recovery")
        
        # =====================================================================
        # SUMMARY
        # =====================================================================
        logger.info("\n" + "=" * 80)
        logger.info("TEST RESULTS SUMMARY")
        logger.info("=" * 80)
        
        pods = deployment.get_pods()
        final_count = sum(len(p) for p in pods.values())
        
        def fmt_rate(stats):
            if stats['requests'] == 0:
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
        logger.info(f"  ✓ CUDA passthrough: {'PASS' if passthrough_success else 'FAIL'}")
        logger.info(f"  ✓ CUDA fault toggle: {'PASS' if toggle_success else 'FAIL'}")
        logger.info(f"  ✓ Node cordoned: {'PASS' if cordoned else 'FAIL'}")
        logger.info(f"  ✓ Pods: {initial_count} → {final_count}")
        logger.info("")
        
        if cordoned:
            logger.info("🎉 SUCCESS: NVSentinel detected XID 79 and cordoned the node!")
        
        logger.info("=" * 80)
        
        # Assertions
        assert passthrough_success, "CUDA passthrough setup failed"
        assert toggle_success, "CUDA fault toggle failed"
        assert cordoned, "NVSentinel did not cordon the node"
        assert load_stats["after_recovery"]["success"] > 0, "Recovery inference failed"
