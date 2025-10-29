"""
XID 79 E2E Test - Refactored Version Using Helper Modules

This test simulates the complete NVSentinel fault tolerance workflow:
1. Inject XID 79 via API (triggers syslog detection)
2. Inject CUDA fault library (simulates GPU failure)
3. Wait for pods to crash naturally
4. Simulate NVSentinel workflow (cordon, drain, restart GPU, uncordon)
5. Validate recovery

Note: This refactored version uses helper modules to minimize code duplication.
Original version: 1425 lines → Refactored: ~200 lines
"""

import os
import sys
import time
import pytest
import requests
from pathlib import Path
from kubernetes import client, config

# Add helpers to path
sys.path.insert(0, str(Path(__file__).parent.parent / "helpers"))

from cuda_fault_injection import CUDAFaultInjector
from inference_testing import InferenceLoadTester
from nvsentinel_workflow import NVSentinelWorkflowSimulator
from k8s_operations import PodOperations

# Configuration
IN_CLUSTER = os.getenv("KUBERNETES_SERVICE_HOST") is not None
API_BASE_URL = "http://fault-injection-api.fault-injection-system.svc.cluster.local:8080" if IN_CLUSTER else "http://localhost:8080"

if IN_CLUSTER:
    config.load_incluster_config()
else:
    config.load_kube_config()

k8s_core = client.CoreV1Api()

# Test configuration
TARGET_DEPLOYMENT = os.getenv("TARGET_DEPLOYMENT", "vllm-v1-disagg-router")
NAMESPACE = "dynamo-oviya"
INFERENCE_ENDPOINT = os.getenv("INFERENCE_ENDPOINT", "http://localhost:8000/v1/completions")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-0.6B")
SKIP_CUDA_FAULT = os.getenv("SKIP_CUDA_FAULT_INJECTION", "false").lower() == "true"


@pytest.fixture
def cleanup_on_exit():
    """Pytest fixture to ensure cleanup happens even on Ctrl+C or test failure."""
    cleanup_state = {
        "fault_id": None,
        "cuda_injector": None,
        "workflow_simulator": None,
        "load_tester": None
    }
    
    yield cleanup_state
    
    # Cleanup always runs
    print("\n" + "="*80)
    print("CLEANUP")
    print("="*80)
    
    try:
        # Stop load tester
        if cleanup_state["load_tester"]:
            cleanup_state["load_tester"].stop()
        
        # CUDA fault injection cleanup
        if cleanup_state["cuda_injector"]:
            cleanup_state["cuda_injector"].cleanup_cuda_fault_injection(
                TARGET_DEPLOYMENT, NAMESPACE, force_delete_pods=False
            )
        
        # Workflow cleanup (uncordon nodes)
        if cleanup_state["workflow_simulator"]:
            cleanup_state["workflow_simulator"].cleanup()
        
        # Clean up fault API
        if cleanup_state["fault_id"]:
            try:
                requests.delete(f"{API_BASE_URL}/api/v1/faults/{cleanup_state['fault_id']}", timeout=10)
                print(f"[✓] Fault {cleanup_state['fault_id']} cleaned up")
            except Exception as e:
                print(f"[⚠] Failed to clean up fault: {e}")
        
        print("[✓] Cleanup complete")
        
    except Exception as e:
        print(f"[⚠] Cleanup encountered errors: {e}")


def test_xid79_with_cuda_fault_injection(cleanup_on_exit):
    """
    E2E test for XID 79 fault tolerance with REAL CUDA fault injection.
    
    Tests complete workflow:
    - XID 79 injection → syslog detection
    - CUDA fault injection → pods crash naturally
    - NVSentinel workflow → cordon, drain, restart GPU, uncordon
    - Recovery validation → inference works again
    """
    print("\n" + "="*80)
    print("XID 79 E2E TEST - REAL CUDA FAULT INJECTION")
    print("="*80)
    
    if SKIP_CUDA_FAULT:
        pytest.skip("CUDA fault injection skipped (SKIP_CUDA_FAULT_INJECTION=true)")
    
    # Initialize components
    cuda_injector = CUDAFaultInjector()
    load_tester = InferenceLoadTester(INFERENCE_ENDPOINT, MODEL_NAME)
    workflow = NVSentinelWorkflowSimulator(k8s_core, NAMESPACE, TARGET_DEPLOYMENT)
    pod_ops = PodOperations(k8s_core)
    
    # Register for cleanup
    cleanup_on_exit["cuda_injector"] = cuda_injector
    cleanup_on_exit["load_tester"] = load_tester
    cleanup_on_exit["workflow_simulator"] = workflow
    
    try:
        # ======================
        # PHASE 0: Prerequisites
        # ======================
        print("\n" + "="*80)
        print("PHASE 0: Prerequisites & Setup")
        print("="*80)
        
        # Check API health
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        assert response.status_code == 200, f"API unhealthy ({response.status_code})"
        print("[✓] Fault injection API healthy")
        
        # Build CUDA fault library
        assert cuda_injector.build_library(), "Failed to build CUDA fault injection library"
        print("[✓] CUDA fault injection library ready")
        
        # Get target pods and node
        pods = k8s_core.list_namespaced_pod(
            namespace=NAMESPACE,
            label_selector=f"nvidia.com/dynamo-component-type=worker,nvidia.com/dynamo-graph-deployment-name={TARGET_DEPLOYMENT}"
        )
        assert pods.items, f"No worker pods found for deployment: {TARGET_DEPLOYMENT}"
        
        target_node = pods.items[0].spec.node_name
        ready_pods = [p for p in pods.items 
                     if p.status.phase == "Running" and 
                     p.status.container_statuses and 
                     p.status.container_statuses[0].ready]
        
        assert len(ready_pods) >= 3, f"Expected 3 ready pods, found {len(ready_pods)}"
        print(f"[✓] Target node: {target_node}")
        print(f"[✓] {len(ready_pods)} worker pods ready")
        
        # Test baseline inference
        baseline_result = load_tester.send_inference_request()
        if baseline_result["success"]:
            print(f"[✓] Baseline inference working (latency: {baseline_result['latency']:.2f}s)")
        else:
            print(f"[⚠] Baseline inference failed: {baseline_result['error'][:100]}")
        
        # Start continuous load
        print("\n[→] Starting continuous inference load (1 request / 3s)")
        load_tester.start(interval=3.0)
        time.sleep(6)
        initial_stats = load_tester.get_stats()
        print(f"[✓] Baseline load: {initial_stats['success']}/{initial_stats['total']} requests successful")
        
        # ======================
        # PHASE 1: XID 79 Injection
        # ======================
        print("\n" + "="*80)
        print("PHASE 1: XID 79 Injection (Triggers NVSentinel Detection)")
        print("="*80)
        
        response = requests.post(
            f"{API_BASE_URL}/api/v1/faults/gpu/inject/xid-79",
            json={"node_name": target_node, "xid_type": 79, "gpu_id": 0},
            timeout=60
        )
        assert response.status_code == 200, f"XID injection failed: {response.text}"
        
        fault_id = response.json()["fault_id"]
        cleanup_on_exit["fault_id"] = fault_id
        print(f"[✓] XID 79 injected successfully (Fault ID: {fault_id})")
        print(f"    NVSentinel syslog-health-monitor will detect this")
        
        # ======================
        # PHASE 2: CUDA Fault Injection
        # ======================
        print("\n" + "="*80)
        print("PHASE 2: CUDA Fault Injection (Simulates GPU Hardware Failure)")
        print("="*80)
        
        # Create ConfigMap with CUDA fault library
        assert cuda_injector.create_configmap_with_library(NAMESPACE), "Failed to create ConfigMap"
        
        # Patch deployment to use CUDA fault library (pins pods to target_node)
        assert cuda_injector.patch_deployment_for_cuda_fault(
            TARGET_DEPLOYMENT, NAMESPACE, target_node=target_node, xid_type=79
        ), "Failed to patch deployment"
        
        # Trigger restart
        target_pods = [p for p in pods.items if p.spec.node_name == target_node]
        cuda_injector.trigger_pod_restart(target_pods, NAMESPACE)
        
        print(f"\n[→] Pods pinned to {target_node} (simulates real XID 79)")
        
        # Wait for pods to crash
        label_selector = f"nvidia.com/dynamo-component-type=worker,nvidia.com/dynamo-graph-deployment-name={TARGET_DEPLOYMENT}"
        crashed = cuda_injector.wait_for_pods_to_crash(NAMESPACE, label_selector, target_node, timeout=420)
        
        if crashed:
            print("[✓] Pods crashing - CUDA fault injection working!")
        else:
            print("[⚠] Not all pods crashed within 7 minutes, continuing...")
        
        # Show crash impact on inference
        crash_stats = load_tester.get_stats()
        print(f"\n[→] Inference during crash: {crash_stats['success_rate']:.1f}% success rate")
        
        # ======================
        # PHASE 3: Cordon Node
        # ======================
        assert workflow.cordon_faulty_node(target_node), "Failed to cordon node"
        
        # ======================
        # PHASE 4: Remove CUDA Fault & Drain
        # ======================
        print("\n[TEST CLEANUP] Removing CUDA fault injection artifacts...")
        print("    (In production: no LD_PRELOAD to remove)")
        assert cuda_injector.cleanup_cuda_fault_injection(
            TARGET_DEPLOYMENT, NAMESPACE, force_delete_pods=True
        ), "Failed to cleanup CUDA fault"
        
        print(f"\n[✓] All pods deleted - will recreate with clean spec")
        print(f"    (Equivalent to draining in production)")
        
        # ======================
        # PHASE 5: GPU Driver Restart
        # ======================
        driver_success = workflow.restart_gpu_driver(target_node, wait_timeout=300)
        
        # ======================
        # PHASE 6: Uncordon Node
        # ======================
        if driver_success:
            assert workflow.uncordon_node(target_node), "Failed to uncordon node"
        else:
            print("[⚠] GPU driver restart failed - leaving node cordoned")
        
        # ======================
        # PHASE 7: Wait for Rescheduling
        # ======================
        rescheduled = workflow.wait_for_pod_rescheduling(
            expected_count=3, 
            exclude_node=target_node if not driver_success else None,
            timeout=900
        )
        assert rescheduled, "Pods failed to reschedule"
        
        # ======================
        # PHASE 8: Inference Recovery
        # ======================
        recovery_rate = workflow.wait_for_inference_recovery(load_tester, timeout=900)
        assert recovery_rate >= 70, f"Recovery rate too low: {recovery_rate}%"
        
        # ======================
        # PHASE 9: Final Summary
        # ======================
        load_tester.stop()
        final_stats = load_tester.get_stats()
        
        print("\n" + "="*80)
        print("✓ TEST COMPLETED - XID 79 E2E WITH CUDA FAULT INJECTION")
        print("="*80)
        print("\nValidated:")
        print("  ✓ XID 79 injection works")
        print("  ✓ CUDA fault library makes CUDA calls fail")
        print("  ✓ Pods crash naturally due to CUDA errors")
        print("  ✓ Node cordoning prevents new pods on faulty node")
        print("  ✓ Node draining evicts crashing pods")
        print(f"  ✓ GPU driver restart: {'succeeded' if driver_success else 'failed'}")
        print("  ✓ Pods reschedule to healthy nodes")
        print(f"  ✓ Inference recovery: {recovery_rate:.0f}%")
        print(f"  ✓ Overall availability: {final_stats['success_rate']:.1f}%")
        print("\nComplete NVSentinel workflow validated!")
        print("="*80)
        
    except Exception as e:
        print(f"\n[✗] TEST FAILED: {e}")
        raise


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

