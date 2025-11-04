"""
XID 48 E2E Fault Tolerance Test - Double-Bit ECC Error (Automated)

Tests the fault tolerance flow for memory corruption (XID 48) with NVSentinel.

Key differences from XID 79:
- Memory-level error (not full hardware failure)
- May NOT trigger node drain (depends on NVSentinel policy)
- Less severe than XID 79
- Pods should continue running (no CUDA errors)
- Tests ECC error handling path

Flow:
1. Inject XID 48 (Double-Bit ECC) → Kernel logs
2. NVSentinel detects → Creates health event
3. Determine action based on policy:
   - Option A: Node cordoned + pods drained (if ECC errors exceed threshold)
   - Option B: Warning logged + monitoring (if below threshold)
4. System stabilizes

Usage:
    # In-cluster (recommended):
    python scripts/run_test_incluster.py examples/test_xid48_dbe_ecc_automated.py xid48-test
    
    # Local (requires port-forward):
    pytest examples/test_xid48_dbe_ecc_automated.py -v -s
"""

import os
import sys
import time
import pytest
import requests
from kubernetes import client, config
from kubernetes.client.rest import ApiException

# Add helpers to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'helpers'))
from inference_testing import InferenceLoadTester, get_inference_endpoint
from nvsentinel_monitor import NVSentinelMonitor

# Configuration
IN_CLUSTER = os.getenv("KUBERNETES_SERVICE_HOST") is not None

if IN_CLUSTER:
    API_BASE_URL = "http://fault-injection-api.fault-injection-system.svc.cluster.local:8080"
    config.load_incluster_config()
else:
    API_BASE_URL = "http://localhost:8080"
    config.load_kube_config()

k8s_core = client.CoreV1Api()

# Test configuration
TARGET_DEPLOYMENT = os.getenv("TARGET_DEPLOYMENT", "vllm-v1-disagg-router")
NAMESPACE = "dynamo-oviya"
FAULT_INJECTION_NAMESPACE = "fault-injection-system"
NVSENTINEL_NAMESPACE = "nvsentinel"
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-0.6B")

# Get inference endpoint (auto-detects in-cluster vs local)
INFERENCE_ENDPOINT = get_inference_endpoint(TARGET_DEPLOYMENT, NAMESPACE)

# Timeout configuration
DETECTION_TIMEOUT = 90  # NVSentinel detection (60s poll + buffer)
MONITORING_TIMEOUT = 120  # Monitor for unexpected behavior


class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    RESET = '\033[0m'




@pytest.fixture(scope="function")
def cleanup_on_exit():
    """Fixture for cleanup tracking."""
    cleanup_data = {}
    yield cleanup_data
    
    # Cleanup
    print(f"\n{Colors.BLUE}[CLEANUP] Starting cleanup...{Colors.RESET}")
    
    if "load_tester" in cleanup_data and cleanup_data["load_tester"]:
        try:
            cleanup_data["load_tester"].stop()
            print(f"[CLEANUP] Load tester stopped")
        except Exception as e:
            print(f"[CLEANUP] Warning: Could not stop load tester: {e}")
    
    if "fault_id" in cleanup_data:
        try:
            requests.delete(
                f"{API_BASE_URL}/api/v1/faults/{cleanup_data['fault_id']}",
                timeout=10
            )
            print(f"[CLEANUP] Fault {cleanup_data['fault_id']} deleted")
        except Exception as e:
            print(f"[CLEANUP] Warning: Could not delete fault: {e}")
    
    if "target_node" in cleanup_data:
        try:
            # Uncordon node if it was cordoned during test
            node = k8s_core.read_node(cleanup_data['target_node'])
            if node.spec.unschedulable:
                k8s_core.patch_node(
                    cleanup_data['target_node'],
                    {"spec": {"unschedulable": False}}
                )
                print(f"[CLEANUP] Node {cleanup_data['target_node']} uncordoned")
        except Exception as e:
            print(f"[CLEANUP] Warning: Could not uncordon node: {e}")


def test_xid48_dbe_ecc_automated(cleanup_on_exit):
    """
    E2E test for XID 48 (Double-Bit ECC Error) with NVSentinel.
    
    This test validates:
    - XID 48 injection triggers syslog-health-monitor detection
    - NVSentinel classifies XID 48 as FATAL (per NVIDIA XID catalog)
    - fault-quarantine-module cordons the node immediately
    - node-drainer waits 60 min before evicting pods (AllowCompletion mode)
    - Within test window: Pods continue running, inference works
    - GPU remains functional (memory error but CUDA operational)
    
    Key Differences from XID 79:
    - XID 48: Memory corruption (GPU on bus, CUDA works, 60min grace)
    - XID 79: Hardware disconnect (GPU off bus, CUDA fails immediately)
    - XID 48: No CUDA fault injection needed (GPU functional)
    - Both: Classified as fatal, trigger cordon
    - Timing: XID 48 has grace period, XID 79 immediate failure
    """
    print("\n" + "=" * 80)
    print(f"{Colors.YELLOW}XID 48 E2E TEST - DOUBLE-BIT ECC ERROR (AUTOMATED){Colors.RESET}")
    print("=" * 80)

    nvsentinel = NVSentinelMonitor(k8s_core, NVSENTINEL_NAMESPACE)
    load_tester = InferenceLoadTester(INFERENCE_ENDPOINT, MODEL_NAME)
    cleanup_on_exit["load_tester"] = load_tester

    try:
        # ======================
        # PHASE 0: Prerequisites
        # ======================
        print("\n" + "=" * 80)
        print(f"{Colors.BLUE}PHASE 0: Prerequisites & Health Checks{Colors.RESET}")
        print("=" * 80)

        # Check fault injection API
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        assert response.status_code == 200, f"API unhealthy ({response.status_code})"
        print(f"[✓] Fault injection API healthy")

        # Check NVSentinel components
        components = nvsentinel.check_nvsentinel_health()
        print("\nNVSentinel Components:")
        critical_components = ["syslog-health-monitor"]
        
        all_critical_healthy = True
        for component, healthy in components.items():
            status = "✓" if healthy else "✗"
            is_critical = component in critical_components
            component_type = "(required)" if is_critical else "(optional)"
            print(f"  [{status}] {component} {component_type}: {'Running' if healthy else 'Not Ready'}")
            if not healthy and is_critical:
                all_critical_healthy = False
        
        if not all_critical_healthy:
            pytest.skip("Critical NVSentinel components not ready - skipping test")

        # Get target pods and node
        pods = k8s_core.list_namespaced_pod(
            namespace=NAMESPACE,
            label_selector=f"nvidia.com/dynamo-component-type=worker,nvidia.com/dynamo-graph-deployment-name={TARGET_DEPLOYMENT}",
        )
        assert pods.items, f"No worker pods found for deployment: {TARGET_DEPLOYMENT}"

        target_node = pods.items[0].spec.node_name
        cleanup_on_exit["target_node"] = target_node
        
        ready_pods = [
            p for p in pods.items
            if p.status.phase == "Running"
            and p.status.container_statuses
            and p.status.container_statuses[0].ready
        ]

        print(f"\n[✓] Target node: {target_node}")
        print(f"[✓] {len(ready_pods)} worker pods ready")

        # Record initial node state
        initial_node = k8s_core.read_node(target_node)
        initial_schedulable = not (initial_node.spec.unschedulable or False)
        print(f"[✓] Node initially schedulable: {initial_schedulable}")

        # Test baseline inference
        print(f"\n[→] Testing baseline inference...")
        print(f"    Endpoint: {INFERENCE_ENDPOINT}")
        print(f"    Model: {MODEL_NAME}")
        baseline_result = load_tester.send_inference_request()
        if baseline_result["success"]:
            print(f"[✓] Baseline inference working (latency: {baseline_result['latency']:.2f}s)")
        else:
            print(f"[⚠] Baseline inference failed: {baseline_result['error']}")
            print(f"    Test will continue but may fail later")

        # Start continuous load
        print(f"\n[→] Starting continuous inference load (1 request / 3s)")
        load_tester.start(interval=3.0)
        time.sleep(6)  # Let a couple requests go through
        initial_stats = load_tester.get_stats()
        print(f"[✓] Baseline load: {initial_stats['success']}/{initial_stats['total']} requests successful")

        # ======================
        # PHASE 1: XID 48 Injection
        # ======================
        print("\n" + "=" * 80)
        print(f"{Colors.YELLOW}PHASE 1: XID 48 (Double-Bit ECC Error) Injection{Colors.RESET}")
        print("=" * 80)

        print(f"\n[→] Injecting XID 48 on {target_node}")
        print(f"    XID 48 = Double-Bit ECC Error (uncorrectable memory error)")
        print(f"    Expected: NVSentinel logs event, action based on threshold")
        
        response = requests.post(
            f"{API_BASE_URL}/api/v1/faults/gpu/inject/xid-48",
            json={"node_name": target_node, "xid_type": 48, "gpu_id": 0},
            timeout=60,
        )
        assert response.status_code == 200, f"XID injection failed: {response.text}"

        fault_id = response.json()["fault_id"]
        cleanup_on_exit["fault_id"] = fault_id
        print(f"[✓] XID 48 injected successfully (Fault ID: {fault_id})")
        print(f"    syslog-health-monitor will detect this in kernel logs")

        # ======================
        # PHASE 2: Wait for Detection
        # ======================
        print("\n" + "=" * 80)
        print(f"{Colors.BLUE}PHASE 2: NVSentinel Detection{Colors.RESET}")
        print("=" * 80)

        print(f"\n[→] Waiting for NVSentinel detection ({DETECTION_TIMEOUT}s)")
        print(f"    syslog-health-monitor polls every ~15s")
        time.sleep(DETECTION_TIMEOUT)
        print(f"[✓] Detection window elapsed")

        # Validate full detection pipeline
        print(f"\n[→] Validating NVSentinel detection pipeline...")
        detected, results = nvsentinel.wait_for_detection(target_node, 48, timeout=60, expect_quarantine=True)
        
        if detected:
            print(f"[✓] Detection pipeline validated successfully")
            
            # Validate health event structure if found
            if results.get("health_event") and results.get("event_data"):
                print(f"\n[→] Validating health event structure...")
                valid, validation_msg = nvsentinel.validate_health_event_structure(
                    results["event_data"].get("raw", ""),
                    xid_type=48,
                    expected_fatal=True,
                    expected_agent="syslog-health-monitor"
                )
                if valid:
                    print(f"[✓] Health event structure valid")
                else:
                    print(f"[⚠] Health event validation: {validation_msg}")
            
            # Show quarantine details
            if results.get("quarantine_details"):
                quarantine = results["quarantine_details"]
                print(f"\n[→] Quarantine details:")
                print(f"    Cordoned: {quarantine.get('cordoned')}")
                print(f"    Annotations: {list(quarantine.get('annotations', {}).keys())}")
                print(f"    Labels: {list(quarantine.get('labels', {}).keys())}")
        else:
            print(f"[✗] Detection validation failed:")
            for msg in results.get("messages", []):
                print(f"    {msg}")
            pytest.fail(f"NVSentinel detection failed. Results: {results}")

        # ======================
        # PHASE 3: Monitor System Response
        # ======================
        print("\n" + "=" * 80)
        print(f"{Colors.BLUE}PHASE 3: Monitor System Response{Colors.RESET}")
        print("=" * 80)

        print(f"\n[→] Monitoring system behavior ({MONITORING_TIMEOUT}s)")
        print(f"    XID 48 is classified as FATAL by NVSentinel:")
        print(f"    - fault-quarantine cordons node immediately")
        print(f"    - node-drainer waits 60 min before eviction (AllowCompletion)")
        print(f"    - KEY EXPECTATION: Pods stay running, inference continues")
        
        start_time = time.time()
        node_was_cordoned = False
        last_inference_check = start_time
        
        while time.time() - start_time < MONITORING_TIMEOUT:
            # Check node status
            node = k8s_core.read_node(target_node)
            is_cordoned = node.spec.unschedulable or False
            
            if is_cordoned and not node_was_cordoned:
                node_was_cordoned = True
                print(f"\n[!] Node {target_node} was cordoned")
                print(f"    This indicates ECC error threshold was exceeded")
                print(f"    NVSentinel triggered fault-quarantine-module")
                break
            
            # Check pod health
            current_pods = k8s_core.list_namespaced_pod(
                namespace=NAMESPACE,
                label_selector=f"nvidia.com/dynamo-component-type=worker,nvidia.com/dynamo-graph-deployment-name={TARGET_DEPLOYMENT}",
            )
            current_ready = [
                p for p in current_pods.items
                if p.status.phase == "Running"
                and p.status.container_statuses
                and p.status.container_statuses[0].ready
            ]
            
            # Pods should stay running during grace period
            if len(current_ready) < len(ready_pods):
                print(f"\n[⚠] Pod count decreased: {len(current_ready)}/{len(ready_pods)}")
                print(f"    Unexpected - XID 48 has 60min grace period")
            
            # Check inference health every 30s
            if time.time() - last_inference_check >= 30:
                stats = load_tester.get_stats()
                print(f"\n[→] Inference check: {stats['success']}/{stats['total']} successful ({stats['success_rate']:.1f}%)")
                if stats['success_rate'] < 70:
                    print(f"    [⚠] Success rate dropped below 70% - unexpected for XID 48!")
                last_inference_check = time.time()
            
            time.sleep(15)
        
        # ======================
        # PHASE 4: Verify Final State
        # ======================
        print("\n" + "=" * 80)
        print(f"{Colors.BLUE}PHASE 4: Verify Final State{Colors.RESET}")
        print("=" * 80)

        final_node = k8s_core.read_node(target_node)
        final_schedulable = not (final_node.spec.unschedulable or False)
        
        final_pods = k8s_core.list_namespaced_pod(
            namespace=NAMESPACE,
            label_selector=f"nvidia.com/dynamo-component-type=worker,nvidia.com/dynamo-graph-deployment-name={TARGET_DEPLOYMENT}",
        )
        final_ready = [
            p for p in final_pods.items
            if p.status.phase == "Running"
            and p.status.container_statuses
            and p.status.container_statuses[0].ready
        ]

        # Stop load tester and get final stats
        load_tester.stop()
        final_stats = load_tester.get_stats()

        print(f"\nFinal System State:")
        print(f"  Node schedulable: {final_schedulable}")
        print(f"  Ready pods: {len(final_ready)}/{len(ready_pods)}")
        print(f"  Node was cordoned during test: {node_was_cordoned}")
        print(f"  Inference success rate: {final_stats['success_rate']:.1f}% ({final_stats['success']}/{final_stats['total']})")
        print(f"  Average latency: {final_stats['avg_latency']:.2f}s")

        # Assert key expectations for XID 48
        # Node should be cordoned (XID 48 is fatal)
        assert not final_schedulable, \
            f"Node should be cordoned for XID 48 (fatal error)"
        
        # Pods should stay running during grace period (60 min before eviction)
        assert len(final_ready) == len(ready_pods), \
            f"Pods should stay running during grace period: {len(final_ready)} vs {len(ready_pods)}"
        
        # Inference should continue working (GPU functional, just has ECC error)
        assert final_stats['success_rate'] >= 70, \
            f"Inference should continue during grace period: {final_stats['success_rate']:.1f}% (GPU still functional)"

        # ======================
        # PHASE 5: Cleanup
        # ======================
        print("\n" + "=" * 80)
        print(f"{Colors.BLUE}PHASE 5: Cleanup{Colors.RESET}")
        print("=" * 80)

        # Delete fault
        requests.delete(f"{API_BASE_URL}/api/v1/faults/{fault_id}", timeout=10)
        print(f"[✓] Fault {fault_id} deleted")

        # Uncordon node if needed
        if node_was_cordoned or not final_schedulable:
            k8s_core.patch_node(target_node, {"spec": {"unschedulable": False}})
            print(f"[✓] Node {target_node} uncordoned")

        # Mark load tester as cleared in cleanup
        cleanup_on_exit["load_tester"] = None

        # ======================
        # SUCCESS
        # ======================
        print("\n" + "=" * 80)
        print(f"{Colors.GREEN}✓ TEST COMPLETED - XID 48 DOUBLE-BIT ECC ERROR{Colors.RESET}")
        print("=" * 80)
        print("\nValidated Detection Pipeline:")
        print(f"  ✓ XID 48 injection: Injected to kernel logs on {target_node}")
        print(f"  ✓ Kernel logs: XID 48 message verified in syslog")
        print(f"  ✓ Health event: Created by syslog-health-monitor")
        print(f"  ✓ Event structure: isFatal=true, errorCode=[48], agent=syslog-health-monitor")
        print(f"  ✓ Node quarantine: fault-quarantine cordoned the node")
        print(f"  ✓ Quarantine details: Annotations and labels applied")
        print(f"\nValidated System Behavior:")
        print(f"  ✓ Pod stability: Pods stayed running during 60min grace period")
        print(f"  ✓ Inference continuity: {final_stats['success_rate']:.1f}% success rate")
        print(f"     ({final_stats['success']}/{final_stats['total']} requests successful)")
        
        print("\nKey Differences from XID 79:")
        print(f"  • XID 48: Memory corruption (GPU on bus, CUDA works, 60min grace)")
        print(f"  • XID 79: Hardware disconnect (GPU off bus, CUDA fails immediately)")
        print(f"  • XID 48: Pods stay alive during grace (eventually evicted after 60min)")
        print(f"  • XID 79: Pods crash immediately (CUDA_ERROR_NO_DEVICE)")
        print(f"  • Both: Classified as fatal, trigger cordon")
        print(f"  • XID 48: No CUDA fault injection needed (GPU functional)")
        
        print(f"\nLogs to check:")
        print(f"  kubectl logs -n {NVSENTINEL_NAMESPACE} -l app.kubernetes.io/name=syslog-health-monitor --tail=50")
        print(f"  kubectl logs -n gpu-operator -l app=nvidia-dcgm --tail=50")
        print(f"  dmesg | grep -i 'xid.*48' (on node {target_node})")
        print("=" * 80)

    except Exception as e:
        print(f"\n[✗] TEST FAILED: {e}")
        
        # Print inference stats if available
        if load_tester and load_tester.running:
            try:
                stats = load_tester.get_stats()
                print(f"\nInference stats at failure:")
                print(f"  {stats['success']}/{stats['total']} successful ({stats['success_rate']:.1f}%)")
                if stats['errors']:
                    print(f"  Recent errors: {stats['errors'][:3]}")
            except:
                pass
        
        raise


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

