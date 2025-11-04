#!/usr/bin/env python3
"""
GPU Fault Tolerance Test Example - Framework Usage Guide

Copy-paste template showing how to use the GPU fault injection framework.
"""
import sys
import time
from pathlib import Path

import pytest
from kubernetes import client, config

sys.path.insert(0, str(Path(__file__).parent.parent / "client"))

from gpu_fault_helpers import GPUFaultHelper
from test_helpers import get_config_from_env, send_completion_request


# ==============================================================================
# Setup
# ==============================================================================

@pytest.fixture(scope="module")
def gpu_helper():
    """Initialize GPU fault helper"""
    config = get_config_from_env()
    helper = GPUFaultHelper(api_url=config['api_url'])
    yield helper
    helper.close()


@pytest.fixture
def gpu_node():
    """Get a GPU node name from your cluster"""
    config.load_kube_config()
    k8s = client.CoreV1Api()
    
    # Find node with worker pods
    pods = k8s.list_namespaced_pod(
        namespace="dynamo-oviya",  # Your namespace
        label_selector="nvidia.com/dynamo-component-type=worker"
    )
    
    if pods.items:
        return pods.items[0].spec.node_name
    
    pytest.skip("No GPU nodes found")


# ==============================================================================
# Example 1: Context Manager (Recommended - Auto-cleanup)
# ==============================================================================

@pytest.mark.fault_tolerance
def test_xid_94_with_context_manager(gpu_helper, gpu_node):
    """Use context manager for automatic fault recovery"""
    
    # 1. Test baseline (before fault)
    send_completion_request("baseline", 10)
    
    # 2. Inject fault with context manager
    with gpu_helper.xid_94(node_name=gpu_node, gpu_id=0):
        # Fault is active here
        time.sleep(10)  # Wait for fault to propagate
        
        # 3. Test during fault (other GPUs should handle requests)
        for i in range(5):
            try:
                send_completion_request(f"test-{i}", 10)
                # System fault-tolerant
            except:
                # Some requests may fail
                pass
    
    # 4. Fault auto-recovered, validate recovery
    time.sleep(5)
    send_completion_request("recovery", 10)  # Should work


# ==============================================================================
# Example 2: Manual Injection (More Control)
# ==============================================================================

@pytest.mark.fault_tolerance
def test_xid_48_manual_injection(gpu_helper, gpu_node):
    """Manual injection with explicit recovery"""
    
    # 1. Baseline
    send_completion_request("baseline", 10)
    
    # 2. Inject fault manually
    fault = gpu_helper.inject_xid_48_dbe_ecc(
        node_name=gpu_node,
        gpu_id=0,
        duration=None  # No auto-expire
    )
    fault_id = fault['fault_id']
    
    try:
        # 3. Test during fault
        time.sleep(30)  # High-severity needs more time
        
        for i in range(5):
            try:
                send_completion_request(f"test-{i}", 10, timeout=45)
            except:
                pass
        
    finally:
        # 4. Always cleanup (even on test failure)
        gpu_helper.recover_fault(fault_id)
    
    # 5. Validate recovery
    time.sleep(10)
    send_completion_request("recovery", 10)


# ==============================================================================
# Example 3: All XID Types Available
# ==============================================================================

@pytest.mark.fault_tolerance
def test_all_xid_types(gpu_helper, gpu_node):
    """Show all available XID error types"""
    
    # Low severity - fast recovery
    with gpu_helper.xid_94(node_name=gpu_node, gpu_id=0):
        time.sleep(5)
        send_completion_request("test", 10)
    
    time.sleep(5)
    
    # Medium severity
    # with gpu_helper.xid_43(node_name=gpu_node, gpu_id=0):
    #     time.sleep(10)
    #     send_completion_request("test", 10)
    
    # with gpu_helper.xid_119(node_name=gpu_node, gpu_id=0):
    #     time.sleep(10)
    
    # High severity - slower recovery
    # with gpu_helper.xid_48(node_name=gpu_node, gpu_id=0):
    #     time.sleep(30)
    
    # with gpu_helper.xid_74(node_name=gpu_node, gpu_id=0):
    #     time.sleep(20)
    
    # Critical - triggers pod rescheduling (very slow)
    # with gpu_helper.xid_79(node_name=gpu_node, gpu_id=0):
    #     time.sleep(60)
    #     # Pods may reschedule to other nodes
    
    # with gpu_helper.xid_95(node_name=gpu_node, gpu_id=0):
    #     time.sleep(60)


# ==============================================================================
# Example 4: Multi-GPU Testing
# ==============================================================================

@pytest.mark.fault_tolerance
def test_multiple_gpus(gpu_helper, gpu_node):
    """Test fault injection on multiple GPUs"""
    
    num_gpus = 4  # Adjust for your hardware
    
    for gpu_id in range(num_gpus):
        # Inject on each GPU sequentially
        with gpu_helper.xid_94(node_name=gpu_node, gpu_id=gpu_id):
            time.sleep(5)
            send_completion_request(f"gpu-{gpu_id}", 10)
        
        time.sleep(2)  # Brief pause between GPUs


# ==============================================================================
# Example 5: Measuring Impact
# ==============================================================================

@pytest.mark.fault_tolerance
def test_measure_latency_impact(gpu_helper, gpu_node):
    """Measure performance impact during fault"""
    
    # 1. Measure baseline latency
    start = time.time()
    send_completion_request("baseline", 10)
    baseline_latency = time.time() - start
    
    # 2. Measure during fault
    with gpu_helper.xid_94(node_name=gpu_node, gpu_id=0):
        time.sleep(10)
        
        start = time.time()
        send_completion_request("degraded", 10, timeout=45)
        fault_latency = time.time() - start
        
        # Compare latencies
        assert fault_latency < baseline_latency * 3  # Not too slow
    
    # 3. Measure recovery
    time.sleep(5)
    start = time.time()
    send_completion_request("recovery", 10)
    recovery_latency = time.time() - start


# ==============================================================================
# Example 6: Fault Tolerance Assertions
# ==============================================================================

@pytest.mark.fault_tolerance
def test_fault_tolerance_threshold(gpu_helper, gpu_node):
    """Assert minimum fault tolerance level"""
    
    with gpu_helper.xid_94(node_name=gpu_node, gpu_id=0):
        time.sleep(10)
        
        # Test multiple requests
        success_count = 0
        for i in range(10):
            try:
                send_completion_request(f"test-{i}", 10)
                success_count += 1
            except:
                pass
        
        # Assert fault tolerance (e.g., ≥80% success)
        assert success_count >= 8, f"Only {success_count}/10 succeeded"


# ==============================================================================
# Example 7: Critical Errors (XID 79) - Pod Rescheduling
# ==============================================================================

@pytest.mark.fault_tolerance
@pytest.mark.slow
def test_xid_79_critical_failure(gpu_helper, gpu_node):
    """Critical errors trigger pod rescheduling"""
    
    # XID 79 causes pods to crash and reschedule
    with gpu_helper.xid_79(node_name=gpu_node, gpu_id=0):
        # Pods on this node will crash
        time.sleep(60)
        
        # System should still respond (other nodes handle requests)
        for i in range(5):
            try:
                send_completion_request(f"test-{i}", 10)
                # Other nodes handling
            except:
                # Expected during rescheduling
                pass
    
    # Wait for pod rescheduling (3-5 minutes)
    time.sleep(180)
    
    # Validate full recovery
    send_completion_request("recovery", 10)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

