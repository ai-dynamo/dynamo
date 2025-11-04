#!/usr/bin/env python3
"""
Network Fault Tolerance Test Example - Framework Usage Guide

Copy-paste template showing how to use the network fault injection framework.
"""
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "client"))

from fault_injection_client import FaultInjectionClient, NetworkMode, NetworkPartition
from test_helpers import get_config_from_env, send_completion_request


# ==============================================================================
# Setup
# ==============================================================================

@pytest.fixture(scope="module")
def client():
    """Initialize fault injection client"""
    config = get_config_from_env()
    client = FaultInjectionClient(api_url=config['api_url'])
    yield client
    client.__exit__(None, None, None)


@pytest.fixture(scope="module")
def config():
    """Get test configuration"""
    return get_config_from_env()


# ==============================================================================
# Example 1: Block NATS (NetworkPolicy - Context Manager)
# ==============================================================================

@pytest.mark.fault_tolerance
def test_worker_nats_partition(client, config):
    """Block one worker's access to NATS"""
    
    # 1. Baseline
    send_completion_request("baseline", 10)
    
    # 2. Inject partition with context manager
    with client.network_partition(
        partition_type=NetworkPartition.CUSTOM,
        mode=NetworkMode.NETWORKPOLICY,
        namespace=config['app_namespace'],
        target_pod_prefix="vllm-agg-0-vllmdecodeworker",
        block_nats=True,
        duration=60  # Auto-expire
    ):
        # Partition active
        time.sleep(5)  # Wait for NetworkPolicy to apply
        
        # 3. Test fault tolerance (other workers handle requests)
        for i in range(10):
            try:
                send_completion_request(f"test-{i}", 10)
                # Should succeed
            except:
                # Some may fail
                pass
    
    # 4. Auto-recovered, validate
    time.sleep(5)
    send_completion_request("recovery", 10)


# ==============================================================================
# Example 2: Block Specific Pods (Pod-to-Pod Isolation)
# ==============================================================================

@pytest.mark.fault_tolerance
def test_worker_to_frontend_blocking(client, config):
    """Block worker from reaching frontend using label selectors"""
    
    # 1. Baseline
    send_completion_request("baseline", 10)
    
    # 2. Inject partition (manual for demo)
    fault = client.inject_network_partition(
        partition_type=NetworkPartition.CUSTOM,
        mode=NetworkMode.NETWORKPOLICY,
        namespace=config['app_namespace'],
        target_pod_prefix="vllm-agg-0-vllmdecodeworker",
        block_specific_pods=[
            {"app.kubernetes.io/name": "vllm-agg-0-frontend"}
        ],
        block_nats=False,  # Keep NATS working
        duration=60
    )
    
    try:
        # 3. Test during partition
        time.sleep(5)
        
        for i in range(5):
            try:
                send_completion_request(f"test-{i}", 10)
            except:
                pass
        
    finally:
        # 4. Cleanup
        client.recover_fault(fault.fault_id)
    
    # 5. Validate recovery
    time.sleep(5)
    send_completion_request("recovery", 10)


# ==============================================================================
# Example 3: ChaosMesh - 50% Packet Loss
# ==============================================================================

@pytest.mark.fault_tolerance
@pytest.mark.requires_chaosmesh
def test_packet_loss(client, config):
    """Test with 50% packet loss (requires ChaosMesh)"""
    
    # 1. Measure baseline latency
    start = time.time()
    send_completion_request("baseline", 10)
    baseline_latency = time.time() - start
    
    # 2. Inject 50% packet loss
    with client.network_partition(
        partition_type=NetworkPartition.CUSTOM,
        mode=NetworkMode.CHAOS_MESH,
        namespace=config['app_namespace'],
        target_pod_prefix="vllm-agg-0-vllmdecodeworker",
        packet_loss_percent=50,
        target_nats=True,
        duration=60
    ):
        time.sleep(10)  # Wait for ChaosMesh
        
        # 3. Test degraded performance
        start = time.time()
        send_completion_request("degraded", 10, timeout=60)
        degraded_latency = time.time() - start
        
        # Should be slower
        assert degraded_latency > baseline_latency
    
    # 4. Validate recovery
    time.sleep(10)
    send_completion_request("recovery", 10)


# ==============================================================================
# Example 4: ChaosMesh - Network Delay
# ==============================================================================

@pytest.mark.fault_tolerance
@pytest.mark.requires_chaosmesh
def test_network_delay(client, config):
    """Add network delay + jitter"""
    
    # Baseline
    start = time.time()
    send_completion_request("baseline", 10)
    baseline = time.time() - start
    
    # Add 100ms delay + 50ms jitter
    with client.network_partition(
        partition_type=NetworkPartition.CUSTOM,
        mode=NetworkMode.CHAOS_MESH,
        namespace=config['app_namespace'],
        target_pod_prefix="vllm-agg-0-vllmdecodeworker",
        delay_ms=100,
        delay_jitter_ms=50,
        target_nats=True,
        duration=60
    ):
        time.sleep(10)
        
        start = time.time()
        send_completion_request("delayed", 10, timeout=60)
        delayed = time.time() - start
        
        assert delayed > baseline  # Noticeably slower
    
    time.sleep(10)
    send_completion_request("recovery", 10)


# ==============================================================================
# Example 5: ChaosMesh - Bandwidth Limiting
# ==============================================================================

@pytest.mark.fault_tolerance
@pytest.mark.requires_chaosmesh
def test_bandwidth_limit(client, config):
    """Limit bandwidth to 1 Mbps"""
    
    with client.network_partition(
        partition_type=NetworkPartition.CUSTOM,
        mode=NetworkMode.CHAOS_MESH,
        namespace=config['app_namespace'],
        target_pod_prefix="vllm-agg-0-vllmdecodeworker",
        bandwidth_limit="1mbps",
        target_nats=True,
        duration=60
    ):
        time.sleep(10)
        send_completion_request("limited", 10, timeout=60)
    
    time.sleep(10)
    send_completion_request("recovery", 10)


# ==============================================================================
# Example 6: Combined Chaos
# ==============================================================================

@pytest.mark.fault_tolerance
@pytest.mark.requires_chaosmesh
def test_combined_chaos(client, config):
    """Combine packet loss + delay + bandwidth limit"""
    
    with client.network_partition(
        partition_type=NetworkPartition.CUSTOM,
        mode=NetworkMode.CHAOS_MESH,
        namespace=config['app_namespace'],
        target_pod_prefix="vllm-agg-0-vllmdecodeworker",
        packet_loss_percent=20,
        delay_ms=50,
        delay_jitter_ms=25,
        bandwidth_limit="10mbps",
        target_nats=True,
        duration=60
    ):
        time.sleep(10)
        
        # Very degraded network
        for i in range(5):
            try:
                send_completion_request(f"test-{i}", 10, timeout=90)
            except:
                pass


# ==============================================================================
# Example 7: Measure Recovery Time
# ==============================================================================

@pytest.mark.fault_tolerance
def test_recovery_time(client, config):
    """Measure time to recover after partition ends"""
    
    # Inject partition
    fault = client.inject_network_partition(
        partition_type=NetworkPartition.CUSTOM,
        mode=NetworkMode.NETWORKPOLICY,
        namespace=config['app_namespace'],
        target_pod_prefix="vllm-agg-0-vllmdecodeworker",
        block_nats=True
    )
    time.sleep(10)
    
    # Remove and measure recovery
    client.recover_fault(fault.fault_id)
    recovery_start = time.time()
    
    # Poll for recovery
    for attempt in range(30):
        try:
            send_completion_request("recovery", 10, timeout=10)
            recovery_time = time.time() - recovery_start
            break
        except:
            time.sleep(1)
    
    # Assert fast recovery
    assert recovery_time < 15


# ==============================================================================
# Example 8: Fault Tolerance Threshold
# ==============================================================================

@pytest.mark.fault_tolerance
def test_fault_tolerance_threshold(client, config):
    """Assert minimum success rate during fault"""
    
    with client.network_partition(
        partition_type=NetworkPartition.CUSTOM,
        mode=NetworkMode.NETWORKPOLICY,
        namespace=config['app_namespace'],
        target_pod_prefix="vllm-agg-0-vllmdecodeworker",
        block_nats=True,
        duration=60
    ):
        time.sleep(5)
        
        # Test success rate
        success_count = 0
        for i in range(10):
            try:
                send_completion_request(f"test-{i}", 10)
                success_count += 1
            except:
                pass
        
        # Assert ≥80% success
        assert success_count >= 8, f"Only {success_count}/10"


# ==============================================================================
# Example 9: All NetworkPolicy Patterns
# ==============================================================================

@pytest.mark.fault_tolerance
def test_networkpolicy_patterns(client, config):
    """Show all NetworkPolicy blocking patterns"""
    
    # Pattern 1: Block NATS only
    with client.network_partition(
        partition_type=NetworkPartition.CUSTOM,
        mode=NetworkMode.NETWORKPOLICY,
        namespace=config['app_namespace'],
        target_pod_prefix="vllm-agg-0-vllmdecodeworker",
        block_nats=True,
        duration=30
    ):
        time.sleep(5)
        send_completion_request("test", 10)
    
    time.sleep(5)
    
    # Pattern 2: Block specific pods only
    # with client.network_partition(
    #     partition_type=NetworkPartition.CUSTOM,
    #     mode=NetworkMode.NETWORKPOLICY,
    #     namespace=config['app_namespace'],
    #     target_pod_prefix="vllm-agg-0-vllmdecodeworker",
    #     block_specific_pods=[{"app.kubernetes.io/name": "frontend"}],
    #     block_nats=False,
    #     duration=30
    # ):
    #     time.sleep(5)
    
    # Pattern 3: Block both NATS and specific pods
    # with client.network_partition(
    #     partition_type=NetworkPartition.CUSTOM,
    #     mode=NetworkMode.NETWORKPOLICY,
    #     namespace=config['app_namespace'],
    #     target_pod_prefix="vllm-agg-0-vllmdecodeworker",
    #     block_nats=True,
    #     block_specific_pods=[{"app.kubernetes.io/name": "frontend"}],
    #     duration=30
    # ):
    #     time.sleep(5)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
