#!/usr/bin/env python3
"""
Test NATS traffic with 50% packet loss using ChaosMesh.

This demonstrates using ChaosMesh to inject partial packet loss instead of
completely blocking traffic. This is useful for testing system behavior under
degraded network conditions rather than complete partitions.

Prerequisites:
- ChaosMesh must be installed in the cluster (already done on AKS dynamo-dev cluster)
- Install with: helm install chaos-mesh chaos-mesh/chaos-mesh --namespace=chaos-mesh --create-namespace
"""
import random
import sys
import time
from pathlib import Path

import pytest

# Add client directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "client"))

from fault_injection_client import FaultInjectionClient, NetworkMode, NetworkPartition
from test_helpers import (
    Colors,
    check_frontend_reachable,
    get_config_from_env,
    send_completion_request,
    validate_completion_response,
)

# Get configuration from environment
config = get_config_from_env()
API_URL = config["api_url"]
FRONTEND_URL = config["frontend_url"]
APP_NAMESPACE = config["app_namespace"]


@pytest.mark.fault_tolerance
@pytest.mark.network_required
@pytest.mark.chaos_mesh
def test_nats_packet_loss_50_percent():
    """
    Test worker behavior with 50% packet loss to NATS.
    
    This injects 50% packet loss on NATS traffic for a worker pod using ChaosMesh.
    Unlike complete partition tests, this tests system behavior under degraded
    but not completely broken network conditions.
    """
    print(f"\n{Colors.CYAN}{'=' * 80}")
    print(
        f"{Colors.BOLD}{Colors.MAGENTA}TEST: NATS Traffic with 50% Packet Loss (ChaosMesh){Colors.RESET}"
    )
    print(f"{Colors.CYAN}{'=' * 80}{Colors.RESET}")

    client = FaultInjectionClient(api_url=API_URL)
    fault_id = None

    print(f"{Colors.GRAY}API URL: {API_URL}")
    print(f"Frontend URL: {FRONTEND_URL}")
    print(f"Namespace: {APP_NAMESPACE}{Colors.RESET}")

    # BEFORE: Check baseline
    print(f"\n{Colors.BOLD}{Colors.BLUE}[BEFORE] Establish Baseline{Colors.RESET}")
    print(f"{Colors.BLUE}{'-' * 80}{Colors.RESET}")
    if not check_frontend_reachable():
        pytest.skip("Frontend not reachable")
    print(f"{Colors.GREEN}[OK]{Colors.RESET} Frontend reachable")

    try:
        unique_prompt = f"Hello {random.randint(1000, 9999)}"
        response = send_completion_request(unique_prompt, 10, timeout=30)
        validate_completion_response(response)
        print(f"{Colors.GREEN}[OK]{Colors.RESET} Baseline request succeeded")
    except Exception as e:
        pytest.fail(f"Baseline request failed: {e}")

    # DURING: Inject 50% packet loss to NATS
    print(
        f"\n{Colors.BOLD}{Colors.YELLOW}[DURING] Inject Fault - 50% Packet Loss to NATS{Colors.RESET}"
    )
    print(f"{Colors.BLUE}{'-' * 80}{Colors.RESET}")
    print(
        f"{Colors.CYAN}Injecting ChaosMesh NetworkChaos: 50% packet loss on NATS traffic...{Colors.RESET}"
    )
    print(
        f"{Colors.GRAY}Using ChaosMesh to simulate degraded network conditions{Colors.RESET}"
    )

    try:
        # Inject 50% packet loss to NATS traffic for a worker
        fault_info = client.inject_network_partition(
            partition_type=NetworkPartition.CUSTOM,
            source=APP_NAMESPACE,
            target=APP_NAMESPACE,
            mode=NetworkMode.CHAOS_MESH,  # Use ChaosMesh instead of NetworkPolicy
            duration=60,
            namespace=APP_NAMESPACE,
            target_pod_prefix="vllm-agg-0-vllmdecodeworker",  # Target worker pod
            packet_loss_percent=50,  # 50% packet loss
            target_nats=True,  # Target NATS traffic
        )
        fault_id = fault_info.fault_id
        print(
            f"{Colors.GREEN}[OK]{Colors.RESET} Fault injected: {Colors.BOLD}{fault_id}{Colors.RESET}"
        )
        print(f"{Colors.GRAY}   Status: {fault_info.status}")
        print(f"   Type: {fault_info.fault_type}")
        print(f"   Message: {fault_info.message}")
        print(f"   NetworkChaos applied to worker pod with 50% packet loss to NATS")
        print(f"   Expected behavior: Degraded but not completely broken communication{Colors.RESET}")

        # Wait for ChaosMesh to take effect
        time.sleep(5)

        # Test during fault
        print(
            f"\n{Colors.BOLD}{Colors.MAGENTA}[FAULT TOLERANCE TEST]{Colors.RESET} Sending requests during packet loss..."
        )
        print(
            f"{Colors.CYAN}[NOTE]{Colors.RESET} 50% packet loss simulates degraded network. System should handle this gracefully."
        )
        
        success_count = 0
        fail_count = 0
        slow_count = 0

        for i in range(5):
            try:
                start_time = time.time()
                unique_prompt = f"Fault tolerance test {random.randint(1000, 9999)}"
                response = send_completion_request(unique_prompt, 10, timeout=30)
                validate_completion_response(response)
                elapsed = time.time() - start_time
                
                success_count += 1
                if elapsed > 5:
                    slow_count += 1
                    print(
                        f"{Colors.YELLOW}[SLOW]{Colors.RESET} Request {i+1}/5 succeeded but took {elapsed:.1f}s (degraded)"
                    )
                else:
                    print(
                        f"{Colors.GREEN}[OK]{Colors.RESET} Request {i+1}/5 succeeded in {elapsed:.1f}s"
                    )
            except Exception as e:
                fail_count += 1
                print(f"{Colors.YELLOW}[WARN]{Colors.RESET} Request {i+1}/5 failed: {e}")
            time.sleep(2)

        print(
            f"\n{Colors.BOLD}Results during 50% packet loss:{Colors.RESET}"
        )
        print(f"  {Colors.GREEN}✓ Succeeded: {success_count}/5{Colors.RESET}")
        print(f"  {Colors.YELLOW}⚠ Slow (>5s): {slow_count}/5{Colors.RESET}")
        print(f"  {Colors.RED}✗ Failed: {fail_count}/5{Colors.RESET}")

        # Validate fault tolerance
        if success_count >= 3:
            print(
                f"{Colors.GREEN}[EXCELLENT]{Colors.RESET} System handled degraded network well ({success_count}/5 success rate)"
            )
        elif success_count >= 1:
            print(
                f"{Colors.YELLOW}[ACCEPTABLE]{Colors.RESET} System partially functional under degraded network ({success_count}/5 success rate)"
            )
        else:
            print(
                f"{Colors.RED}[POOR]{Colors.RESET} System failed to handle 50% packet loss (0/5 success rate)"
            )
            print(
                f"{Colors.CYAN}[NOTE]{Colors.RESET} Consider implementing retry logic or connection pooling"
            )

        # Recover fault
        print(f"\n{Colors.BOLD}{Colors.BLUE}[AFTER] Recover Fault{Colors.RESET}")
        print(f"{Colors.BLUE}{'-' * 80}{Colors.RESET}")
        client.recover_fault(fault_id)
        print(f"{Colors.GREEN}[OK]{Colors.RESET} NetworkChaos removed, fault recovered")
        fault_id = None

        # Wait for recovery
        print(f"{Colors.GRAY}Waiting for system to stabilize...{Colors.RESET}")
        time.sleep(10)

        # Validate recovery
        recovery_success = False
        for attempt in range(3):
            try:
                print(f"{Colors.GRAY}Recovery validation attempt {attempt+1}/3...{Colors.RESET}")
                unique_prompt = f"Hello after recovery {random.randint(1000, 9999)}"
                response = send_completion_request(unique_prompt, 10, timeout=30)
                validate_completion_response(response)
                print(f"{Colors.GREEN}[OK]{Colors.RESET} Recovery request succeeded")
                recovery_success = True
                break
            except Exception as e:
                if attempt < 2:
                    print(
                        f"{Colors.YELLOW}[WARN]{Colors.RESET} Attempt {attempt+1} failed, retrying in 5s: {e}"
                    )
                    time.sleep(5)
                else:
                    print(f"{Colors.RED}[ERROR]{Colors.RESET} All recovery attempts failed: {e}")

        if not recovery_success:
            pytest.fail("Recovery validation failed after 3 attempts")

        print(f"\n{Colors.GREEN}{'=' * 80}")
        print(f"{Colors.BOLD}[PASS] TEST PASSED{Colors.RESET}")
        print(f"{Colors.GREEN}{'=' * 80}{Colors.RESET}")

    except Exception as e:
        print(f"\n{Colors.RED}[FAIL] TEST FAILED: {e}{Colors.RESET}")
        import traceback

        traceback.print_exc()
        
        # Check if ChaosMesh is installed
        if "chaos-mesh" in str(e).lower() or "networkchaos" in str(e).lower():
            print(f"\n{Colors.YELLOW}[INFO]{Colors.RESET} ChaosMesh might not be installed.")
            print(f"{Colors.CYAN}Install ChaosMesh with:{Colors.RESET}")
            print(f"  kubectl create ns chaos-mesh")
            print(f"  helm repo add chaos-mesh https://charts.chaos-mesh.org")
            print(f"  helm install chaos-mesh chaos-mesh/chaos-mesh -n chaos-mesh")
        
        pytest.fail(str(e))

    finally:
        if fault_id:
            print(f"\n{Colors.YELLOW}[CLEANUP]{Colors.RESET} Recovering fault...")
            try:
                client.recover_fault(fault_id)
                print(f"{Colors.GREEN}[OK]{Colors.RESET} Cleanup: Fault recovered")
            except Exception as cleanup_error:
                print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} Cleanup failed: {cleanup_error}")


@pytest.mark.fault_tolerance
@pytest.mark.network_required
@pytest.mark.chaos_mesh
def test_nats_variable_delay():
    """
    Test worker behavior with variable delay to NATS traffic.
    
    This adds 100ms ± 50ms delay to NATS traffic to simulate high latency.
    """
    print(f"\n{Colors.CYAN}{'=' * 80}")
    print(
        f"{Colors.BOLD}{Colors.MAGENTA}TEST: NATS Traffic with Variable Delay (ChaosMesh){Colors.RESET}"
    )
    print(f"{Colors.CYAN}{'=' * 80}{Colors.RESET}")

    client = FaultInjectionClient(api_url=API_URL)
    fault_id = None

    try:
        # Inject variable delay to NATS traffic
        fault_info = client.inject_network_partition(
            partition_type=NetworkPartition.CUSTOM,
            source=APP_NAMESPACE,
            target=APP_NAMESPACE,
            mode=NetworkMode.CHAOS_MESH,
            duration=60,
            namespace=APP_NAMESPACE,
            target_pod_prefix="vllm-agg-0-vllmdecodeworker",
            delay_ms=100,  # Add 100ms delay
            delay_jitter_ms=50,  # ± 50ms jitter
            target_nats=True,
        )
        fault_id = fault_info.fault_id
        print(
            f"{Colors.GREEN}[OK]{Colors.RESET} Fault injected: {Colors.BOLD}{fault_id}{Colors.RESET}"
        )
        print(f"{Colors.GRAY}   {fault_info.message}{Colors.RESET}")

        time.sleep(5)

        # Test during fault
        print(f"\n{Colors.BOLD}Testing with {fault_info.message}...{Colors.RESET}")
        
        latencies = []
        for i in range(3):
            try:
                start_time = time.time()
                unique_prompt = f"Latency test {random.randint(1000, 9999)}"
                response = send_completion_request(unique_prompt, 10, timeout=30)
                validate_completion_response(response)
                elapsed = time.time() - start_time
                latencies.append(elapsed)
                print(f"{Colors.GREEN}[OK]{Colors.RESET} Request {i+1}/3 completed in {elapsed:.2f}s")
            except Exception as e:
                print(f"{Colors.YELLOW}[WARN]{Colors.RESET} Request {i+1}/3 failed: {e}")
            time.sleep(2)

        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            print(f"\n{Colors.CYAN}Average latency: {avg_latency:.2f}s{Colors.RESET}")

        # Recover
        client.recover_fault(fault_id)
        fault_id = None
        print(f"\n{Colors.GREEN}[OK]{Colors.RESET} Fault recovered")

    except Exception as e:
        print(f"\n{Colors.RED}[FAIL]{Colors.RESET} {e}")
        pytest.fail(str(e))
    finally:
        if fault_id:
            try:
                client.recover_fault(fault_id)
            except:
                pass


if __name__ == "__main__":
    # Run with: python3 test_nats_packet_loss_50_percent.py
    pytest.main([__file__, "-v", "-s"])

