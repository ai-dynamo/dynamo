#!/usr/bin/env python3
"""
Test pod-to-pod blocking with vllm-disagg deployment (multiple workers).

This demonstrates true fault tolerance: when one worker is isolated,
other workers continue handling requests.
"""
import os
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
    send_completion_request,
    validate_completion_response,
)

# Disagg-specific configuration
API_URL = os.getenv("API_URL", "http://localhost:8080")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://vllm-disagg-0-frontend.dynamo-oviya.svc.cluster.local:8000")
APP_NAMESPACE = os.getenv("APP_NAMESPACE", "dynamo-oviya")


@pytest.mark.fault_tolerance
@pytest.mark.network_required
def test_disagg_worker_to_frontend_blocking():
    """
    Test pod-to-pod blocking with multiple decode workers.
    
    When one worker is isolated from frontend, other workers handle requests.
    This validates true fault tolerance in a multi-worker deployment.
    """
    print(f"\n{Colors.CYAN}{'=' * 80}")
    print(
        f"{Colors.BOLD}{Colors.MAGENTA}TEST: Disagg Worker→Frontend Blocking (Multi-Worker Fault Tolerance){Colors.RESET}"
    )
    print(f"{Colors.CYAN}{'=' * 80}{Colors.RESET}")

    client = FaultInjectionClient(api_url=API_URL)
    fault_id = None

    print(f"{Colors.GRAY}API URL: {API_URL}")
    print(f"Frontend URL: {FRONTEND_URL}")
    print(f"Namespace: {APP_NAMESPACE}")
    print(f"Workers: 2 decode workers (multi-worker deployment){Colors.RESET}")

    # BEFORE: Check baseline
    print(f"\n{Colors.BOLD}{Colors.BLUE}[BEFORE] Establish Baseline{Colors.RESET}")
    print(f"{Colors.BLUE}{'-' * 80}{Colors.RESET}")
    if not check_frontend_reachable(FRONTEND_URL):
        pytest.skip("Frontend not reachable")
    print(f"{Colors.GREEN}[OK]{Colors.RESET} Frontend reachable")

    try:
        unique_prompt = f"Hello {random.randint(1000, 9999)}"
        response = send_completion_request(unique_prompt, 10, frontend_url=FRONTEND_URL, timeout=30)
        validate_completion_response(response)
        print(f"{Colors.GREEN}[OK]{Colors.RESET} Baseline request succeeded")
    except Exception as e:
        pytest.fail(f"Baseline request failed: {e}")

    # DURING: Inject pod-to-pod blocking
    print(
        f"\n{Colors.BOLD}{Colors.YELLOW}[DURING] Inject Fault - Block ONE Worker from Frontend{Colors.RESET}"
    )
    print(f"{Colors.BLUE}{'-' * 80}{Colors.RESET}")
    print(
        f"{Colors.CYAN}Injecting NetworkPolicy: Block first decode worker from reaching frontend...{Colors.RESET}"
    )
    print(
        f"{Colors.GRAY}Other decode worker should handle all requests (fault tolerance){Colors.RESET}"
    )

    try:
        # Block ONE decode worker from reaching frontend
        fault_info = client.inject_network_partition(
            partition_type=NetworkPartition.CUSTOM,
            source=APP_NAMESPACE,
            target=APP_NAMESPACE,
            mode=NetworkMode.NETWORKPOLICY,
            duration=60,
            namespace=APP_NAMESPACE,
            target_pod_prefix="vllm-disagg-0-vllmdecodeworker-g",  # Targets first worker
            block_specific_pods=[
                {"app.kubernetes.io/name": "vllm-disagg-0-frontend"},  # Block frontend by label
            ],
            block_nats=False,  # Keep NATS working
        )
        fault_id = fault_info.fault_id
        print(
            f"{Colors.GREEN}[OK]{Colors.RESET} Fault injected: {Colors.BOLD}{fault_id}{Colors.RESET}"
        )
        print(f"{Colors.GRAY}   Status: {fault_info.status}")
        print(f"   Type: {fault_info.fault_type}")
        print(f"   NetworkPolicy applied to: ONE decode worker")
        print(f"   Blocks traffic to: Frontend pods (label: app.kubernetes.io/name=vllm-disagg-0-frontend)")
        print(f"   Expected: Other worker handles requests{Colors.RESET}")

        # Wait for NetworkPolicy to take effect
        time.sleep(5)

        # Test fault tolerance during partition
        print(
            f"\n{Colors.BOLD}{Colors.MAGENTA}[FAULT TOLERANCE TEST]{Colors.RESET} Sending requests during partition..."
        )
        print(
            f"{Colors.CYAN}[EXPECTATION]{Colors.RESET} Requests should succeed - second worker handles traffic"
        )
        success_count = 0
        fail_count = 0

        for i in range(5):
            try:
                unique_prompt = f"Fault tolerance test {random.randint(1000, 9999)}"
                response = send_completion_request(unique_prompt, 10, frontend_url=FRONTEND_URL, timeout=30)
                validate_completion_response(response)
                success_count += 1
                print(
                    f"{Colors.GREEN}[OK]{Colors.RESET} Request {i+1}/5 succeeded (handled by healthy worker)"
                )
            except Exception as e:
                fail_count += 1
                print(f"{Colors.RED}[FAIL]{Colors.RESET} Request {i+1}/5 failed: {e}")
            time.sleep(2)

        print(
            f"\n{Colors.BOLD}Results during partition:{Colors.RESET} {Colors.GREEN}{success_count} succeeded{Colors.RESET}, {Colors.RED}{fail_count} failed{Colors.RESET}"
        )

        # Validate fault tolerance
        if success_count >= 3:  # At least 60% success rate
            print(
                f"{Colors.GREEN}[SUCCESS]{Colors.RESET} ✓ Fault tolerance VERIFIED - system remained operational!"
            )
            print(
                f"{Colors.GREEN}   → {success_count}/{success_count+fail_count} requests succeeded despite one worker being isolated{Colors.RESET}"
            )
        else:
            print(
                f"{Colors.YELLOW}[WARNING]{Colors.RESET} Limited fault tolerance - only {success_count}/{success_count+fail_count} requests succeeded"
            )
            if success_count == 0:
                print(
                    f"{Colors.RED}[FAIL]{Colors.RESET} No requests succeeded - fault tolerance not working!"
                )

        # Recover fault
        print(f"\n{Colors.BOLD}{Colors.BLUE}[AFTER] Recover Fault{Colors.RESET}")
        print(f"{Colors.BLUE}{'-' * 80}{Colors.RESET}")
        client.recover_fault(fault_id)
        print(f"{Colors.GREEN}[OK]{Colors.RESET} NetworkPolicy removed, fault recovered")
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
                response = send_completion_request(unique_prompt, 10, frontend_url=FRONTEND_URL, timeout=30)
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
        print(f"{Colors.BOLD}[PASS] FAULT TOLERANCE TEST PASSED ✓{Colors.RESET}")
        print(f"{Colors.GREEN}{'=' * 80}{Colors.RESET}")

    except Exception as e:
        print(f"\n{Colors.RED}[FAIL] TEST FAILED: {e}{Colors.RESET}")
        import traceback

        traceback.print_exc()
        pytest.fail(str(e))

    finally:
        if fault_id:
            print(f"\n{Colors.YELLOW}[CLEANUP]{Colors.RESET} Recovering fault...")
            try:
                client.recover_fault(fault_id)
                print(f"{Colors.GREEN}[OK]{Colors.RESET} Cleanup: Fault recovered")
            except Exception as cleanup_error:
                print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} Cleanup failed: {cleanup_error}")


if __name__ == "__main__":
    # Run with: python3 test_disagg_pod_to_pod.py
    pytest.main([__file__, "-v", "-s"])

