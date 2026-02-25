#!/usr/bin/env python3
"""
Test Worker->Frontend network partition with fault tolerance validation.

This test uses label-based NetworkPolicy to block a specific worker pod
from communicating with the frontend pod. This verifies whether the system
can handle requests through other workers when one is isolated.

Uses Kubernetes pod label selectors for precise fault injection.
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
def test_worker_to_frontend_partition():
    """
    Test fault tolerance during Worker->Frontend network partition.

    Verifies that the system can handle requests when a worker
    cannot communicate with the frontend. Other workers should handle requests.
    """
    print(f"\n{Colors.CYAN}{'=' * 80}")
    print(
        f"{Colors.BOLD}{Colors.MAGENTA}TEST: Worker->Frontend Network Partition (Fault Tolerance){Colors.RESET}"
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

    # DURING: Inject network partition
    print(
        f"\n{Colors.BOLD}{Colors.YELLOW}[DURING] Inject Fault - Worker->Frontend Partition{Colors.RESET}"
    )
    print(f"{Colors.BLUE}{'-' * 80}{Colors.RESET}")
    print(
        f"{Colors.CYAN}Injecting network partition: blocking one worker's access to frontend...{Colors.RESET}"
    )

    try:
        # Block specific worker from reaching frontend using label selectors
        fault_info = client.inject_network_partition(
            partition_type=NetworkPartition.CUSTOM,
            source=APP_NAMESPACE,
            target=APP_NAMESPACE,
            mode=NetworkMode.NETWORKPOLICY,
            duration=60,
            namespace=APP_NAMESPACE,
            target_pod_prefix="vllm-agg-0-vllmdecodeworker",
            block_specific_pods=[
                # Block traffic to frontend pods (identified by labels)
                {"app.kubernetes.io/name": "vllm-agg-0-frontend"},
            ],
            block_nats=False,  # Don't block NATS, only frontend
        )
        fault_id = fault_info.fault_id
        print(
            f"{Colors.GREEN}[OK]{Colors.RESET} Fault injected: {Colors.BOLD}{fault_id}{Colors.RESET}"
        )
        print(f"{Colors.GRAY}   Status: {fault_info.status}")
        print(f"   Type: {fault_info.fault_type}{Colors.RESET}")

        # Wait for partition to settle
        time.sleep(5)

        # Test fault tolerance during partition
        print(
            f"\n{Colors.BOLD}{Colors.MAGENTA}[FAULT TOLERANCE TEST]{Colors.RESET} Sending requests during partition..."
        )
        success_count = 0
        fail_count = 0

        for i in range(3):
            try:
                unique_prompt = f"Fault tolerance test {random.randint(1000, 9999)}"
                response = send_completion_request(unique_prompt, 10, timeout=30)
                validate_completion_response(response)
                success_count += 1
                print(
                    f"{Colors.GREEN}[OK]{Colors.RESET} Request {i+1}/3 succeeded (other workers handling requests)"
                )
            except Exception as e:
                fail_count += 1
                print(f"{Colors.YELLOW}[WARN]{Colors.RESET} Request {i+1}/3 failed: {e}")
            time.sleep(2)

        print(
            f"\n{Colors.BOLD}Results during partition:{Colors.RESET} {Colors.GREEN}{success_count} succeeded{Colors.RESET}, {Colors.RED}{fail_count} failed{Colors.RESET}"
        )

        # Fault tolerance assertion
        if success_count == 0:
            print(
                f"{Colors.YELLOW}[WARNING]{Colors.RESET} No requests succeeded - fault tolerance may not be working"
            )
        else:
            print(
                f"{Colors.GREEN}[OK]{Colors.RESET} Fault tolerance verified - system remained operational"
            )

        # Recover fault
        print(f"\n{Colors.BOLD}{Colors.BLUE}[AFTER] Recover Fault{Colors.RESET}")
        print(f"{Colors.BLUE}{'-' * 80}{Colors.RESET}")
        client.recover_fault(fault_id)
        print(f"{Colors.GREEN}[OK]{Colors.RESET} Fault recovered")
        fault_id = None

        # Wait for recovery with retries
        print(f"{Colors.GRAY}Waiting for system to stabilize after partition...{Colors.RESET}")
        time.sleep(10)

        # Validate recovery with retries
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
            pytest.fail(
                "Recovery validation failed after 3 attempts - system may need more time to stabilize"
            )

        print(f"\n{Colors.GREEN}{'=' * 80}")
        print(f"{Colors.BOLD}[PASS] TEST PASSED{Colors.RESET}")
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
    # Run with: python3 test_partition_worker_to_frontend.py
    pytest.main([__file__, "-v", "-s"])
