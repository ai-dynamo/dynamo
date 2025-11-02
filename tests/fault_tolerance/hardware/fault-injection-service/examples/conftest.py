"""Pytest configuration for network partition tests."""
import pytest
import sys
from pathlib import Path

# Add client directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "client"))

from fault_injection_client import FaultInjectionClient
from test_helpers import get_config_from_env


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--verbose-tests",
        action="store_true",
        default=False,
        help="Enable verbose output for partition tests (shows detailed diagnostics)",
    )


@pytest.fixture(scope="session")
def verbose_mode(request):
    """Fixture to get verbose mode flag."""
    return request.config.getoption("--verbose-tests")


@pytest.fixture(scope="session")
def fault_injection_client():
    """Create a session-scoped fault injection client."""
    config = get_config_from_env()
    return FaultInjectionClient(api_url=config["api_url"])


@pytest.fixture
def cleanup_faults(fault_injection_client, verbose_mode):
    """
    Fixture to ensure network partitions are cleaned up after each test.
    
    Usage:
        def test_something(cleanup_faults):
            fault_id = cleanup_faults.track(client.inject_network_partition(...))
            # Test code here
            # Cleanup happens automatically even if test fails or is interrupted
    """
    fault_ids = []
    
    class FaultTracker:
        def track(self, fault_info):
            """Track a fault for cleanup. Returns the fault_info for convenience."""
            if fault_info and hasattr(fault_info, 'fault_id'):
                fault_ids.append(fault_info.fault_id)
            return fault_info
        
        def track_id(self, fault_id):
            """Track a fault ID directly."""
            if fault_id:
                fault_ids.append(fault_id)
            return fault_id
    
    tracker = FaultTracker()
    
    # Let test run
    yield tracker
    
    # Cleanup after test (even if it fails or is interrupted)
    if fault_ids:
        if verbose_mode:
            print(f"\n[CLEANUP] Recovering {len(fault_ids)} fault(s)...")
        
        for fault_id in fault_ids:
            try:
                fault_injection_client.recover_fault(fault_id)
                if verbose_mode:
                    print(f"[CLEANUP] ✓ Recovered fault: {fault_id}")
            except Exception as e:
                # Don't fail the test due to cleanup errors
                print(f"[CLEANUP WARNING] Failed to recover fault {fault_id}: {e}")


@pytest.fixture(scope="session", autouse=True)
def cleanup_on_interrupt(request, verbose_mode):
    """
    Session-level fixture to cleanup all faults on Ctrl+C or unexpected termination.
    """
    def cleanup_handler():
        """Called on pytest exit/interrupt."""
        try:
            config = get_config_from_env()
            client = FaultInjectionClient(api_url=config["api_url"])
            
            # List all active faults
            try:
                response = client._make_request("GET", "/faults")
                active_faults = response.get("faults", [])
                
                if active_faults:
                    print(f"\n[SESSION CLEANUP] Found {len(active_faults)} active fault(s)")
                    for fault in active_faults:
                        try:
                            fault_id = fault.get("fault_id")
                            client.recover_fault(fault_id)
                            if verbose_mode:
                                print(f"[SESSION CLEANUP] ✓ Recovered: {fault_id}")
                        except Exception as e:
                            print(f"[SESSION CLEANUP WARNING] Failed to recover {fault_id}: {e}")
            except Exception as e:
                print(f"[SESSION CLEANUP WARNING] Could not list faults: {e}")
        except Exception as e:
            print(f"[SESSION CLEANUP ERROR] {e}")
    
    # Register cleanup on session finish
    request.addfinalizer(cleanup_handler)
