# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
"""
Pytest configuration for hardware fault injection tests.

This conftest provides automatic setup of:
- DCGM DaemonSet for GPU monitoring
- Fault Injection API client
- Cleanup handlers

Simply import this file or place it in your test directory to enable auto-setup.
"""

import logging
import sys
from pathlib import Path

import pytest

# Add client library to path
sys.path.insert(0, str(Path(__file__).parent.parent / "client"))

from fault_injection_client import FaultInjectionClient

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def dcgm_namespace():
    """
    DCGM namespace configuration.

    Override this in your own conftest.py if you need a different namespace:

        @pytest.fixture(scope="session")
        def dcgm_namespace():
            return "my-custom-namespace"
    """
    return "gpu-operator"


@pytest.fixture(scope="session")
def dcgm_yaml_path():
    """
    Path to dcgm-daemonset.yaml file.

    Override this in your own conftest.py if you need a custom path:

        @pytest.fixture(scope="session")
        def dcgm_yaml_path():
            return "/path/to/my/dcgm-daemonset.yaml"
    """
    # Default: auto-detect
    return None


@pytest.fixture(scope="session", autouse=True)
def dcgm_infrastructure(dcgm_namespace, dcgm_yaml_path):
    """
    Verify DCGM infrastructure for GPU tests.

    This fixture automatically runs once per test session and:
    1. Checks if DCGM is already deployed
    2. Skips tests if DCGM is not present

    With autouse=True, this runs automatically for all tests - no need to
    explicitly request it in test signatures!

    To disable auto-check, set autouse=False above or override in your conftest.py.
    """
    import os
    
    logger.info("=" * 80)
    logger.info("Verifying DCGM Infrastructure")
    logger.info("=" * 80)

    # Support both in-cluster and local execution
    api_url = os.environ.get("API_URL", "http://localhost:8080")
    client = FaultInjectionClient(api_url=api_url)

    # Check if DCGM is already deployed
    status = client.check_dcgm_status(dcgm_namespace)

    if status.get("deployed"):
        logger.info(f"✓ DCGM deployed in namespace '{dcgm_namespace}'")
        logger.info(f"  Ready pods: {status.get('ready_pods')}/{status.get('desired_pods')}")
        logger.info("✓ DCGM infrastructure verified - ready for testing")
    else:
        logger.warning(f"⚠ DCGM not found in namespace '{dcgm_namespace}'")
        logger.warning("  Install GPU Operator with: helm install gpu-operator nvidia/gpu-operator")
        pytest.skip(f"DCGM not deployed in '{dcgm_namespace}' - required for GPU fault detection")

    yield dcgm_namespace

    logger.info("DCGM infrastructure check complete")


@pytest.fixture(scope="session")
def fault_client_with_dcgm(dcgm_infrastructure):
    """
    Fault injection client with DCGM infrastructure ready.

    This combines the fault client with automatic DCGM setup.

    Usage:
        def test_gpu_fault(fault_client_with_dcgm):
            client = fault_client_with_dcgm
            # Test code here
            pass
    """
    import os
    
    # Support both in-cluster and local execution
    api_url = os.environ.get("API_URL", "http://localhost:8080")
    client = FaultInjectionClient(api_url=api_url)

    # Verify API is accessible
    if not client.health_check():
        pytest.skip("Fault Injection API not accessible at http://localhost:8080")

    logger.info("Fault Injection Client ready")

    yield client


    logger.info("Test session complete")


@pytest.fixture(scope="function")
def fault_client():
    """
    Basic fault injection client without DCGM setup.

    Use this if you want to control DCGM deployment manually or
    if you don't need GPU monitoring.

    Usage:
        def test_network_fault(fault_client):
            # Test code here
            pass
    """
    import os
    
    # Support both in-cluster and local execution
    api_url = os.environ.get("API_URL", "http://localhost:8080")
    client = FaultInjectionClient(api_url=api_url)

    if not client.health_check():
        pytest.skip("Fault Injection API not accessible")

    yield client


# Pytest markers for categorizing tests
def pytest_configure(config):
    """Register custom markers"""
    config.addinivalue_line("markers", "gpu_required: mark test as requiring GPU hardware")
    config.addinivalue_line("markers", "dcgm_required: mark test as requiring DCGM infrastructure")
    config.addinivalue_line("markers", "fault_tolerance: mark test as a fault tolerance test")
