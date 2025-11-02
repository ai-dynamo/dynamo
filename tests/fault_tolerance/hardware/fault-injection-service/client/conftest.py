"""
Pytest configuration for fault injection tests.

This file registers custom pytest markers to avoid warnings.
"""

import pytest


def pytest_configure(config):
    """Register custom markers to avoid warnings."""
    config.addinivalue_line(
        "markers", "fault_tolerance: mark test as a fault tolerance test"
    )
    config.addinivalue_line(
        "markers", "network_required: mark test as requiring network fault injection"
    )


@pytest.fixture(scope="session", autouse=True)
def dcgm_infrastructure():
    """Override DCGM fixture for network partition tests that don't require DCGM."""
    yield None
