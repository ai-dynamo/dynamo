# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
"""
Pytest configuration for fault tolerance tests.

This file imports all fixtures from fault_test_fixtures.py and makes them
available to all tests in this directory.
"""

import sys
from pathlib import Path

# Add helpers to path
sys.path.insert(0, str(Path(__file__).parent.parent / "helpers"))

# Import all fixtures that exist
from fault_test_fixtures import (  # noqa: E402
    default_deployment,
    default_namespace,
    ensure_clean_test_environment,
    expect_cordon_and_drain,
    expect_cordon_only,
    expect_full_automation,
    fault_test,
    network_partition_test,
    skip_if_insufficient_gpus,
    skip_if_no_nvsentinel,
    test_config,
    xid74_test,
    xid79_test,
    xid79_with_custom_validation,
)

# Make fixtures available to pytest
__all__ = [
    "test_config",
    "default_deployment",
    "default_namespace",
    "fault_test",
    "xid79_test",
    "xid74_test",
    "xid79_with_custom_validation",
    "network_partition_test",
    "expect_full_automation",
    "expect_cordon_and_drain",
    "expect_cordon_only",
    "ensure_clean_test_environment",
    "skip_if_no_nvsentinel",
    "skip_if_insufficient_gpus",
]
