# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Scale testing tool for Dynamo DGD deployments on Kubernetes.

This module provides utilities to deploy configurable numbers of Dynamo mocker
instances as DynamoGraphDeployment resources on Kubernetes, with the Dynamo
operator managing infrastructure and load generation to verify functionality.
"""

from tests.scale_test.config import LoadTestConfig, ScaleManagerConfig, ScaleTestConfig
from tests.scale_test.dgd_builder import ScaleTestDGDBuilder, create_scale_test_specs
from tests.scale_test.load_generator import LoadGenerator
from tests.scale_test.scale_manager import ScaleManager

__all__ = [
    "ScaleManager",
    "LoadGenerator",
    "ScaleTestDGDBuilder",
    "create_scale_test_specs",
    "ScaleTestConfig",
    "LoadTestConfig",
    "ScaleManagerConfig",
]
