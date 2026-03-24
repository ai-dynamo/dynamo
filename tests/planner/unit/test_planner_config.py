# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for PlannerConfig validation."""

import pytest
from pydantic import ValidationError

from dynamo.planner.utils.planner_config import PlannerConfig

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


def test_global_planner_mode():
    """Test PlannerConfig accepts global-planner environment with namespace."""
    config = PlannerConfig(
        namespace="test-ns",
        environment="global-planner",
        global_planner_namespace="global-ns",
    )
    assert config.environment == "global-planner"
    assert config.global_planner_namespace == "global-ns"


def test_global_planner_mode_without_namespace():
    """Test validation fails for global-planner environment without namespace."""
    with pytest.raises(ValidationError, match="global_planner_namespace is required"):
        PlannerConfig(
            namespace="test-ns",
            environment="global-planner",
        )


def test_invalid_environment():
    """Test PlannerConfig rejects invalid environment."""
    with pytest.raises(ValidationError):
        PlannerConfig(
            namespace="test-ns",
            environment="invalid-environment",
        )


def test_all_fields_work():
    """Test that PlannerConfig accepts all fields."""
    config = PlannerConfig(
        namespace="test-ns",
        backend="vllm",
        environment="kubernetes",
        ttft=200,
        itl=50,
        max_gpu_budget=16,
        throughput_adjustment_interval=60,
    )
    assert config.namespace == "test-ns"
    assert config.backend == "vllm"
    assert config.environment == "kubernetes"
    assert config.ttft == 200
    assert config.itl == 50
    assert config.max_gpu_budget == 16
    assert config.throughput_adjustment_interval == 60


def test_throughput_metrics_source_default():
    """throughput_metrics_source defaults to 'frontend'."""
    config = PlannerConfig(namespace="test-ns")
    assert config.throughput_metrics_source == "frontend"


def test_throughput_metrics_source_frontend():
    """throughput_metrics_source accepts 'frontend'."""
    config = PlannerConfig(namespace="test-ns", throughput_metrics_source="frontend")
    assert config.throughput_metrics_source == "frontend"


def test_throughput_metrics_source_router():
    """throughput_metrics_source accepts 'router'."""
    config = PlannerConfig(namespace="test-ns", throughput_metrics_source="router")
    assert config.throughput_metrics_source == "router"


def test_throughput_metrics_source_invalid():
    """throughput_metrics_source rejects invalid values."""
    with pytest.raises(ValidationError):
        PlannerConfig(namespace="test-ns", throughput_metrics_source="invalid")
