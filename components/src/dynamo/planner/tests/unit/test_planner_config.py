# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for PlannerConfig validation."""

import pytest
from pydantic import ValidationError

from dynamo.planner.config.planner_config import PlannerConfig

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


@pytest.mark.parametrize("bucket_size", [1, 4, 9, 16, 25])
def test_fpm_sample_bucket_size_accepts_perfect_squares(bucket_size):
    """fpm_sample_bucket_size must be a perfect square (valid values)."""
    config = PlannerConfig(namespace="test-ns", fpm_sample_bucket_size=bucket_size)
    assert config.fpm_sample_bucket_size == bucket_size


@pytest.mark.parametrize("bucket_size", [2, 3, 5, 7, 10])
def test_fpm_sample_bucket_size_rejects_non_squares(bucket_size):
    """fpm_sample_bucket_size rejects values that are not perfect squares."""
    with pytest.raises(ValidationError, match="perfect square"):
        PlannerConfig(namespace="test-ns", fpm_sample_bucket_size=bucket_size)


def test_max_num_fpm_samples_field():
    """max_num_fpm_samples configures the FPM sample retention (formerly load_learning_window)."""
    config = PlannerConfig(namespace="test-ns", max_num_fpm_samples=100)
    assert config.max_num_fpm_samples == 100


def test_agg_mode_supports_throughput_scaling():
    """Agg mode supports throughput-based scaling."""
    config = PlannerConfig(
        namespace="test-ns",
        mode="agg",
        optimization_target="sla",
        enable_throughput_scaling=True,
        enable_load_scaling=False,
    )
    assert config.mode == "agg"
    assert config.enable_throughput_scaling is True
    assert config.scaling_enabled() is True
