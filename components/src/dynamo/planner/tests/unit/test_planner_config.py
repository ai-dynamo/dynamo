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


def test_encode_mode_accepts_config():
    config = PlannerConfig(
        namespace="test-ns",
        mode="encode",
        encode_engine_num_gpu=2,
        model_name="test-model",
    )
    assert config.mode == "encode"
    assert config.encode_engine_num_gpu == 2


def test_encode_global_planner_requires_encode_engine_num_gpu():
    with pytest.raises(
        ValidationError,
        match="encode_engine_num_gpu is required when mode='encode' and environment='global-planner'",
    ):
        PlannerConfig(
            namespace="test-ns",
            environment="global-planner",
            global_planner_namespace="global-ns",
            mode="encode",
            model_name="test-model",
        )


def test_encode_global_planner_requires_model_name():
    with pytest.raises(
        ValidationError,
        match="model_name is required when mode='encode' and environment='global-planner'",
    ):
        PlannerConfig(
            namespace="test-ns",
            environment="global-planner",
            global_planner_namespace="global-ns",
            mode="encode",
            encode_engine_num_gpu=1,
        )


def test_encode_global_planner_accepts_required_fields():
    config = PlannerConfig(
        namespace="test-ns",
        environment="global-planner",
        global_planner_namespace="global-ns",
        mode="encode",
        model_name="test-model",
        encode_engine_num_gpu=1,
    )
    assert config.environment == "global-planner"
    assert config.mode == "encode"


def test_encode_mode_rejects_load_scaling():
    with pytest.raises(
        ValidationError, match="mode='encode' does not support enable_load_scaling=True"
    ):
        PlannerConfig(
            namespace="test-ns",
            mode="encode",
            model_name="test-model",
            encode_engine_num_gpu=1,
            enable_load_scaling=True,
        )


def test_encode_mode_rejects_virtual_environment():
    with pytest.raises(
        ValidationError,
        match="mode='encode' is not supported when environment='virtual' in Phase 1",
    ):
        PlannerConfig(
            namespace="test-ns",
            environment="virtual",
            mode="encode",
            model_name="test-model",
            encode_engine_num_gpu=1,
        )


def test_encode_mode_rejects_mocker_backend():
    with pytest.raises(
        ValidationError,
        match="backend='mocker' does not support mode='encode' in Phase 1",
    ):
        PlannerConfig(
            namespace="test-ns",
            backend="mocker",
            mode="encode",
            model_name="test-model",
            encode_engine_num_gpu=1,
        )


def test_encode_mode_rejects_pre_swept_results():
    with pytest.raises(
        ValidationError,
        match="mode='encode' does not support .*use-pre-swept-results",
    ):
        PlannerConfig(
            namespace="test-ns",
            mode="encode",
            model_name="test-model",
            encode_engine_num_gpu=1,
            profile_results_dir="use-pre-swept-results:H200:model:vllm:1.0:1:1:1:16:1",
        )


@pytest.mark.parametrize("mode", ["disagg", "prefill", "decode", "agg"])
def test_existing_modes_still_validate(mode: str):
    config = PlannerConfig(namespace="test-ns", mode=mode)
    assert config.mode == mode
