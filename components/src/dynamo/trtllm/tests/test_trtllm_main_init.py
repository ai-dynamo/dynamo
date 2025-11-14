# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Simple test for TensorRT-LLM MetricsCollector import and basic functionality.
"""

from unittest.mock import Mock

import pytest

# Mark all tests in this module to run only in TensorRT-LLM container
pytestmark = pytest.mark.trtllm


def test_tensorrt_llm_metrics_collector_import():
    """Test that we can import MetricsCollector from TensorRT-LLM."""
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore warnings during import
            from tensorrt_llm.metrics.collector import MetricsCollector

        # Test basic initialization (only once to avoid registry conflicts)
        metrics_collector = MetricsCollector(
            {"model_name": "test-model-unique", "engine_type": "trtllm"}
        )

        assert metrics_collector is not None
        print("✅ MetricsCollector imported and initialized successfully")

    except ImportError as e:
        pytest.skip(f"TensorRT-LLM not available: {e}")
    except Exception as e:
        pytest.fail(f"Failed to initialize MetricsCollector: {e}")


def test_prometheus_registry_import():
    """Test that we can import Prometheus registry."""
    try:
        from prometheus_client import REGISTRY

        assert REGISTRY is not None
        print("✅ Prometheus REGISTRY imported successfully")

    except ImportError as e:
        pytest.skip(f"Prometheus client not available: {e}")


def test_prometheus_metrics_integration():
    """Test Prometheus metrics integration as used in main.py init() function."""
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore warnings during import
            from prometheus_client import REGISTRY

            from dynamo.common.utils.prometheus import register_engine_metrics_callback

        # Mock endpoint for registration (simulating what init() does)
        mock_endpoint = Mock()

        # Test the exact call that main.py init() makes
        register_engine_metrics_callback(
            endpoint=mock_endpoint,
            registry=REGISTRY,
            exclude_prefixes=["python_", "process_"],
            add_prefix="trtllm_",
        )

        print("✅ Prometheus metrics integration test passed")

    except ImportError as e:
        pytest.skip(f"Required modules not available: {e}")
    except Exception as e:
        pytest.fail(f"Prometheus integration test failed: {e}")


if __name__ == "__main__":
    # Run tests directly for quick verification
    test_tensorrt_llm_metrics_collector_import()
    test_prometheus_registry_import()
    test_prometheus_metrics_integration()
    print("🎉 All tests passed!")
