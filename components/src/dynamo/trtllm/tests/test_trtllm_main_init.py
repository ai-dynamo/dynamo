# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Simple test for TensorRT-LLM MetricsCollector import and basic functionality.
"""

from unittest.mock import Mock

import pytest

# Mark all tests in this module to run only in TensorRT-LLM container
pytestmark = [
    pytest.mark.unit,
    pytest.mark.trtllm,
    pytest.mark.gpu_1,
    pytest.mark.pre_merge,
    pytest.mark.profiled_vram_gib(0),
]


def test_tensorrt_llm_metrics_collector_import():
    """Test that we can import MetricsCollector from TensorRT-LLM."""
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore warnings during import
            from tensorrt_llm.metrics.collector import MetricsCollector

        # TRT-LLM's MetricsCollector registers `trtllm_request_success_*`
        # against the global prometheus_client REGISTRY at construction.
        # When pytest-xdist groups this test in the same worker as any
        # other test that already instantiated a MetricsCollector (via
        # LegacyWorker startup paths or transitive imports), the second
        # construction raises ValueError("Duplicated timeseries"). Clear
        # any pre-existing trtllm_* collectors before constructing so the
        # test is self-isolating.
        from prometheus_client import REGISTRY

        for collector in list(REGISTRY._collector_to_names.keys()):
            names = REGISTRY._collector_to_names.get(collector, set())
            if any(n.startswith("trtllm_") for n in names):
                REGISTRY.unregister(collector)

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
            metric_prefix_filters=["trtllm_"],
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
