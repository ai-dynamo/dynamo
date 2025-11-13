# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for GPU utilities."""

from unittest.mock import Mock, patch

import pytest
from prometheus_client import REGISTRY

from dynamo.common.utils.gpu import (
    get_gpu_info,
    get_gpu_uuids,
    initialize_gpu_info_metric,
)

pytestmark = [
    pytest.mark.unit,
]


class TestGetGpuUuids:
    """Test GPU UUID retrieval functionality."""

    @pytest.fixture
    def mock_pynvml_single_gpu(self):
        """Mock pynvml for a single GPU system."""
        mock_nvml = Mock()
        mock_nvml.nvmlDeviceGetCount.return_value = 1
        mock_nvml.nvmlDeviceGetUUID.return_value = (
            "GPU-abc12345-def6-7890-abcd-ef1234567890"
        )

        with patch.dict("sys.modules", {"pynvml": mock_nvml}):
            yield mock_nvml

    @pytest.fixture
    def mock_pynvml_multi_gpu(self):
        """Mock pynvml for a multi-GPU system (tensor parallel)."""
        mock_nvml = Mock()
        mock_nvml.nvmlDeviceGetCount.return_value = 4
        mock_nvml.nvmlDeviceGetUUID.side_effect = [
            "GPU-00000000-0000-0000-0000-000000000000",
            "GPU-11111111-1111-1111-1111-111111111111",
            "GPU-22222222-2222-2222-2222-222222222222",
            "GPU-33333333-3333-3333-3333-333333333333",
        ]

        with patch.dict("sys.modules", {"pynvml": mock_nvml}):
            yield mock_nvml

    def test_single_gpu(self, mock_pynvml_single_gpu):
        """Test retrieving UUID from a single GPU system."""
        uuids = get_gpu_uuids()

        # Should return exactly one UUID
        assert len(uuids) == 1
        assert uuids[0] == "GPU-abc12345-def6-7890-abcd-ef1234567890"

        # Should initialize and shutdown NVML
        mock_pynvml_single_gpu.nvmlInit.assert_called_once()
        mock_pynvml_single_gpu.nvmlShutdown.assert_called_once()

    def test_multi_gpu(self, mock_pynvml_multi_gpu):
        """Test retrieving UUIDs from a multi-GPU system (tensor parallel)."""
        uuids = get_gpu_uuids()

        # Should return all four UUIDs
        assert len(uuids) == 4, f"Expected 4 UUIDs, got {len(uuids)}"
        assert all(uuid.startswith("GPU-") for uuid in uuids)
        assert len(set(uuids)) == 4, "All UUIDs should be unique"

    def test_bytes_to_string_conversion(self):
        """Test handling of pynvml versions that return bytes instead of strings."""
        mock_nvml = Mock()
        mock_nvml.nvmlDeviceGetCount.return_value = 1
        # Some pynvml versions return bytes
        mock_nvml.nvmlDeviceGetUUID.return_value = (
            b"GPU-abc12345-def6-7890-abcd-ef1234567890"
        )

        with patch.dict("sys.modules", {"pynvml": mock_nvml}):
            uuids = get_gpu_uuids()

            # Should convert bytes to string
            assert isinstance(uuids[0], str)
            assert uuids[0] == "GPU-abc12345-def6-7890-abcd-ef1234567890"

    def test_no_gpus(self):
        """Test behavior when no GPUs are present (CPU-only system)."""
        mock_nvml = Mock()
        mock_nvml.nvmlDeviceGetCount.return_value = 0

        with patch.dict("sys.modules", {"pynvml": mock_nvml}):
            uuids = get_gpu_uuids()

            # Should return empty list
            assert uuids == []

    def test_nvml_failure(self):
        """Test graceful handling when NVML queries fail."""
        mock_nvml = Mock()
        mock_nvml.nvmlInit.side_effect = RuntimeError("NVML not available")

        with patch.dict("sys.modules", {"pynvml": mock_nvml}):
            uuids = get_gpu_uuids()

            # Should return empty list on error
            assert uuids == []


class TestGetGpuInfo:
    """Test comprehensive GPU information retrieval."""

    @pytest.fixture
    def mock_pynvml_complete_info(self):
        """Mock pynvml with complete GPU metadata."""
        mock_nvml = Mock()

        # System info
        mock_nvml.nvmlDeviceGetCount.return_value = 1
        mock_nvml.nvmlSystemGetDriverVersion.return_value = "535.129.03"
        mock_nvml.nvmlSystemGetCudaDriverVersion.return_value = 12020  # CUDA 12.2

        # Device info
        mock_nvml.nvmlDeviceGetUUID.return_value = (
            "GPU-ef6ef310-6bb8-8bfa-1b66-9fb6e8479ee8"
        )
        mock_nvml.nvmlDeviceGetName.return_value = "NVIDIA RTX 6000 Ada Generation"

        # Memory info
        mock_memory_info = Mock()
        mock_memory_info.total = 51539607552  # 48 GB in bytes
        mock_nvml.nvmlDeviceGetMemoryInfo.return_value = mock_memory_info

        # Compute capability
        mock_nvml.nvmlDeviceGetCudaComputeCapability.return_value = (8, 9)

        # PCI info
        mock_pci_info = Mock()
        mock_pci_info.busId = b"00000000:01:00.0"
        mock_nvml.nvmlDeviceGetPciInfo.return_value = mock_pci_info

        # Power and MIG
        mock_nvml.nvmlDeviceGetPowerManagementDefaultLimit.return_value = 300000  # 300W
        mock_nvml.nvmlDeviceGetMigMode.return_value = (0, 0)  # MIG disabled

        with patch.dict("sys.modules", {"pynvml": mock_nvml}):
            yield mock_nvml

    def test_complete_metadata(self, mock_pynvml_complete_info):
        """Test retrieving all 10 metadata fields for a GPU."""
        gpu_info_list = get_gpu_info()

        # Should return one GPU with complete metadata
        assert len(gpu_info_list) == 1
        gpu_info = gpu_info_list[0]

        # Verify all metadata fields are present and correct
        assert gpu_info["gpu_uuid"] == "GPU-ef6ef310-6bb8-8bfa-1b66-9fb6e8479ee8"
        assert gpu_info["gpu_index"] == "0"
        assert gpu_info["gpu_model"] == "NVIDIA RTX 6000 Ada Generation"
        assert gpu_info["gpu_memory_gb"] == "48.0"
        assert gpu_info["compute_capability"] == "8.9"
        assert gpu_info["pci_bus_id"] == "00000000:01:00.0"
        assert gpu_info["driver_version"] == "535.129.03"
        assert gpu_info["cuda_version"] == "12.2"
        assert gpu_info["power_limit_w"] == "300"
        assert gpu_info["mig_mode"] == "Disabled"

    def test_multi_gpu_distinct_indices(self):
        """Test that multi-GPU systems get distinct gpu_index values."""
        mock_nvml = Mock()
        mock_nvml.nvmlDeviceGetCount.return_value = 2
        mock_nvml.nvmlSystemGetDriverVersion.return_value = "535.129.03"
        mock_nvml.nvmlSystemGetCudaDriverVersion.return_value = 12020

        mock_nvml.nvmlDeviceGetUUID.side_effect = [
            "GPU-00000000-0000-0000-0000-000000000000",
            "GPU-11111111-1111-1111-1111-111111111111",
        ]
        mock_nvml.nvmlDeviceGetName.return_value = "NVIDIA H100 80GB HBM3"

        mock_memory_info = Mock()
        mock_memory_info.total = 85899345920  # 80 GB
        mock_nvml.nvmlDeviceGetMemoryInfo.return_value = mock_memory_info

        mock_nvml.nvmlDeviceGetCudaComputeCapability.return_value = (9, 0)

        mock_pci_info = Mock()
        mock_pci_info.busId = b"00000000:01:00.0"
        mock_nvml.nvmlDeviceGetPciInfo.return_value = mock_pci_info

        mock_nvml.nvmlDeviceGetPowerManagementDefaultLimit.return_value = 700000
        mock_nvml.nvmlDeviceGetMigMode.return_value = (0, 0)

        with patch.dict("sys.modules", {"pynvml": mock_nvml}):
            gpu_info_list = get_gpu_info()

            # Should return two GPUs with distinct indices
            assert len(gpu_info_list) == 2
            assert gpu_info_list[0]["gpu_index"] == "0"
            assert gpu_info_list[1]["gpu_index"] == "1"
            assert gpu_info_list[0]["gpu_uuid"] != gpu_info_list[1]["gpu_uuid"]

    @pytest.mark.parametrize(
        "cuda_version_int,expected_str",
        [
            (12020, "12.2"),  # CUDA 12.2
            (12010, "12.1"),  # CUDA 12.1
            (12000, "12.0"),  # CUDA 12.0
            (11080, "11.8"),  # CUDA 11.8
            (11000, "11.0"),  # CUDA 11.0
        ],
    )
    def test_cuda_version_formatting(self, cuda_version_int, expected_str):
        """Test CUDA version number formatting (12020 -> 12.2, etc.)."""
        mock_nvml = Mock()
        mock_nvml.nvmlDeviceGetCount.return_value = 1
        mock_nvml.nvmlSystemGetDriverVersion.return_value = "535.129.03"
        mock_nvml.nvmlSystemGetCudaDriverVersion.return_value = cuda_version_int

        mock_nvml.nvmlDeviceGetUUID.return_value = "GPU-test"
        mock_nvml.nvmlDeviceGetName.return_value = "Test GPU"

        mock_memory_info = Mock()
        mock_memory_info.total = 51539607552
        mock_nvml.nvmlDeviceGetMemoryInfo.return_value = mock_memory_info

        mock_nvml.nvmlDeviceGetCudaComputeCapability.return_value = (8, 9)

        mock_pci_info = Mock()
        mock_pci_info.busId = b"00000000:01:00.0"
        mock_nvml.nvmlDeviceGetPciInfo.return_value = mock_pci_info

        mock_nvml.nvmlDeviceGetPowerManagementDefaultLimit.return_value = 300000
        mock_nvml.nvmlDeviceGetMigMode.return_value = (0, 0)

        with patch.dict("sys.modules", {"pynvml": mock_nvml}):
            gpu_info_list = get_gpu_info()

            # Should format CUDA version correctly
            assert gpu_info_list[0]["cuda_version"] == expected_str

    def test_mig_mode_detection(self):
        """Test detection of MIG (Multi-Instance GPU) mode enabled/disabled."""
        mock_nvml = Mock()
        mock_nvml.nvmlDeviceGetCount.return_value = 1
        mock_nvml.nvmlSystemGetDriverVersion.return_value = "535.129.03"
        mock_nvml.nvmlSystemGetCudaDriverVersion.return_value = 12020

        mock_nvml.nvmlDeviceGetUUID.return_value = "GPU-mig-test"
        mock_nvml.nvmlDeviceGetName.return_value = "NVIDIA H100"

        mock_memory_info = Mock()
        mock_memory_info.total = 85899345920
        mock_nvml.nvmlDeviceGetMemoryInfo.return_value = mock_memory_info

        mock_nvml.nvmlDeviceGetCudaComputeCapability.return_value = (9, 0)

        mock_pci_info = Mock()
        mock_pci_info.busId = b"00000000:01:00.0"
        mock_nvml.nvmlDeviceGetPciInfo.return_value = mock_pci_info

        mock_nvml.nvmlDeviceGetPowerManagementDefaultLimit.return_value = 700000

        # MIG is enabled
        mock_nvml.nvmlDeviceGetMigMode.return_value = (1, 1)

        with patch.dict("sys.modules", {"pynvml": mock_nvml}):
            gpu_info_list = get_gpu_info()

            # Should detect MIG enabled
            assert gpu_info_list[0]["mig_mode"] == "Enabled"

    def test_partial_info_on_query_failure(self):
        """Test that function returns partial info when some queries fail."""
        mock_nvml = Mock()
        mock_nvml.nvmlDeviceGetCount.return_value = 1
        mock_nvml.nvmlSystemGetDriverVersion.return_value = "535.129.03"
        mock_nvml.nvmlSystemGetCudaDriverVersion.return_value = 12020

        mock_nvml.nvmlDeviceGetUUID.return_value = "GPU-test-uuid"

        # Name query fails
        mock_nvml.nvmlDeviceGetName.side_effect = RuntimeError("Name query failed")

        # Memory query fails
        mock_nvml.nvmlDeviceGetMemoryInfo.side_effect = RuntimeError(
            "Memory query failed"
        )

        # Other queries succeed
        mock_nvml.nvmlDeviceGetCudaComputeCapability.return_value = (8, 9)

        mock_pci_info = Mock()
        mock_pci_info.busId = b"00000000:01:00.0"
        mock_nvml.nvmlDeviceGetPciInfo.return_value = mock_pci_info

        mock_nvml.nvmlDeviceGetPowerManagementDefaultLimit.return_value = 300000
        mock_nvml.nvmlDeviceGetMigMode.return_value = (0, 0)

        with patch.dict("sys.modules", {"pynvml": mock_nvml}):
            gpu_info_list = get_gpu_info()

            # Should still return GPU with fallback values for failed queries
            assert len(gpu_info_list) == 1
            assert gpu_info_list[0]["gpu_model"] == "Unknown GPU"
            assert gpu_info_list[0]["gpu_memory_gb"] == "Unknown"
            assert (
                gpu_info_list[0]["compute_capability"] == "8.9"
            ), "Successful query should work"


class TestInitializeGpuInfoMetric:
    """Test Prometheus info metric initialization."""

    @pytest.fixture(autouse=True)
    def cleanup_prometheus_registry(self):
        """Clean up Prometheus registry after each test to prevent duplicate metric errors.

        Note: The conftest.py fixture already resets the global _gpu_info_metric variable.
        This fixture specifically handles unregistering from the Prometheus REGISTRY.
        """
        yield
        # After test runs, unregister the GPU info metric from Prometheus
        import dynamo.common.utils.gpu as gpu_module

        if gpu_module._gpu_info_metric is not None:
            try:
                REGISTRY.unregister(gpu_module._gpu_info_metric)
            except Exception:
                # Ignore errors if metric was already unregistered
                pass

    def test_metric_creation_with_labels(self):
        """Test that metric is created with GPU metadata and extra labels."""
        mock_nvml = Mock()
        mock_nvml.nvmlDeviceGetCount.return_value = 1
        mock_nvml.nvmlSystemGetDriverVersion.return_value = "535.129.03"
        mock_nvml.nvmlSystemGetCudaDriverVersion.return_value = 12020

        mock_nvml.nvmlDeviceGetUUID.return_value = "GPU-test-metric"
        mock_nvml.nvmlDeviceGetName.return_value = "NVIDIA RTX 6000 Ada"

        mock_memory_info = Mock()
        mock_memory_info.total = 51539607552
        mock_nvml.nvmlDeviceGetMemoryInfo.return_value = mock_memory_info

        mock_nvml.nvmlDeviceGetCudaComputeCapability.return_value = (8, 9)

        mock_pci_info = Mock()
        mock_pci_info.busId = b"00000000:01:00.0"
        mock_nvml.nvmlDeviceGetPciInfo.return_value = mock_pci_info

        mock_nvml.nvmlDeviceGetPowerManagementDefaultLimit.return_value = 300000
        mock_nvml.nvmlDeviceGetMigMode.return_value = (0, 0)

        with patch.dict("sys.modules", {"pynvml": mock_nvml}):
            # Initialize with extra labels
            initialize_gpu_info_metric(
                extra_labels={
                    "model": "Qwen/Qwen3-0.6B",
                    "dynamo_component": "backend",
                    "dynamo_namespace": "dynamo",
                }
            )

            # Verify metric was created (check via REGISTRY)
            from prometheus_client import REGISTRY, generate_latest

            metrics_text = generate_latest(REGISTRY).decode("utf-8")

            # Should contain the info metric with labels
            assert "dynamo_worker_gpu_info" in metrics_text
            assert "GPU-test-metric" in metrics_text
            assert "Qwen/Qwen3-0.6B" in metrics_text
            assert "backend" in metrics_text

    def test_metric_multi_gpu_series(self):
        """Test that separate series are created for multi-GPU systems."""
        mock_nvml = Mock()
        mock_nvml.nvmlDeviceGetCount.return_value = 2
        mock_nvml.nvmlSystemGetDriverVersion.return_value = "535.129.03"
        mock_nvml.nvmlSystemGetCudaDriverVersion.return_value = 12020

        mock_nvml.nvmlDeviceGetUUID.side_effect = [
            "GPU-multi-0",
            "GPU-multi-1",
        ]
        mock_nvml.nvmlDeviceGetName.return_value = "NVIDIA H100"

        mock_memory_info = Mock()
        mock_memory_info.total = 85899345920
        mock_nvml.nvmlDeviceGetMemoryInfo.return_value = mock_memory_info

        mock_nvml.nvmlDeviceGetCudaComputeCapability.return_value = (9, 0)

        mock_pci_info = Mock()
        mock_pci_info.busId = b"00000000:01:00.0"
        mock_nvml.nvmlDeviceGetPciInfo.return_value = mock_pci_info

        mock_nvml.nvmlDeviceGetPowerManagementDefaultLimit.return_value = 700000
        mock_nvml.nvmlDeviceGetMigMode.return_value = (0, 0)

        with patch.dict("sys.modules", {"pynvml": mock_nvml}):
            initialize_gpu_info_metric(
                extra_labels={"model": "Llama-2-70B", "dynamo_component": "backend"}
            )

            from prometheus_client import REGISTRY, generate_latest

            metrics_text = generate_latest(REGISTRY).decode("utf-8")

            # Should have two separate series
            assert "GPU-multi-0" in metrics_text
            assert "GPU-multi-1" in metrics_text
            assert 'gpu_index="0"' in metrics_text
            assert 'gpu_index="1"' in metrics_text

    def test_metric_no_gpus_graceful(self):
        """Test that initialization handles no GPUs gracefully without error."""
        mock_nvml = Mock()
        mock_nvml.nvmlDeviceGetCount.return_value = 0

        with patch.dict("sys.modules", {"pynvml": mock_nvml}):
            # Should not raise an error
            initialize_gpu_info_metric(extra_labels={"model": "test"})
