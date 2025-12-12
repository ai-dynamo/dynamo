# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the KVBM Tensor wrapper.

The Tensor class wraps PyTorch tensors and validates that they are on CUDA devices.
Non-CUDA tensors should raise an error.
"""

import pytest
import torch
from kvbm._core import v2

Tensor = v2.Tensor


class TestTensorCudaOnly:
    """Tests that Tensor only accepts CUDA tensors."""

    def test_cpu_tensor_raises_error(self):
        """CPU tensors should be rejected with a clear error message."""
        cpu_tensor = torch.randn(4, 8, 16)

        with pytest.raises(RuntimeError) as exc_info:
            Tensor(cpu_tensor)

        assert "Only CUDA tensors are supported" in str(exc_info.value)
        assert "cpu" in str(exc_info.value)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_cuda_tensor_succeeds(self):
        """CUDA tensors should be accepted without error."""
        cuda_tensor = torch.randn(4, 8, 16, device="cuda:0")

        tensor = Tensor(cuda_tensor)

        assert tensor.shape == [4, 8, 16]
        assert tensor.device_index == 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_tensor_properties(self):
        """Verify tensor properties are correctly extracted."""
        shape = (2, 3, 4, 5)
        cuda_tensor = torch.randn(*shape, device="cuda:0", dtype=torch.float32)

        tensor = Tensor(cuda_tensor)

        # Check shape
        assert tensor.shape == list(shape)

        # Check stride (contiguous tensor)
        expected_stride = [60, 20, 5, 1]  # product of remaining dims
        assert tensor.stride == expected_stride

        # Check element size (float32 = 4 bytes)
        assert tensor.element_size == 4

        # Check size in bytes
        assert tensor.size_bytes == 2 * 3 * 4 * 5 * 4  # numel * element_size

        # Check data pointer is non-zero
        assert tensor.data_ptr > 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_different_dtypes(self):
        """Verify different dtypes report correct element sizes."""
        dtypes_and_sizes = [
            (torch.float16, 2),
            (torch.bfloat16, 2),
            (torch.float32, 4),
            (torch.float64, 8),
        ]

        for dtype, expected_size in dtypes_and_sizes:
            cuda_tensor = torch.randn(4, 4, device="cuda:0", dtype=dtype)
            tensor = Tensor(cuda_tensor)
            assert tensor.element_size == expected_size, f"Failed for {dtype}"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_non_contiguous_tensor(self):
        """Non-contiguous tensors should also work, with their actual stride."""
        cuda_tensor = torch.randn(4, 8, 16, device="cuda:0")
        transposed = cuda_tensor.transpose(0, 2)  # Shape: [16, 8, 4]

        tensor = Tensor(transposed)

        assert tensor.shape == [16, 8, 4]
        # Stride should reflect the transposed layout
        assert tensor.stride == [
            1,
            16,
            128,
        ]  # [1, 16, 128] for [16, 8, 4] from [4, 8, 16]

    @pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.device_count() < 2,
        reason="Multiple CUDA devices required",
    )
    def test_different_cuda_device(self):
        """Tensor on a different CUDA device should work."""
        cuda_tensor = torch.randn(4, 4, device="cuda:1")

        tensor = Tensor(cuda_tensor)

        assert tensor.device_index == 1
