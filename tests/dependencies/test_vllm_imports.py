# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests to sanity check that required dependencies can be imported."""

import pytest


@pytest.mark.vllm
@pytest.mark.unit
@pytest.mark.gpu_1
@pytest.mark.pre_merge
def test_import_deep_ep():
    """Test that deep_ep module can be imported."""
    try:
        import deep_ep

        assert deep_ep is not None
    except ImportError as e:
        pytest.fail(f"Failed to import deep_ep: {e}")


@pytest.mark.vllm
@pytest.mark.unit
@pytest.mark.gpu_1
@pytest.mark.pre_merge
def test_import_pplx_kernels():
    """Test that pplx_kernels module can be imported."""
    try:
        import pplx_kernels

        assert pplx_kernels is not None
    except ImportError as e:
        pytest.fail(f"Failed to import pplx_kernels: {e}")
