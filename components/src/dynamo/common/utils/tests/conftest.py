# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Conftest for common.utils unit tests."""

import sys
from unittest.mock import MagicMock

import pytest


# Mock pynvml globally for tests
@pytest.fixture(scope="session", autouse=True)
def mock_pynvml_module():
    """Mock pynvml module to avoid requiring nvidia-ml-py in test environment."""
    if "pynvml" not in sys.modules:
        sys.modules["pynvml"] = MagicMock()
    yield
    # Don't clean up - leave mock in place for other tests


@pytest.fixture(autouse=True)
def reset_gpu_info_metric():
    """Reset the global GPU info metric between tests."""
    try:
        import dynamo.common.utils.gpu as gpu_module

        gpu_module._gpu_info_metric = None
        yield
        gpu_module._gpu_info_metric = None
    except ImportError:
        # If module can't be imported, skip cleanup
        yield
