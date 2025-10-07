# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Conftest for dynamo backend unit tests.

Handles conditional test collection to prevent import errors when backend
frameworks are not installed in the current container.
"""

import importlib.util


def pytest_ignore_collect(collection_path, config):
    """Skip collecting backend test files if their framework isn't installed.

    Checks test file naming pattern: test_<backend>_*.py
    """
    filename = collection_path.name

    # Map test file prefixes to required modules
    backend_requirements = {
        "test_vllm_": "vllm",
        "test_sglang_": "sglang",
        "test_trtllm_": "tensorrt_llm",
    }

    for prefix, required_module in backend_requirements.items():
        if filename.startswith(prefix):
            if importlib.util.find_spec(required_module) is None:
                return True  # Module not available, skip this file

    return None  # Not a backend test or module available
