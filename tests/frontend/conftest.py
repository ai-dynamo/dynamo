# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Conftest for frontend tests.

Handles conditional test collection to prevent import errors when required
dependencies are not installed in the current environment.
"""

import importlib.util


def pytest_ignore_collect(collection_path, config):
    """Skip collecting test files if required dependencies aren't installed."""
    filename = collection_path.name

    # Skip prompt_embeds tests if openai or torch aren't available
    if filename == "test_prompt_embeds.py":
        if importlib.util.find_spec("openai") is None:
            return True  # openai not available, skip this file
        if importlib.util.find_spec("torch") is None:
            return True  # torch not available, skip this file

    return None
