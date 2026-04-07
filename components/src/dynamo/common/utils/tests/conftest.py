# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Conftest for dynamo.common.utils unit tests.

Skips tests that require unavailable dynamo runtime extensions.
"""



def pytest_ignore_collect(collection_path, config):
    """Skip test files that require unavailable runtime modules."""
    return None
