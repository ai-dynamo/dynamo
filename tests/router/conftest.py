# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Router-specific pytest configuration.

Overrides global test fixtures to default to file storage backend for router tests.
"""

import pytest


@pytest.fixture
def store_kv(request):
    """
    KV store for router tests. Defaults to "file" for local development.
    To iterate over multiple stores in a test:
        @pytest.mark.parametrize("store_kv", ["file", "etcd"], indirect=True)
    """
    return getattr(request, "param", "file")