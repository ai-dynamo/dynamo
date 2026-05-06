# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for HTTP facade tests.

Autouse fixture ensures the shared HTTP client singleton is closed after
each test so ``DYN_HTTP_BACKEND`` changes take effect between runs and
no "Unclosed client session" warning bleeds across tests.
"""

from __future__ import annotations

import pytest_asyncio

from dynamo.common.http import close_http_client


@pytest_asyncio.fixture(autouse=True)
async def _close_shared_http_client():
    yield
    await close_http_client()
