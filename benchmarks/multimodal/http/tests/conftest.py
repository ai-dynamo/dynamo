# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for the tests of the HTTP sweep harness.

An autouse fixture closes the shared HTTP client after each test. The current
tests never create that client, so the fixture only guards a future test that
fetches for real.
"""

from __future__ import annotations

import pytest_asyncio

from dynamo.common.http import close_http_client


@pytest_asyncio.fixture(autouse=True)
async def _close_shared_http_client():
    yield
    await close_http_client()
