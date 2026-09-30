# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for HTTP facade tests.

One autouse fixture closes the shared HTTP client singleton after each test,
so ``DYN_HTTP_BACKEND`` changes take effect between runs and no "Unclosed
client session" warning bleeds across tests.

A second clears the proxy environment. The client refuses a policy-protected
fetch that an egress proxy would carry (see ``DYN_MM_TRUST_EGRESS_PROXY``), so
a developer machine or CI runner with an ambient ``http_proxy`` would other-
wise fail tests that have nothing to do with proxies. A test that wants a
proxy sets one itself with ``monkeypatch``.
"""

from __future__ import annotations

import pytest
import pytest_asyncio

from dynamo.common.http import close_http_client

_PROXY_VARS = (
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
)


@pytest.fixture(autouse=True)
def _no_ambient_proxy(monkeypatch):
    for name in _PROXY_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest_asyncio.fixture(autouse=True)
async def _close_shared_http_client():
    yield
    await close_http_client()
