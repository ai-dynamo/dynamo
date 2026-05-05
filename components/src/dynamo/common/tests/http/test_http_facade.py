# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Facade-level tests: backend resolution + SSRF revalidation loop.

Per-backend exception mapping lives in
``test_aiohttp_client.py`` / ``test_httpx_client.py``.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from dynamo.common import http as mm_http
from dynamo.common.http import AiohttpClient, HttpxClient
from dynamo.common.http.url_validator import UrlValidationError, UrlValidationPolicy

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


@pytest.fixture(autouse=True)
def _reset_backend_cache():
    """Every test resolves the backend from scratch.

    The reset before ``yield`` ensures each test starts with no cached
    singleton; the reset after ``yield`` prevents this module-level
    state from leaking into other tests or fixtures that run later in
    the session.
    """
    mm_http._default = None
    yield
    mm_http._default = None


# --- Backend selection ---


async def test_default_backend_is_aiohttp(monkeypatch) -> None:
    monkeypatch.delenv("DYN_MM_HTTP_BACKEND", raising=False)
    assert isinstance(mm_http.get_default_client(), AiohttpClient)


async def test_httpx_backend_selected(monkeypatch) -> None:
    monkeypatch.setenv("DYN_MM_HTTP_BACKEND", "httpx")
    assert isinstance(mm_http.get_default_client(), HttpxClient)


async def test_invalid_backend_raises(monkeypatch) -> None:
    monkeypatch.setenv("DYN_MM_HTTP_BACKEND", "requests")
    with pytest.raises(ValueError, match="DYN_MM_HTTP_BACKEND"):
        mm_http.get_default_client()


# --- Backend-neutral SSRF revalidation via fetch_bytes(policy=...) ---
#
# These exercise the facade path through an injected
# ``fetch_body_or_redirect`` stub on a concrete client instance, so they
# cover the shared redirect-loop + validate-each-hop logic independent
# of the chosen backend. We then parametrize over both backends to
# confirm routing.


_PERMISSIVE = UrlValidationPolicy(allow_http=True, allow_private_ips=True)


def _client_for(name: str):
    return {"aiohttp": AiohttpClient, "httpx": HttpxClient}[name]


@pytest.mark.parametrize("backend_name", ["aiohttp", "httpx"])
async def test_fetch_with_policy_returns_first_response(
    monkeypatch, backend_name
) -> None:
    monkeypatch.setenv("DYN_MM_HTTP_BACKEND", backend_name)
    client = mm_http.get_default_client()
    assert isinstance(client, _client_for(backend_name))

    call_count = {"n": 0}

    async def _fake(url, timeout):
        call_count["n"] += 1
        return b"body-bytes", None

    with patch.object(client, "fetch_body_or_redirect", _fake):
        result = await mm_http.fetch_bytes(
            "https://example.com/x.png", 30.0, policy=_PERMISSIVE
        )
    assert result == b"body-bytes"
    assert call_count["n"] == 1


@pytest.mark.parametrize("backend_name", ["aiohttp", "httpx"])
async def test_fetch_with_policy_follows_safe_redirect(
    monkeypatch, backend_name
) -> None:
    monkeypatch.setenv("DYN_MM_HTTP_BACKEND", backend_name)
    client = mm_http.get_default_client()

    hops: list[str] = []

    async def _fake(url, timeout):
        hops.append(url)
        if url == "https://example.com/x.png":
            return None, "https://example.com/final.png"
        return b"final-bytes", None

    with patch.object(client, "fetch_body_or_redirect", _fake):
        result = await mm_http.fetch_bytes(
            "https://example.com/x.png", 30.0, policy=_PERMISSIVE
        )
    assert result == b"final-bytes"
    assert hops == ["https://example.com/x.png", "https://example.com/final.png"]


@pytest.mark.parametrize("backend_name", ["aiohttp", "httpx"])
async def test_fetch_with_policy_blocks_redirect_to_private_ip(
    monkeypatch, backend_name
) -> None:
    monkeypatch.setenv("DYN_MM_HTTP_BACKEND", backend_name)
    client = mm_http.get_default_client()

    strict = UrlValidationPolicy(allow_private_ips=False)

    async def _fake(url, timeout):
        return None, "http://169.254.169.254/latest/meta-data/"

    with patch.object(client, "fetch_body_or_redirect", _fake):
        with pytest.raises(UrlValidationError):
            await mm_http.fetch_bytes("https://8.8.8.8/x.png", 30.0, policy=strict)


@pytest.mark.parametrize("backend_name", ["aiohttp", "httpx"])
async def test_fetch_with_policy_enforces_redirect_limit(
    monkeypatch, backend_name
) -> None:
    monkeypatch.setenv("DYN_MM_HTTP_BACKEND", backend_name)
    client = mm_http.get_default_client()

    # _MAX_REDIRECTS=3 → 4 hops trip the cap.
    chain = {
        "https://example.com/a": "https://example.com/b",
        "https://example.com/b": "https://example.com/c",
        "https://example.com/c": "https://example.com/d",
        "https://example.com/d": "https://example.com/e",
    }

    async def _fake(url, timeout):
        return None, chain[url]

    with patch.object(client, "fetch_body_or_redirect", _fake):
        with pytest.raises(UrlValidationError, match="Too many redirects"):
            await mm_http.fetch_bytes("https://example.com/a", 30.0, policy=_PERMISSIVE)
