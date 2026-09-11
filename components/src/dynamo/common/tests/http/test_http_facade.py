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
from dynamo.common.http import AiohttpClient, HttpxClient, base, from_env
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
    monkeypatch.delenv("DYN_HTTP_BACKEND", raising=False)
    assert isinstance(mm_http.get_default_client(), AiohttpClient)


async def test_httpx_backend_selected(monkeypatch) -> None:
    monkeypatch.setenv("DYN_HTTP_BACKEND", "httpx")
    assert isinstance(mm_http.get_default_client(), HttpxClient)


async def test_invalid_backend_raises(monkeypatch) -> None:
    monkeypatch.setenv("DYN_HTTP_BACKEND", "requests")
    with pytest.raises(ValueError, match="DYN_HTTP_BACKEND"):
        mm_http.get_default_client()


async def test_legacy_mm_http_env_var_still_honored(monkeypatch) -> None:
    """Legacy ``DYN_MM_HTTP_*`` env vars from the deleted
    ``dynamo.common.multimodal.http_client`` module still take effect.
    """
    monkeypatch.delenv("DYN_HTTP_MAX_CONNECTIONS", raising=False)
    monkeypatch.setenv("DYN_MM_HTTP_MAX_CONNECTIONS", "77")
    config = from_env()
    assert config.max_connections == 77


# --- Backend-neutral SSRF revalidation via fetch_bytes(policy=...) ---
#
# These exercise the facade path through an injected stub on the
# subclass-impl ``_fetch_body_or_redirect`` seam — the cleanest place
# to inject test responses without standing up a real HTTP server.
# Patching a private method is acceptable here because we're testing
# the parent class's redirect-loop logic, not the seam itself.


_PERMISSIVE = UrlValidationPolicy(allow_http=True, allow_private_ips=True)


def _client_for(name: str):
    return {"aiohttp": AiohttpClient, "httpx": HttpxClient}[name]


@pytest.mark.parametrize("backend_name", ["aiohttp", "httpx"])
async def test_fetch_with_policy_returns_first_response(
    monkeypatch, backend_name
) -> None:
    monkeypatch.setenv("DYN_HTTP_BACKEND", backend_name)
    client = mm_http.get_default_client()
    assert isinstance(client, _client_for(backend_name))

    call_count = {"n": 0}

    async def _fake(url, timeout, *, max_bytes=None):
        call_count["n"] += 1
        return b"body-bytes", None

    with patch.object(client, "_fetch_body_or_redirect", _fake):
        result = await mm_http.fetch_bytes(
            "https://example.com/x.png", 30.0, policy=_PERMISSIVE
        )
    assert result == b"body-bytes"
    assert call_count["n"] == 1


@pytest.mark.parametrize("backend_name", ["aiohttp", "httpx"])
async def test_fetch_with_policy_follows_safe_redirect(
    monkeypatch, backend_name
) -> None:
    monkeypatch.setenv("DYN_HTTP_BACKEND", backend_name)
    client = mm_http.get_default_client()

    hops: list[str] = []

    async def _fake(url, timeout, *, max_bytes=None):
        hops.append(url)
        if url == "https://example.com/x.png":
            return None, "https://example.com/final.png"
        return b"final-bytes", None

    with patch.object(client, "_fetch_body_or_redirect", _fake):
        result = await mm_http.fetch_bytes(
            "https://example.com/x.png", 30.0, policy=_PERMISSIVE
        )
    assert result == b"final-bytes"
    assert hops == ["https://example.com/x.png", "https://example.com/final.png"]


@pytest.mark.parametrize("backend_name", ["aiohttp", "httpx"])
async def test_fetch_with_policy_blocks_redirect_to_private_ip(
    monkeypatch, backend_name
) -> None:
    monkeypatch.setenv("DYN_HTTP_BACKEND", backend_name)
    client = mm_http.get_default_client()

    strict = UrlValidationPolicy(allow_private_ips=False)

    async def _fake(url, timeout, *, max_bytes=None):
        return None, "http://169.254.169.254/latest/meta-data/"

    with patch.object(client, "_fetch_body_or_redirect", _fake):
        with pytest.raises(UrlValidationError):
            await mm_http.fetch_bytes("https://8.8.8.8/x.png", 30.0, policy=strict)


@pytest.mark.parametrize("backend_name", ["aiohttp", "httpx"])
async def test_fetch_with_policy_enforces_redirect_limit(
    monkeypatch, backend_name
) -> None:
    monkeypatch.setenv("DYN_HTTP_BACKEND", backend_name)
    client = mm_http.get_default_client()

    # _MAX_REDIRECTS=3 → 4 hops trip the cap.
    chain = {
        "https://example.com/a": "https://example.com/b",
        "https://example.com/b": "https://example.com/c",
        "https://example.com/c": "https://example.com/d",
        "https://example.com/d": "https://example.com/e",
    }

    async def _fake(url, timeout, *, max_bytes=None):
        return None, chain[url]

    with patch.object(client, "_fetch_body_or_redirect", _fake):
        with pytest.raises(UrlValidationError, match="Too many redirects"):
            await mm_http.fetch_bytes("https://example.com/a", 30.0, policy=_PERMISSIVE)


# --- Download cap (collect_capped) ---
#
# The cap exists because moving the diffusion download out of SGLang left its
# media_url_max_file_size_mb behind. It has to hold while the body streams: a
# declared Content-Length is attacker-controlled and absent when chunked.


async def _chunks(*sizes: int):
    for n in sizes:
        yield b"x" * n


async def test_collect_capped_joins_a_body_under_the_limit() -> None:
    assert await base.collect_capped(_chunks(4, 4, 2), "u", 100) == b"x" * 10


async def test_collect_capped_without_a_limit_reads_everything() -> None:
    assert len(await base.collect_capped(_chunks(50, 50), "u", None)) == 100


async def test_collect_capped_refuses_a_body_over_the_limit() -> None:
    with pytest.raises(UrlValidationError, match="download limit"):
        await base.collect_capped(_chunks(60, 60), "u", 100)


async def test_collect_capped_counts_across_short_chunks() -> None:
    """Each chunk is well under the limit; only the running total exceeds it.

    aiohttp's ``read(n)`` returns at most n bytes and in practice returns far
    fewer, so a single capped read would let this body through.
    """
    with pytest.raises(UrlValidationError, match="download limit"):
        await base.collect_capped(_chunks(*([10] * 20)), "u", 100)


async def test_collect_capped_does_not_echo_an_unbounded_url() -> None:
    url = "https://example.com/" + "a" * 200_000
    with pytest.raises(UrlValidationError) as excinfo:
        await base.collect_capped(_chunks(200), url, 100)

    assert len(str(excinfo.value)) < 500


async def test_http_status_error_bounds_both_halves_of_its_message() -> None:
    """The video diffusion handler puts str(exc) straight in its response body,
    and httpx's own status-error text repeats the URL, so neither the url nor
    the backend message can go in at full length."""
    url = "https://example.com/" + "u" * 200_000
    err = base.HttpStatusError(404, "Not Found " + "m" * 200_000, url)

    assert len(str(err)) < 500
    assert err.url == url  # the attribute still carries the full value
