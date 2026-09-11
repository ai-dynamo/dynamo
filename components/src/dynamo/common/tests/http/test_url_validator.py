# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``dynamo.common.http.url_validator``.

These cover scheme / IP / hostname / path / redirect logic in isolation of
the media loaders, so they run quickly with no network and no vLLM imports.
"""

from __future__ import annotations

import ipaddress
import re
import socket
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from dynamo.common.http import url_validator
from dynamo.common.http.url_validator import (
    UrlValidationError,
    UrlValidationPolicy,
    describe_error_detail,
    is_blocked_ip,
    validate_local_path,
    validate_media_reference,
    validate_media_url,
    validate_url,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
]


STRICT_HTTPS = UrlValidationPolicy()

PERMISSIVE = UrlValidationPolicy(
    allow_http=True,
    allow_private_ips=True,
)

_RUST_STRING_LITERAL = re.compile(r'"([^"\\]+)"')


def _find_rust_media_loader_source() -> Path:
    for parent in Path(__file__).resolve().parents:
        candidate = (
            parent / "lib" / "llm" / "src" / "preprocessor" / "media" / "loader.rs"
        )
        if candidate.is_file():
            return candidate
    raise AssertionError("Could not locate the Rust media loader source")


def _rust_static_strings(source: str, name: str) -> set[str]:
    marker = f"static {name}:"
    declaration = source.find(marker)
    assert declaration >= 0, f"Rust {name} declaration not found"

    initializer = source.find("LazyLock::new(|| {", declaration)
    assert initializer >= 0, f"Rust {name} LazyLock initializer not found"
    array_start = source.find("[", initializer)
    array_end = source.find("]", array_start)
    assert (
        array_start >= 0 and array_end >= 0
    ), f"Rust {name} initializer is no longer a string array"

    values = set(_RUST_STRING_LITERAL.findall(source[array_start:array_end]))
    assert values, f"Rust {name} contains no string literals"
    return values


# ---------------------------------------------------------------------------
# is_blocked_ip()
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ip",
    [
        "127.0.0.1",
        "10.0.0.1",
        "172.16.5.5",
        "192.168.1.1",
        "169.254.169.254",  # AWS metadata
        "100.64.0.1",  # CGNAT
        "0.0.0.0",
        "::1",
        "fe80::1",
        "fc00::1",
        "240.0.0.1",  # reserved
    ],
)
def test_is_blocked_ip_blocks_known_ranges(ip: str) -> None:
    assert is_blocked_ip(ip) is True


@pytest.mark.parametrize(
    "ip",
    [
        "8.8.8.8",
        "1.1.1.1",
        "93.184.216.34",  # example.com
        "2606:4700:4700::1111",  # Cloudflare
    ],
)
def test_is_blocked_ip_allows_public(ip: str) -> None:
    assert is_blocked_ip(ip) is False


def test_is_blocked_ip_non_ip_literal_returns_false() -> None:
    # A hostname, not an IP — is_blocked_ip only classifies literals.
    assert is_blocked_ip("example.com") is False


def test_python_and_rust_policy_constants_match() -> None:
    """Test that the production blocklists are synchronized."""
    rust_source = _find_rust_media_loader_source().read_text(encoding="utf-8")
    rust_networks = {
        ipaddress.ip_network(network)
        for network in _rust_static_strings(rust_source, "BLOCKED_IP_NETWORKS")
    }
    rust_hosts = _rust_static_strings(rust_source, "BLOCKED_HOSTS")

    assert set(url_validator._BLOCKED_IP_NETWORKS) == rust_networks
    assert set(url_validator._BLOCKED_HOSTS) == rust_hosts


# ---------------------------------------------------------------------------
# validate_url() — scheme handling
# ---------------------------------------------------------------------------


async def test_validate_url_rejects_empty() -> None:
    with pytest.raises(UrlValidationError, match="empty"):
        await validate_url("", STRICT_HTTPS)


async def test_validate_url_rejects_http_by_default() -> None:
    with pytest.raises(UrlValidationError, match="not allowed"):
        await validate_url("http://example.com/x.png", STRICT_HTTPS)


async def test_validate_url_rejects_file_scheme() -> None:
    with pytest.raises(UrlValidationError, match="scheme 'file'"):
        await validate_url("file:///etc/passwd", STRICT_HTTPS)


async def test_validate_url_rejects_ftp() -> None:
    with pytest.raises(UrlValidationError, match="scheme 'ftp'"):
        await validate_url("ftp://example.com/x.png", STRICT_HTTPS)


async def test_validate_url_accepts_data_url_by_default() -> None:
    # data: URLs never touch the network — we allow them without further checks.
    await validate_url("data:image/png;base64,iVBORw0KGgoAAAA=", STRICT_HTTPS)


async def test_validate_url_http_allowed_when_opted_in() -> None:
    policy = PERMISSIVE
    # With public hostname + allow_private_ips=True (keeps DNS out of the test)
    await validate_url("http://example.com/x.png", policy)


# ---------------------------------------------------------------------------
# validate_url() — IP literal handling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "https://127.0.0.1/x.png",
        "https://169.254.169.254/latest/meta-data/",
        "https://10.0.0.5/x.png",
        "https://192.168.1.1/x.png",
        "https://[::1]/x.png",
    ],
)
async def test_validate_url_rejects_private_ip_literal(url: str) -> None:
    with pytest.raises(UrlValidationError, match="blocked range"):
        await validate_url(url, STRICT_HTTPS)


async def test_validate_url_allows_public_ip_literal() -> None:
    # Public IP literal and allow_private_ips=False — should pass the IP test
    # and skip DNS resolution (the IP path short-circuits).
    await validate_url("https://8.8.8.8/x.png", STRICT_HTTPS)


async def test_validate_url_allows_private_ip_when_opted_in() -> None:
    await validate_url("https://127.0.0.1/x.png", PERMISSIVE)


# ---------------------------------------------------------------------------
# validate_url() — blocked hostnames
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "host",
    [
        "localhost",
        "metadata.google.internal",
        "metadata",
        "kubernetes.default.svc",
    ],
)
async def test_validate_url_rejects_blocked_hostname(host: str) -> None:
    with pytest.raises(UrlValidationError, match="blocked"):
        await validate_url(f"https://{host}/path", STRICT_HTTPS)


# ---------------------------------------------------------------------------
# validate_url() — DNS resolution
# ---------------------------------------------------------------------------


def _fake_getaddrinfo(addrs: list[str]):
    def _impl(host: str, *_args: Any, **_kwargs: Any):
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", (addr, 0)) for addr in addrs
        ]

    return _impl


async def test_validate_url_rejects_host_resolving_to_private_ip() -> None:
    with patch(
        "dynamo.common.http.url_validator.socket.getaddrinfo",
        side_effect=_fake_getaddrinfo(["10.0.0.5"]),
    ):
        with pytest.raises(UrlValidationError, match="blocked IP"):
            await validate_url("https://attacker.example.com/x.png", STRICT_HTTPS)


async def test_validate_url_rejects_host_if_any_ip_is_private() -> None:
    # Even if the host resolves to a public IP too, any blocked IP is fatal.
    with patch(
        "dynamo.common.http.url_validator.socket.getaddrinfo",
        side_effect=_fake_getaddrinfo(["8.8.8.8", "169.254.169.254"]),
    ):
        with pytest.raises(UrlValidationError, match="169.254.169.254"):
            await validate_url("https://mixed.example.com/x.png", STRICT_HTTPS)


async def test_validate_url_accepts_public_host() -> None:
    with patch(
        "dynamo.common.http.url_validator.socket.getaddrinfo",
        side_effect=_fake_getaddrinfo(["93.184.216.34"]),
    ):
        await validate_url("https://example.com/x.png", STRICT_HTTPS)


async def test_validate_url_resolution_failure_raises() -> None:
    with patch(
        "dynamo.common.http.url_validator.socket.getaddrinfo",
        side_effect=socket.gaierror("nodename nor servname provided"),
    ):
        with pytest.raises(UrlValidationError, match="Could not resolve"):
            await validate_url("https://does-not-exist.invalid/x.png", STRICT_HTTPS)


async def test_validate_url_skips_resolution_when_private_allowed() -> None:
    # In developer mode we short-circuit DNS to keep tests deterministic.
    with patch("dynamo.common.http.url_validator.socket.getaddrinfo") as resolver:
        await validate_url("https://example.com/x.png", PERMISSIVE)
        resolver.assert_not_called()


# ---------------------------------------------------------------------------
# validate_local_path()
# ---------------------------------------------------------------------------


def test_validate_local_path_rejected_when_disabled() -> None:
    with pytest.raises(UrlValidationError, match="not permitted"):
        validate_local_path("/etc/passwd", STRICT_HTTPS)


def test_validate_local_path_accepts_inside_prefix(tmp_path) -> None:
    media = tmp_path / "media"
    media.mkdir()
    target = media / "sample.png"
    target.write_bytes(b"\x89PNG\r\n")

    policy = UrlValidationPolicy(allowed_local_path=str(media))
    resolved = validate_local_path(str(target), policy)
    assert resolved == target.resolve()


def test_validate_local_path_rejects_outside_prefix(tmp_path) -> None:
    media = tmp_path / "media"
    media.mkdir()
    other = tmp_path / "secrets"
    other.mkdir()
    secret = other / "creds.txt"
    secret.write_text("hunter2")

    policy = UrlValidationPolicy(allowed_local_path=str(media))
    with pytest.raises(UrlValidationError, match="outside the allowed directory"):
        validate_local_path(str(secret), policy)


def test_validate_local_path_rejects_symlink_escape(tmp_path) -> None:
    media = tmp_path / "media"
    media.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    link = media / "link.png"
    link.symlink_to(outside)

    policy = UrlValidationPolicy(allowed_local_path=str(media))
    # Path.resolve() follows the symlink; the target is outside the prefix.
    with pytest.raises(UrlValidationError, match="outside the allowed directory"):
        validate_local_path(str(link), policy)


def test_validate_local_path_missing_file(tmp_path) -> None:
    policy = UrlValidationPolicy(allowed_local_path=str(tmp_path))
    with pytest.raises(UrlValidationError, match="File not found"):
        validate_local_path(str(tmp_path / "nope.png"), policy)


def test_validate_local_path_missing_prefix(tmp_path) -> None:
    target = tmp_path / "sample.png"
    target.write_bytes(b"x")
    policy = UrlValidationPolicy(allowed_local_path=str(tmp_path / "does-not-exist"))
    with pytest.raises(UrlValidationError, match="allowed_local_path does not exist"):
        validate_local_path(str(target), policy)


# ---------------------------------------------------------------------------
# UrlValidationPolicy.from_env()
# ---------------------------------------------------------------------------


def test_policy_from_env_defaults(monkeypatch) -> None:
    monkeypatch.delenv("DYN_MM_ALLOW_INTERNAL", raising=False)
    monkeypatch.delenv("DYN_MM_LOCAL_PATH", raising=False)

    policy = UrlValidationPolicy.from_env()
    assert policy.allow_http is False
    assert policy.allow_private_ips is False
    assert policy.allowed_local_path is None


def test_policy_from_env_allow_internal(monkeypatch) -> None:
    monkeypatch.setenv("DYN_MM_ALLOW_INTERNAL", "1")
    monkeypatch.setenv("DYN_MM_LOCAL_PATH", "/data/media")

    policy = UrlValidationPolicy.from_env()
    assert policy.allow_http is True
    assert policy.allow_private_ips is True
    assert policy.allowed_local_path == "/data/media"


# Fetch-with-revalidation tests now live in test_http_backends.py where
# they exercise the backend-neutral facade path against both httpx and
# aiohttp. See ``test_fetch_with_policy_*``.


# ---------------------------------------------------------------------------
# validate_media_url() — high-level facade used by media loaders
#
# Exercised here once with a generic file extension; the per-loader test files
# only keep a single wiring smoke test confirming the loader plumbs its
# url_policy through to this function.
# ---------------------------------------------------------------------------


async def test_validate_media_url_converts_local_paths(tmp_path) -> None:
    media_path = tmp_path / "sample.bin"
    media_path.write_bytes(b"data")

    policy = UrlValidationPolicy(allowed_local_path=str(tmp_path))

    assert (
        await validate_media_url(str(media_path), policy)
        == media_path.resolve().as_uri()
    )


async def test_validate_media_url_preserves_data_urls() -> None:
    data_url = "data:application/octet-stream;base64,Zm9v"
    policy = UrlValidationPolicy()
    assert await validate_media_url(data_url, policy) == data_url


async def test_validate_media_url_preserves_http_urls() -> None:
    url = "https://example.com/sample.bin"
    assert await validate_media_url(url, PERMISSIVE) == url


async def test_validate_media_url_rejects_bare_path_by_default(tmp_path) -> None:
    media_path = tmp_path / "sample.bin"
    media_path.write_bytes(b"data")

    policy = UrlValidationPolicy()

    with pytest.raises(UrlValidationError, match="Local media paths are not permitted"):
        await validate_media_url(str(media_path), policy)


async def test_validate_media_url_rejects_private_ip() -> None:
    policy = UrlValidationPolicy()

    with pytest.raises(UrlValidationError):
        await validate_media_url("https://169.254.169.254/sample.bin", policy)


async def test_validate_media_url_accepts_file_uri_inside_prefix(tmp_path) -> None:
    media_path = tmp_path / "sample.bin"
    media_path.write_bytes(b"data")
    policy = UrlValidationPolicy(allowed_local_path=str(tmp_path))

    file_uri = media_path.resolve().as_uri()
    assert await validate_media_url(file_uri, policy) == file_uri


async def test_validate_media_url_rejects_file_uri_outside_prefix(tmp_path) -> None:
    allowed = tmp_path / "media"
    allowed.mkdir()
    other = tmp_path / "secret.bin"
    other.write_bytes(b"data")
    policy = UrlValidationPolicy(allowed_local_path=str(allowed))

    with pytest.raises(UrlValidationError, match="outside the allowed directory"):
        await validate_media_url(other.resolve().as_uri(), policy)


async def test_validate_media_url_rejects_a_percent_encoded_nul(tmp_path) -> None:
    """%00 must stay a UrlValidationError, not leak out as a bare ValueError.

    file:// paths are percent-decoded, so %00 reaches Path.resolve() as a real
    NUL and lstat() raises ValueError rather than OSError. Callers key their
    4xx-vs-5xx decision on the UrlValidationError type (see image_loader and
    video_loader), so letting a plain ValueError escape turns a rejected input
    into a server error.
    """
    policy = UrlValidationPolicy(allowed_local_path=str(tmp_path))

    with pytest.raises(UrlValidationError):
        await validate_media_url(f"file://{tmp_path}/x%00.png", policy)


def test_validate_local_path_bounds_the_path_in_its_message(tmp_path) -> None:
    """The rejected path is client-supplied and unbounded; the message is not.

    It lands in an error response and in a log line, so one request would
    otherwise amplify into that much text at every sink.
    """
    policy = UrlValidationPolicy(allowed_local_path=str(tmp_path))
    path = "/nope/" + "A" * 200_000 + ".png"

    with pytest.raises(UrlValidationError) as excinfo:
        validate_local_path(path, policy)

    assert len(str(excinfo.value)) < 500
    assert "200010 chars" in str(excinfo.value)  # true size stays visible


def test_validate_local_path_keeps_an_ordinary_path_intact(tmp_path) -> None:
    """Control: bounding must not change the message for a normal path."""
    policy = UrlValidationPolicy(allowed_local_path=str(tmp_path))

    with pytest.raises(UrlValidationError, match=r"Path '/etc/passwd' is outside"):
        validate_local_path("/etc/passwd", policy)


async def test_validate_media_reference_rejects_empty(tmp_path) -> None:
    """Parity with validate_media_url, which guards the empty string."""
    policy = UrlValidationPolicy(allowed_local_path=str(tmp_path))

    with pytest.raises(UrlValidationError, match="empty"):
        await validate_media_reference("", policy)


@pytest.mark.parametrize(
    "url",
    [
        "https:///" + "A" * 200_000,  # no host component
        "A" * 200_000 + "://x",  # scheme is client-supplied too
    ],
)
async def test_validate_url_bounds_the_url_in_its_message(url) -> None:
    """These messages became client-visible once the diffusion handlers
    started mapping UrlValidationError to a 400 that carries the reason."""
    with pytest.raises(UrlValidationError) as excinfo:
        await validate_url(url, STRICT_HTTPS)

    assert len(str(excinfo.value)) < 500


async def test_redirect_chain_in_the_limit_message_is_bounded() -> None:
    """The chain is entirely attacker-chosen URLs."""
    from dynamo.common.http.base import HttpClient

    long_hop = "https://example.com/" + "A" * 200_000

    class _Client(HttpClient):
        async def _fetch_simple(self, url, timeout, *, max_bytes=None):
            raise AssertionError("unused")

        async def _fetch_body_or_redirect(self, url, timeout, *, max_bytes=None):
            return None, long_hop

        async def close(self):
            return None

    policy = UrlValidationPolicy(allow_http=True, allow_private_ips=True)
    with pytest.raises(UrlValidationError, match="Too many redirects") as excinfo:
        await _Client().fetch_bytes(long_hop, 1.0, policy=policy)

    assert len(str(excinfo.value)) < 1000


async def test_a_data_uri_does_not_pay_for_a_label_it_cannot_use(monkeypatch) -> None:
    """describe_media_source copies the source, and a data: URI is the payload.

    The data branch returns before any message is built, so building the label
    first made validate_url O(payload): 1.35 ms for a 32 MiB URI against
    0.03 ms after.
    """
    calls = []
    monkeypatch.setattr(
        url_validator,
        "describe_media_source",
        lambda src, *a, **kw: calls.append(src) or src,
    )

    url = "data:image/png;base64,AAAA"
    assert await url_validator.validate_url(url, STRICT_HTTPS) == url
    assert calls == []


def test_missing_allowed_dir_does_not_name_it(tmp_path) -> None:
    """Same disclosure as the 'outside the allowed directory' message below it:
    this one reaches the client too, on an ordinary misconfiguration."""
    policy = UrlValidationPolicy(allowed_local_path=str(tmp_path / "gone"))
    # The input itself must resolve, or "File not found" fires first.
    media = tmp_path / "x.png"
    media.write_bytes(b"x")

    with pytest.raises(UrlValidationError, match="does not exist") as excinfo:
        validate_local_path(str(media), policy)

    assert "gone" not in str(excinfo.value)


def test_describe_error_detail_keeps_a_short_detail_intact() -> None:
    assert describe_error_detail("[Errno 8] not known") == "[Errno 8] not known"


def test_describe_error_detail_bounds_from_both_ends() -> None:
    detail = "head-of-the-message " + "x" * 40_000 + " tail-of-the-message"
    label = describe_error_detail(detail)

    assert len(label) < 300
    assert label.startswith("head-of-the-message")
    assert label.endswith("tail-of-the-message")
    assert str(len(detail)) in label  # true size stays visible
