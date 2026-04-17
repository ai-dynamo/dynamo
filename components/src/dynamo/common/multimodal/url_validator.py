# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""URL / path validation and SSRF-safe HTTP fetching for multimodal loaders.

By default (``UrlValidationPolicy()``), only ``https://`` and ``data:`` URLs
are allowed; private / internal IPs and local filesystem access are both
blocked. Individual loaders can add stricter rules on top — ``ImageLoader``,
for example, refuses every local input regardless of policy.

To loosen the defaults, either build a ``UrlValidationPolicy(...)`` directly
or call ``UrlValidationPolicy.from_env()`` to pick up the ``DYN_MM_*`` vars
below.

Environment variables
---------------------
``DYN_MM_ALLOW_INTERNAL`` (``1``/``0``, default ``0``)
    Allow ``http://`` and private-network / internal IP targets. Intended
    for on-prem or local-dev setups where media lives on an internal
    network. **Never** set this on anything public-facing.
``DYN_MM_LOCAL_PATH`` (absolute directory path)
    Let opted-in callers load ``file://`` and bare filesystem paths inside
    this prefix. Empty (the default) blocks local access entirely.
"""

import ipaddress
import os
import socket
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import httpx


class UrlValidationError(ValueError):
    """Raised when a URL or filesystem path fails the configured policy."""


# IP ranges that must never be reachable from a user-controlled URL.
# Source: RFC1918 (private), RFC6598 (CGNAT), RFC5735 (loopback, link-local,
# 0.0.0.0/8), RFC4193 (ULA), RFC4291 (IPv6 loopback / link-local), RFC6890
# (reserved). Link-local 169.254/16 covers the AWS / OpenStack metadata IP.
_BLOCKED_IP_NETWORKS: tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...] = (
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("100.64.0.0/10"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.0.0.0/24"),
    ipaddress.ip_network("192.0.2.0/24"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("198.18.0.0/15"),
    ipaddress.ip_network("198.51.100.0/24"),
    ipaddress.ip_network("203.0.113.0/24"),
    ipaddress.ip_network("224.0.0.0/4"),
    ipaddress.ip_network("240.0.0.0/4"),
    ipaddress.ip_network("255.255.255.255/32"),
    ipaddress.ip_network("::/128"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("::ffff:0:0/96"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
    ipaddress.ip_network("ff00::/8"),
)

# Hostnames that resolve to cloud metadata / internal services regardless of
# DNS records. Matched case-insensitively.
_BLOCKED_HOSTS: frozenset[str] = frozenset(
    {
        "localhost",
        "localhost.localdomain",
        "ip6-localhost",
        "ip6-loopback",
        "metadata",
        "metadata.google.internal",
        "metadata.goog",
        "kubernetes.default",
        "kubernetes.default.svc",
    }
)

def is_blocked_ip(ip_text: str) -> bool:
    """Return True if ``ip_text`` parses as an IP inside one of the blocked ranges."""
    try:
        ip = ipaddress.ip_address(ip_text)
    except ValueError:
        return False
    return any(ip in net for net in _BLOCKED_IP_NETWORKS)


@dataclass(frozen=True)
class UrlValidationPolicy:
    """Frozen policy describing which media URLs and local paths are allowed."""

    allow_http: bool = False
    allow_private_ips: bool = False
    allowed_local_path: str | None = None

    @classmethod
    def from_env(cls) -> "UrlValidationPolicy":
        """Build a policy by reading the ``DYN_MM_*`` environment variables."""
        allow_internal = os.getenv("DYN_MM_ALLOW_INTERNAL", "0") == "1"
        return cls(
            allow_http=allow_internal,
            allow_private_ips=allow_internal,
            allowed_local_path=os.getenv("DYN_MM_LOCAL_PATH", "").strip() or None,
        )


def validate_url(url: str, policy: UrlValidationPolicy) -> str:
    """Check ``url`` against ``policy`` and return it unchanged if it passes.

    ``https://`` and ``data:`` always pass. ``http://`` needs
    ``allow_http=True``. Anything else is rejected outright.

    For URLs with a hostname, we resolve it here and check the resulting
    IPs against the blocked ranges. This catches obvious DNS rebinding but
    not an attacker who changes their answer between this lookup and
    httpx's actual connect — real per-connection IP pinning lives in the
    Rust loader (issue #14).

    Raises ``UrlValidationError`` on any policy violation.
    """
    if not url:
        raise UrlValidationError("URL is empty")

    parsed = urlparse(url)
    scheme = parsed.scheme.lower()

    if scheme == "data":
        return url

    if scheme not in ("http", "https"):
        raise UrlValidationError(f"URL scheme '{scheme}' not allowed")

    if scheme == "http" and not policy.allow_http:
        raise UrlValidationError(
            "http:// URLs are not allowed; set DYN_MM_ALLOW_INTERNAL=1 to enable"
        )

    host = (parsed.hostname or "").lower()
    if not host:
        raise UrlValidationError(f"URL has no host component: {url!r}")

    if not policy.allow_private_ips and host in _BLOCKED_HOSTS:
        raise UrlValidationError(
            f"Host '{host}' is blocked (resolves to internal service)"
        )

    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        if not policy.allow_private_ips and is_blocked_ip(host):
            raise UrlValidationError(f"IP literal '{host}' is in a blocked range")
        return url

    if policy.allow_private_ips:
        return url

    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise UrlValidationError(f"Could not resolve host '{host}': {exc}") from exc
    for info in infos:
        addr = info[4][0]
        if is_blocked_ip(addr):
            raise UrlValidationError(f"Host '{host}' resolves to blocked IP '{addr}'")
    return url


def validate_local_path(path: str, policy: UrlValidationPolicy) -> Path:
    """Resolve ``path`` and confirm it sits inside ``allowed_local_path``.

    We call ``Path.resolve()`` first, so symlinks that point outside the
    allowed prefix are caught. Local access is refused outright when
    ``allowed_local_path`` is unset (the default).

    Raises ``UrlValidationError`` if the feature is off or the resolved
    path escapes the prefix.
    """
    if not policy.allowed_local_path:
        raise UrlValidationError(
            "Local media paths are not permitted; set " "DYN_MM_LOCAL_PATH to enable"
        )

    try:
        resolved = Path(path).expanduser().resolve(strict=True)
    except FileNotFoundError as exc:
        raise UrlValidationError(f"File not found: {path}") from exc
    except OSError as exc:
        raise UrlValidationError(f"Could not resolve path '{path}': {exc}") from exc

    try:
        allowed = Path(policy.allowed_local_path).expanduser().resolve(strict=True)
    except FileNotFoundError as exc:
        raise UrlValidationError(
            f"Configured allowed_local_path does not exist: {policy.allowed_local_path}"
        ) from exc

    try:
        resolved.relative_to(allowed)
    except ValueError as exc:
        raise UrlValidationError(
            f"Path '{path}' is outside the allowed directory '{policy.allowed_local_path}'"
        ) from exc

    return resolved


def validate_media_url(url: str, policy: UrlValidationPolicy) -> str:
    """Validate any media input and return a canonical URL string.

    Bare filesystem paths and ``file://`` URIs go through
    ``validate_local_path`` and come back as a resolved ``file://`` URI.
    Everything else goes through ``validate_url`` and is returned
    unchanged. Callers can still reject the result afterwards —
    ``ImageLoader``, for example, refuses local files regardless.

    Raises ``UrlValidationError`` on any policy violation.
    """
    if not url:
        raise UrlValidationError("URL is empty")

    parsed = urlparse(url)
    scheme = parsed.scheme.lower()

    if scheme in ("", "file"):
        raw_path = parsed.path if scheme == "file" else url
        resolved = validate_local_path(raw_path, policy)
        return resolved.as_uri()

    return validate_url(url, policy)


_MAX_REDIRECTS = 3


async def fetch_with_revalidation(
    client: httpx.AsyncClient,
    url: str,
    policy: UrlValidationPolicy,
) -> httpx.Response:
    """Safely fetch a URL while checking security policy at every redirect.

    Only ``_MAX_REDIRECTS`` hops allowed. ``client`` must have
    ``follow_redirects=False`` (the default from ``get_http_client``).
    We follow redirects ourselves and validate each ``Location`` header
    against the policy first.

    Only plain ``GET`` with no custom headers is supported. httpx normally
    strips credentials on cross-origin redirects only when
    ``follow_redirects=True``.

    Raises ``UrlValidationError`` on any policy violation or when the
    redirect chain exceeds ``_MAX_REDIRECTS``.
    """
    current_url = url
    hops_remaining = _MAX_REDIRECTS
    visited: list[str] = []

    while True:
        validate_url(current_url, policy)
        visited.append(current_url)

        request = client.build_request("GET", current_url)
        response = await client.send(request, follow_redirects=False)

        if not response.is_redirect:
            return response

        location = response.headers.get("location")
        if not location:
            return response

        if hops_remaining <= 0:
            await response.aclose()
            raise UrlValidationError(
                f"Too many redirects (max={_MAX_REDIRECTS}); chain={visited}"
            )
        hops_remaining -= 1

        next_url = str(response.url.join(location))
        await response.aclose()
        current_url = next_url
