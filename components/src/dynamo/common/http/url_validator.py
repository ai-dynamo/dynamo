# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""URL / path validation and SSRF-safe HTTP fetching for multimodal loaders.

By default (``UrlValidationPolicy()``), only ``https://`` and ``data:`` URLs
are allowed; private / internal IPs and local filesystem access are both
blocked. Individual loaders can add stricter rules on top — ``ImageLoader``,
for example, refuses every local input regardless of policy.

To loosen the defaults, either build a ``UrlValidationPolicy(...)`` directly
or call ``UrlValidationPolicy.from_env()`` to pick up the ``DYN_MM_*`` vars
below.

"""

import asyncio
import ipaddress
import os
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Final
from urllib.parse import unquote, urlparse


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


# Longest a media source may render as inside an error message or log line.
# Generous enough to keep an ordinary URL intact and identifiable.
SOURCE_LABEL_LIMIT: Final = 120


def describe_media_source(source: str, limit: int = SOURCE_LABEL_LIMIT) -> str:
    """Render ``source`` as a bounded label safe to put in an error or log.

    A ``data:`` URI carries the whole media payload inline, so echoing one into
    an error message serializes megabytes of base64 -- to the client, and to
    every log sink that records the failure. Describe those by media type and
    size instead, never by content. Other sources are truncated, since a URL
    identifies the request without being unbounded.

    Lives here rather than in ``multimodal.media_source`` so the validators
    below can bound their own messages: importing that package pulls in torch.
    """
    if not isinstance(source, str):
        return "<non-string media source>"
    if source.startswith("data:"):
        meta = source[len("data:") :].partition(",")[0]
        media_type = meta.split(";")[0] or "application/octet-stream"
        return f"data:{media_type} ({len(source)} chars, payload elided)"
    if len(source) > limit:
        return f"{source[:limit]}... ({len(source)} chars)"
    return source


def describe_error_detail(detail: str, limit: int = SOURCE_LABEL_LIMIT) -> str:
    """Bound a backend exception's text, keeping both ends.

    Unlike a media source, the useful part of one of these is usually at the
    end: aiohttp renders the client-supplied host *before* the errno, so a
    head-only truncation would keep the attacker's string and drop the
    diagnosis.
    """
    if not isinstance(detail, str):
        return "<non-string error detail>"
    if len(detail) <= limit:
        return detail
    # The marker comes out of the budget, not on top of it: a caller that sizes
    # a buffer by ``limit`` should not be handed something longer.
    marker = f"... ({len(detail)} chars) ..."
    budget = limit - len(marker)
    if budget <= 0:
        return marker
    head = budget // 2
    return f"{detail[:head]}{marker}{detail[-(budget - head):]}"


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


async def validate_url(url: str, policy: UrlValidationPolicy) -> str:
    """Check ``url`` against ``policy`` and return it unchanged if it passes.

    ``https://`` and ``data:`` always pass. ``http://`` needs
    ``allow_http=True``. Anything else is rejected outright.

    For URLs with a hostname, we resolve it here (off the event loop via
    ``loop.getaddrinfo``) and check the resulting IPs against the blocked
    ranges. This catches obvious DNS rebinding but not an attacker who
    changes their answer between this lookup and httpx's actual connect.

    Raises ``UrlValidationError`` on any policy violation.
    """
    if not url:
        raise UrlValidationError("URL is empty")

    parsed = urlparse(url)
    scheme = parsed.scheme.lower()

    # Before the label: describe_media_source copies the source, and a data:
    # URI carries the whole payload inline, so building one for the branch that
    # returns without using it dominates the call (98% of it at 32 MiB).
    if scheme == "data":
        return url

    # Every message below is surfaced to the caller (the diffusion handlers
    # turn it into a 400 body) and logged, so nothing client-supplied goes in
    # at full length.
    label = describe_media_source(url)

    if scheme not in ("http", "https"):
        raise UrlValidationError(
            f"URL scheme '{describe_media_source(scheme)}' not allowed"
        )

    if scheme == "http" and not policy.allow_http:
        raise UrlValidationError(
            "http:// URLs are not allowed; set DYN_MM_ALLOW_INTERNAL=1 to enable"
        )

    host = (parsed.hostname or "").lower()
    if not host:
        raise UrlValidationError(f"URL has no host component: {label!r}")

    if not policy.allow_private_ips and host in _BLOCKED_HOSTS:
        raise UrlValidationError(
            f"Host '{describe_media_source(host)}' is blocked "
            "(resolves to internal service)"
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

    loop = asyncio.get_running_loop()
    try:
        infos = await loop.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise UrlValidationError(
            f"Could not resolve host '{describe_media_source(host)}': {exc}"
        ) from exc
    for info in infos:
        addr = info[4][0]
        if is_blocked_ip(addr):
            raise UrlValidationError(
                f"Host '{describe_media_source(host)}' resolves to blocked IP '{addr}'"
            )
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

    # ``path`` is client-supplied and unbounded: describe_media_source keeps it
    # out of an error response and a log line at full length. Short, ordinary
    # paths render unchanged.
    label = describe_media_source(path)

    try:
        resolved = Path(path).expanduser().resolve(strict=True)
    except FileNotFoundError as exc:
        raise UrlValidationError(f"File not found: {label}") from exc
    except OSError as exc:
        raise UrlValidationError(f"Could not resolve path '{label}': {exc}") from exc
    except ValueError as exc:
        # An embedded NUL makes lstat() raise ValueError, not OSError. Reachable
        # since file:// paths are percent-decoded, so %00 becomes a real NUL.
        # Callers map UrlValidationError to a client error; a bare ValueError
        # would reach them as a server error instead.
        raise UrlValidationError(f"Invalid path '{label}': {exc}") from exc

    try:
        allowed = Path(policy.allowed_local_path).expanduser().resolve(strict=True)
    except FileNotFoundError as exc:
        # Same reason as the message below: callers put this in a client
        # response, and the configured directory is deployment detail.
        raise UrlValidationError(
            "Configured allowed_local_path does not exist"
        ) from exc

    try:
        resolved.relative_to(allowed)
    except ValueError as exc:
        # The configured directory is deployment detail; naming it here puts
        # it in the client's error body, since callers surface this message.
        raise UrlValidationError(
            f"Path '{label}' is outside the allowed directory"
        ) from exc

    return resolved


async def validate_media_url(url: str, policy: UrlValidationPolicy) -> str:
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
        # file:// paths are percent-encoded; a bare path is literal.
        raw_path = unquote(parsed.path) if scheme == "file" else url
        resolved = validate_local_path(raw_path, policy)
        return resolved.as_uri()

    return await validate_url(url, policy)


async def validate_media_reference(reference: str, policy: UrlValidationPolicy) -> str:
    """Like :func:`validate_media_url` but return a plain filesystem path for
    local references instead of a ``file://`` URI, for callers that pass the
    result to a loader expecting a bare path.
    """
    if not reference:
        raise UrlValidationError("Media reference is empty")

    parsed = urlparse(reference)
    if parsed.scheme.lower() in ("", "file"):
        # file:// paths are percent-encoded; a bare path is literal.
        raw_path = unquote(parsed.path) if parsed.scheme else reference
        return str(validate_local_path(raw_path, policy))
    return await validate_url(reference, policy)


_MAX_REDIRECTS = 3
