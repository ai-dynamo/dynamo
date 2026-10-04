# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Turn a client-supplied media URL into bytes, under one policy and one bound.

:func:`load_media_bytes` is the entrypoint for every media fetch: it validates
the URL, then reads it by scheme -- http(s) through ``fetch_bytes`` (SSRF
revalidation on every redirect hop, connect-time address filtering), ``data:``
by decoding, ``file://`` from disk -- applying the caller's size bound to all
three. Callers decode the bytes; none of them reads a source itself.

The non-HTTP readers exist for the hardware decoder:

``fetch_bytes`` only speaks HTTP(S), so ``file://`` and ``data:`` media never
produced bytes at the routing layer and could not reach NVDEC -- they fell
through to the software decoder instead. The codec-compliant runtime images
ship no software video decoder, so that fallback no longer resolves and those
schemes lost video support entirely. This module supplies the missing bytes so
H.264/H.265 from a local file or a data URI decodes on the GPU like any other
source.

``file://`` stays behind ``validate_local_path``: local access is refused
unless ``DYN_MM_LOCAL_PATH`` is set, paths are resolved before the prefix check
(so symlinks cannot escape), and this adds no read surface beyond what the
existing media connectors already allow under the same policy.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import logging
from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

from dynamo.common.http import fetch_bytes
from dynamo.common.http.url_validator import (
    SOURCE_LABEL_LIMIT,
    UrlValidationError,
    UrlValidationPolicy,
    describe_media_source,
    validate_local_path,
    validate_media_url,
)

logger = logging.getLogger(__name__)

# describe_media_source now lives beside the validators, which have to bound a
# client-supplied source in their own messages and cannot import this package
# (it pulls in torch). Re-exported here because callers import it from here.
__all__ = [
    "LOCAL_MEDIA_SCHEMES",
    "SOURCE_LABEL_LIMIT",
    "decode_data_uri",
    "describe_media_source",
    "is_local_media_url",
    "load_media_bytes",
    "read_local_media_bytes",
]

# Schemes this module can turn into bytes. http(s) is deliberately absent: it
# belongs to fetch_bytes, which applies SSRF revalidation on every redirect hop.
LOCAL_MEDIA_SCHEMES = frozenset({"file", "data"})


def is_local_media_url(url: str) -> bool:
    """True when :func:`read_local_media_bytes` can produce bytes for ``url``."""
    return urlparse(url).scheme in LOCAL_MEDIA_SCHEMES


def decode_data_uri(url: str, max_bytes: int | None = None) -> bytes:
    """Decode a ``data:`` URI body to bytes.

    Only base64 payloads are accepted: a percent-encoded body would have to be
    re-encoded to bytes by guessing a charset, and media data URIs are base64
    in practice. ``max_bytes`` bounds the decoded size before anything is
    decoded, as ``fetch_bytes`` bounds a download.
    """
    _, _, remainder = url.partition(":")
    meta, sep, payload = remainder.partition(",")
    if not sep:
        raise UrlValidationError("Malformed data URI: missing ',' separator")
    if "base64" not in meta.split(";"):
        raise UrlValidationError("Unsupported data URI: expected base64 payload")
    if max_bytes is not None:
        # unquote() below copies its whole input, so reject an oversized payload
        # before it runs. The guard has to use a LOWER bound on the decoded size,
        # or it rejects payloads that fit: percent-escaping inflates a payload
        # up to 3x, so the raw length overstates it. Unescaping shrinks text by
        # at most 3x (each `%XY` triplet becomes one character), so the body is
        # at least len(payload) / 3 long, and a body of n characters decodes to
        # at least n // 4 * 3 - 2 bytes. len() is O(1); the exact check below
        # makes the final call.
        if (len(payload) // 3) // 4 * 3 - 2 > max_bytes:
            raise UrlValidationError(
                f"Data URI payload exceeds the maximum allowed size ({max_bytes} bytes)"
            )
    body = unquote(payload)
    if max_bytes is not None:
        padding = min(2, len(body) - len(body.rstrip("=")))
        if len(body) // 4 * 3 - padding > max_bytes:
            raise UrlValidationError(
                f"Data URI payload exceeds the maximum allowed size ({max_bytes} bytes)"
            )
    try:
        return base64.b64decode(body, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise UrlValidationError(f"Malformed base64 in data URI: {exc}") from exc


async def read_local_media_bytes(
    url: str, policy: UrlValidationPolicy, *, max_bytes: int | None = None
) -> bytes:
    """Return the bytes behind a ``file://`` or ``data:`` media URL.

    ``max_bytes`` bounds the result for both schemes, as ``fetch_bytes`` bounds
    a download, so a source cannot get past the limit by being local.

    Raises ``UrlValidationError`` when the scheme is unsupported, when local
    access is disabled or the path escapes ``allowed_local_path``, when a data
    URI is malformed, or when the content exceeds ``max_bytes``.
    """
    scheme = urlparse(url).scheme
    if scheme == "data":
        return decode_data_uri(url, max_bytes=max_bytes)
    if scheme != "file":
        raise UrlValidationError(f"Unsupported local media scheme: {scheme!r}")

    # url2pathname handles the platform's file:// -> path rules; the policy
    # check below is what actually authorizes the read.
    parsed = urlparse(url)
    path = validate_local_path(url2pathname(parsed.path), policy)
    return await asyncio.to_thread(_read_bounded, path, max_bytes)


def _read_bounded(path: Path, max_bytes: int | None) -> bytes:
    # Read one byte past the bound rather than trusting st_size: the file can
    # grow between a stat and the read, and a FIFO or device reports no size.
    with path.open("rb") as fh:
        data = fh.read() if max_bytes is None else fh.read(max_bytes + 1)
    if max_bytes is not None and len(data) > max_bytes:
        raise UrlValidationError(
            f"Media exceeds the {max_bytes} byte read limit: {describe_media_source(path.as_uri())}"
        )
    return data


async def load_media_bytes(
    url: str,
    policy: UrlValidationPolicy,
    *,
    timeout: float,
    max_bytes: int | None,
) -> bytes:
    """Validate a client media URL and return its bytes.

    What is checked, for every scheme ``validate_media_url`` accepts:

    - http(s): the URL and each redirect hop against ``policy`` (scheme, blocked
      hostnames, every resolved address against the blocked ranges), and the
      addresses again at connect time, so a DNS answer that changes between
      check and connect is refused. A proxied fetch is refused unless the
      deployment trusts the proxy to enforce destinations
      (``DYN_MM_TRUST_EGRESS_PROXY``).
    - ``data:``: the encoded length (``DYN_MM_MAX_DATA_URL_MB``), then the
      decoded size against ``max_bytes``.
    - ``file://`` and bare paths: refused unless ``policy`` allows a local
      prefix; the resolved path must stay inside it; the read is bounded.

    ``max_bytes`` bounds the result for every scheme and is enforced while
    reading, never after buffering. It is required so each caller states the
    limit it owns; ``None`` deliberately removes it.

    Raises ``UrlValidationError`` (a ``ValueError``: the client's source was
    refused) or, for http(s), the ``HttpError`` family. Callers map those to
    their own error contract; ``HttpConfigurationError`` is an operator fault
    and must not be reported to the client as a bad request.
    """
    normalized = await validate_media_url(url, policy)
    if urlparse(normalized).scheme in ("http", "https"):
        return await fetch_bytes(
            normalized, timeout, policy=policy, max_bytes=max_bytes
        )
    # validate_media_url returns only http(s), data: or file://.
    return await read_local_media_bytes(normalized, policy, max_bytes=max_bytes)
