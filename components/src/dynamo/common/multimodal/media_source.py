# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read media bytes from the non-HTTP sources the hardware decoder can use.

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
from typing import Final
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

from dynamo.common.http.url_validator import (
    UrlValidationError,
    UrlValidationPolicy,
    validate_local_path,
)

logger = logging.getLogger(__name__)

# Schemes this module can turn into bytes. http(s) is deliberately absent: it
# belongs to fetch_bytes, which applies SSRF revalidation on every redirect hop.
LOCAL_MEDIA_SCHEMES = frozenset({"file", "data"})


def is_local_media_url(url: str) -> bool:
    """True when :func:`read_local_media_bytes` can produce bytes for ``url``."""
    return urlparse(url).scheme in LOCAL_MEDIA_SCHEMES


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


def _decode_data_uri(url: str) -> bytes:
    """Decode a ``data:`` URI body to bytes.

    Only base64 payloads are accepted: a percent-encoded body would have to be
    re-encoded to bytes by guessing a charset, and media data URIs are base64
    in practice.
    """
    _, _, remainder = url.partition(":")
    meta, sep, payload = remainder.partition(",")
    if not sep:
        raise UrlValidationError("Malformed data URI: missing ',' separator")
    if "base64" not in meta.split(";"):
        raise UrlValidationError("Unsupported data URI: expected base64 payload")
    try:
        return base64.b64decode(unquote(payload), validate=True)
    except (binascii.Error, ValueError) as exc:
        raise UrlValidationError(f"Malformed base64 in data URI: {exc}") from exc


async def read_local_media_bytes(url: str, policy: UrlValidationPolicy) -> bytes:
    """Return the bytes behind a ``file://`` or ``data:`` media URL.

    Raises ``UrlValidationError`` when the scheme is unsupported, when local
    access is disabled or the path escapes ``allowed_local_path``, or when a
    data URI is malformed.
    """
    scheme = urlparse(url).scheme
    if scheme == "data":
        return _decode_data_uri(url)
    if scheme != "file":
        raise UrlValidationError(f"Unsupported local media scheme: {scheme!r}")

    # url2pathname handles the platform's file:// -> path rules; the policy
    # check below is what actually authorizes the read.
    parsed = urlparse(url)
    path = validate_local_path(url2pathname(parsed.path), policy)
    return await asyncio.to_thread(path.read_bytes)
