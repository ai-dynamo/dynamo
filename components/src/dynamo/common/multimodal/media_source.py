# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Turn a client-supplied media URL into bytes under one policy and one bound.

Backends call :func:`load_media_bytes` and decode the result. The non-HTTP
readers also let H.264/H.265 from a local file or a data URI reach NVDEC.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import logging
import os
import stat
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

# describe_media_source lives beside the validators, which cannot import this
# package (it pulls in torch); re-exported because callers import it from here.
__all__ = [
    "SOURCE_LABEL_LIMIT",
    "decode_data_uri",
    "describe_media_source",
    "load_media_bytes",
]


def decode_data_uri(url: str, *, max_bytes: int | None) -> bytes:
    """Decode a ``data:`` URI body to bytes.

    Only base64 payloads are accepted: a percent-encoded body would have to be
    re-encoded to bytes by guessing a charset, and media data URIs are base64
    in practice. ``max_bytes`` bounds the decoded size before anything is
    decoded; it is required so a caller states it (``None`` for no bound).
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


async def load_media_bytes(
    url: str,
    policy: UrlValidationPolicy,
    *,
    timeout: float,
    max_bytes: int,
) -> bytes:
    """Validate a client media URL against ``policy`` and return its bytes.

    http(s) goes through ``fetch_bytes`` (per-hop revalidation and connect-time
    address filtering); ``data:`` is decoded; ``file://`` is read only under the
    policy's local prefix and only if it is a regular file. ``max_bytes``
    bounds every scheme. All arguments are required, so no caller can fall back
    to an unchecked or unbounded read by leaving one out.

    Raises ``UrlValidationError`` (a ``ValueError``: the source was refused) or,
    for http(s), the ``HttpError`` family -- of which ``HttpConfigurationError``
    is an operator fault, not the caller's.
    """
    normalized = await validate_media_url(url, policy)
    scheme = urlparse(normalized).scheme
    if scheme in ("http", "https"):
        return await fetch_bytes(
            normalized, timeout, policy=policy, max_bytes=max_bytes
        )
    if scheme == "data":
        return decode_data_uri(normalized, max_bytes=max_bytes)
    # validate_media_url returns only http(s), data: or file://.
    path = validate_local_path(url2pathname(urlparse(normalized).path), policy)
    return await asyncio.to_thread(_read_bounded, path, max_bytes, url)


def _read_bounded(path: Path, max_bytes: int, url: str) -> bytes:
    # Errors name the client's URL, never the resolved path: the allowed
    # directory is deployment detail.
    label = describe_media_source(url)
    try:
        # O_NONBLOCK so a FIFO cannot hang the open; fstat on the opened
        # descriptor so the check and the read see the same file.
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    except OSError as exc:
        raise UrlValidationError(
            f"Media could not be read ({exc.strerror}): {label}"
        ) from exc
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise UrlValidationError(f"Media is not a regular file: {label}")
    with os.fdopen(fd, "rb") as fh:
        # Read one byte past the bound rather than trusting st_size.
        data = fh.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise UrlValidationError(
            f"Media exceeds the {max_bytes} byte read limit: {label}"
        )
    return data
