# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Materialize a client media reference to a trusted local path.

`validate_media_reference` blocks the *initial* URL, but a validated URL must
still not be handed to a downstream generator that fetches it and follows
redirects on its own — an allowed origin can `302` to an internal address
(redirect SSRF). This fetches URL references through the SSRF-safe client (which
revalidates every redirect hop against the policy) into a temp file and yields
that local path; local references pass through unchanged. Callers hand the
generator only a trusted local path, never a URL.
"""

from __future__ import annotations

import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator
from urllib.parse import urlparse

from .url_validator import (
    UrlValidationError,
    UrlValidationPolicy,
    describe_media_source,
    validate_media_reference,
)

logger = logging.getLogger(__name__)

# A temp-file suffix is derived from the client's URL path, so it has to be
# bounded: mkstemp() raises ENAMETOOLONG past the filesystem's limit, and that
# OSError carries the server's temp directory back to the caller. Long enough
# for any real media extension.
_MAX_SUFFIX_LEN = 16

# Before this module existed, a URL input_reference was downloaded by SGLang's
# own get_image_bytes -> download_remote_media, which streams under
# ``media_url_max_file_size_mb`` (default 64 MiB, sglang 0.5.18
# server_args.py:2804). Handing the generator a local path moves it to the
# open() branch, where that cap no longer applies -- so carry the same default
# here rather than silently dropping the bound.
#
# That knob was operator-tunable as a SGLang server arg, so carry the knob too:
# ``DYN_MM_MAX_FILE_SIZE_MB`` overrides the default, in megabytes, matching both
# the SGLang arg it replaces and trtllm's ``DYN_TRTLLM_MAX_FILE_SIZE_MB``.
DYN_MM_MAX_FILE_SIZE_MB = "DYN_MM_MAX_FILE_SIZE_MB"
DEFAULT_MAX_MEDIA_MB = 64
MAX_MEDIA_BYTES = DEFAULT_MAX_MEDIA_MB * 1024 * 1024

# A parameter default binds at definition time, so it cannot call the resolver
# below -- the env has to be read per call for a worker that is configured after
# import and for tests that monkeypatch it. This sentinel means "resolve it".
_FROM_ENV = -1


def max_media_bytes() -> int:
    """Media download cap in bytes, from ``DYN_MM_MAX_FILE_SIZE_MB``.

    Mirrors ``media_decoder._video_num_frames``: read at call time, and an
    empty, unparseable or non-positive value falls back to the default with a
    warning rather than raising. A malformed operator value must not take the
    worker down, and must not silently remove the bound either.
    """
    raw = os.getenv(DYN_MM_MAX_FILE_SIZE_MB, "").strip()
    if not raw:
        return MAX_MEDIA_BYTES
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "Ignoring invalid %s=%r; using %s MB",
            DYN_MM_MAX_FILE_SIZE_MB,
            raw,
            DEFAULT_MAX_MEDIA_MB,
        )
        return MAX_MEDIA_BYTES
    if value <= 0:
        logger.warning(
            "Ignoring non-positive %s=%r; using %s MB",
            DYN_MM_MAX_FILE_SIZE_MB,
            raw,
            DEFAULT_MAX_MEDIA_MB,
        )
        return MAX_MEDIA_BYTES
    return value * 1024 * 1024


def _temp_suffix(url: str) -> str:
    """Extension to give the temp file, bounded and stripped of path separators.

    Only a plausible extension survives: the generator picks a decoder from it,
    and everything else in a client-supplied path is noise at best.
    """
    suffix = Path(urlparse(url).path).suffix
    if len(suffix) > _MAX_SUFFIX_LEN or not suffix[1:].isalnum():
        return ""
    return suffix


@asynccontextmanager
async def local_media_reference(
    reference: str,
    policy: UrlValidationPolicy,
    *,
    timeout: float = 30.0,
    max_bytes: int | None = _FROM_ENV,
) -> AsyncIterator[str]:
    """Yield a trusted local filesystem path for ``reference``.

    URL references are fetched through the policy-aware client (per-hop redirect
    revalidation) into a temp file that is removed on exit. Local references are
    validated by ``validate_media_reference`` and yielded as-is.

    ``max_bytes`` bounds the download while it is read; exceeding it raises
    ``UrlValidationError``, as an oversized client-chosen source is a bad
    request rather than a server fault. Left unset it resolves from
    ``DYN_MM_MAX_FILE_SIZE_MB`` per call; ``None`` disables the bound.

    Raises ``UrlValidationError`` for anything that is not a local path or an
    http(s) URL — notably ``data:``, which ``validate_url`` allows but which is
    a URI, not a path, and would reach the generator as one.
    """
    if max_bytes == _FROM_ENV:
        max_bytes = max_media_bytes()

    resolved = await validate_media_reference(reference, policy)
    scheme = urlparse(resolved).scheme
    if scheme in ("http", "https"):
        # Imported here, not at module scope, so tests can monkeypatch
        # ``dynamo.common.http.fetch_bytes`` and see the patched function.
        from . import fetch_bytes

        data = await fetch_bytes(resolved, timeout, policy=policy, max_bytes=max_bytes)
        fd, tmp = tempfile.mkstemp(suffix=_temp_suffix(resolved))
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
            yield tmp
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass
    elif scheme:
        raise UrlValidationError(
            f"Media reference scheme '{scheme}' cannot be resolved to a local "
            f"file: {describe_media_source(reference)}"
        )
    else:
        yield resolved
