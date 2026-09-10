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

import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator
from urllib.parse import urlparse

from .url_validator import UrlValidationPolicy, validate_media_reference


@asynccontextmanager
async def local_media_reference(
    reference: str, policy: UrlValidationPolicy, *, timeout: float = 30.0
) -> AsyncIterator[str]:
    """Yield a trusted local filesystem path for ``reference``.

    URL references are fetched through the policy-aware client (per-hop redirect
    revalidation) into a temp file that is removed on exit. Local references are
    validated by ``validate_media_reference`` and yielded as-is.
    """
    resolved = await validate_media_reference(reference, policy)
    if urlparse(resolved).scheme in ("http", "https"):
        # Lazy import: the http package __init__ pulls in the client stack.
        from . import fetch_bytes

        data = await fetch_bytes(resolved, timeout, policy=policy)
        suffix = Path(urlparse(resolved).path).suffix
        fd, tmp = tempfile.mkstemp(suffix=suffix)
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
            yield tmp
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass
    else:
        yield resolved
