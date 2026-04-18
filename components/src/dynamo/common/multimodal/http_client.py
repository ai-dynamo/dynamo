# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# The shared client has ``follow_redirects=False`` to prevent redirect-based
# SSRF filter bypass. Callers must follow redirects manually via
# :func:`dynamo.common.multimodal.url_validator.fetch_with_revalidation` so
# that each hop is re-validated against the SSRF policy.
_global_http_client: Optional[httpx.AsyncClient] = None


def get_http_client(timeout: float = 60.0) -> httpx.AsyncClient:
    """Return a shared async HTTP client for media fetches.

    The client intentionally disables automatic redirect following. Callers
    that need to follow redirects must route the request through
    :func:`fetch_with_revalidation`, which revalidates every redirect hop
    against the SSRF policy.
    """
    global _global_http_client

    if _global_http_client is None or _global_http_client.is_closed:
        _global_http_client = httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=False,
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
        )
        logger.info(
            "Shared HTTP client initialized (timeout=%ss, follow_redirects=False)",
            timeout,
        )

    return _global_http_client
