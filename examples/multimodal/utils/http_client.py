# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# Global HTTP client instance
_global_http_client: Optional[httpx.AsyncClient] = None


def get_http_client(timeout: float = 60.0) -> httpx.AsyncClient:
    """
    Get or create a shared HTTP client instance.

    Args:
        timeout: Timeout for HTTP requests

    Returns:
        Shared HTTP client instance
    """
    global _global_http_client

    if _global_http_client is None or _global_http_client.is_closed:
        _global_http_client = httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
        )
        logger.info(f"Shared HTTP client initialized with timeout={timeout}s")

    return _global_http_client
