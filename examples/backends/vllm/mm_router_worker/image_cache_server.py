# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lightweight HTTP server that serves cached images from MM Router.

The MM Router loads and processes images during multimodal request handling.
Instead of forwarding ~1MB base64 data URIs through the Python→Rust FFI
boundary and TCP serialization path, we cache the raw image bytes here and
let the downstream vLLM backend fetch them via a fast loopback HTTP GET.

This reduces the forwarded payload from ~1MB to ~60 bytes per image.
"""

import asyncio
import hashlib
import logging
from collections import OrderedDict

from aiohttp import web

logger = logging.getLogger(__name__)


class ImageCacheServer:
    """In-memory image cache with HTTP serving."""

    def __init__(self, max_entries: int = 1024):
        self._cache: OrderedDict[str, tuple[bytes, str]] = OrderedDict()
        self._max_entries = max_entries
        self._port: int | None = None
        self._runner: web.AppRunner | None = None

    @property
    def port(self) -> int | None:
        return self._port

    def base_url(self) -> str:
        """Return the base URL for cached images."""
        if self._port is None:
            raise RuntimeError("Image cache server not started")
        return f"http://127.0.0.1:{self._port}"

    def put(self, image_bytes: bytes, content_type: str = "image/png") -> str:
        """Cache image bytes and return the cache key (content hash)."""
        key = hashlib.sha256(image_bytes).hexdigest()[:16]
        if key not in self._cache:
            if len(self._cache) >= self._max_entries:
                self._cache.popitem(last=False)
            self._cache[key] = (image_bytes, content_type)
        else:
            self._cache.move_to_end(key)
        return key

    def url_for(self, key: str) -> str:
        """Return the full HTTP URL for a cached image."""
        return f"{self.base_url()}/images/{key}"

    async def _handle_get(self, request: web.Request) -> web.Response:
        key = request.match_info["key"]
        entry = self._cache.get(key)
        if entry is None:
            return web.Response(status=404, text="Not found")
        image_bytes, content_type = entry
        return web.Response(body=image_bytes, content_type=content_type)

    async def start(self) -> int:
        """Start the HTTP server on a random available port. Returns the port."""
        app = web.Application()
        app.router.add_get("/images/{key}", self._handle_get)

        self._runner = web.AppRunner(app, access_log=None)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()

        # Extract the actual port
        self._port = site._server.sockets[0].getsockname()[1]
        logger.info(f"Image cache server started on port {self._port}")
        return self._port

    async def stop(self) -> None:
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
            self._port = None
