#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""
Generic model-router handler.

Wires together a :class:`BaseModelRouter` (routing logic) and a
:class:`PoolManager` (downstream clients) into an async ``generate``
method that can be served as a Dynamo endpoint.
"""

import logging

from dynamo._core import Context

from .base import BaseModelRouter
from .pool import PoolManager

logger = logging.getLogger(__name__)


class ModelRouterHandler:
    """Routes incoming requests through the configured router and forwards
    them to the selected pool.

    Args:
        router: A concrete :class:`BaseModelRouter` implementation.
        pool_manager: A :class:`PoolManager` with one client per pool.
    """

    def __init__(self, router: BaseModelRouter, pool_manager: PoolManager):
        self.router = router
        self.pool_manager = pool_manager

    async def initialize(self) -> None:
        """Initialize pool clients and the router."""
        await self.pool_manager.initialize()
        await self.router.initialize()
        logger.info("ModelRouterHandler initialized successfully")

    def cleanup(self) -> None:
        """Synchronous cleanup — delegates to the router if needed.

        Called from a ``finally`` block in the worker, matching the
        pattern used by vllm/sglang handlers.
        """
        # Router cleanup is async; schedule it if there's a running loop.
        import asyncio

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.router.cleanup())
        except RuntimeError:
            # No running loop — run synchronously
            asyncio.run(self.router.cleanup())

    async def generate(self, request, context: Context):
        """Classify *request*, forward to the chosen pool, and yield responses.

        Args:
            request: PreprocessedRequest dict (with token_ids, etc.).
            context: Dynamo request context for cancellation and lifecycle.
        """
        pool_name = await self.router.route(request)
        client = self.pool_manager.get_client(pool_name)

        async for response in await client.round_robin(request, annotated=False):
            yield response
