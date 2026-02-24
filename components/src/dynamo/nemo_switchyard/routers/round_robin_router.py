#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""
Round-robin router — zero-dependency reference implementation.

Cycles through the configured pools in order, distributing requests
evenly across all pools.

Usage::

    python -m dynamo.nemo_switchyard \\
        --router-type round_robin \\
        --pool strong=strong_pool.router.generate \\
        --pool weak=weak_pool.router.generate
"""

import itertools

from ..base import BaseModelRouter, RouterConfig
from ..registry import register_router


class RoundRobinRouter(BaseModelRouter):
    """Routes requests to pools in a round-robin cycle."""

    def __init__(self, config: RouterConfig):
        super().__init__(config)
        self._cycle = itertools.cycle(config.pool_names)

    async def route(self, request: dict) -> str:
        return next(self._cycle)


register_router("round_robin", RoundRobinRouter)
