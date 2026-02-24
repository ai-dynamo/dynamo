#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""
Random router — zero-dependency reference implementation.

Routes each request to a randomly chosen pool.  Supports optional
per-pool weights via ``config.extra["weights"]`` (a dict mapping pool
names to numeric weights).

Usage::

    python -m dynamo.nemo_switchyard \\
        --router-type random \\
        --pool strong=strong_pool.router.generate \\
        --pool weak=weak_pool.router.generate
"""

import random as _random

from ..base import BaseModelRouter, RouterConfig
from ..registry import register_router


class RandomRouter(BaseModelRouter):
    """Randomly routes requests across pools."""

    def __init__(self, config: RouterConfig):
        super().__init__(config)
        weights_raw = config.extra.get("weights")
        if weights_raw:
            # Validate that every key is a known pool
            for name in weights_raw:
                if name not in config.pool_names:
                    raise ValueError(
                        f"Weight key '{name}' is not in pool_names {config.pool_names}"
                    )
            self._pools = list(weights_raw.keys())
            self._weights = [float(weights_raw[p]) for p in self._pools]
        else:
            self._pools = list(config.pool_names)
            self._weights = None  # uniform

    async def route(self, request: dict) -> str:
        if self._weights:
            return _random.choices(self._pools, weights=self._weights, k=1)[0]
        return _random.choice(self._pools)


register_router("random", RandomRouter)
