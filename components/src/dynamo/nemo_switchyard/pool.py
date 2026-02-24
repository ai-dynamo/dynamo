#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""
Pool management for the pluggable router abstraction layer.

Provides :class:`Pool` (a named endpoint with a dyn:// client) and
:class:`PoolManager` (creates and looks up clients for N pools).
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

from dynamo.runtime import DistributedRuntime

logger = logging.getLogger(__name__)


@dataclass
class Pool:
    """A single downstream pool that the router can forward requests to."""

    name: str
    endpoint_path: str  # "namespace.component.endpoint"
    client: Any = field(default=None, repr=False)  # Dynamo client, set during initialize()


class PoolManager:
    """Creates and manages dyn:// clients for every configured pool.

    Args:
        runtime: The :class:`DistributedRuntime` used to create clients.
        pool_endpoints: Mapping of ``pool_name`` -> ``"namespace.component.endpoint"``.
    """

    def __init__(
        self,
        runtime: DistributedRuntime,
        pool_endpoints: Dict[str, str],
    ):
        self._runtime = runtime
        self._pools: Dict[str, Pool] = {
            name: Pool(name=name, endpoint_path=ep)
            for name, ep in pool_endpoints.items()
        }

    async def initialize(self) -> None:
        """Create dyn:// clients for all pools."""
        for pool in self._pools.values():
            parts = pool.endpoint_path.split(".")
            if len(parts) != 3:
                raise ValueError(
                    f"Invalid endpoint path for '{pool.name}' pool: {pool.endpoint_path}. "
                    "Expected format: namespace.component.endpoint"
                )
            ns, comp, ep = parts
            pool.client = (
                await self._runtime.namespace(ns).component(comp).endpoint(ep).client()
            )
            logger.info("Connected to '%s' pool at %s", pool.name, pool.endpoint_path)

    def get_client(self, pool_name: str) -> Any:
        """Return the dyn:// client for *pool_name*."""
        pool = self._pools.get(pool_name)
        if pool is None:
            raise KeyError(f"Unknown pool: {pool_name!r}")
        if pool.client is None:
            raise RuntimeError(
                f"Pool '{pool_name}' has not been initialized — call initialize() first"
            )
        return pool.client

    @property
    def pool_names(self) -> List[str]:
        return list(self._pools)
