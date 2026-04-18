#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""
Global Router Handler for hierarchical routing to worker pools.

Supports two modes:
- "disagg": Routes prefill and decode requests to separate pool types
  based on (ISL, TTFT) and (context_length, ITL) respectively.
- "agg": Routes generate requests to unified pools that handle both
  prefill and decode, based on (ISL, ITL).

Both modes support priority-based pool overrides from agent hints.
"""

import logging
from typing import Any, AsyncGenerator, Dict, Optional

from dynamo.runtime import Client, DistributedRuntime

from .pool_selection import load_config

logger = logging.getLogger(__name__)


class GlobalRouterHandler:
    """
    Handler for the Global Router that routes requests to worker pools.

    The global router sits between the frontend and local routers. It:
    - In disagg mode: routes prefill/decode requests to separate pool types
    - In agg mode: routes generate requests to unified pools
    - Uses grid-based selection strategy from config to choose pools
    - Supports priority-based pool overrides from agent hints
    """

    def __init__(
        self,
        runtime: DistributedRuntime,
        config_path: str,
        model_name: str,
        default_ttft_target: Optional[float] = None,
        default_itl_target: Optional[float] = None,
    ):
        self.runtime = runtime
        self.config = load_config(config_path)
        self.model_name = model_name
        self.default_ttft_target = default_ttft_target
        self.default_itl_target = default_itl_target

        # Clients to local routers in each pool namespace
        # Will be populated in initialize()
        self.prefill_clients: Dict[str, Client] = {}
        self.decode_clients: Dict[str, Client] = {}
        self.agg_clients: Dict[str, Client] = {}

        if self.config.mode == "disagg":
            assert self.config.prefill_pool_dynamo_namespaces is not None
            assert self.config.decode_pool_dynamo_namespaces is not None
            self.prefill_namespace_to_idx: Dict[str, int] = {
                ns: idx
                for idx, ns in enumerate(self.config.prefill_pool_dynamo_namespaces)
            }
            self.decode_namespace_to_idx: Dict[str, int] = {
                ns: idx
                for idx, ns in enumerate(self.config.decode_pool_dynamo_namespaces)
            }
        elif self.config.mode == "agg":
            assert self.config.agg_pool_dynamo_namespaces is not None
            self.agg_namespace_to_idx: Dict[str, int] = {
                ns: idx for idx, ns in enumerate(self.config.agg_pool_dynamo_namespaces)
            }

    async def initialize(self) -> None:
        """
        Initialize clients to all local routers.

        This connects to the local router in each pool's namespace.
        Local routers are expected at: {namespace}.router.generate
        """
        logger.info(f"Initializing Global Router Handler (mode={self.config.mode})...")

        if self.config.mode == "disagg":
            await self._initialize_disagg()
        elif self.config.mode == "agg":
            await self._initialize_agg()

    async def _initialize_disagg(self) -> None:
        """Initialize disagg mode clients to prefill and decode pools."""
        assert self.config.prefill_pool_dynamo_namespaces is not None
        assert self.config.decode_pool_dynamo_namespaces is not None

        # Connect to prefill pool local routers
        for idx, namespace in enumerate(self.config.prefill_pool_dynamo_namespaces):
            try:
                endpoint = self.runtime.endpoint(f"{namespace}.router.generate")
                client = await endpoint.client()
                self.prefill_clients[namespace] = client
                logger.info(
                    f"Connected to prefill pool {idx}: {namespace}.router.generate"
                )
            except Exception as e:
                logger.error(
                    f"Failed to connect to prefill pool {idx} ({namespace}): {e}"
                )
                raise

        # Connect to decode pool local routers
        for idx, namespace in enumerate(self.config.decode_pool_dynamo_namespaces):
            try:
                endpoint = self.runtime.endpoint(f"{namespace}.router.generate")
                client = await endpoint.client()
                self.decode_clients[namespace] = client
                logger.info(
                    f"Connected to decode pool {idx}: {namespace}.router.generate"
                )
            except Exception as e:
                logger.error(
                    f"Failed to connect to decode pool {idx} ({namespace}): {e}"
                )
                raise

        logger.info(
            f"Global Router initialized (disagg): {len(self.prefill_clients)} prefill pools, "
            f"{len(self.decode_clients)} decode pools"
        )

    async def _initialize_agg(self) -> None:
        """Initialize agg mode clients to unified pools."""
        assert self.config.agg_pool_dynamo_namespaces is not None

        for idx, namespace in enumerate(self.config.agg_pool_dynamo_namespaces):
            try:
                endpoint = self.runtime.endpoint(f"{namespace}.router.generate")
                client = await endpoint.client()
                self.agg_clients[namespace] = client
                logger.info(f"Connected to agg pool {idx}: {namespace}.router.generate")
            except Exception as e:
                logger.error(f"Failed to connect to agg pool {idx} ({namespace}): {e}")
                raise

        logger.info(f"Global Router initialized (agg): {len(self.agg_clients)} pools")

    async def handle_prefill(
        self, request: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Handle prefill requests from the frontend (disagg mode).

        Selects the appropriate prefill pool based on ISL, TTFT target,
        and optional priority, then forwards the request to the local
        router in that pool.
        """
        assert self.config.prefill_pool_selection_strategy is not None
        assert self.config.prefill_pool_dynamo_namespaces is not None

        # Extract ISL (input sequence length)
        token_ids = request.get("token_ids", [])
        isl = len(token_ids)

        # Extract TTFT target from extra_args if provided, fallback to CLI default
        extra_args = request.get("extra_args") or {}
        ttft_target = extra_args.get("ttft_target") or self.default_ttft_target

        # Extract priority from routing hints (set by nvext.agent_hints.priority)
        routing = request.get("routing") or {}
        priority = routing.get("priority")

        # Select prefill pool
        pool_idx = self.config.prefill_pool_selection_strategy.select_pool(
            isl=isl, ttft_target=ttft_target, priority=priority
        )
        namespace = self.config.prefill_pool_dynamo_namespaces[pool_idx]
        client = self.prefill_clients[namespace]

        logger.info(
            f"Routing prefill request: ISL={isl}, TTFT_target={ttft_target}, "
            f"priority={priority} -> pool {pool_idx} ({namespace})"
        )

        # Forward request to local router and stream back responses
        try:
            stream = await client.generate(request)
            async for output in stream:
                # Extract data from stream response object
                data = output.data() if hasattr(output, "data") else output
                yield data
        except Exception as e:
            logger.error(f"Error forwarding prefill request to {namespace}: {e}")
            raise

    async def handle_decode(
        self, request: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Handle decode requests from the frontend (disagg mode).

        Selects the appropriate decode pool based on context length, ITL target,
        and optional priority, then forwards the request to the local
        router in that pool.
        """
        assert self.config.decode_pool_selection_strategy is not None
        assert self.config.decode_pool_dynamo_namespaces is not None

        # Extract context length (input tokens + any previously generated)
        token_ids = request.get("token_ids", [])
        # context_length should be averaged ISL + OSL // 2
        # TODO: predict OSL based on ISL
        context_length = len(token_ids)

        # Extract ITL target from extra_args if provided, fallback to CLI default
        extra_args = request.get("extra_args") or {}
        itl_target = extra_args.get("itl_target") or self.default_itl_target

        # Extract priority from routing hints (set by nvext.agent_hints.priority)
        routing = request.get("routing") or {}
        priority = routing.get("priority")

        # Select decode pool
        pool_idx = self.config.decode_pool_selection_strategy.select_pool(
            context_length=context_length,
            itl_target=itl_target,
            priority=priority,
        )
        namespace = self.config.decode_pool_dynamo_namespaces[pool_idx]
        client = self.decode_clients[namespace]

        logger.info(
            f"Routing decode request: context_length={context_length}, "
            f"ITL_target={itl_target}, priority={priority} -> "
            f"pool {pool_idx} ({namespace})"
        )

        # Forward request to local router and stream back responses
        try:
            stream = await client.generate(request)
            async for output in stream:
                # Extract data from stream response object
                data = output.data() if hasattr(output, "data") else output
                yield data
        except Exception as e:
            logger.error(f"Error forwarding decode request to {namespace}: {e}")
            raise

    async def handle_generate(
        self, request: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Handle generate requests (agg mode).

        Selects the appropriate agg pool based on TTFT target, ITL target, and
        optional priority, then forwards the request to the local router in
        that pool. The pool's workers handle both prefill and decode.
        """
        assert self.config.agg_pool_selection_strategy is not None
        assert self.config.agg_pool_dynamo_namespaces is not None

        # Extract SLA targets from extra_args, fallback to CLI defaults.
        # Use `is None` checks to preserve explicit 0 values.
        extra_args = request.get("extra_args") or {}
        ttft_target = extra_args.get("ttft_target")
        if ttft_target is None:
            ttft_target = self.default_ttft_target
        itl_target = extra_args.get("itl_target")
        if itl_target is None:
            itl_target = self.default_itl_target

        # Extract priority from routing hints (set by nvext.agent_hints.priority)
        routing = request.get("routing") or {}
        priority = routing.get("priority")

        # Select agg pool
        pool_idx = self.config.agg_pool_selection_strategy.select_pool(
            ttft_target=ttft_target, itl_target=itl_target, priority=priority
        )
        namespace = self.config.agg_pool_dynamo_namespaces[pool_idx]
        client = self.agg_clients[namespace]

        logger.info(
            f"Routing agg request: TTFT_target={ttft_target}, ITL_target={itl_target}, "
            f"priority={priority} -> pool {pool_idx} ({namespace})"
        )

        # Forward request to local router and stream back responses
        try:
            stream = await client.generate(request)
            async for output in stream:
                data = output.data() if hasattr(output, "data") else output
                yield data
        except Exception as e:
            logger.error(f"Error forwarding agg request to {namespace}: {e}")
            raise

    def get_pool_info(self) -> Dict[str, Any]:
        """Get information about connected pools for debugging/monitoring."""
        info: Dict[str, Any] = {
            "model_name": self.model_name,
            "mode": self.config.mode,
        }
        if self.config.mode == "disagg":
            info.update(
                {
                    "num_prefill_pools": self.config.num_prefill_pools,
                    "num_decode_pools": self.config.num_decode_pools,
                    "prefill_pools": self.config.prefill_pool_dynamo_namespaces,
                    "decode_pools": self.config.decode_pool_dynamo_namespaces,
                    "prefill_connected": list(self.prefill_clients.keys()),
                    "decode_connected": list(self.decode_clients.keys()),
                }
            )
        elif self.config.mode == "agg":
            info.update(
                {
                    "num_agg_pools": self.config.num_agg_pools,
                    "agg_pools": self.config.agg_pool_dynamo_namespaces,
                    "agg_connected": list(self.agg_clients.keys()),
                }
            )
        return info
