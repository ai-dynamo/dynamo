#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""
Global Router Handler for hierarchical routing to prefill/decode pools.

This handler:
1. Receives requests from the frontend (acts as both prefill and decode worker)
2. Selects the appropriate pool based on config-driven grid selection
3. Forwards requests to local routers in the selected pool's namespace
"""

import logging
from typing import Any, AsyncGenerator, Dict, Optional

from dynamo.runtime import Client, DistributedRuntime

from .pool_selection import load_config

logger = logging.getLogger(__name__)


class GlobalRouterHandler:
    """
    Handler for the Global Router that routes requests to prefill/decode pools.

    The global router sits between the frontend and local routers. It:
    - Receives prefill requests and routes to appropriate prefill pool
    - Receives decode requests and routes to appropriate decode pool
    - Uses grid-based selection strategy from config to choose pools
    """

    def __init__(
        self,
        runtime: DistributedRuntime,
        config_path: str,
        model_name: str,
        default_ttft_target: Optional[float] = None,
        default_itl_target: Optional[float] = None,
    ):
        """
        Initialize the Global Router Handler.

        Args:
            runtime: Dynamo distributed runtime for creating clients
            config_path: Path to the JSON configuration file
            model_name: Model name for logging/debugging
            default_ttft_target: Default TTFT target (ms) when not in request
            default_itl_target: Default ITL target (ms) when not in request
        """
        self.runtime = runtime
        self.config = load_config(config_path)
        self.model_name = model_name
        self.default_ttft_target = default_ttft_target
        self.default_itl_target = default_itl_target

        # Clients to local routers in each pool namespace
        # Will be populated in initialize()
        self.prefill_clients: Dict[str, Client] = {}
        self.decode_clients: Dict[str, Client] = {}

        # Keep track of namespace -> pool index mapping for easy access
        self.prefill_namespace_to_idx: Dict[str, int] = {
            ns: idx for idx, ns in enumerate(self.config.prefill_pool_dynamo_namespaces)
        }
        self.decode_namespace_to_idx: Dict[str, int] = {
            ns: idx for idx, ns in enumerate(self.config.decode_pool_dynamo_namespaces)
        }

    async def initialize(self) -> None:
        """
        Initialize clients to all local routers.

        This connects to the local router in each pool's namespace.
        Local routers are expected at: {namespace}.router.generate
        """
        logger.info("Initializing Global Router Handler...")

        # Connect to prefill pool local routers
        for idx, namespace in enumerate(self.config.prefill_pool_dynamo_namespaces):
            try:
                endpoint = (
                    self.runtime.namespace(namespace)
                    .component("router")
                    .endpoint("generate")
                )
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
                endpoint = (
                    self.runtime.namespace(namespace)
                    .component("router")
                    .endpoint("generate")
                )
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
            f"Global Router initialized: {len(self.prefill_clients)} prefill pools, "
            f"{len(self.decode_clients)} decode pools"
        )

    async def handle_prefill(
        self, request: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Handle prefill requests from the frontend.

        Selects the appropriate prefill pool based on ISL and TTFT target,
        then forwards the request to the local router in that pool.

        Args:
            request: PreprocessedRequest dict with token_ids, etc.

        Yields:
            LLMEngineOutput dicts from the prefill worker
        """
        # Extract ISL (input sequence length)
        token_ids = request.get("token_ids", [])
        isl = len(token_ids)

        # Extract TTFT target from extra_args if provided, fallback to CLI default
        extra_args = request.get("extra_args") or {}
        ttft_target = extra_args.get("ttft_target") or self.default_ttft_target

        # Select prefill pool
        pool_idx = self.config.prefill_pool_selection_strategy.select_pool(
            isl=isl, ttft_target=ttft_target
        )
        namespace = self.config.prefill_pool_dynamo_namespaces[pool_idx]
        client = self.prefill_clients[namespace]

        logger.info(
            f"Routing prefill request: ISL={isl}, TTFT_target={ttft_target} -> "
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
            logger.error(f"Error forwarding prefill request to {namespace}: {e}")
            raise

    async def handle_decode(
        self, request: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Handle decode requests from the frontend.

        Selects the appropriate decode pool based on context length and ITL target,
        then forwards the request to the local router in that pool.

        Args:
            request: PreprocessedRequest dict with token_ids, prefill_result, etc.

        Yields:
            LLMEngineOutput dicts from the decode worker
        """
        # Extract context length (input tokens + any previously generated)
        token_ids = request.get("token_ids", [])
        # context_length should be averaged ISL + OSL // 2
        # TODO: predict OSL based on ISL
        context_length = len(token_ids)

        # Extract ITL target from extra_args if provided, fallback to CLI default
        extra_args = request.get("extra_args") or {}
        itl_target = extra_args.get("itl_target") or self.default_itl_target

        # Select decode pool
        pool_idx = self.config.decode_pool_selection_strategy.select_pool(
            context_length=context_length, itl_target=itl_target
        )
        namespace = self.config.decode_pool_dynamo_namespaces[pool_idx]
        client = self.decode_clients[namespace]

        logger.info(
            f"Routing decode request: context_length={context_length}, ITL_target={itl_target} -> "
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

    def get_pool_info(self) -> Dict[str, Any]:
        """
        Get information about connected pools for debugging/monitoring.

        Returns:
            Dict with pool information
        """
        return {
            "model_name": self.model_name,
            "num_prefill_pools": self.config.num_prefill_pools,
            "num_decode_pools": self.config.num_decode_pools,
            "prefill_pools": self.config.prefill_pool_dynamo_namespaces,
            "decode_pools": self.config.decode_pool_dynamo_namespaces,
            "prefill_connected": list(self.prefill_clients.keys()),
            "decode_connected": list(self.decode_clients.keys()),
        }
