#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""
Global Router Service for Hierarchical Routing

Usage: python -m dynamo.global_router --config <config.json> --model-name <model>

This service acts as both a prefill and decode worker from the frontend's perspective,
but internally routes requests to local routers in different namespaces based on
a grid-based pool selection strategy.

Key features:
- Registers as BOTH prefill AND decode worker via register_model()
- Routes prefill requests based on (ISL, TTFT) to prefill pools
- Routes decode requests based on (context_length, ITL) to decode pools
- Connects to local routers in each pool's namespace
"""

import argparse
import asyncio
import logging

import uvloop

from dynamo.llm import ModelInput, ModelType, register_model
from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging

from .backend_args import DynamoGlobalRouterArgGroup, DynamoGlobalRouterConfig
from .handler import GlobalRouterHandler

configure_dynamo_logging()
logger = logging.getLogger(__name__)


def parse_args() -> DynamoGlobalRouterConfig:
    """Parse command-line arguments for the Global Router service."""
    parser = argparse.ArgumentParser(
        description="Dynamo Global Router Service: Hierarchical routing to prefill/decode pools",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    DynamoGlobalRouterArgGroup().add_arguments(parser)
    args = parser.parse_args()
    config = DynamoGlobalRouterConfig.from_cli_args(args)
    config.validate()
    return config


@dynamo_worker()
async def worker(runtime: DistributedRuntime):
    """Main worker function for the Global Router service."""

    config = parse_args()
    logger.info("Starting Global Router Service")
    logger.info(f"Config: {config.config_path}")
    logger.info(f"Model name: {config.model_name}")
    logger.info(f"Namespace: {config.namespace}")

    # Create handler
    handler = GlobalRouterHandler(
        runtime=runtime,
        config_path=config.config_path,
        model_name=config.model_name,
        default_ttft_target=config.default_ttft_target,
        default_itl_target=config.default_itl_target,
    )

    # Initialize connections to local routers
    await handler.initialize()

    # Create endpoints for prefill and decode
    # Note: We use separate endpoints so we can register them with different ModelTypes
    prefill_endpoint = runtime.endpoint(
        f"{config.namespace}.{config.component_name}.prefill_generate"
    )
    decode_endpoint = runtime.endpoint(
        f"{config.namespace}.{config.component_name}.decode_generate"
    )

    logger.info("Registering as prefill worker...")
    # Register as prefill worker - frontend will send prefill requests here
    # Use model_name as model_path since we don't need tokenizer/model files
    await register_model(
        model_input=ModelInput.Tokens,
        model_type=ModelType.Prefill,
        endpoint=prefill_endpoint,
        model_path=config.model_name,
        model_name=config.model_name,
    )
    logger.info(
        f"Registered prefill endpoint: {config.namespace}.{config.component_name}.prefill_generate"
    )

    logger.info("Registering as decode worker...")
    # Register as decode worker - frontend will send decode requests here
    await register_model(
        model_input=ModelInput.Tokens,
        model_type=ModelType.Chat | ModelType.Completions,
        endpoint=decode_endpoint,
        model_path=config.model_name,
        model_name=config.model_name,
    )
    logger.info(
        f"Registered decode endpoint: {config.namespace}.{config.component_name}.decode_generate"
    )

    logger.info("Global Router ready - serving endpoints...")
    logger.info(f"Pool info: {handler.get_pool_info()}")

    # Serve both endpoints concurrently
    try:
        await asyncio.gather(
            prefill_endpoint.serve_endpoint(
                handler.handle_prefill,
                graceful_shutdown=True,
                metrics_labels=[("service", "global_router"), ("type", "prefill")],
            ),
            decode_endpoint.serve_endpoint(
                handler.handle_decode,
                graceful_shutdown=True,
                metrics_labels=[("service", "global_router"), ("type", "decode")],
            ),
        )
    except Exception as e:
        logger.error(f"Failed to serve endpoints: {e}")
        raise
    finally:
        logger.info("Global Router Service shutting down")


def main():
    """Entry point for the Global Router service."""
    uvloop.run(worker())


if __name__ == "__main__":
    main()
