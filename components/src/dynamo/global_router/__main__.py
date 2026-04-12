#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""
Global Router Service for Hierarchical Routing

Usage: python -m dynamo.global_router --config <config.json> --model-name <model>

This service routes requests to local routers in different namespaces based on
a grid-based pool selection strategy. It supports two modes:

- "disagg" mode: Registers as BOTH prefill AND decode worker. Routes prefill
  requests based on (ISL, TTFT) and decode requests based on (context_length, ITL)
  to separate pool types.

- "agg" mode: Registers as a single generate worker. Routes all requests based
  on (ISL, ITL) to unified pools that handle both prefill and decode.

Both modes support priority-based pool overrides from agent hints.
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
        description="Dynamo Global Router Service: Hierarchical routing to worker pools",
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
    # validate() ensures these are non-None; assert to narrow types for mypy
    assert config.config_path is not None
    assert config.model_name is not None
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

    logger.info(f"Mode: {handler.config.mode}")
    logger.info(f"Pool info: {handler.get_pool_info()}")

    if handler.config.mode == "disagg":
        await _serve_disagg(runtime, config, handler)
    elif handler.config.mode == "agg":
        await _serve_agg(runtime, config, handler)
    else:
        raise ValueError(f"Unknown mode: {handler.config.mode}")


async def _serve_disagg(
    runtime: DistributedRuntime,
    config: DynamoGlobalRouterConfig,
    handler: GlobalRouterHandler,
) -> None:
    """Register and serve disagg-mode endpoints (prefill + decode)."""
    assert config.model_name is not None
    prefill_endpoint = runtime.endpoint(
        f"{config.namespace}.{config.component_name}.prefill_generate"
    )
    decode_endpoint = runtime.endpoint(
        f"{config.namespace}.{config.component_name}.decode_generate"
    )

    logger.info("Registering as prefill worker...")
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

    logger.info("Global Router ready (disagg mode) - serving endpoints...")

    try:
        await asyncio.gather(
            prefill_endpoint.serve_endpoint(
                handler.handle_prefill,
                graceful_shutdown=True,
                metrics_labels=[
                    ("service", "global_router"),
                    ("type", "prefill"),
                ],
            ),
            decode_endpoint.serve_endpoint(
                handler.handle_decode,
                graceful_shutdown=True,
                metrics_labels=[
                    ("service", "global_router"),
                    ("type", "decode"),
                ],
            ),
        )
    except Exception as e:
        logger.error(f"Failed to serve disagg endpoints: {e}")
        raise
    finally:
        logger.info("Global Router Service shutting down")


async def _serve_agg(
    runtime: DistributedRuntime,
    config: DynamoGlobalRouterConfig,
    handler: GlobalRouterHandler,
) -> None:
    """Register and serve agg-mode endpoint (single generate)."""
    assert config.model_name is not None
    generate_endpoint = runtime.endpoint(
        f"{config.namespace}.{config.component_name}.generate"
    )

    logger.info("Registering as agg worker (Chat + Completions)...")
    await register_model(
        model_input=ModelInput.Tokens,
        model_type=ModelType.Chat | ModelType.Completions,
        endpoint=generate_endpoint,
        model_path=config.model_name,
        model_name=config.model_name,
    )
    logger.info(
        f"Registered agg endpoint: {config.namespace}.{config.component_name}.generate"
    )

    logger.info("Global Router ready (agg mode) - serving endpoint...")

    try:
        await generate_endpoint.serve_endpoint(
            handler.handle_generate,
            graceful_shutdown=True,
            metrics_labels=[
                ("service", "global_router"),
                ("type", "agg"),
            ],
        )
    except Exception as e:
        logger.error(f"Failed to serve agg endpoint: {e}")
        raise
    finally:
        logger.info("Global Router Service shutting down")


def main():
    """Entry point for the Global Router service."""
    uvloop.run(worker())


if __name__ == "__main__":
    main()
