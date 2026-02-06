#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""
Global Router Service for Hierarchical Routing

Usage: python -m dynamo.global_router --config <config.json> --model-name <model>

This service acts as both a prefill and decode worker from the frontend's perspective,
but internally routes requests to local routers in different namespaces based on
a grid-based pool selection strategy.

Key features:
- Registers as BOTH prefill AND decode worker via register_llm()
- Routes prefill requests based on (ISL, TTFT) to prefill pools
- Routes decode requests based on (context_length, ITL) to decode pools
- Connects to local routers in each pool's namespace
"""

import argparse
import asyncio
import logging
import os

import uvloop

from dynamo.llm import ModelInput, ModelType, register_llm
from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging

from .handler import GlobalRouterHandler

configure_dynamo_logging()
logger = logging.getLogger(__name__)

DYN_NAMESPACE = os.environ.get("DYN_NAMESPACE", "dynamo")


def parse_args():
    """Parse command-line arguments for the Global Router service."""
    parser = argparse.ArgumentParser(
        description="Dynamo Global Router Service: Hierarchical routing to prefill/decode pools",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the JSON configuration file defining pool namespaces and selection strategy",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Model name for registration (must match workers)",
    )

    parser.add_argument(
        "--namespace",
        type=str,
        default=DYN_NAMESPACE,
        help=f"Dynamo namespace for the global router (default: {DYN_NAMESPACE})",
    )

    parser.add_argument(
        "--component-name",
        type=str,
        default="global_router",
        help="Component name for the global router (default: global_router)",
    )

    parser.add_argument(
        "--default-ttft-target",
        type=float,
        default=None,
        help="Default TTFT target (ms) for prefill pool selection when SLA not present in request",
    )

    parser.add_argument(
        "--default-itl-target",
        type=float,
        default=None,
        help="Default ITL target (ms) for decode pool selection when SLA not present in request",
    )

    return parser.parse_args()


@dynamo_worker()
async def worker(runtime: DistributedRuntime):
    """Main worker function for the Global Router service."""

    args = parse_args()

    logger.info("Starting Global Router Service")
    logger.info(f"Config: {args.config}")
    logger.info(f"Model name: {args.model_name}")
    logger.info(f"Namespace: {args.namespace}")

    # Create handler
    handler = GlobalRouterHandler(
        runtime=runtime,
        config_path=args.config,
        model_name=args.model_name,
        default_ttft_target=args.default_ttft_target,
        default_itl_target=args.default_itl_target,
    )

    # Initialize connections to local routers
    await handler.initialize()

    # Create component in the global router namespace
    component = runtime.namespace(args.namespace).component(args.component_name)

    # Create endpoints for prefill and decode
    # Note: We use separate endpoints so we can register them with different ModelTypes
    prefill_endpoint = component.endpoint("prefill_generate")
    decode_endpoint = component.endpoint("decode_generate")

    logger.info("Registering as prefill worker...")
    # Register as prefill worker - frontend will send prefill requests here
    # Use model_name as model_path since we don't need tokenizer/model files
    await register_llm(
        model_input=ModelInput.Tokens,
        model_type=ModelType.Prefill,
        endpoint=prefill_endpoint,
        model_path=args.model_name,
        model_name=args.model_name,
    )
    logger.info(
        f"Registered prefill endpoint: {args.namespace}.{args.component_name}.prefill_generate"
    )

    logger.info("Registering as decode worker...")
    # Register as decode worker - frontend will send decode requests here
    await register_llm(
        model_input=ModelInput.Tokens,
        model_type=ModelType.Chat | ModelType.Completions,
        endpoint=decode_endpoint,
        model_path=args.model_name,
        model_name=args.model_name,
    )
    logger.info(
        f"Registered decode endpoint: {args.namespace}.{args.component_name}.decode_generate"
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
