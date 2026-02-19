# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
MM Router Worker - Multimodal-aware KV cache routing for TRT-LLM.

This worker receives OpenAI-format requests from the frontend, computes
mm_hash for any images, finds the best TRT-LLM worker based on KV cache
overlap, and forwards the request to that worker.

Usage:
    python -m examples.backends.trtllm.mm_router_worker \
        --model Qwen/Qwen2-VL-2B-Instruct \
        --model-type qwen2_vl \
        --namespace default \
        --component mm_router \
        --endpoint generate \
        --downstream-component trtllm \
        --downstream-endpoint generate
"""

import argparse
import asyncio
import logging
import signal

import uvloop
from tensorrt_llm.llmapi.tokenizer import tokenizer_factory
from transformers import AutoProcessor

from dynamo.llm import KvRouter, KvRouterConfig, ModelInput, ModelType, register_llm
from dynamo.runtime import DistributedRuntime, dynamo_worker

from .handler import MMRouterHandler

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="MM Router Worker - Multimodal-aware KV cache routing"
    )

    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2-VL-2B-Instruct",
        help="Model path or HuggingFace model ID",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="qwen2_vl",
        help="Model type for TRT-LLM multimodal loader",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=32,
        help="KV cache block size",
    )

    # This worker's endpoint configuration
    parser.add_argument(
        "--namespace",
        type=str,
        default="default",
        help="Dynamo namespace",
    )
    parser.add_argument(
        "--component",
        type=str,
        default="mm_router",
        help="This worker's component name",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default="generate",
        help="This worker's endpoint name",
    )

    # Downstream TRT-LLM worker configuration
    parser.add_argument(
        "--downstream-component",
        type=str,
        default="trtllm",
        help="Downstream TRT-LLM workers' component name",
    )
    parser.add_argument(
        "--downstream-endpoint",
        type=str,
        default="generate",
        help="Downstream TRT-LLM workers' endpoint name",
    )

    return parser.parse_args()


async def graceful_shutdown(runtime: DistributedRuntime) -> None:
    """Handle graceful shutdown."""
    logger.info("Received shutdown signal, shutting down...")
    runtime.shutdown()
    logger.info("Shutdown complete")


@dynamo_worker()
async def worker(runtime: DistributedRuntime) -> None:
    """
    Main worker function.

    Sets up connections to downstream TRT-LLM workers, creates KvRouter
    for KV-aware routing, and serves the MM router endpoint.
    """
    args = parse_args()

    # Set up signal handlers for graceful shutdown
    loop = asyncio.get_running_loop()

    def signal_handler():
        asyncio.create_task(graceful_shutdown(runtime))

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    logger.info("MM Router Worker starting...")
    logger.info(f"Model: {args.model}")
    logger.info(f"This worker: {args.namespace}.{args.component}.{args.endpoint}")
    logger.info(
        f"Downstream: {args.namespace}.{args.downstream_component}.{args.downstream_endpoint}"
    )

    # Connect to downstream TRT-LLM workers
    downstream_endpoint = (
        runtime.namespace(args.namespace)
        .component(args.downstream_component)
        .endpoint(args.downstream_endpoint)
    )
    downstream_client = await downstream_endpoint.client()

    logger.info("Waiting for downstream TRT-LLM workers...")
    instance_ids = await downstream_client.wait_for_instances()
    logger.info(f"Found {len(instance_ids)} workers: {list(instance_ids)}")

    # Create KvRouter to select workers based on KV overlap
    kv_router = KvRouter(
        endpoint=downstream_endpoint,
        block_size=args.block_size,
        kv_router_config=KvRouterConfig(),
    )
    logger.info("KvRouter created successfully")

    # Initialize tokenizer and processor for MM processing
    logger.info(f"Loading tokenizer from {args.model}...")
    tokenizer = tokenizer_factory(args.model)

    processor = None
    try:
        logger.info(f"Loading HuggingFace processor from {args.model}...")
        processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    except Exception as e:
        logger.warning(f"Failed to load HF processor: {e}")
        logger.warning("Visual token expansion will not be available")

    # Create handler
    handler = MMRouterHandler(
        kv_router=kv_router,
        tokenizer=tokenizer,
        processor=processor,
        model=args.model,
        model_type=args.model_type,
        block_size=args.block_size,
    )

    # Register this worker's endpoint
    component = runtime.namespace(args.namespace).component(args.component)
    endpoint = component.endpoint(args.endpoint)

    # Use ModelInput.Tokens so Frontend preprocesses the request
    # Request format: {token_ids, sampling_options, stop_conditions, extra_args: {messages}}
    await register_llm(
        ModelInput.Tokens,
        ModelType.Chat,
        endpoint,
        args.model,
        kv_cache_block_size=args.block_size,
    )

    logger.info(f"MM Router Worker ready, serving {args.endpoint} endpoint...")

    # Serve the endpoint
    await endpoint.serve_endpoint(handler.generate)


def main() -> None:
    """Entry point for the MM Router Worker."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    uvloop.install()
    asyncio.run(worker())


if __name__ == "__main__":
    main()
