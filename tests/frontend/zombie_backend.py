#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""Zombie backend for testing frontend inactivity timeout (issue #7545).

This is a standalone Dynamo worker that registers as a model backend via the
discovery plane (etcd) and request plane (TCP/NATS), but intentionally stalls
when handling requests -- simulating a zombie backend that holds a live TCP
connection but never (or only partially) produces output.

Usage:
    python zombie_backend.py --model-path Qwen/Qwen3-0.6B --stall-after-tokens 0

    --stall-after-tokens 0  : Accept request, send SSE headers, never send tokens (pure zombie)
    --stall-after-tokens N  : Send N tokens normally, then stall forever (mid-stream zombie)

Without --stall-after-tokens the backend behaves normally (sends all tokens).
"""

import argparse
import asyncio
import logging
import os
import signal

import uvloop

from dynamo.llm import ModelInput, ModelType, register_model
from dynamo.runtime import DistributedRuntime, dynamo_worker

logger = logging.getLogger(__name__)

DYN_NAMESPACE = os.environ.get("DYN_NAMESPACE", "dynamo")
DEFAULT_ENDPOINT = f"dyn://{DYN_NAMESPACE}.backend.generate"


def parse_args():
    parser = argparse.ArgumentParser(description="Zombie backend for timeout testing")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="HuggingFace model ID (used for tokenizer and model registration)",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default=None,
        help=f"Dynamo endpoint (default: {DEFAULT_ENDPOINT})",
    )
    parser.add_argument(
        "--stall-after-tokens",
        type=int,
        default=None,
        help="Number of tokens to send before stalling. "
        "0 = pure zombie (never send tokens). "
        "N = send N tokens then stall. "
        "Omit = normal mode (send all tokens).",
    )
    return parser.parse_args()


class ZombieHandler:
    """Request handler that can simulate zombie behavior."""

    def __init__(self, stall_after_tokens=None):
        self.stall_after_tokens = stall_after_tokens

    async def generate(self, request):
        """Handle a generate request.

        The request dict contains token_ids, sampling_options, stop_conditions, etc.
        We yield dicts with token_ids and optionally finish_reason.
        """
        max_tokens = request.get("stop_conditions", {}).get("max_tokens", 16)
        logger.info(
            "Received request: max_tokens=%d, stall_after=%s",
            max_tokens,
            self.stall_after_tokens,
        )

        if self.stall_after_tokens is not None:
            # Zombie mode: send some tokens then stall forever
            tokens_to_send = self.stall_after_tokens
            for i in range(min(tokens_to_send, max_tokens)):
                yield {"token_ids": [1000 + i]}
                await asyncio.sleep(0.01)  # Small delay between tokens

            # Now stall forever -- this simulates the zombie
            logger.info("Stalling after %d tokens (zombie mode)", tokens_to_send)
            await asyncio.Event().wait()  # Block forever
        else:
            # Normal mode: send all tokens
            for i in range(max_tokens):
                is_last = i == max_tokens - 1
                output = {"token_ids": [1000 + i]}
                if is_last:
                    output["finish_reason"] = "length"
                yield output
                await asyncio.sleep(0.01)


@dynamo_worker()
async def worker(runtime: DistributedRuntime):
    args = parse_args()

    endpoint_str = args.endpoint or DEFAULT_ENDPOINT
    # Strip dyn:// prefix for runtime.endpoint()
    endpoint_id = endpoint_str.replace("dyn://", "", 1)

    endpoint = runtime.endpoint(endpoint_id)

    await register_model(
        ModelInput.Tokens,
        ModelType.Chat | ModelType.Completions,
        endpoint,
        args.model_path,
    )

    handler = ZombieHandler(stall_after_tokens=args.stall_after_tokens)

    loop = asyncio.get_running_loop()

    def signal_handler():
        asyncio.create_task(graceful_shutdown(runtime))

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    logger.info(
        "Zombie backend started: model=%s stall_after=%s endpoint=%s",
        args.model_path,
        args.stall_after_tokens,
        endpoint_id,
    )

    await endpoint.serve_endpoint(handler.generate)


async def graceful_shutdown(runtime: DistributedRuntime):
    logger.info("Received shutdown signal, shutting down")
    runtime.shutdown()


if __name__ == "__main__":
    uvloop.run(worker())
