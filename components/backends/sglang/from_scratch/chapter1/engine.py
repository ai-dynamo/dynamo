# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import asyncio
import logging
import sys
from typing import Any, Dict

import sglang
import uvloop
from sglang.srt.server_args import ServerArgs

from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging

configure_dynamo_logging()


class SGLangEngineHandler:
    def __init__(self, server_args: ServerArgs):
        self.server_args = server_args
        self.engine = sglang.Engine(server_args=server_args)
        logging.info("SGLang engine worker initialized successfully")

    async def generate(self, request: Dict[str, Any]):
        """Generate response using SGLang"""
        try:
            logging.info("SGLang engine received request")

            token_ids = request.get("token_ids", [])
            sampling_options = request.get("sampling_options", {})
            stop_conditions = request.get("stop_conditions", {})

            # Build sampling params in SGLang format
            sampling_params = {
                "temperature": sampling_options.get("temperature", 0.7),
                "max_new_tokens": stop_conditions.get("max_tokens", 25),
            }

            # We want to stream but accumulate at the processor
            stream = await self.engine.async_generate(
                input_ids=token_ids, sampling_params=sampling_params, stream=True
            )

            logging.info("Processing SGLang stream...")
            num_output_tokens_so_far = 0

            async for res in stream:
                finish_reason = res["meta_info"]["finish_reason"]

                if finish_reason:
                    # Send completion signal
                    out = {"token_ids": [], "finish_reason": finish_reason["type"]}
                    logging.info(f"Engine yielding finish: {out}")
                    yield out
                else:
                    # Send new tokens only
                    next_total_toks = len(res["output_ids"])
                    new_tokens = res["output_ids"][num_output_tokens_so_far:]
                    if new_tokens:  # Only yield if we have new tokens
                        out = {"token_ids": new_tokens}
                        logging.info(f"Engine yielding {len(new_tokens)} new tokens")
                        yield out
                    num_output_tokens_so_far = next_total_toks

        except Exception as e:
            logging.error(f"Error in SGLang engine generate: {e}")
            yield {"error": str(e)}


def parse_sglang_args(args: list[str]) -> ServerArgs:
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    parsed_args = parser.parse_args(args)
    return ServerArgs.from_cli_args(parsed_args)


@dynamo_worker(static=False)
async def worker(runtime: DistributedRuntime):
    """Main worker function"""
    # Parse command line arguments
    args = parse_sglang_args(sys.argv[1:])

    logging.info(f"Starting SGLang engine worker with model {args.model_path}")

    # Create engine component
    component = runtime.namespace("inference").component("engine")
    await component.create_service()

    # Initialize SGLang engine handler
    engine = SGLangEngineHandler(server_args=args)

    # Create generate endpoint
    endpoint = component.endpoint("generate")
    await endpoint.serve_endpoint(engine.generate)


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(worker())
