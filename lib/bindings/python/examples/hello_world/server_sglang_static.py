# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#
# Static version of server_sglang.py - see there for most details.
#
# The key differences between this and `server_sglang.py` are:
# - We do not call register_llm to advertise ourself in etcd. There is no etcd.
# - The frontend must know up-front all the details for the model: name, pre-processor path, and type.
#
# Window 1: `python server_sglang_static.py`. Wait for log "Starting endpoint".
# Window 2: `dynamo-run out=dyn://dynamo.backend.generate --model-name "Qwen/Qwen3-0.6B" --model-path <hf_path> --model-type Backend

import argparse
import asyncio
import os
import sys

import sglang
import uvloop
from sglang.srt.server_args import ServerArgs

from dynamo.runtime import DistributedRuntime, dynamo_worker

DYN_NAMESPACE = os.environ.get("DYN_NAMESPACE", "dynamo")
DEFAULT_ENDPOINT = f"dyn://{DYN_NAMESPACE}.backend.generate"
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_TEMPERATURE = 0.7


class Config:
    """Command line parameters or defaults"""

    namespace: str
    component: str
    endpoint: str
    model: str


class RequestHandler:
    """
    Request handler for the generate endpoint
    """

    def __init__(self, engine):
        self.engine_client = engine

    async def generate(self, request):
        # print(f"Received request: {request}")
        sampling_params = {
            "temperature": request["sampling_options"]["temperature"]
            or DEFAULT_TEMPERATURE,
            # sglang defaults this to 128
            "max_new_tokens": request["stop_conditions"]["max_tokens"],
        }
        num_output_tokens_so_far = 0
        gen = await self.engine_client.async_generate(
            input_ids=request["token_ids"], sampling_params=sampling_params, stream=True
        )
        async for res in gen:
            # res is a dict

            finish_reason = res["meta_info"]["finish_reason"]
            if finish_reason:
                # Don't forward the stop token
                out = {"token_ids": [], "finish_reason": finish_reason["type"]}
                next_total_toks = num_output_tokens_so_far
            else:
                next_total_toks = len(res["output_ids"])
                out = {"token_ids": res["output_ids"][num_output_tokens_so_far:]}
            yield out
            num_output_tokens_so_far = next_total_toks


@dynamo_worker(static=True)
async def worker(runtime: DistributedRuntime):
    await init(runtime, cmd_line_args())


async def init(runtime: DistributedRuntime, config: Config):
    """
    Instantiate and serve
    """
    component = runtime.namespace(config.namespace).component(config.component)
    await component.create_service()

    endpoint = component.endpoint(config.endpoint)

    engine_args = ServerArgs(
        model_path=config.model,
        skip_tokenizer_init=True,
    )

    engine_client = sglang.Engine(server_args=engine_args)

    # the server will gracefully shutdown (i.e., keep opened TCP streams finishes)
    # after the lease is revoked
    await endpoint.serve_endpoint(RequestHandler(engine_client).generate)


def cmd_line_args():
    parser = argparse.ArgumentParser(
        description="SGLang server integrated with Dynamo runtime."
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default=DEFAULT_ENDPOINT,
        help=f"Dynamo endpoint string in 'dyn://namespace.component.endpoint' format. Default: {DEFAULT_ENDPOINT}",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Path to disk model or HuggingFace model identifier to load. Default: {DEFAULT_MODEL}",
    )
    args = parser.parse_args()

    config = Config()
    config.model = args.model

    endpoint_str = args.endpoint.replace("dyn://", "", 1)
    endpoint_parts = endpoint_str.split(".")
    if len(endpoint_parts) != 3:
        print(
            f"Invalid endpoint format: '{args.endpoint}'. Expected 'dyn://namespace.component.endpoint' or 'namespace.component.endpoint'."
        )
        sys.exit(1)

    parsed_namespace, parsed_component_name, parsed_endpoint_name = endpoint_parts

    config.namespace = parsed_namespace
    config.component = parsed_component_name
    config.endpoint = parsed_endpoint_name

    return config


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(worker())
