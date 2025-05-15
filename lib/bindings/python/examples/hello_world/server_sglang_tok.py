# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#
# A very basic example of sglang worker handling pre-processed requests.
#
# Dynamo does the HTTP handling and load balancing, then forwards the
# request via NATS to this python script, which runs sglang. sglang will
# do the pre/post-processing.
#
# The key differences between this and `server_sglang.py` are:
# - The `register_llm` function registers us a `Chat` model
# - The `generate` function receives a chat completion request and must return matching response
#
# Setup a virtualenv with dynamo.llm, dynamo.runtime and sglang[all] installed
#  in lib/bindings/python `maturin develop` and `pip install -e .` should do it
# Start nats and etcd:
#  - nats-server -js
#
# Window 1: `python server_sglang.py`. Wait for log "Starting endpoint".
# Window 2: `dynamo-run out=dyn://dynamo.backend.generate`

import argparse
import asyncio
import sys
import time

import uvloop
from sglang.srt.entrypoints.engine import _launch_subprocesses
from sglang.srt.openai_api.adapter import v1_chat_generate_request
from sglang.srt.openai_api.protocol import ChatCompletionRequest
from sglang.srt.server_args import ServerArgs

from dynamo.llm import ModelType, register_llm
from dynamo.runtime import DistributedRuntime, dynamo_worker

DEFAULT_ENDPOINT = "dyn://dynamo.backend.generate"
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"


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

    def __init__(self, tokenizer_manager):
        self.tokenizer_manager = tokenizer_manager

    async def generate(self, request):
        # Request is dict matching OpenAI Chat Completions
        # https://platform.openai.com/docs/api-reference/chat
        # Return type must be the matching Response

        # print(f"Received request: {request}")

        count = 0
        adapted_request, _ = v1_chat_generate_request(
            [ChatCompletionRequest(**request)], self.tokenizer_manager
        )
        async for res in self.tokenizer_manager.generate_request(adapted_request, None):
            index = res.get("index", 0)
            text = res["text"]

            finish_reason = res["meta_info"]["finish_reason"]
            finish_reason_type = finish_reason["type"] if finish_reason else None
            next_count = len(text)
            delta = text[count:]
            choice_data = {
                "index": index,
                "delta": {"role": "assistant", "content": delta},
                "finish_reason": finish_reason_type,
            }
            created = int(time.time())
            response = {
                "id": res["meta_info"]["id"],
                "created": created,
                "choices": [choice_data],
                "model": request["model"],
                "object": "chat.completion",
            }
            yield response
            count = next_count


@dynamo_worker(static=False)
async def worker(runtime: DistributedRuntime):
    await init(runtime, cmd_line_args())


async def init(runtime: DistributedRuntime, config: Config):
    """
    Instantiate and serve
    """
    component = runtime.namespace(config.namespace).component(config.component)
    await component.create_service()

    endpoint = component.endpoint(config.endpoint)
    await register_llm(ModelType.Chat, endpoint, config.model)

    server_args = ServerArgs(model_path=config.model)
    tokenizer_manager, _scheduler_info = _launch_subprocesses(server_args=server_args)

    # the server will gracefully shutdown (i.e., keep opened TCP streams finishes)
    # after the lease is revoked
    await endpoint.serve_endpoint(RequestHandler(tokenizer_manager).generate)


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
