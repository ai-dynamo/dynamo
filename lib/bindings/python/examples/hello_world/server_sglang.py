# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#
# A very basic example of sglang worker handling pre-processed requests.
#
# Dynamo does the HTTP handling, prompt templating and tokenization, then forwards the
# request via NATS to this python script, which runs sglang.
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

import sglang
import uvloop
from sglang.srt.server_args import ServerArgs

from dynamo.llm import ModelType, register_llm
from dynamo.runtime import DistributedRuntime, dynamo_worker

DEFAULT_ENDPOINT = "dyn://dynamo.backend.generate"
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


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
            "temperature": request["sampling_options"]["temperature"],
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
            else:
                next_total_toks = len(res["output_ids"])
                out = {"token_ids": res["output_ids"][num_output_tokens_so_far:]}
            yield out
            num_output_tokens_so_far = next_total_toks


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
    await register_llm(ModelType.Backend, endpoint, config.model)

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
