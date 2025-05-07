# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
    base_gpu_id: int
    tensor_parallel_size: int
    nnodes: int
    node_rank: int
    dist_init_addr: str
    extra_engine_args: str


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
    print("Started server instance")

    await register_llm(endpoint, config.model, ModelType.Backend)

    arg_map = {
        "model_path": config.model,
        "skip_tokenizer_init": True,
        "tp_size": config.tensor_parallel_size,
        "base_gpu_id": config.base_gpu_id,
    }
    if config.dist_init_addr != "":
        arg_map["trust_remote_code"] = True
        arg_map["nnodes"] = config.nnodes
        arg_map["dist_init_addr"] = config.dist_init_addr
        # In practice this is always 0 because Dynamo only manages the leader
        arg_map["node_rank"] = config.node_rank

    if config.extra_engine_args != "":
        json_map = {}
        # extra_engine_args is a filename
        try:
            with open(config.extra_engine_args) as f:
                json_map = json.load(f)
        except FileNotFoundError:
            logging.error(f"File {config.extra_engine_args} not found.")
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON in {config.extra_engine_args}: {e}")
        logging.debug(f"Adding extra engine arguments: {json_map}")
        arg_map = {**arg_map, **json_map}  # json_map gets precedence

    engine_args = ServerArgs(**arg_map)
    engine_client = sglang.Engine(server_args=engine_args)

    # the server will gracefully shutdown (i.e., keep opened TCP streams finishes)
    # after the lease is revoked
    await endpoint.serve_endpoint(RequestHandler(engine_client).generate, None)


def cmd_line_args():
    parser = argparse.ArgumentParser(
        description="SGLang server integrated with Dynamo LLM."
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
    parser.add_argument(
        "--base-gpu-id",
        type=int,
        default=0,
        help="The base GPU ID to start allocating GPUs from. Useful when running multiple instances on the same machine.",
    )
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=1, help="Number of GPUs to use."
    )
    parser.add_argument(
        "--nnodes", type=int, default=1, help="The number of machines SGLang will use"
    )
    parser.add_argument(
        "--node-rank",
        type=int,
        default=0,
        help="Unique number for each node. 0 for the leader.",
    )
    parser.add_argument(
        "--dist-init-addr",
        type=str,
        default="",
        help="Host address (e.g., `192.168.0.2:25000`) of the node with rank 0",
    )
    parser.add_argument(
        "--extra-engine-args",
        type=str,
        default="",
        help="Path to a JSON file containing additional keyword arguments to pass to the SGLang Engine.",
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
    config.base_gpu_id = args.base_gpu_id
    config.tensor_parallel_size = args.tensor_parallel_size
    config.nnodes = args.nnodes
    config.node_rank = args.node_rank
    config.dist_init_addr = args.dist_init_addr
    config.extra_engine_args = args.extra_engine_args

    return config


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(worker())
