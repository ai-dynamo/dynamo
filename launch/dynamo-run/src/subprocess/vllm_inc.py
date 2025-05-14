# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# `dynamo-run out=vllm` runs this script
# Can also be used standalone: `python3 vllm_inc.py` - lots of optional cmd line params

# Setup checklist:
# - We are in a virtualenv with vllm installed - and patched if using kv routing.
# - `libdynamo_llm_capi.so` is in system lib path or it's containing folder is in LD_LIBRARY_PATH
#   It builds in target/debug/ by default.

import argparse
import asyncio
import logging
import os
import sys
import uuid
from typing import Optional

import uvloop
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.openai.api_server import (
    build_async_engine_client_from_engine_args,
)
from vllm.inputs import TokensPrompt

from dynamo.llm import KvMetricsPublisher, ModelType, register_llm
from dynamo.runtime import DistributedRuntime, dynamo_worker

# Only used if you run it manually from the command line
DEFAULT_ENDPOINT = "dyn://dynamo.backend.generate"
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"

logging.basicConfig(level=logging.DEBUG)


class Config:
    """Command line parameters or defaults"""

    namespace: str
    component: str
    endpoint: str
    model_path: str
    model_name: Optional[str]
    tensor_parallel_size: int
    kv_block_size: int
    extra_engine_args: str


class RequestHandler:
    """
    Request handler for the generate endpoint
    """

    def __init__(self, component, engine, default_sampling_params):
        self.component = component
        self.engine_client = engine
        self.default_sampling_params = default_sampling_params
        self.metrics_publisher = KvMetricsPublisher()

    def setup_kv_metrics(self):
        if not hasattr(self.engine_client, "set_metrics_publisher"):
            logging.debug("VLLM version does not support KV metrics")
            return

        self.engine_client.set_metrics_publisher(self.metrics_publisher)
        # Initially send dummy metrics to kick start,
        # vLLM will not update stat until forward pass is triggered
        self.metrics_publisher.publish(
            0,  # request_active_slots
            1024,  # request_total_slots
            0,  # kv_active_blocks
            1024,  # kv_total_blocks
            0,  # num_requests_waiting
            0.0,  # gpu_cache_usage_perc
            0.0,  # gpu_prefix_cache_hit_rate
        )
        task = asyncio.create_task(self.create_metrics_publisher_endpoint())
        task.add_done_callback(
            lambda _: logging.debug("metrics publisher endpoint created")
        )

    async def create_metrics_publisher_endpoint(self):
        logging.debug("Creating metrics publisher endpoint")
        await self.metrics_publisher.create_endpoint(self.component)

    async def generate(self, request):
        # logging.debug(f"Received request: {request}")
        request_id = str(uuid.uuid4().hex)

        prompt = TokensPrompt(prompt_token_ids=request["token_ids"])

        sampling_params = SamplingParams(**self.default_sampling_params)
        for key, value in request["sampling_options"].items():
            if not value:
                continue
            if hasattr(sampling_params, key):
                setattr(sampling_params, key, value)

        max_tokens = request["stop_conditions"]["max_tokens"]
        if max_tokens:
            sampling_params.max_tokens = max_tokens

        num_output_tokens_so_far = 0
        gen = self.engine_client.generate(prompt, sampling_params, request_id)
        async for res in gen:
            # res is vllm's RequestOutput

            # This is the expected way for a request to end.
            # The new token ID will be eos, don't forward it.
            if res.finished:
                yield {"finish_reason": "stop", "token_ids": []}
                break

            if not res.outputs:
                yield {"finish_reason": "error", "token_ids": []}
                break

            output = res.outputs[0]
            next_total_toks = len(output.token_ids)
            out = {"token_ids": output.token_ids[num_output_tokens_so_far:]}
            if output.finish_reason:
                out["finish_reason"] = output.finish_reason
            if output.stop_reason:
                out["stop_reason"] = output.stop_reason
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
    await register_llm(
        ModelType.Backend, endpoint, config.model_path, config.model_name
    )

    arg_map = {
        "model": config.model_path,
        "task": "generate",
        "tensor_parallel_size": config.tensor_parallel_size,
        "skip_tokenizer_init": True,
        "disable_log_requests": True,
        "enable_prefix_caching": True,
        "block_size": config.kv_block_size,
        # KV routing relies on logging KV metrics
        "disable_log_stats": False,
    }
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

    # Patch won't start KVCacheEventManager unless these four are set
    os.environ["VLLM_WORKER_ID"] = str(endpoint.lease_id())
    os.environ[
        "VLLM_KV_CAPI_PATH"
    ] = "libdynamo_llm_capi.so"  # Must be on LD_LIBRARY_PATH
    os.environ["VLLM_KV_NAMESPACE"] = config.namespace
    os.environ["VLLM_KV_COMPONENT"] = config.component

    os.environ["VLLM_NO_USAGE_STATS"] = "1"  # Avoid internal HTTP requests
    engine_args = AsyncEngineArgs(**arg_map)
    model_config = engine_args.create_model_config()
    # Load default sampling params from `generation_config.json`
    default_sampling_params = model_config.get_diff_sampling_param()

    engine_context = build_async_engine_client_from_engine_args(engine_args)
    engine_client = await engine_context.__aenter__()

    handler = RequestHandler(component, engine_client, default_sampling_params)
    handler.setup_kv_metrics()

    # the server will gracefully shutdown (i.e., keep opened TCP streams finishes)
    # after the lease is revoked
    await endpoint.serve_endpoint(handler.generate)


def cmd_line_args():
    parser = argparse.ArgumentParser(
        description="vLLM server integrated with Dynamo LLM."
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default=DEFAULT_ENDPOINT,
        help=f"Dynamo endpoint string in 'dyn://namespace.component.endpoint' format. Default: {DEFAULT_ENDPOINT}",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Path to disk model or HuggingFace model identifier to load. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="",
        help="Name to serve the model under. Defaults to deriving it from model path.",
    )
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=1, help="Number of GPUs to use."
    )
    parser.add_argument(
        "--kv-block-size", type=int, default=16, help="Size of a KV cache block."
    )
    parser.add_argument(
        "--extra-engine-args",
        type=str,
        default="",
        help="Path to a JSON file containing additional keyword arguments to pass to the vLLM AsyncLLMEngine.",
    )
    args = parser.parse_args()

    config = Config()
    config.model_path = args.model_path
    if args.model_name:
        config.model_name = args.model_name
    else:
        # This becomes an `Option` on the Rust side
        config.model_name = None

    endpoint_str = args.endpoint.replace("dyn://", "", 1)
    endpoint_parts = endpoint_str.split(".")
    if len(endpoint_parts) != 3:
        logging.error(
            f"Invalid endpoint format: '{args.endpoint}'. Expected 'dyn://namespace.component.endpoint' or 'namespace.component.endpoint'."
        )
        sys.exit(1)

    parsed_namespace, parsed_component_name, parsed_endpoint_name = endpoint_parts

    config.namespace = parsed_namespace
    config.component = parsed_component_name
    config.endpoint = parsed_endpoint_name
    config.tensor_parallel_size = args.tensor_parallel_size
    config.kv_block_size = args.kv_block_size
    config.extra_engine_args = args.extra_engine_args

    return config


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(worker())
