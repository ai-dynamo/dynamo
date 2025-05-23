# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# TODO:
# - Add event and metrics publishers
# - Support default dynamo-run out=trtllm launch
# - Support disaggregated serving
#
# Can be used standalone: `python3 trtllm_inc.py` - lots of optional cmd line params

import argparse
import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

import uvloop

# Import TRTLLM and related modules
from tensorrt_llm import LLM, LlmArgs, SamplingParams
from tensorrt_llm.llmapi.llm_utils import update_llm_args_with_extra_options
from tensorrt_llm.llmapi.tokenizer import tokenizer_factory

from dynamo.llm import KvMetricsPublisher, ModelType, register_llm
from dynamo.runtime import DistributedRuntime, dynamo_worker

# Only used if you run it manually from the command line
DEFAULT_ENDPOINT = "dyn://dynamo.backend.generate"
# Qwen/Qwen3-0.6B is not supported by TRTLLM yet.
DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


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
        self.engine = engine
        self.component = component
        self.default_sampling_params = default_sampling_params
        self.metrics_publisher = KvMetricsPublisher()

    def setup_kv_metrics(self):
        # Initially send dummy metrics to kick start,
        # TRTLLM will not update stat until forward pass is triggered
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
        inputs = request["token_ids"]

        sampling_params = self.default_sampling_params
        for key, value in request["sampling_options"].items():
            if not value:
                continue
            if hasattr(sampling_params, key):
                setattr(sampling_params, key, value)

        max_tokens = request["stop_conditions"]["max_tokens"]
        if max_tokens:
            sampling_params.max_tokens = max_tokens

        num_output_tokens_so_far = 0
        # TODO: Disable streaming for context only requests when adding disagg support
        async for res in self.engine.llm.generate_async(
            inputs=inputs, sampling_params=sampling_params, streaming=True
        ):
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


class AsyncLLMEngine:
    def __init__(self, engine_args):
        self.engine_args = engine_args
        self._llm: Optional[LLM] = None
        self._initialized = False

    async def initialize(self):
        if not self._initialized:
            model = self.engine_args.pop("model")
            self._llm = LLM(
                model=model,
                **self.engine_args,
            )
            self._initialized = True

    async def cleanup(self):
        if self._initialized:
            try:
                self._llm.shutdown()
            except Exception as e:
                logging.error(f"Error during cleanup: {e}")
            finally:
                self._initialized = False

    @property
    def llm(self):
        if not self._initialized:
            raise RuntimeError("Engine not initialized")
        return self._llm


@asynccontextmanager
async def get_llm_engine(engine_args: LlmArgs) -> AsyncGenerator[AsyncLLMEngine, None]:
    engine = AsyncLLMEngine(engine_args)
    try:
        await engine.initialize()
        yield engine
    except Exception as e:
        logging.error(f"Error in engine context: {e}")
        raise
    finally:
        await engine.cleanup()


async def init(runtime: DistributedRuntime, config: Config):
    """
    Instantiate and serve
    """
    component = runtime.namespace(config.namespace).component(config.component)
    await component.create_service()

    # Convert model path to Path object if it's a local path, otherwise keep as string
    model_path = str(config.model_path)

    arg_map = {
        "model": model_path,
        "tensor_parallel_size": config.tensor_parallel_size,
        "skip_tokenizer_init": True,
        "disable_log_requests": True,
        "enable_prefix_caching": True,
        # KV routing relies on logging KV metrics
        "disable_log_stats": False,
    }
    if config.extra_engine_args != "":
        arg_map = update_llm_args_with_extra_options(arg_map, config.extra_engine_args)

    logging.debug(f"TRTLLM engine args: {arg_map}")
    engine_args = arg_map

    # Populate default sampling params from the model
    tokenizer = tokenizer_factory(arg_map["model"])
    default_sampling_params = SamplingParams()
    default_sampling_params._setup(tokenizer)
    default_sampling_params.stop = None

    async with get_llm_engine(engine_args) as engine:
        endpoint = component.endpoint(config.endpoint)
        await register_llm(
            ModelType.Backend, endpoint, config.model_path, config.model_name
        )
        handler = RequestHandler(component, engine, default_sampling_params)
        handler.setup_kv_metrics()

        # the server will gracefully shutdown (i.e., keep opened TCP streams finishes)
        # after the lease is revoked
        await endpoint.serve_endpoint(handler.generate)


def cmd_line_args():
    parser = argparse.ArgumentParser(
        description="TensorRT-LLM server integrated with Dynamo LLM."
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
        "--kv-block-size", type=int, default=32, help="Size of a KV cache block."
    )
    parser.add_argument(
        "--extra-engine-args",
        type=str,
        default="",
        help="Path to a YAML file containing additional keyword arguments to pass to the TRTLLM engine.",
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
