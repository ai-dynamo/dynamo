# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# TODO:
# - Support disaggregated serving
# - Update examples to use this engine.
#
# `dynamo-run out=trtllm` runs this script
# Can be used standalone: `python3 trtllm_inc.py` - lots of optional cmd line params
#
# Disaggregated serving:
# - Ingress: dynamo run in=http out=dyn
# - Decode Worker: python3 trtllm_inc.py --disagg-mode=decode --extra-engine-args=trtllm_config/sample.yaml
# - Prefill Worker: python3 trtllm_inc.py --disagg-mode=prefill --extra-engine-args=trtllm_config/sample.yaml


import argparse
import asyncio
import base64
import copy
import logging
import sys
import warnings
from dataclasses import asdict
from typing import Optional

import uvloop

# Import TRTLLM and related modules
from tensorrt_llm import SamplingParams
from tensorrt_llm.llmapi import DisaggregatedParams
from tensorrt_llm.llmapi.llm_utils import update_llm_args_with_extra_options
from tensorrt_llm.llmapi.tokenizer import tokenizer_factory

from dynamo.llm import (
    ModelType,
    get_tensorrtllm_engine,
    get_tensorrtllm_publisher,
    register_llm,
)
from dynamo.runtime import DistributedRuntime, dynamo_worker

# Only used if you run it manually from the command line
DEFAULT_ENDPOINT = "dyn://dynamo.backend.generate"
# Qwen/Qwen3-0.6B is not supported by TRTLLM yet.
DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# Default endpoint for the remote prefill service.
DEFAULT_PREFILL_ENDPOINT = "dyn://dynamo.backend.prefill"

# Default buffer size for kv cache events.
DEFAULT_KV_EVENT_BUFFER_MAX_SIZE = 1024

logging.basicConfig(level=logging.DEBUG)


def parse_endpoint(endpoint: str) -> tuple[str, str, str]:
    endpoint_str = endpoint.replace("dyn://", "", 1)
    endpoint_parts = endpoint_str.split(".")
    if len(endpoint_parts) != 3:
        logging.error(
            f"Invalid endpoint format: '{endpoint}'. Expected 'dyn://namespace.component.endpoint' or 'namespace.component.endpoint'."
        )
        sys.exit(1)

    return tuple(endpoint_parts)


class DisaggregatedParamsCodec:
    """
    Codec for encoding and decoding disaggregated params for network transfer.
    """

    @staticmethod
    def decode(
        disaggregated_params: DisaggregatedParams,
    ) -> DisaggregatedParams:
        if disaggregated_params is None:
            return None
        else:
            opaque_state = (
                base64.b64decode(disaggregated_params.opaque_state)
                if disaggregated_params.opaque_state is not None
                else None
            )
            disaggregated_params.opaque_state = opaque_state
            return disaggregated_params

    @staticmethod
    def encode(
        disaggregated_params: DisaggregatedParams,
    ) -> DisaggregatedParams:
        if disaggregated_params is None:
            return None
        else:
            encoded_opaque_state = (
                base64.b64encode(disaggregated_params.opaque_state).decode("utf-8")
                if disaggregated_params.opaque_state is not None
                else None
            )
            disaggregated_params.opaque_state = encoded_opaque_state
            return disaggregated_params


class Config:
    """Command line parameters or defaults"""

    namespace: str
    component: str
    endpoint: str
    model_path: str
    model_name: Optional[str] = None
    tensor_parallel_size: int
    kv_block_size: int
    extra_engine_args: str
    publish_events_and_metrics: bool
    disaggregation_mode: str
    remote_prefill_endpoint: str


class RequestHandler:
    """
    Request handler for the generate endpoint
    """

    def __init__(
        self,
        component,
        engine,
        default_sampling_params,
        publishers,
        disaggregation_mode,
        remote_prefill_client,
    ):
        self.engine = engine
        self.component = component
        self.default_sampling_params = default_sampling_params
        self.publishers = publishers
        self.disaggregation_mode = disaggregation_mode
        self.remote_prefill_client = remote_prefill_client
        self.first_generation = True

    async def remote_prefill(self, request):
        prefill_request = copy.deepcopy(request)
        prefill_request["stop_conditions"]["max_tokens"] = 1

        # Set the disaggregated params to context_only for remote prefill
        prefill_request["disaggregated_params"] = asdict(
            DisaggregatedParamsCodec.encode(
                DisaggregatedParams(request_type="context_only")
            )
        )

        if self.remote_prefill_client is None:
            raise ValueError("Prefill client not initialized")

        # TODO: Use smart KV router to determine which prefill worker to use. This would also require supporting publishing events for prefill workers.
        remote_prefill_responses = [
            remote_prefill_response
            async for remote_prefill_response in await self.remote_prefill_client.round_robin(
                prefill_request
            )
        ]
        if len(remote_prefill_responses) > 1:
            raise ValueError(
                "Prefill worker returned more than one response. This is currently not supported in remote prefill mode."
            )
        remote_prefill_response = remote_prefill_responses[0]
        return remote_prefill_response

    async def generate(self, request):
        # Check if there is an error in the publishers error queue
        publishers_error = (
            self.publishers.check_error_queue() if self.publishers else None
        )
        if publishers_error:
            raise publishers_error

        inputs = request["token_ids"]

        # Decode the disaggregated params from the request
        if "disaggregated_params" in request:
            disaggregated_params = DisaggregatedParamsCodec.decode(
                DisaggregatedParams(**request["disaggregated_params"])
            )
        else:
            disaggregated_params = None

        num_output_tokens_so_far = 0

        if self.disaggregation_mode == "decode":
            # Run prefill/context phase remotely if disaggregation mode is decode.
            prefill_result = await self.remote_prefill(request)
            remote_prefill_response = prefill_result.data()
            if (
                remote_prefill_response["finish_reason"] == "stop"
                or remote_prefill_response["finish_reason"] == "error"
            ):
                yield remote_prefill_response
                return
            num_output_tokens_so_far = len(remote_prefill_response["token_ids"])

            # Decode the disaggregated params from the remote prefill response
            disaggregated_params = DisaggregatedParamsCodec.decode(
                DisaggregatedParams(**remote_prefill_response["disaggregated_params"])
            )

            # Send the first token response to the client
            first_token_response = remote_prefill_response
            first_token_response.pop("disaggregated_params")
            yield first_token_response

            # Set the disaggregated params to generation_only for the rest of the generation
            disaggregated_params.request_type = "generation_only"

        sampling_params = self.default_sampling_params
        for key, value in request["sampling_options"].items():
            if not value:
                continue
            if hasattr(sampling_params, key):
                setattr(sampling_params, key, value)

        max_tokens = request["stop_conditions"]["max_tokens"]
        if max_tokens:
            sampling_params.max_tokens = max_tokens

        # TODO: Disable streaming for context only requests when adding disagg support
        async for res in self.engine.llm.generate_async(
            inputs=inputs,
            sampling_params=sampling_params,
            disaggregated_params=disaggregated_params,
            streaming=(self.disaggregation_mode != "prefill"),
        ):
            # TRTLLM engine needs to start generating tokens first before stats
            # can be retrieved.
            if self.first_generation and self.publishers:
                self.publishers.start()
                self.first_generation = False

            if res.finished and self.disaggregation_mode != "prefill":
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
            if self.disaggregation_mode == "prefill":
                # Return the disaggregated params only when operating in prefill mode.
                out["disaggregated_params"] = asdict(
                    DisaggregatedParamsCodec.encode(output.disaggregated_params)
                )
            yield out
            num_output_tokens_so_far = next_total_toks


@dynamo_worker(static=False)
async def worker(runtime: DistributedRuntime):
    await init(runtime, cmd_line_args())


async def init(runtime: DistributedRuntime, config: Config):
    """
    Instantiate and serve
    """

    remote_prefill_client = None
    if config.disaggregation_mode == "decode":
        logging.info(
            f"Initializing remote prefill client for endpoint: {config.remote_prefill_endpoint}"
        )
        parsed_namespace, parsed_component_name, parsed_endpoint_name = parse_endpoint(
            config.remote_prefill_endpoint
        )
        remote_prefill_client = (
            await runtime.namespace(parsed_namespace)
            .component(parsed_component_name)
            .endpoint(parsed_endpoint_name)
            .client()
        )

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
        # TODO: Support extra engine args from json file as well.
        arg_map = update_llm_args_with_extra_options(arg_map, config.extra_engine_args)
    if config.publish_events_and_metrics:
        # 'event_buffer_max_size' is required to enable TRTLLM to publish kv cache events.
        kv_cache_config = None
        if "kv_cache_config" not in arg_map:
            kv_cache_config = {}
            kv_cache_config["event_buffer_max_size"] = DEFAULT_KV_EVENT_BUFFER_MAX_SIZE
        else:
            kv_cache_config = arg_map["kv_cache_config"]
            if not kv_cache_config.event_buffer_max_size:
                kv_cache_config.event_buffer_max_size = DEFAULT_KV_EVENT_BUFFER_MAX_SIZE
        arg_map["kv_cache_config"] = kv_cache_config

        # Only pytorch backend is supported for now to publish events and metrics.
        if "backend" not in arg_map:
            arg_map["backend"] = "pytorch"
        elif arg_map["backend"] != "pytorch":
            logging.error(
                "Only pytorch backend is supported for now to publish events and metrics."
            )
            sys.exit(1)

    logging.info(f"TRTLLM engine args: {arg_map}")
    engine_args = arg_map

    # Populate default sampling params from the model
    tokenizer = tokenizer_factory(arg_map["model"])
    default_sampling_params = SamplingParams()
    default_sampling_params._setup(tokenizer)
    default_sampling_params.stop = None

    async with get_tensorrtllm_engine(engine_args) as engine:
        endpoint = component.endpoint(config.endpoint)

        if config.disaggregation_mode != "prefill":
            # Register the model with the endpoint if disaggregation mode is not prefill.
            # Prefill worker will get the request directly from the Decode worker and not
            # through the ingress.
            # FIXME: Enable publishing events and metrics for disaggregated prefill.
            # Currently prefill workers are chosen in round-robin fashion.
            await register_llm(
                ModelType.Backend,
                endpoint,
                config.model_path,
                config.model_name,
                kv_cache_block_size=config.kv_block_size,
            )

        if (
            config.publish_events_and_metrics
            and config.disaggregation_mode != "prefill"
        ):
            # Initialize and pass in the publishers to the request handler to
            # publish events and metrics.
            kv_listener = runtime.namespace(config.namespace).component(
                config.component
            )
            async with get_tensorrtllm_publisher(
                component,
                engine,
                kv_listener,
                int(endpoint.lease_id()),
                config.kv_block_size,
            ) as publisher:
                handler = RequestHandler(
                    component,
                    engine,
                    default_sampling_params,
                    publisher,
                    config.disaggregation_mode,
                    remote_prefill_client,
                )
                await endpoint.serve_endpoint(handler.generate)
        else:
            # No publishers, so just pass in None to the request handler.
            handler = RequestHandler(
                component,
                engine,
                default_sampling_params,
                None,
                config.disaggregation_mode,
                remote_prefill_client,
            )
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
    # IMPORTANT: We should ideally not expose this to users. We should be able to
    # query the block size from the TRTLLM engine.
    parser.add_argument(
        "--kv-block-size", type=int, default=32, help="Size of a KV cache block."
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=None,
        help="This argument is not used by TRTLLM. Please provide max_input_len, max_seq_len and max_output_len in yaml file and point --extra-engine-args to the yaml file.",
    )
    parser.add_argument(
        "--extra-engine-args",
        type=str,
        default="",
        help="Path to a YAML file containing additional keyword arguments to pass to the TRTLLM engine.",
    )
    parser.add_argument(
        "--publish-events-and-metrics",
        action="store_true",
        help="Publish events and metrics to the dynamo components. Note: This is not supported when running in prefill disaggregation mode.",
    )
    parser.add_argument(
        "--disagg-mode",
        type=str,
        choices=["none", "prefill", "decode"],
        default="none",
        help="Specifies the disaggregation mode for the engine. none: aggregated serving, prefill: disaggregated prefill, decode: disaggregated decode",
    )
    parser.add_argument(
        "--remote-prefill-endpoint",
        type=str,
        default=DEFAULT_PREFILL_ENDPOINT,
        help=f"Endpoint(in 'dyn://namespace.component.endpoint' format) to send prefill requests to when running in decode disaggregation mode. Default: {DEFAULT_PREFILL_ENDPOINT}",
    )
    args = parser.parse_args()

    # Validate arguments
    if args.context_length is not None:
        warnings.warn(
            "--context-length is accepted for compatibility but will be ignored for TensorRT-LLM. Please provide max_input_len, max_seq_len and max_output_len in yaml file and point --extra-engine-args to the yaml file.",
            UserWarning,
        )

    endpoint = args.endpoint
    if args.disagg_mode == "prefill":
        if args.remote_prefill_endpoint != DEFAULT_PREFILL_ENDPOINT:
            logging.error(
                "--remote-prefill-endpoint is not supported when running in prefill disaggregation mode."
            )
            sys.exit(1)
        else:
            endpoint = DEFAULT_PREFILL_ENDPOINT

        if args.publish_events_and_metrics:
            warnings.warn(
                "--publish-events-and-metrics is not supported when running in prefill disaggregation mode.",
                UserWarning,
            )

    config = Config()
    config.model_path = args.model_path
    if args.model_name:
        config.model_name = args.model_name
    else:
        # This becomes an `Option` on the Rust side
        config.model_name = None

    parsed_namespace, parsed_component_name, parsed_endpoint_name = parse_endpoint(
        endpoint
    )

    config.namespace = parsed_namespace
    config.component = parsed_component_name
    config.endpoint = parsed_endpoint_name
    config.tensor_parallel_size = args.tensor_parallel_size
    config.kv_block_size = args.kv_block_size
    config.extra_engine_args = args.extra_engine_args
    config.publish_events_and_metrics = args.publish_events_and_metrics
    config.disaggregation_mode = args.disagg_mode
    config.remote_prefill_endpoint = args.remote_prefill_endpoint

    return config


if __name__ == "__main__":
    uvloop.install()
    try:
        asyncio.run(worker())
    except KeyboardInterrupt:
        logging.info("Received SIGINT, shutting down...")
