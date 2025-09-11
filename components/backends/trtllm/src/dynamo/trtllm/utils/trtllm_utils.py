# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
from typing import Optional

from tensorrt_llm.llmapi import BuildConfig

from dynamo._core import get_reasoning_parser_names, get_tool_parser_names
from dynamo.trtllm import __version__
from dynamo.trtllm.request_handlers.handler_base import (
    DisaggregationMode,
    DisaggregationStrategy,
)

DYN_NAMESPACE = os.environ.get("DYN_NAMESPACE", "dynamo")

# Default endpoint for the next worker.
DEFAULT_ENDPOINT = f"dyn://{DYN_NAMESPACE}.tensorrt_llm.generate"
DEFAULT_MODEL_PATH = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_NEXT_ENDPOINT = f"dyn://{DYN_NAMESPACE}.tensorrt_llm_next.generate"
DEFAULT_ENCODE_ENDPOINT = f"dyn://{DYN_NAMESPACE}.tensorrt_llm_encode.generate"
DEFAULT_DISAGGREGATION_STRATEGY = DisaggregationStrategy.DECODE_FIRST
DEFAULT_DISAGGREGATION_MODE = DisaggregationMode.AGGREGATED


class Config:
    """Command line parameters or defaults"""

    def __init__(self) -> None:
        self.namespace: str = ""
        self.component: str = ""
        self.endpoint: str = ""
        self.model_path: str = ""
        self.served_model_name: Optional[str] = None
        self.tensor_parallel_size: int = 1
        self.pipeline_parallel_size: int = 1
        self.expert_parallel_size: Optional[int] = None
        self.kv_block_size: int = 32
        self.migration_limit: int = 0
        self.gpus_per_node: Optional[int] = None
        self.max_batch_size: int = BuildConfig.max_batch_size
        self.max_num_tokens: int = BuildConfig.max_num_tokens
        self.max_seq_len: int = BuildConfig.max_seq_len
        self.max_beam_width: int = BuildConfig.max_beam_width
        self.free_gpu_memory_fraction: Optional[float] = None
        self.extra_engine_args: str = ""
        self.override_engine_args: str = ""
        self.publish_events_and_metrics: bool = False
        self.disaggregation_mode: DisaggregationMode = DEFAULT_DISAGGREGATION_MODE
        self.disaggregation_strategy: DisaggregationStrategy = (
            DEFAULT_DISAGGREGATION_STRATEGY
        )
        self.next_endpoint: str = ""
        self.encode_endpoint: str = ""
        self.modality: str = "text"
        self.allowed_local_media_path: str = ""
        self.max_file_size_mb: int = 50
        self.reasoning_parser: Optional[str] = None
        self.tool_call_parser: Optional[str] = None

    def __str__(self) -> str:
        return (
            f"Config(namespace={self.namespace}, "
            f"component={self.component}, "
            f"endpoint={self.endpoint}, "
            f"model_path={self.model_path}, "
            f"served_model_name={self.served_model_name}, "
            f"tensor_parallel_size={self.tensor_parallel_size}, "
            f"pipeline_parallel_size={self.pipeline_parallel_size}, "
            f"expert_parallel_size={self.expert_parallel_size}, "
            f"kv_block_size={self.kv_block_size}, "
            f"gpus_per_node={self.gpus_per_node}, "
            f"max_batch_size={self.max_batch_size}, "
            f"max_num_tokens={self.max_num_tokens}, "
            f"max_seq_len={self.max_seq_len}, "
            f"max_beam_width={self.max_beam_width}, "
            f"free_gpu_memory_fraction={self.free_gpu_memory_fraction}, "
            f"extra_engine_args={self.extra_engine_args}, "
            f"override_engine_args={self.override_engine_args}, "
            f"migration_limit={self.migration_limit}, "
            f"publish_events_and_metrics={self.publish_events_and_metrics}, "
            f"disaggregation_mode={self.disaggregation_mode}, "
            f"disaggregation_strategy={self.disaggregation_strategy}, "
            f"next_endpoint={self.next_endpoint}, "
            f"encode_endpoint={self.encode_endpoint}, "
            f"modality={self.modality}, "
            f"allowed_local_media_path={self.allowed_local_media_path}, "
            f"max_file_size_mb={self.max_file_size_mb}, "
            f"reasoning_parser={self.reasoning_parser}, "
            f"tool_call_parser={self.tool_call_parser}"
        )


def is_first_worker(config):
    """
    Check if the current worker is the first worker in the disaggregation chain.
    """
    is_primary_worker = config.disaggregation_mode == DisaggregationMode.AGGREGATED
    if not is_primary_worker:
        is_primary_worker = (
            config.disaggregation_strategy == DisaggregationStrategy.PREFILL_FIRST
        ) and (config.disaggregation_mode == DisaggregationMode.PREFILL)

    if not is_primary_worker:
        is_primary_worker = (
            config.disaggregation_strategy == DisaggregationStrategy.DECODE_FIRST
        ) and (config.disaggregation_mode == DisaggregationMode.DECODE)

    return is_primary_worker


def parse_endpoint(endpoint: str) -> tuple[str, str, str]:
    endpoint_str = endpoint.replace("dyn://", "", 1)
    endpoint_parts = endpoint_str.split(".")
    if len(endpoint_parts) != 3:
        raise ValueError(
            f"Invalid endpoint format: '{endpoint}'. "
            "Expected 'dyn://namespace.component.endpoint' or 'namespace.component.endpoint'."
        )
    namespace, component, endpoint_name = endpoint_parts
    return namespace, component, endpoint_name


def cmd_line_args():
    parser = argparse.ArgumentParser(
        description="TensorRT-LLM server integrated with Dynamo LLM."
    )
    parser.add_argument(
        "--version", action="version", version=f"Dynamo Backend TRTLLM {__version__}"
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default="",
        help=f"Dynamo endpoint string in 'dyn://namespace.component.endpoint' format. Default: {DEFAULT_ENDPOINT} if first worker, {DEFAULT_NEXT_ENDPOINT} if next worker",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to disk model or HuggingFace model identifier to load. Default: {DEFAULT_MODEL_PATH}",
    )
    parser.add_argument(
        "--served-model-name",
        type=str,
        default="",
        help="Name to serve the model under. Defaults to deriving it from model path.",
    )
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=1, help="Tensor parallelism size."
    )
    parser.add_argument(
        "--pipeline-parallel-size",
        type=int,
        default=None,
        help="Pipeline parallelism size.",
    )
    parser.add_argument(
        "--expert-parallel-size",
        type=int,
        default=None,
        help="expert parallelism size.",
    )

    # IMPORTANT: We should ideally not expose this to users. We should be able to
    # query the block size from the TRTLLM engine.
    parser.add_argument(
        "--kv-block-size", type=int, default=32, help="Size of a KV cache block."
    )
    parser.add_argument(
        "--migration-limit",
        type=int,
        default=0,
        help="Maximum number of times a request may be migrated to a different engine worker. The number may be overridden by the engine.",
    )
    parser.add_argument(
        "--gpus-per-node",
        type=int,
        default=None,
        help="Number of GPUs per node. If not provided, will be inferred from the environment.",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=BuildConfig.max_batch_size,
        help="Maximum number of requests that the engine can schedule.",
    )
    parser.add_argument(
        "--max-num-tokens",
        type=int,
        default=BuildConfig.max_num_tokens,
        help="Maximum number of batched input tokens after padding is removed in each batch.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=BuildConfig.max_seq_len,
        help="Maximum total length of one request, including prompt and outputs. "
        "If unspecified, the value is deduced from the model config.",
    )
    parser.add_argument(
        "--max-beam-width",
        type=int,
        default=BuildConfig.max_beam_width,
        help="Maximum number of beams for beam search decoding.",
    )
    parser.add_argument(
        "--free-gpu-memory-fraction",
        type=float,
        default=None,
        help="Free GPU memory fraction reserved for KV Cache, after allocating model weights and buffers.",
    )

    parser.add_argument(
        "--extra-engine-args",
        type=str,
        default="",
        help="Path to a YAML file containing additional keyword arguments to pass to the TRTLLM engine.",
    )
    parser.add_argument(
        "--override-engine-args",
        type=str,
        default="",
        help='Python dictionary string to override specific engine arguments from the YAML file. Example: \'{"tensor_parallel_size": 2, "kv_cache_config": {"enable_block_reuse": false}}\'',
    )
    parser.add_argument(
        "--publish-events-and-metrics",
        action="store_true",
        help="If set, publish events and metrics to the dynamo components.",
    )
    parser.add_argument(
        "--disaggregation-mode",
        type=str,
        default=DEFAULT_DISAGGREGATION_MODE,
        choices=[mode.value for mode in DisaggregationMode],
        help=f"Mode to use for disaggregation. Default: {DEFAULT_DISAGGREGATION_MODE}",
    )
    parser.add_argument(
        "--use-nixl-connect",
        type=bool,
        default=False,
        help="Use NIXL Connect for communication between workers.",
    )
    parser.add_argument(
        "--disaggregation-strategy",
        type=str,
        default=DEFAULT_DISAGGREGATION_STRATEGY,
        choices=[strategy.value for strategy in DisaggregationStrategy],
        help=f"Strategy to use for disaggregation. Default: {DEFAULT_DISAGGREGATION_STRATEGY}",
    )
    parser.add_argument(
        "--modality",
        type=str,
        default="text",
        choices=["text", "multimodal"],
        help="Modality to use for the model. Default: text. Current supported modalities are image.",
    )
    parser.add_argument(
        "--next-endpoint",
        type=str,
        default="",
        help=f"Endpoint(in 'dyn://namespace.component.endpoint' format) to send requests to when running in disaggregation mode. Default: {DEFAULT_NEXT_ENDPOINT} if first worker, empty if next worker",
    )
    parser.add_argument(
        "--encode-endpoint",
        type=str,
        default="",
        help=f"Endpoint(in 'dyn://namespace.component.endpoint' format) for the encode worker. Default: {DEFAULT_ENCODE_ENDPOINT}",
    )
    parser.add_argument(
        "--allowed-local-media-path",
        type=str,
        default="",
        help="Path to a directory that is allowed to be accessed by the model. Default: empty",
    )
    parser.add_argument(
        "--max-file-size-mb",
        type=int,
        default=50,
        help="Maximum size of downloadable embedding files/Image URLs. Default: 50MB",
    )
    # To avoid name conflicts with different backends, adoped prefix "dyn-" for dynamo specific args
    parser.add_argument(
        "--dyn-tool-call-parser",
        type=str,
        default=None,
        choices=get_tool_parser_names(),
        help="Tool call parser name for the model.",
    )
    parser.add_argument(
        "--dyn-reasoning-parser",
        type=str,
        default=None,
        choices=get_reasoning_parser_names(),
        help="Reasoning parser name for the model. If not specified, no reasoning parsing is performed.",
    )

    args = parser.parse_args()

    config = Config()
    # Set the model path and served model name.
    config.model_path = args.model_path
    if args.served_model_name:
        config.served_model_name = args.served_model_name
    else:
        # This becomes an `Option` on the Rust side
        config.served_model_name = None

    # Set the disaggregation mode and strategy.
    config.disaggregation_mode = DisaggregationMode(args.disaggregation_mode)
    config.disaggregation_strategy = DisaggregationStrategy(
        args.disaggregation_strategy
    )

    # Set the appropriate defaults for the endpoint and next endpoint.
    if is_first_worker(config):
        if args.endpoint == "":
            args.endpoint = DEFAULT_ENDPOINT
        if (
            args.next_endpoint == ""
            and config.disaggregation_mode != DisaggregationMode.AGGREGATED
        ):
            args.next_endpoint = DEFAULT_NEXT_ENDPOINT
    elif config.disaggregation_mode == DisaggregationMode.ENCODE:
        if args.endpoint == "":
            args.endpoint = DEFAULT_ENCODE_ENDPOINT
    else:
        if args.endpoint == "":
            args.endpoint = DEFAULT_NEXT_ENDPOINT
        if args.next_endpoint != "":
            raise ValueError("Next endpoint is not allowed for the next worker")
    endpoint = args.endpoint
    parsed_namespace, parsed_component_name, parsed_endpoint_name = parse_endpoint(
        endpoint
    )

    config.namespace = parsed_namespace
    config.component = parsed_component_name
    config.endpoint = parsed_endpoint_name
    config.next_endpoint = args.next_endpoint
    config.encode_endpoint = args.encode_endpoint
    config.allowed_local_media_path = args.allowed_local_media_path
    config.max_file_size_mb = args.max_file_size_mb

    config.tensor_parallel_size = args.tensor_parallel_size
    if args.pipeline_parallel_size is not None:
        config.pipeline_parallel_size = args.pipeline_parallel_size
    if args.expert_parallel_size is not None:
        config.expert_parallel_size = args.expert_parallel_size
    if args.gpus_per_node is not None:
        config.gpus_per_node = args.gpus_per_node
    if args.free_gpu_memory_fraction is not None:
        config.free_gpu_memory_fraction = args.free_gpu_memory_fraction
    config.max_batch_size = args.max_batch_size
    config.max_num_tokens = args.max_num_tokens
    config.max_seq_len = args.max_seq_len
    config.max_beam_width = args.max_beam_width
    config.kv_block_size = args.kv_block_size
    config.migration_limit = args.migration_limit
    config.extra_engine_args = args.extra_engine_args
    config.override_engine_args = args.override_engine_args
    config.publish_events_and_metrics = args.publish_events_and_metrics
    config.modality = args.modality

    config.reasoning_parser = args.dyn_reasoning_parser
    config.tool_call_parser = args.dyn_tool_call_parser

    return config
