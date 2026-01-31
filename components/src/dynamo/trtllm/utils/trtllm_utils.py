# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
from typing import Optional

from tensorrt_llm.llmapi import BuildConfig

from dynamo._core import get_reasoning_parser_names, get_tool_parser_names
from dynamo.common.config_dump import add_config_dump_args, register_encoder
from dynamo.trtllm import __version__
from dynamo.trtllm.request_handlers.handler_base import DisaggregationMode

DYN_NAMESPACE = os.environ.get("DYN_NAMESPACE", "dynamo")

# Default endpoints for TensorRT-LLM workers
DEFAULT_ENDPOINT = (
    f"dyn://{DYN_NAMESPACE}.tensorrt_llm.generate"  # Decode/aggregated workers
)
DEFAULT_PREFILL_ENDPOINT = f"dyn://{DYN_NAMESPACE}.prefill.generate"  # Prefill workers
DEFAULT_ENCODE_ENDPOINT = (
    f"dyn://{DYN_NAMESPACE}.tensorrt_llm_encode.generate"  # Encode workers
)
DEFAULT_MODEL_PATH = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
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
        self.max_batch_size: int = BuildConfig.model_fields["max_batch_size"].default
        self.max_num_tokens: int = BuildConfig.model_fields["max_num_tokens"].default
        self.max_seq_len: int = BuildConfig.model_fields["max_seq_len"].default
        self.max_beam_width: int = BuildConfig.model_fields["max_beam_width"].default
        self.free_gpu_memory_fraction: float = 0.9
        self.extra_engine_args: str = ""
        self.override_engine_args: str = ""
        self.publish_events_and_metrics: bool = False
        self.disaggregation_mode: DisaggregationMode = DEFAULT_DISAGGREGATION_MODE
        self.encode_endpoint: str = ""
        self.modality: str = "text"
        self.allowed_local_media_path: str = ""
        self.max_file_size_mb: int = 50
        self.reasoning_parser: Optional[str] = None
        self.tool_call_parser: Optional[str] = None
        self.dump_config_to: Optional[str] = None
        self.custom_jinja_template: Optional[str] = None
        self.dyn_endpoint_types: str = "chat,completions"
        self.store_kv: str = ""
        self.request_plane: str = ""
        self.event_plane: str = ""
        self.enable_local_indexer: bool = False
        # Whether to enable NATS for KV events (derived from publish_events_and_metrics)
        self.use_kv_events: bool = False

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
            f"encode_endpoint={self.encode_endpoint}, "
            f"modality={self.modality}, "
            f"allowed_local_media_path={self.allowed_local_media_path}, "
            f"max_file_size_mb={self.max_file_size_mb}, "
            f"reasoning_parser={self.reasoning_parser}, "
            f"tool_call_parser={self.tool_call_parser}, "
            f"dump_config_to={self.dump_config_to}, "
            f"custom_jinja_template={self.custom_jinja_template}, "
            f"store_kv={self.store_kv}, "
            f"request_plane={self.request_plane}, "
            f"event_plane={self.event_plane}, "
            f"enable_local_indexer={self.enable_local_indexer}, "
            f"use_kv_events={self.use_kv_events}"
        )


@register_encoder(Config)
def _preprocess_for_encode_config(
    obj: Config,
) -> dict:  # pyright: ignore[reportUnusedFunction]
    """Convert Config object to dictionary for encoding."""
    return obj.__dict__


def parse_endpoint(endpoint: str) -> tuple[str, str, str]:
    """Parse a Dynamo endpoint string into its components.

    Args:
        endpoint: Endpoint string in format 'namespace.component.endpoint'
            or 'dyn://namespace.component.endpoint'.

    Returns:
        Tuple of (namespace, component, endpoint_name).

    Raises:
        ValueError: If endpoint format is invalid.
    """
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
    """Parse command-line arguments for the TensorRT-LLM backend.

    Returns:
        Config: Parsed configuration object.
    """
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
        help=f"Dynamo endpoint string in 'dyn://namespace.component.endpoint' format. Default: {DEFAULT_ENDPOINT} for decode/aggregated, {DEFAULT_PREFILL_ENDPOINT} for prefill workers, or {DEFAULT_ENCODE_ENDPOINT} for encode workers",
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
        default=BuildConfig.model_fields["max_batch_size"].default,
        help="Maximum number of requests that the engine can schedule.",
    )
    parser.add_argument(
        "--max-num-tokens",
        type=int,
        default=BuildConfig.model_fields["max_num_tokens"].default,
        help="Maximum number of batched input tokens after padding is removed in each batch.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=BuildConfig.model_fields["max_seq_len"].default,
        help="Maximum total length of one request, including prompt and outputs. "
        "If unspecified, the value is deduced from the model config.",
    )
    parser.add_argument(
        "--max-beam-width",
        type=int,
        default=BuildConfig.model_fields["max_beam_width"].default,
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
        "--modality",
        type=str,
        default="text",
        choices=["text", "multimodal"],
        help="Modality to use for the model. Default: text. Current supported modalities are image.",
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
    parser.add_argument(
        "--connector",
        type=str,
        default="none",
        choices=["none", "kvbm"],
        help="Connector to use for the model.",
    )
    add_config_dump_args(parser)
    parser.add_argument(
        "--custom-jinja-template",
        type=str,
        default=None,
        help="Path to a custom Jinja template file to override the model's default chat template. This template will take precedence over any template found in the model repository.",
    )
    parser.add_argument(
        "--dyn-endpoint-types",
        type=str,
        default="chat,completions",
        help="Comma-separated list of endpoint types to enable. Options: 'chat', 'completions'. Default: 'chat,completions'. Use 'completions' for models without chat templates.",
    )
    parser.add_argument(
        "--store-kv",
        type=str,
        choices=["etcd", "file", "mem"],
        default=os.environ.get("DYN_STORE_KV", "etcd"),
        help="Which key-value backend to use: etcd, mem, file. Etcd uses the ETCD_* env vars (e.g. ETCD_ENDPOINTS) for connection details. File uses root dir from env var DYN_FILE_KV or defaults to $TMPDIR/dynamo_store_kv.",
    )
    parser.add_argument(
        "--request-plane",
        type=str,
        choices=["nats", "http", "tcp"],
        default=os.environ.get("DYN_REQUEST_PLANE", "tcp"),
        help="Determines how requests are distributed from routers to workers. 'tcp' is fastest [nats|http|tcp]",
    )
    parser.add_argument(
        "--event-plane",
        type=str,
        choices=["nats", "zmq"],
        default=os.environ.get("DYN_EVENT_PLANE", "nats"),
        help="Determines how events are published [nats|zmq]",
    )
    parser.add_argument(
        "--enable-local-indexer",
        type=str,
        choices=["true", "false"],
        default=os.environ.get("DYN_LOCAL_INDEXER", "false"),
        help="Enable worker-local KV indexer for tracking this worker's own KV cache state (can also be toggled with env var DYN_LOCAL_INDEXER).",
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

    # Set the disaggregation mode.
    config.disaggregation_mode = DisaggregationMode(args.disaggregation_mode)

    # Set the appropriate default for the endpoint based on disaggregation mode
    if args.endpoint == "":
        if config.disaggregation_mode == DisaggregationMode.ENCODE:
            args.endpoint = DEFAULT_ENCODE_ENDPOINT
        elif config.disaggregation_mode == DisaggregationMode.PREFILL:
            args.endpoint = DEFAULT_PREFILL_ENDPOINT
        else:
            # Decode and aggregated workers use "tensorrt_llm" component
            args.endpoint = DEFAULT_ENDPOINT
    endpoint = args.endpoint
    parsed_namespace, parsed_component_name, parsed_endpoint_name = parse_endpoint(
        endpoint
    )

    config.namespace = parsed_namespace
    config.component = parsed_component_name
    config.endpoint = parsed_endpoint_name
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
    config.dump_config_to = args.dump_config_to
    config.dyn_endpoint_types = args.dyn_endpoint_types
    config.store_kv = args.store_kv
    config.request_plane = args.request_plane
    config.event_plane = args.event_plane
    config.enable_local_indexer = str(args.enable_local_indexer).lower() == "true"
    # Derive use_kv_events from publish_events_and_metrics
    config.use_kv_events = config.publish_events_and_metrics
    config.connector = args.connector

    # Handle custom jinja template path expansion (environment variables and home directory)
    if args.custom_jinja_template:
        expanded_template_path = os.path.expandvars(
            os.path.expanduser(args.custom_jinja_template)
        )
        # Validate custom Jinja template file exists
        if not os.path.isfile(expanded_template_path):
            raise FileNotFoundError(
                f"Custom Jinja template file not found: {expanded_template_path}"
            )
        config.custom_jinja_template = expanded_template_path
    else:
        config.custom_jinja_template = None

    return config


def deep_update(target: dict, source: dict) -> None:
    """
    Recursively update nested dictionaries.

    Args:
        target: Dictionary to update
        source: Dictionary with new values
    """
    for key, value in source.items():
        if isinstance(value, dict) and key in target and isinstance(target[key], dict):
            deep_update(target[key], value)
        else:
            target[key] = value
