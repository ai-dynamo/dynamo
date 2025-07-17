# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
from typing import Optional

from dynamo.trtllm.utils.request_handlers.handler_base import (
    DisaggregationMode,
    DisaggregationStrategy,
)

# Default endpoint for the next worker.
DEFAULT_ENDPOINT = "dyn://dynamo.tensorrt_llm.generate"
DEFAULT_MODEL_PATH = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_NEXT_ENDPOINT = "dyn://dynamo.tensorrt_llm_next.generate"
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
        self.kv_block_size: int = 32
        self.extra_engine_args: str = ""
        self.publish_events_and_metrics: bool = False
        self.disaggregation_mode: DisaggregationMode = DEFAULT_DISAGGREGATION_MODE
        self.disaggregation_strategy: DisaggregationStrategy = (
            DEFAULT_DISAGGREGATION_STRATEGY
        )
        self.next_endpoint: str = ""

    def __str__(self) -> str:
        return (
            f"Config(namespace={self.namespace}, "
            f"component={self.component}, "
            f"endpoint={self.endpoint}, "
            f"model_path={self.model_path}, "
            f"served_model_name={self.served_model_name}, "
            f"tensor_parallel_size={self.tensor_parallel_size}, "
            f"kv_block_size={self.kv_block_size}, "
            f"extra_engine_args={self.extra_engine_args}, "
            f"publish_events_and_metrics={self.publish_events_and_metrics}, "
            f"disaggregation_mode={self.disaggregation_mode}, "
            f"disaggregation_strategy={self.disaggregation_strategy}, "
            f"next_endpoint={self.next_endpoint})"
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
        "--tensor-parallel-size", type=int, default=1, help="Number of GPUs to use."
    )
    # IMPORTANT: We should ideally not expose this to users. We should be able to
    # query the block size from the TRTLLM engine.
    parser.add_argument(
        "--kv-block-size", type=int, default=32, help="Size of a KV cache block."
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
        "--disaggregation-strategy",
        type=str,
        default=DEFAULT_DISAGGREGATION_STRATEGY,
        choices=[strategy.value for strategy in DisaggregationStrategy],
        help=f"Strategy to use for disaggregation. Default: {DEFAULT_DISAGGREGATION_STRATEGY}",
    )
    parser.add_argument(
        "--next-endpoint",
        type=str,
        default="",
        help=f"Endpoint(in 'dyn://namespace.component.endpoint' format) to send requests to when running in disaggregation mode. Default: {DEFAULT_NEXT_ENDPOINT} if first worker, empty if next worker",
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

    config.tensor_parallel_size = args.tensor_parallel_size
    config.kv_block_size = args.kv_block_size
    config.extra_engine_args = args.extra_engine_args
    config.publish_events_and_metrics = args.publish_events_and_metrics

    return config
