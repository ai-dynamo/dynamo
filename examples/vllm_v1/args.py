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

import argparse
import json
import logging
import sys
from typing import Optional

from vllm.distributed.kv_events import KVEventsConfig

logger = logging.getLogger(__name__)

# Only used if you run it manually from the command line
DEFAULT_ENDPOINT = "dyn://dynamo.backend.generate"
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_METRICS_ENDPOINT_PORT = 5557


class Config:
    """Command line parameters or defaults"""

    namespace: str
    component: str
    endpoint: str
    metrics_endpoint_port: int
    model_path: str
    model_name: Optional[str]
    tensor_parallel_size: int
    kv_block_size: int
    context_length: int
    extra_engine_args: str


def create_vllm_arg_map(config: Config) -> dict:
    """
    Create vLLM engine argument mapping from config.

    Args:
        config: Configuration object containing model and engine parameters

    Returns:
        Dictionary of arguments for vLLM AsyncEngineArgs
    """
    arg_map = {
        "model": config.model_path,
        "task": "generate",
        "tensor_parallel_size": config.tensor_parallel_size,
        "skip_tokenizer_init": True,
        "disable_log_requests": True,
        "enable_prefix_caching": True,
        # KV routing relies on logging KV metrics
        "disable_log_stats": False,
        "kv_events_config": KVEventsConfig(
            enable_kv_cache_events=True,
            publisher="zmq",
            endpoint=f"tcp://*:{config.metrics_endpoint_port}",
        ),
    }

    if config.context_length:
        # Usually we want it to default to the max (from tokenizer_config.json)
        arg_map["max_model_len"] = config.context_length

    if config.kv_block_size > 0:
        arg_map["block_size"] = config.kv_block_size

    if config.extra_engine_args != "":
        json_map = {}
        # extra_engine_args is a filename
        try:
            with open(config.extra_engine_args) as f:
                json_map = json.load(f)
        except FileNotFoundError:
            logger.error(f"File {config.extra_engine_args} not found.")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {config.extra_engine_args}: {e}")
        logger.debug(f"Adding extra engine arguments: {json_map}")
        arg_map = {**arg_map, **json_map}  # json_map gets precedence

    return arg_map


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
        "--metrics-endpoint-port",
        type=int,
        default=DEFAULT_METRICS_ENDPOINT_PORT,
        help="Endpoint where vLLM publishes metrics for dynamo. For DP, we handle the port iteration, however if running multiple instances of a model this has to be changed.",
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
        "--context-length",
        type=int,
        default=None,
        help="Max model context length. Defaults to models max, usually model_max_length from tokenizer_config.json. Reducing this reduces VRAM requirements.",
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
        logger.error(
            f"Invalid endpoint format: '{args.endpoint}'. Expected 'dyn://namespace.component.endpoint' or 'namespace.component.endpoint'."
        )
        sys.exit(1)

    parsed_namespace, parsed_component_name, parsed_endpoint_name = endpoint_parts

    config.namespace = parsed_namespace
    config.component = parsed_component_name
    config.endpoint = parsed_endpoint_name
    config.metrics_endpoint_port = args.metrics_endpoint_port
    config.tensor_parallel_size = args.tensor_parallel_size
    config.kv_block_size = args.kv_block_size
    config.context_length = args.context_length
    config.extra_engine_args = args.extra_engine_args

    return config
